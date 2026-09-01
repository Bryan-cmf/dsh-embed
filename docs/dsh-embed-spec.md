# SPEC · dsh-embed 技術規格 v1.2

> 狀態：**待用戶評審**（spec-driven-development 工作流：SPECIFY→PLAN→TASKS→IMPLEMENT，本檔為 SPECIFY 產物）
> v1.1（t2 實現反饋）：§4 請求體 `backend` 字段；§4.2 批量池化公式；§8 torchvision。
> v1.2（t7 驗收反饋）：§6 融合策略改語義主導；§10 測量獨佔紀律。
> v1.3（t15/t16 修復反饋，2026-09-01）：§7 dim 按後端全維默認（text 2560 / visual 2048，MRL-512 雙雙實測劣勢降為選項）；§6 融合公式定稿 sem-primary+kw 補位（消融 96.2%）；§6.5 圖文雙路 RRF 否決（弱列表稀釋 84.6%<92.3%），圖像 @2048 定生產路，caption 合成列 Phase 3a 升級路徑。**系統級融合定律：弱列表不可等權融合（D1/D2 兩場景獨立驗證）。**
> 配套：[PRD](dsh-embed-prd.md) · [Plan](dsh-wemm-integration-plan.md) · 實驗證據 `dsh-wemm-poc/results/`

---

## 0. 假設（Assumptions，實施前請糾正）

1. sidecar 部署形態是**本機雙 Python 進程**，不是容器/服務集群（單用戶 Mac 工作站）
2. `dsh-embed` 是新的獨立 npm 插件（host composition 行）；`dsh-insights` 僅做 memory v2 消費改動，兩者通過 `embedder` 服務解耦
3. 模型權重預下載至 HF 緩存（已就位，47G），sidecar 離線啟動；升級模型屬手動操作
4. Phase 1/2 不修改任何 DSH 出貨插件（vendor/bundle），只動 Bryan-cmf 自有插件與 profile
5. macOS ≥ 14（M3 Ultra 實測環境）；不承諾 Linux/CUDA 路徑（vLLM 服務化屬未來擴展）

## 1. Objective

為 DSH 提供全本地、低延遲、多後端路由的嵌入服務，並以最小侵入接入記憶/技能/資產三個檢索面。成功標準見 §12；所有性能/質量數字以 Phase 0 實測為基準。

## 2. 架構與進程拓撲

```
┌─ DSH host (Node/Cordis) ──────────────────────────────┐
│  dsh-embed 插件（host composition 行）                  │
│   ├ provide('embedder')  ← dsh-insights 等注入消費      │
│   ├ 進程監管：spawn/health/keep-alive/restart           │
│   └ runtime 目錄 ~/.dsh/run/dsh-embed/                 │
│        ├ mlx.json  {port, token, pid}                  │
│        └ tf.json   {port, token, pid}                  │
└──────┬──────────────────────┬─────────────────────────┘
       │ 127.0.0.1 HTTP       │
┌──────▼──────────┐   ┌───────▼────────────┐
│ sidecar-mlx     │   │ sidecar-tf         │
│ venv-mlx        │   │ venv-tf            │
│ mlx 0.32.2      │   │ torch 2.13 +       │
│ mlx-vlm 0.6.17  │   │ transformers 5.2.0 │
│ WeMM-2B MLX-4bit│   │ Qwen3-4B fp16      │
│ (1.4GB)         │   │ + WeMM-2B fp16 fb  │
│ 文本+圖像        │   │ (8GB + 5.4GB 惰性) │
└─────────────────┘   └────────────────────┘
```

**為何雙進程**：transformers 必須鎖 5.2.0（WeMM 預處理一致性），與 mlx-vlm 的依賴樹存在衝突風險；venv 隔離是已驗證的安全解（`dsh-wemm-poc/.venv` 與 `.venv-mlx` 並存實證）。

**進程監管規則**：
- 懶啟動：首次 `embed*` 調用觸發 spawn（冷啟動實測：mlx ~8s / tf ~30s）
- 健康檢查：每 30s `GET /health`；連續 3 次失敗 → kill + 重啟（指數退避，上限 3 次）
- keep-alive：最後一次調用後 15 分鐘空閒關閉（可配）
- 握手：sidecar 啟動時把 `{port, token}` 寫入 runtime 文件；插件輪該文件完成發現（port=0 隨機分配）

## 3. `embedder` 服務契約（Host TS）

```ts
export interface BackendInfo {
  name: string              // 'wemm2b-mlx4b' | 'qwen3-4b-fp16' | 'wemm2b-fp16'
  model: string             // HF repo id（含社區轉換則鎖 hash）
  dims: number[]            // 支持的 MRL 維度，如 [64,128,256,512,1024,2048]
  modalities: ('text'|'image')[]
  fingerprint: string       // 如 'wemm2b-mlx4b@512'
  alive: boolean
}

export interface EmbedderService {
  backends(): Promise<BackendInfo[]>
  embedTexts(texts: string[], opts?: {
    backend?: string        // 默認 tf 側 'qwen3-4b-fp16'
    dim?: number            // 默認 512（MRL 截斷+重歸一）
    instruct?: string       // 僅 Qwen3 家族生效（query 側）
  }): Promise<Float32Array[]>
  embedImage(path: string, opts?: { backend?: string; dim?: number }): Promise<Float32Array>
  health(): Promise<{ mlx: 'up'|'down'|'starting'; tf: 'up'|'down'|'starting' }>
}
```

**失敗語義**：目標後端不可用 → 立即拋 `EmbedderUnavailableError`；調用方（dsh-insights）必須捕獲並降級 keyword。禁止自動跨後端替補（指紋不可互換，cos 漂移 0.968 實測）。

## 4. Sidecar HTTP API（兩進程同一契約）

```
POST /embed/texts   {texts: string[], dim?: int, instruct?: string, backend?: string}
  → 200 {vectors: number[][], fingerprint, dim, ms}
POST /embed/image   {path: string, dim?: int, backend?: string}
  → 200 {vector: number[], fingerprint, dim, ms}
GET  /backends      → BackendInfo[]
GET  /health        → {ok, uptime_s, backend, ready, loaded}
Header: X-Embed-Token: <隨機 32B hex；僅 127.0.0.1 監聽>
```

- `backend` 僅 tf 側雙後端路由需要（`qwen3-4b-fp16` / `wemm2b-fp16`）；mlx 側單後端忽略該字段（v1.1）

- 向量序列化 v1 用 JSON number[]（522×512 ≈ 5MB，可接受）；>2k 條批量再升級 base64-float32
- 請求上限：texts ≤64 條/次；image path 必須存在且 ≤30MB

### 4.1 sidecar-mlx 嵌入算法（官方語義復現，已驗證）

```
prompt = f"<|im_start|>user\n{text}<|im_end|>\n"
ids = tokenizer.encode(prompt)          # post-processor 自動追加 <embedding>(248077)
feats = model.get_input_embeddings(input_ids=ids, mask=None)
h = language_model.model(ids, inputs_embeds=feats.inputs_embeds,
                         mask=None, cache=None, position_ids=feats.position_ids)
v = l2_normalize(h[last])               # 最後非 pad 位（單條=序列末尾）
dim<full 時: v = l2_normalize(v[:dim])  # MRL 截斷+重歸一
```

圖像路徑：`prepare_inputs(processor, images=[img], prompts=[...含 image_pad...])` → `get_input_embeddings(pixel_values, image_grid_thw)`（實現細節見已驗證的 `scripts/run_eval_mlx.py`）。

### 4.2 sidecar-tf 嵌入算法

Qwen3 家族：官方 last-token pooling + L2 歸一——**批量 left-padding 時取 `h[:, -1]`**（注意：早期 PoC `mask.sum(-1)-1` 公式僅 batch=1 正確，批量時索引落入 pad 區；t2 已按官方公式修復，backfill/批量場景依賴此修復）；query 側拼 `instruct` 前綴，文檔側裸文本（對稱檢索按官方配方）。WeMM fp16 fallback：官方 `model.embedding(**inputs)`（惰性加載，首次調用 30-60s，調用方超時需容納）。

## 5. 數據模型（storageDomain）

```
vector_memory 域（dsh-insights 既有域內新表）
  table 'memory_vectors':  id → {fp: string, dim: int, vec: number[], ts: int}
                           # vec 存 float32 值（JSON number[]），512 維 ≈ 5KB/條

asset_index 域（Phase 3a，dsh-embed 自有；v1.3：圖像路定 @2048 全維——@512 對同版式頁 -5.8pp，圖文雙路 RRF 因弱列表稀釋被否決 84.6%<92.3%）
  table 'entries':  absPath → { mtime, modality: 'image'|'pdfpage',
                               doc?: string, page?: int,
                               vecText?: {fp, dim, vec},   # 頁文本 → tf 側 qwen3-4b@2560
                               vecImage?: {fp, dim, vec}}  # 頁圖像 → wemm2b-mlx4b@2048
  # 生產檢索：image@2048 主路；頁文本為實驗補充路（raw text 82.7% 弱於圖像）
  # 升級路徑（封面/空 text 頁為模型邊界，4 miss 記錄在案）：頁級 caption 合成
  # ——索引期視覺模型生成 caption 併入文本路（t16 救援對照表為證據基礎）
  table 'meta':     'stats' → {fps, count, lastBackfill}
  # v1.2 前置要求（F-QA-3）：MLX 冷 shape 編譯 3.7s/頁——首次全量索引前需
  # shape 分桶 padding 或預編譯熱身（117 頁 7.2min → 目標 ≤2min，PRD R7）

skill_index 域（Phase 3b，dsh-embed 自有）
  table 'vectors':  dir → {name, fp, dim, vec, mtime}
```

**指紋規則**：任何 `vec.fp ≠ 當前配置指紋` 的行視為過期——記憶表觸發單條異步重嵌；資產/技能表觸發整表重建 backfill。指紋 = `{backend}@{dim}`。

## 6. 混合檢索算法（dsh-insights memory v2）

```ts
// query 期（embedding.enabled = true 且 embedder 可用時）
kwHits   = 現行 score() 排序 top-100          // 不變，永遠計算
semHits  = cosine(qv, memory_vectors) top-100  // 後端 qwen3-4b-fp16@2560
fused    = sem 主列表 + kw 補位（v1.2 定型：sem 排名為骨架，kw 命中且 sem 未收錄者
           補入尾部——線上入口 fuseHybrid()。消融依據（52 查詢×522 記憶，Recall@5）：
           kw-only 23.1% / sem-only@2560 96.2% / 等權 RRF k60 90.4%（違反不變式，
           棄用）/ 加權 RRF w3 96.2% / sem-primary+kw補位 96.2%（同分取最簡，選定））
result   = fused top-N（附 kw/sem 各自來源標記）
// embedder 不可用或查詢向量缺失 → 純 kwHits（現行為，零退化）
```

寫入期：`mem_save` 存原文即返回 → 隊列 `{id, content+tags}` → 異步 `embedTexts`（批量 16 條/次）→ 寫 `memory_vectors`；失敗重試 3 次（5s/30s/120s）後放棄並記 health 事件，下次 backfill 兜住。

## 7. 配置面

```yaml
# dsh-embed（host composition 行 config）
mlxSidecar: {enabled: true,  venv: '~/.dsh/dsh-embed/venv-mlx', keepAliveSec: 900}
tfSidecar:  {enabled: true,  venv: '~/.dsh/dsh-embed/venv-tf',  keepAliveSec: 900,
             eagerBackends: ['qwen3-4b-fp16']}          # WeMM fp16 惰性加載
defaults:   {textBackend: 'qwen3-4b-fp16', visualBackend: 'wemm2b-mlx4b',
             textDim: 2560, visualDim: 2048}   # v1.3：dim 按後端全維默認——MRL-512
                                                # 對兩後端均有實測損失（text -9.7pp、
                                                # visual -5.8pp），截斷降為配置選項

# dsh-insights（memory 模組 config 擴展）
memory:
  embedding:
    enabled: false            # 灰度開關，默認 off
    backend: 'qwen3-4b-fp16'
    dim: 2560                 # v1.2：按後端默認（qwen3-4b → 2560；512 對其文本質量
                              # 有 -7.7pp 實測損失。WeMM 視覺側仍 512 為默認）
```

## 8. 項目結構與命令

```
dsh-embed/                      # 新插件倉庫（Phase 1 建）
  src/index.ts                  # host 入口：embedder 服務 + 監管
  src/sidecar/                  # 隨插件分發的 Python（tsx 直讀不了，構建時複製）
    mlx_serve.py  tf_serve.py   # sidecar 進程入口
    mlx_embed.py  tf_embed.py   # 嵌入算法（§4.1/4.2 移植自已驗證腳本）
  cordis.patch.yml
  package.json                  # peerDeps: cordis / dsh-tools（照 dsh-insights 模式）

dsh-insights/                   # Phase 2 改動僅三處
  src/host/memory.ts            # 混合檢索 + 異步隊列（inject embedder，optional）
  src/host/domains.ts           # memory_vectors 表註冊
  cordis.patch.yml              # memory.embedding config

dsh-wemm-poc/                   # 既有評測資產 → 回歸集（保持只讀）
  scripts/run_regression.py     # Phase 1 新增：走 sidecar HTTP 的端到端回歸
```

```bash
# 環境裝配（一次性，腳本化進 dsh-embed/scripts/bootstrap.sh）
uv venv ~/.dsh/dsh-embed/venv-mlx --python 3.12
uv pip install --python ~/.dsh/dsh-embed/venv-mlx/bin/python mlx==0.32.2 mlx-vlm==0.6.17
uv venv ~/.dsh/dsh-embed/venv-tf --python 3.12
uv pip install --python ~/.dsh/dsh-embed/venv-tf/bin/python torch transformers==5.2.0 \
  torchvision==0.28.0 'qwen-vl-utils==0.0.14' numpy

# 開發/測試（dsh-embed 倉庫內）
pnpm bundle && pnpm test          # 單元（RRF/指紋/監管狀態機 mock HTTP）
dsh-wemm-poc/.venv/bin/python scripts/run_regression.py   # 端到端三數據集

# 手工冒煙
curl -s -H "X-Embed-Token: $(jq -r .token ~/.dsh/run/dsh-embed/mlx.json)" \
  "http://127.0.0.1:$(jq -r .port ~/.dsh/run/dsh-embed/mlx.json)/health"
```

## 9. 代碼風格（示例即規約）

```ts
// TS：服務方法可失敗即顯式可失敗；不跨後端隱式替補
const qv = await embedder.embedTexts([q], { dim: 512 })      // may throw EmbedderUnavailableError
  .catch(() => null)                                          // 調用方顯式降級
if (qv) hits = fuse(kwHits, cosine(qv[0], vecTable))
```

```python
# Python sidecar：無全局狀態、無熱路徑分配、錯誤帶 fingerprint 上下文
@app.post('/embed/texts')
def embed_texts(req: TextsReq):
    vecs = backend.embed_texts(req.texts, req.dim)   # 全部拋 EmbedError(code, fingerprint)
    return {'vectors': [v.tolist() for v in vecs], 'fingerprint': backend.fp, ...}
```

## 10. 測試策略

| 層 | 內容 | 位置 |
|---|---|---|
| 單元 | RRF 融合、MRL 截斷重歸一、指紋過期判定、監管狀態機（spawn/crash/backoff，HTTP mock） | `dsh-embed/test/*.test.ts` |
| 契約 | sidecar HTTP schema（兩進程共用同一組用例）、token 鑑權、127.0.0.1 only | `dsh-embed/test/contract.test.ts` |
| 端到端回歸 | Phase 0 三數據集走真實 sidecar：D1≥93 / D2≥90 / D3≥91（基線-2pp；D2 為**圖文雙路**口徑，v1.2）；延遲預算：文本 p95≤50ms（mlx）/≤60ms（tf）、圖像 ≤1s/頁 | `dsh-wemm-poc/scripts/run_regression.py` |
| 測量紀律 | v1.2（F-QA-4）：延遲測量必須 sidecar 獨佔機器（GPU 共享實測放大 4-6 倍）；回歸/驗收報告須標註測量條件 | 同上 |
| 降級 | embedder 停機時 mem_search 純 kw 輸出與舊版逐字節一致；flag off 同 | `dsh-insights/test/memory.test.ts` |

## 11. 回滾

| 層 | 動作 |
|---|---|
| 秒級 | `memory.embedding.enabled=false`（config patch，HMR 生效） |
| 分鐘級 | 插件行 disable / npm 版本回退（dsh-insights pin 舊版） |
| 後端級 | `visualBackend` 從 mlx 切 transformers fp16（同模型官方路徑，重嵌資產/技能索引，記憶表不受影響——文本後端未變） |
| 徹底 | 卸載 dsh-embed；memory_vectors 表棄置不清理也無害（無人讀） |

## 12. 成功標準（可測試）

1. 回歸三數據集全部 ≥ 基線-2pp（§10 表）
2. `mem_save` 延遲增加 <10ms；`mem_search` p95 ≤400ms（含嵌入與融合）
3. sidecar 混亂測試（kill -9 ×3）後 5 分鐘內自愈，期間 mem_search 持續可用（kw 降級）
4. 空閒 15 分鐘後 sidecar 退出，RSS 歸零；再調用 30s 內恢復服務
5. 出網流量嗅探為零（除顯式模型下載操作）

## 13. Boundaries

- **Always**：回歸集綠燈才合併；指紋變更必須帶 backfill 計劃；錯誤消息帶 fingerprint；127.0.0.1 + token
- **Ask first**：改 storageDomain schema、升級 transformers/mlx-vlm 版本、動 dsh-insights 非 memory 模組、擴查詢標註集
- **Never**：接入 hosted embedder、跨後端向量混存同一索引、修改 vendor/ 出貨插件、繞過 flag 直接改 mem_search 行為

## 14. 任務分解（Phase 1，供 PLAN→TASKS 階段細化）

1. dsh-embed 插件骨架 + embedder 服務（mock sidecar）＋單元測試
2. sidecar-mlx（移植 `mlx_embed.py`）＋契約測試
3. sidecar-tf ＋契約測試
4. 進程監管狀態機 + runtime 握手文件
5. `run_regression.py` 端到端回歸接入三數據集
6. （並行可做）回歸查詢集擴至 ≥50 條/場景

> ⚠️ Phase 1 動工前必須先載入 `editing-cordis-compositions` skill 再寫任何 composition；dsh-insights 發版遵循其現有 CHANGEFLOW（改 workspace 倉庫 → npm 發版 → 更新 web profile 引用）。
