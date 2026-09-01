# @bryan-cmf/dsh-embed

DSH host 插件:**全本地語義嵌入基礎設施**。發布 `embedder` 服務(文本/圖像嵌入、多後端路由、MRL 維度),背後由進程監管器管理雙 Python sidecar。零出網(僅 127.0.0.1 + token)、零外部向量庫。

> 配套文檔:[Plan](docs/dsh-wemm-integration-plan.md) · [PRD](docs/dsh-embed-prd.md) · [SPEC](docs/dsh-embed-spec.md) · 實驗證據 `../dsh-wemm-poc/results/`

## 架構(SPEC §2)

```
┌─ DSH host (Node/Cordis) ──────────────────────────────┐
│  dsh-embed(host composition 行,發布 'embedder' 服務)  │
│   ├ embedTexts / embedImage / backends / health        │
│   ├ 進程監管:懶啟動/握手/健康/退避重啟/keep-alive     │
│   └ runtime 目錄 ~/.dsh/run/dsh-embed/{mlx,tf}.json    │
└──────┬──────────────────────┬─────────────────────────┘
       │ 127.0.0.1 + X-Embed-Token
┌──────▼──────────┐   ┌───────▼────────────┐
│ sidecar-mlx     │   │ sidecar-tf         │
│ venv-mlx        │   │ venv-tf            │
│ WeMM-2B MLX-4bit│   │ Qwen3-4B fp16      │
│ 視覺+技能(文本) │   │ + WeMM-2B fp16 fb  │
└─────────────────┘   └────────────────────┘
```

### 後端目錄(Phase 0 選型鎖定)

| name | model | dims | 用途 | sidecar |
|---|---|---|---|---|
| `wemm2b-mlx4b` | `hfadam/WeMM-Embedding-2B-MLX-4bit` | 512, 2048 | 視覺文檔 + 技能匹配 | mlx |
| `qwen3-4b-fp16` | `Qwen/Qwen3-Embedding-4B` | 512, 2560 | 在線文本(記憶/一般檢索) | tf |
| `wemm2b-fp16` | `tencent/WeMM-Embedding-2B` | 512, 2048 | MLX 復現路徑的官方 fallback | tf |

指紋 = `{backend}@{dim}`,**跨後端不可互換**(MLX vs bf16 cos 漂移 0.968 實測)→ 換後端必須重建索引。

## 服務契約(SPEC §3)

```ts
interface EmbedderService {
  backends(): Promise<BackendInfo[]>
  embedTexts(texts: string[], opts?: { backend?: string; dim?: number; instruct?: string }): Promise<Float32Array[]>
  embedImage(path: string, opts?: { backend?: string; dim?: number }): Promise<Float32Array>
  health(): Promise<{ mlx: 'up'|'down'|'starting'; tf: 'up'|'down'|'starting' }>
}
```

**失敗語義**:目標後端不可用 → 立即拋 `EmbedderUnavailableError`,調用方(dsh-insights memory v2)捕獲並降級 keyword。**禁止自動跨後端替補**。`instruct` 僅 Qwen3 家族生效。請求上限:texts ≤64 條/次、圖像 ≤30MB。

**輸入驗證**(`EmbedderValidationError`,永久性、不被降級路徑捕獲):未知後端名、texts 超限/空、dim 非法或不屬於該後端 catalog 支持維度(預校驗擋掉顯然非法 dim 免一次往返;sidecar `/embed` 端仍為最終權威)。

## 進程監管規則(SPEC §2)

- **懶啟動**:首次 embed 調用觸發 spawn(冷啟動 mlx ~8s / tf ~30s);`eagerBackends` 打破懶啟動——tf 默認 `['qwen3-4b-fp16']` 預熱(SPEC §7),mlx 默認懶啟動
- **握手**:sidecar 綁 127.0.0.1:0 後把 `{port, token, pid}` 原子寫入 runtime 文件,插件輪詢發現 + `/health` 就緒
- **孤兒收養**:host 重啟後發現存活健康的同位 sidecar → 直接收養,不重複 spawn
- **健康檢查**:每 30s `GET /health`;連續 3 次失敗 → kill + 重啟(指數退避 1s/4s,上限 3 次)
- **failed 冷卻**:連續 3 次啟動失敗 → 冷卻 30s,期間需求立即拋;冷卻後自愈(kill -9 混亂測試 5 分鐘內恢復)
- **keep-alive**:最後一次調用後 `keepAliveSec`(默認 900s)空閒 → SIGTERM(寬限 5s→SIGKILL),RSS 歸零;再調用即恢復
- **sidecar 自帶空閒看門狗**(host keep-alive + 60s)作孤兒兜底

## 配置(SPEC §7)

```yaml
# host composition 行 config(默認值;全部可覆寫)
runtimeDir: '~/.dsh/run/dsh-embed'
defaults: { textBackend: 'qwen3-4b-fp16', visualBackend: 'wemm2b-mlx4b', dim: 512 }
mlxSidecar: { enabled: true, venv: '~/.dsh/dsh-embed/venv-mlx', keepAliveSec: 900, eagerBackends: [] }
tfSidecar:  { enabled: true, venv: '~/.dsh/dsh-embed/venv-tf',  keepAliveSec: 900, eagerBackends: ['qwen3-4b-fp16'] }
healthIntervalMs: 30000        # 健康檢查間隔
healthFailureLimit: 3          # 連敗殺線
startupTimeoutMs: 180000       # 握手+就緒窗口(tf 冷啟動 30s 留餘量)
requestTimeoutMs: 120000       # 單次嵌入超時
backoffBaseMs: 1000            # 退避基數(×4 遞增)
maxRestartAttempts: 3
```

eager 預設與 SPEC §7 一致:**tf 默認預熱 `['qwen3-4b-fp16']`**(文本後端常駐就緒,免首次 embedTexts ~30s 冷啟動;WeMM fp16 fallback 仍惰性),**mlx 默認 `[]` 懶啟動**。要全懶啟動可顯式 `tfSidecar.eagerBackends: []`。

## 安裝與接線

```bash
# 1) 環境裝配(雙 venv,一次性)
bash scripts/bootstrap.sh
# 2) 構建(含 sidecar .py 複製到 lib/sidecar/)
pnpm build && pnpm test
# 3) host composition 加行(cordis.patch.yml 的 insert 已備好)
#    - id: dsh-embed
#      name: '@bryan-cmf/dsh-embed'
# 4) 契約煙霧(隔離 runtime,--fake 後端,無需權重)
pnpm smoke
```

**本插件發布 `embedder` 服務,必須掛 HOST composition**(跨 session 消費),不可鬆放進 per-session preset(見 cordis.patch.yml 註釋)。

## 測試

| 層 | 內容 | 命令 |
|---|---|---|
| 單元 | RRF 融合、MRL 截斷重歸一、指紋、runtime 文件、監管狀態機(假時鐘+真 HTTP mock)、服務路由/驗證/降級、入口裝配 | `pnpm test` |
| 契約煙霧 | 真 sidecar(--fake)× 生產客戶端:CLI/握手/HTTP/指紋/鑑權 | `pnpm smoke` |
| 端到端回歸 | Phase 0 三數據集走真實 sidecar + 權重 | `../dsh-wemm-poc/scripts/run_regression.py`(qa) |

## 消費方式(dsh-insights memory v2 等)

```ts
// ctx.get('embedder') 讀取可選服務;顯式降級:
const embedder = ctx.get('embedder')
const qv = embedder ? await embedder.embedTexts([query], { dim: 512 })
  .catch(() => null) : null   // EmbedderUnavailableError → 純 keyword
```

## 回滾(SPEC §11)

- 停用 sidecar:`mlxSidecar.enabled: false` / `tfSidecar.enabled: false`(embedder 即拋 unavailable,消費方降級)
- 卸載:移除 host composition 行;`memory_vectors` 表棄置無害(無人讀)
- 後端級:defaults.visualBackend 從 `wemm2b-mlx4b` 切 `wemm2b-fp16`(同模型官方路徑;**須重嵌**視覺/技能索引)

## License

MIT
