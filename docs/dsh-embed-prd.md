# PRD · dsh-embed：DSH 本地語義嵌入基礎設施

> 版本 v1.0（2026-09-01）· 狀態：**待用戶評審** · 配套：[Plan](dsh-wemm-integration-plan.md) · [SPEC](dsh-embed-spec.md)
> 所有基線數據來自 Phase 0 四輪實測（`dsh-wemm-poc/results/`），非估計值。

---

## 1. 背景與問題

DSH（DeepSeek Harness）當前的三個檢索面全部是確定論關鍵詞匹配：

| 檢索面 | 現狀 | 實測召回 | 痛點 |
|---|---|---|---|
| 記憶 `mem_search` | token overlap + recency（`dsh-insights/src/host/memory.ts`） | **35%** | 改寫查詢完全失效：「沙箱被拒的教訓」搜不到 "permission denied" |
| 財報/文檔 | 無產品化檢索 | 17%（頁文本 kw） | 117 頁真實港股報告裏找「核數師是哪家」基本靠翻 |
| 技能路由 | 模型通讀 150+ 描述 | 36%（kw 模擬） | 每輪燒數千 token context，且匹配不準 |

同時存在兩條硬約束：**數據不出機**（財報、記憶含敏感內容，禁用 hosted embedder）；**現有 keyword 行為不可退化**（兜底必須永遠在）。

## 2. 用戶與場景

**U1 · DSH agent（主用戶）**——在長會話中需要：
- *US-1 記憶召回*：「上次 FS_STALE_VERSION 怎麼解的？」→ 語義命中 522 條存量記憶（含中英混合、無共同詞彙）
- *US-2 技能匹配*：「幫我調研這個開源項目」→ 語義定位 `agent-reach`，替代每輪全文目錄掃描
- *US-3 資產檢索*：「找講現金流折現的那頁圖表」→ 文本查詢直接檢索 PDF 頁圖像（**無需 OCR**，WeMM 視覺文檔能力）

**U2 · 人類用戶（Bryan）**——港股研究工作流：本地報告庫（ak-sdd-web 的數十份年報/中報/ESG）語義檢索；對話裏的截圖/圖表回找。

## 3. 產品目標

交付一個 **host 插件 `dsh-embed`**（發布 `embedder` 服務 + 本地 sidecar 進程群），以及三個消費場景（記憶 v2 / 技能工具 / 資產索引）。

### 非目標
- ❌ 不做視頻嵌入（decord/macOS 風險，需求未出現）
- ❌ 不做向量數據庫選型（storageDomain JSON 在現有量級足夠，>5k 條另案遷 sqlite）
- ❌ 不做 hosted embedder 接入（隱私硬約束）
- ❌ 不攔截/改寫系統提示注入技能目錄（改為顯式工具）
- ❌ 不做 4B/9B WeMM 縮放（縮放實驗已否決）

## 4. 需求分級

### P0（Phase 1+2，Gate G1→G3）

| ID | 需求 | 驗收標準 |
|---|---|---|
| R1 | `embedder` 服務：`embedTexts` / `embedImage` / `backends` / `health` | 後端可枚舉、可路由（`{backend, dim}` 參數）；SPEC §3 契約 |
| R2 | 雙 sidecar：MLX-4bit（WeMM 視覺/技能）+ transformers-fp16（Qwen3-4B 文本、WeMM fallback） | 進程懶啟動、健康檢查、keep-alive 窗口、崩潰自動重啟；合計內存 ≤12GB |
| R3 | 記憶 v2 混合檢索 | 對 Phase 0 D1 集 Recall@5 ≥90%（現 35%）；`mem_search` p95 ≤400ms（含嵌入）；keyword 結果永遠保留在融合候選池 |
| R4 | 異步嵌入寫入路徑 | `mem_save` 返回延遲不因嵌入增加（>10ms 視為違約）；隊列失敗重試；backfill 一鍵補全存量 |
| R5 | 灰度與回滾 | `memory.embedding.enabled` 默認 **off**；off 時行為與現版 byte-level 等價 |
| R6 | 回歸測試集 | Phase 0 三數據集接入：D1≥93 / D2≥90 / D3≥91（基線-2pp），每次 sidecar/插件變更必跑 |

### P1（Phase 3）

| ID | 需求 | 驗收標準 |
|---|---|---|
| R7 | `asset_search` 工具：文本查詢 → 工作區圖片/PDF 頁命中 | D2 集 Recall@5 ≥90%；117 頁索引耗時 ≤2 分鐘；文件 mtime 變化觸發增量重嵌 |
| R8 | `skill_semantic_search` 工具 | D3 集 Recall@3 ≥90%；技能目錄 mtime 變化觸發重嵌；返回 top-k 帶分數 |

### P2（暫緩）
- 嵌入看板 UI（記憶/資產索引狀態可視化）
- 多工作區資產索引共享
- MLX 權重官方化後的自動遷移工具

## 5. 成功指標（上線後驗收）

| 指標 | 基線 | 目標 |
|---|---|---|
| 記憶語義召回（D1 集） | 35% | **≥90%** |
| 報告頁檢索（D2 集） | 17% | **≥90%** |
| 技能匹配（D3 集） | 36% | **≥90%** |
| `mem_search` p95 | ~5ms（kw） | ≤400ms（混合，用戶可接受） |
| `mem_save` 延遲 | ~5ms | 不增加（異步） |
| 隱私 | — | 出網流量為零（僅模型下載一次性） |
| 進程穩定性 | — | sidecar 連續 3 天無人工干預（Gate G2） |

## 6. 依賴與風險（摘要，全文見 Plan §4）

- 依賴：Qwen3-Embedding-4B（Apache-2.0）、WeMM-Embedding-2B（Apache-2.0 官方權重 + hfadam MLX 轉換）、mlx 0.32.2 / mlx-vlm 0.6.17 / transformers 5.2.0 / torch 2.13
- 最大風險：MLX 嵌入路徑為官方語義的復現實現 → 保留 transformers 官方路徑一鍵 fallback（SPEC §11）
- **R7 前置風險（t7 實測新增）**：MLX 圖像冷 shape 編譯 3.7s/頁，首次索引 117 頁 ~7.2min 超 R7 目標 ≤2min——Phase 3a 動工前必須先做 shape 分桶/預編譯熱身（熱態 0.42s/頁 已達標）

## 7. 開放問題（需用戶輸入）

1. Phase 2 灰度策略：先在哪些 session/preset 開 `embedding.enabled`？（建議：僅當前 financial-advisor preset 試運行一週）
2. `asset_search` 索引範圍：僅 workspace，還是含 `~/Desktop/Projects/ak-sdd-web` 報告庫？（後者建 P1 內做路徑白名單配置）
3. 是否需要為 46 條標註查詢擴集到 ≥50 條/場景以提高回歸置信度？（建議 Phase 1 順帶完成）
