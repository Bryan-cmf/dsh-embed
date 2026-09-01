# WeMM-Embedding × DSH 引進 · 總體規劃 v3

> 日期：2026-09-01 · 狀態：**Phase 0 完成，選型鎖定，PRD/SPEC 已分離，待用戶批准 Phase 1 開工**
> 文檔體系：本檔（路線圖與決策記錄）→ [`dsh-embed-prd.md`](dsh-embed-prd.md)（產品需求）→ [`dsh-embed-spec.md`](dsh-embed-spec.md)（技術規格）→ `dsh-wemm-poc/results/`（實驗證據）

---

## 1. 四輪實驗 → 四個決策（全部實測閉環）

| # | 問題 | 實驗 | 決策 | 證據 |
|---|---|---|---|---|
| 1 | WeMM 對 DSH 有用嗎？ | Phase 0 三場景評測 | 記憶/視覺/技能三場景語義檢索全部立項；**分場景選型**而非單模型 | kw 基線 35/17/36% → 語義 90/92/93% |
| 2 | 大模型更好嗎？ | 2B/4B/9B + 0.6B/4B 縮放 | 視覺場景 **2B 是甜蜜點**（4B 反降 17pp）；文本場景升級到 **Qwen3-4B**（95%，37ms）；9B 全線不值 | `results_scaling.md` |
| 3 | 精度怎麼選？ | fp32/fp16/bf16 | **fp16**：質量零差異、比 bf16 快 23%、漂移更小；fp32 無收益；FP8 本機無硬件 | `results_precision.md` |
| 4 | MLX 4-bit 呢？ | MLX 4-bit 全量復測 | **WeMM 視覺+技能後端改用 MLX 4-bit**：質量零損失、文本快 8-17 倍、圖像快 1.9 倍、內存省 3.9 倍 | `results_mlx.md` |

## 2. 鎖定選型

| 角色 | 後端 | 質量（實測） | 延遲（實測） | 內存 |
|---|---|---|---|---|
| 在線文本（記憶/一般檢索） | **Qwen3-Embedding-4B @fp16**（transformers） | D1 95% / MRR 0.82 | p95 ~37ms | 8GB |
| 視覺文檔 + 技能匹配 | **WeMM-Embedding-2B @MLX-4bit** | D2 92% · D3 93%（與 bf16 零差異） | 文本 8-17ms · 圖像 417ms/頁 | **1.4GB** |
| 官方 fallback | WeMM-2B @fp16（transformers） | 同 MLX | 慢 2-8 倍 | 5.4GB |

- 存儲維度：**MRL-512**（三輪實驗均與全維持平）
- 向量指紋：`{backend, model, dim}`，**跨後端不可互換**（MLX vs bf16 cos 漂移 0.968）→ 換後端必須重建索引
- 硬約束：全本地推理，數據不出機（財報/記憶隱私）

## 3. 路線圖（Gate 制，每關用戶審批）

```
✅ Phase 0 評測（4 輪實驗，全部通過，資產存 dsh-wemm-poc/）
─── Gate G1（已過，等用戶批准開工）───
Phase 1 · dsh-embed 插件 + 雙 sidecar（1.5-2 天）
  ├ dsh-embed host 插件（embedder 服務 / 進程監管 / 指紋）
  ├ sidecar-mlx：WeMM MLX 4-bit（產品化 mlx_probe 的前向實現）
  ├ sidecar-tf：Qwen3-4B fp16 + WeMM fp16 fallback
  └ 驗收 G2：回歸三數據集 ≥ 基線-2pp；sidecar 連續 3 天無人工干預存活
─── Gate G2 ───
Phase 2 · 記憶 v2（dsh-insights，1 天）
  ├ 異步嵌入隊列 + memory_vectors 表 + RRF 混合檢索（flag 默認 off）
  ├ 存量 522 條 backfill；keyword 永遠兜底
  └ 發版 @bryan-cmf/dsh-insights → 更新 web profile
─── Gate G3（灰度驗收）───
Phase 3 · 消費場景（各自獨立，可拆）
  ├ 3a 視覺資產索引 + asset_search 工具（2-3 天）
  └ 3b skill_semantic_search 工具（0.5-1 天）
```

## 4. 風險登記簿（v3 修訂）

| 風險 | 等級 | 緩解 |
|---|---|---|
| MLX 前向為復現實現（非官方代碼路徑） | 中 | 質量已驗證持平；保留 transformers 官方 fallback 一鍵切換 |
| hfadam 社區轉換權重無官方背書 | 中 | 鎖版本+權重 hash；可隨時用官方權重自轉（convert 腳本已驗證標準 mlx-vlm 路徑） |
| transformers 5.2.0 與 mlx-vlm 依賴衝突 | 高（已規避） | 雙 venv 雙進程隔離（SPEC §2） |
| 小樣本結論（12-20 查詢/集） | 中 | 方向已閉環；Phase 1 回歸集納入 CI 式驗證，後續擴查詢集 |
| storageDomain JSON 膨脹 | 低→中 | 512 維；>5k 條或 >20MB 遷 sqlite（屆時另案） |
| 9B/4B 未來誘惑 | — | 已有縮放曲線數據否決；除非任務分佈變化 |

## 5. 實驗證據索引（dsh-wemm-poc/）

| 文件 | 內容 |
|---|---|
| `results/results.md` + `results.json` | Phase 0 主評測（三場景五系統） |
| `results/results_scaling.md/.json` | 模型縮放實驗（2B/4B/9B、0.6B/4B） |
| `results/results_precision.md/.json` | 精度實驗（fp32/fp16/bf16 + FP8/INT8 可行性結論） |
| `results/results_mlx.md/.json` | MLX 4-bit 全量復測 |
| `data/d1..d3_*` | 三個真實語料數據集 + 46 條標註查詢（回歸集基礎） |
| `scripts/*.py` | 全套可複現腳本（含嵌入緩存） |

## 6. 歷史決策記錄

- **v1（對話輪 3）**：初版口頭方案，重審發現 12 項遺漏（對照組/落點/侵入性/寫入路徑等）
- **v2（同日）**：重審修訂 + Phase 0 執行；Gate G1 四項全過；初選 0.6B+2B
- **v2.1**：縮放實驗 → 改選 Qwen3-4B + WeMM-2B（13.4GB）
- **v2.2**：精度實驗 → 計算精度 bf16→fp16
- **v3（本檔）**：MLX 實驗 → 視覺/技能後端改 MLX 4-bit（9.4GB）；文檔三分（Plan/PRD/SPEC）；v2 附錄的實驗記錄收斂至 §1 決策表
