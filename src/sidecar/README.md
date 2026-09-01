# dsh-embed sidecar 契約(供 sidecar-mlx / sidecar-tf 實現,Phase 1 · T2)

本目錄存放隨插件分發的 Python 進程入口(`mlx_serve.py` / `tf_serve.py`)
與嵌入算法模組(`mlx_embed.py` / `tf_embed.py`)。host 端(supervisor.ts)
按以下契約 spawn 與通信——**實現方(py-engineer)與 host 方(ts-engineer)
共同遵守,任何變更需同步修改兩側**。

## 1. 進程 CLI 契約

supervisor 以如下命令行啟動 sidecar(detached 進程組,stdio 重定向到
`<runtimeDir>/<id>.log`):

```
<venv>/bin/python <script> --runtime-file <abs/path/to/{mlx|tf}.json> [--eager name1,name2]
```

- `--runtime-file`(必填):握手文件路徑。sidecar **綁定 `127.0.0.1:0`**
  (隨機端口)成功後,把 `{"port": <int>, "token": "<hex32>", "pid": <int>}`
  **原子寫入**該文件(先寫臨時文件再 rename)。
  - `token`:隨機 32 bytes hex(64 字符),每個進程實例唯一
  - `pid`:sidecar 自身 pid
  - 寫入時機:HTTP server 開始 listen 之後(不要求模型加載完成);
    host 會輪詢該文件並反覆 `GET /health` 直到 `ok:true`
- `--eager`(可選):插件啟動即需加載的後端名列表(逗號分隔);
  tf 側默認 `qwen3-4b-fp16`,WeMM fp16 保持惰性
- 僅監聽 `127.0.0.1`;對任何缺/錯 `X-Embed-Token` 的請求返回 401
- 退出語義:收到 SIGTERM 應及時優雅退出(<5s,host 寬限期後 SIGKILL)

## 2. HTTP API(SPEC §4;兩進程同一契約)

```
POST /embed/texts   {texts: string[], dim?: int, instruct?: string, backend?: string}
  → 200 {vectors: number[][], fingerprint: string, dim: int, ms: int}
POST /embed/image   {path: string, dim?: int, backend?: string}
  → 200 {vector: number[], fingerprint: string, dim: int, ms}
GET  /backends      → BackendInfo[]
GET  /health        → {ok: boolean, uptime_s: number, backend: string}
Header(所有請求): X-Embed-Token: <token>
```

錯誤返回 `400/401/500` + `{"error": "<message>", "fingerprint": "<若已知>"}`。

### 2.1 `backend` 字段(契約擴展,host 端已實現)

tf sidecar 同進程服務兩個後端(`qwen3-4b-fp16` 默認 + `wemm2b-fp16`
惰性 fallback);請求體可帶可選 `backend` 顯式選擇。mlx sidecar 單後端
`wemm2b-mlx4b`,字段冗餘但必須接受(不匹配時返回 400)。

### 2.2 `instruct` 字段

僅 Qwen3 家族生效(query 側);sidecar 按官方配方拼前綴
`Instruct: Given a query, retrieve relevant {instruct}\nQuery: {text}`
(文檔側裸文本)。非 Qwen3 後端收到 `instruct` 應返回 400
(host 端亦已前置攔截)。

## 3. 請求限制(host 與 sidecar 雙側強制)

- `texts` ≤ 64 條/次;每條非空字符串
- `image path` 必須存在且 ≤ 30MB
- 向量序列化 v1 用 JSON `number[]`(522×512 ≈ 5MB,可接受)

## 4. 後端目錄(與 host catalog.ts 一致)

| name            | model                              | dims      | modalities    | sidecar |
|-----------------|------------------------------------|-----------|---------------|---------|
| `wemm2b-mlx4b`  | `hfadam/WeMM-Embedding-2B-MLX-4bit`| 512, 2048 | text, image   | mlx     |
| `qwen3-4b-fp16` | `Qwen/Qwen3-Embedding-4B`          | 512, 2560 | text          | tf      |
| `wemm2b-fp16`   | `tencent/WeMM-Embedding-2B`        | 512, 2048 | text, image   | tf      |

`fingerprint` 恒為 `{backend}@{dim}`(如 `wemm2b-mlx4b@512`)。
MRL:`dim <` 全維時截斷 + 重新 L2 歸一(host 端會再做一次,冪等)。

## 5. `GET /backends` 返回示例

```json
[
  {"name": "qwen3-4b-fp16", "model": "Qwen/Qwen3-Embedding-4B",
   "dims": [512, 2560], "modalities": ["text"],
   "fingerprint": "qwen3-4b-fp16@512", "alive": true},
  {"name": "wemm2b-fp16", "model": "tencent/WeMM-Embedding-2B",
   "dims": [512, 2048], "modalities": ["text", "image"],
   "fingerprint": "wemm2b-fp16@512", "alive": false}
]
```

`alive` = 該後端模型當前已加載可服務(惰性後端 false)。

## 6. 環境裝配見 `scripts/bootstrap.sh`(uv venv 雙隔離,SPEC §8)
