/**
 * dsh-embed · 服務契約類型（SPEC dsh-embed-spec.md §3/§4）。
 *
 * 本檔只放跨模組共享的型別與常量;實現在 service.ts / supervisor.ts。
 */

/** sidecar 進程標識:mlx(MLX-4bit WeMM)/ tf(transformers fp16)。 */
export type SidecarId = 'mlx' | 'tf'

/** sidecar 啟動握手文件內容(由 sidecar 寫入,插件輪詢發現)。SPEC §2。 */
export interface RuntimeHandshake {
  port: number
  token: string
  pid: number
}

/** 後端描述。SPEC §3 BackendInfo。 */
export interface BackendInfo {
  /** 如 'wemm2b-mlx4b' | 'qwen3-4b-fp16' | 'wemm2b-fp16'。 */
  name: string
  /** HF repo id(社區轉換權重由 sidecar 鎖 hash)。 */
  model: string
  /** 支持的 MRL 維度(poc 實測:WeMM-2B 全維 2048,Qwen3-4B 全維 2560)。 */
  dims: number[]
  modalities: ('text' | 'image')[]
  /** 指紋 `{backend}@{dim}`,如 'wemm2b-mlx4b@512'。跨後端不可互換。 */
  fingerprint: string
  alive: boolean
}

export interface EmbedTextsOptions {
  /** 默認 config.defaults.textBackend('qwen3-4b-fp16')。 */
  backend?: string
  /** 默認 config.defaults.dim(512;MRL 截斷+重歸一)。 */
  dim?: number
  /** 僅 Qwen3 家族生效(query 側任務前綴);其他後端傳入即拋驗證錯。 */
  instruct?: string
}

export interface EmbedImageOptions {
  /** 默認 config.defaults.visualBackend('wemm2b-mlx4b')。 */
  backend?: string
  dim?: number
}

/** `embedder` 服務契約。SPEC §3 EmbedderService。 */
export interface EmbedderService {
  backends(): Promise<BackendInfo[]>
  embedTexts(texts: string[], opts?: EmbedTextsOptions): Promise<Float32Array[]>
  embedImage(path: string, opts?: EmbedImageOptions): Promise<Float32Array>
  health(): Promise<{ mlx: 'up' | 'down' | 'starting'; tf: 'up' | 'down' | 'starting' }>
}

// ── 契約常量(SPEC §4 請求上限) ────────────────────────────────────────────

/** 單次 /embed/texts 上限。 */
export const MAX_TEXTS_PER_REQUEST = 64
/** 單張圖像文件上限(30MB)。 */
export const MAX_IMAGE_BYTES = 30 * 1024 * 1024

/** sidecar HTTP API 路由(SPEC §4)。 */
export const SIDECAR_ROUTES = {
  embedTexts: '/embed/texts',
  embedImage: '/embed/image',
  backends: '/backends',
  health: '/health',
} as const
