/**
 * dsh-embed · sidecar HTTP 客戶端(SPEC §4 契約)。
 *
 * 兩個 sidecar 進程共用同一契約:
 *   POST /embed/texts {texts, dim?, instruct?, backend?} → {vectors, fingerprint, dim, ms}
 *   POST /embed/image  {path, dim?, backend?}             → {vector, fingerprint, dim, ms}
 *   GET  /backends  → BackendInfo[]
 *   GET  /health    → {ok, uptime_s, backend}
 * 鑑權:每請求帶 `X-Embed-Token`;sidecar 僅監聽 127.0.0.1。
 *
 * 錯誤映射(SPEC §3/§9):
 * - 網絡層失敗/非 2xx → EmbedderUnavailableError(帶指紋上下文,若已知)
 * - 回應形狀/指紋不符 → EmbedderResponseError
 * - health/backends 為探測語義,失敗返回 null(調用方自行計數/降級)
 *
 * 契約擴展約定(記錄於 src/sidecar/README.md):請求體可帶可選 `backend`
 * 字段(tf sidecar 同進程服務 'qwen3-4b-fp16' 與 'wemm2b-fp16' 兩後端,
 * 需顯式選擇;mlx sidecar 單後端,字段冗餘但自證)。
 */
import { EmbedderResponseError, EmbedderUnavailableError } from './errors.ts'
import { SIDECAR_ROUTES, type BackendInfo } from './types.ts'

export type FetchLike = (input: string, init?: {
  method?: string
  headers?: Record<string, string>
  body?: string
  signal?: AbortSignal
}) => Promise<{ ok: boolean; status: number; json: () => Promise<unknown> }>

export interface EmbedTextsResult {
  vectors: number[][]
  fingerprint: string
  dim: number
  ms: number
}

export interface EmbedImageResult {
  vector: number[]
  fingerprint: string
  dim: number
  ms: number
}

export interface SidecarEndpoint {
  port: number
  token: string
}

export class SidecarClient {
  private readonly endpoint: SidecarEndpoint
  private readonly fetchImpl: FetchLike

  constructor(endpoint: SidecarEndpoint, fetchImpl?: FetchLike) {
    this.endpoint = endpoint
    this.fetchImpl = fetchImpl ?? (globalThis.fetch as unknown as FetchLike)
  }

  private async request(route: string, init: { method: string; body?: unknown; timeoutMs: number }): Promise<unknown> {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), init.timeoutMs)
    try {
      return await this.fetchImpl(`http://127.0.0.1:${this.endpoint.port}${route}`, {
        method: init.method,
        headers: {
          'content-type': 'application/json',
          'x-embed-token': this.endpoint.token,
        },
        body: init.body === undefined ? undefined : JSON.stringify(init.body),
        signal: controller.signal,
      }).then(async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => null)
          // sidecar 錯誤封套:{"error": {code, message, fingerprint?}}(common.py)
          const err = body !== null && typeof body === 'object' && 'error' in body
            ? (body as { error: unknown }).error
            : null
          const detail = err !== null && typeof err === 'object' && err !== null && 'message' in err
            ? `${String((err as { code?: unknown }).code ?? 'error')}: ${String((err as { message: unknown }).message)}`
            : err !== null && err !== undefined
              ? String(err)
              : ''
          // 4xx(401 除外)= 調用方錯誤(永久,不觸發降級重試);401/5xx = 可用性問題。
          const message = `sidecar ${route} -> HTTP ${res.status}${detail !== '' ? `: ${detail}` : ''}`
          if (res.status >= 400 && res.status < 500 && res.status !== 401 && res.status !== 408) {
            throw new EmbedderResponseError(message)
          }
          throw new EmbedderUnavailableError(message)
        }
        return await res.json()
      })
    } catch (error) {
      if (error instanceof EmbedderUnavailableError || error instanceof EmbedderResponseError) throw error
      const reason = error instanceof Error ? error.message : String(error)
      throw new EmbedderUnavailableError(`sidecar ${route} unreachable: ${reason}`)
    } finally {
      clearTimeout(timer)
    }
  }

  /** 探測 /health;任何失敗返回 null(不拋)。 */
  async health(timeoutMs: number): Promise<{ ok: boolean; uptime_s?: number; backend?: string } | null> {
    try {
      const out = await this.request(SIDECAR_ROUTES.health, { method: 'GET', timeoutMs })
      if (out === null || typeof out !== 'object' || typeof (out as { ok?: unknown }).ok !== 'boolean') return null
      return out as { ok: boolean; uptime_s?: number; backend?: string }
    } catch {
      return null
    }
  }

  /** 探測 /backends;任何失敗返回 null(不拋)。 */
  async backends(timeoutMs: number): Promise<BackendInfo[] | null> {
    try {
      const out = await this.request(SIDECAR_ROUTES.backends, { method: 'GET', timeoutMs })
      if (!Array.isArray(out)) return null
      return out as BackendInfo[]
    } catch {
      return null
    }
  }

  async embedTexts(
    texts: string[],
    opts: { dim?: number; instruct?: string; backend?: string },
    timeoutMs: number,
  ): Promise<EmbedTextsResult> {
    const body: Record<string, unknown> = { texts }
    if (opts.dim !== undefined) body.dim = opts.dim
    if (opts.instruct !== undefined) body.instruct = opts.instruct
    if (opts.backend !== undefined) body.backend = opts.backend
    const out = await this.request(SIDECAR_ROUTES.embedTexts, { method: 'POST', body, timeoutMs }) as Record<string, unknown>
    const { vectors, fingerprint, dim, ms } = out
    if (!Array.isArray(vectors) || vectors.length !== texts.length ||
      !vectors.every((v) => Array.isArray(v) && v.every((x) => typeof x === 'number' && Number.isFinite(x)))) {
      throw new EmbedderResponseError(`/embed/texts malformed 'vectors' (expected ${texts.length} finite number[])`)
    }
    if (typeof fingerprint !== 'string' || typeof dim !== 'number' || typeof ms !== 'number') {
      throw new EmbedderResponseError('/embed/texts malformed envelope {fingerprint, dim, ms}')
    }
    return { vectors: vectors as number[][], fingerprint, dim, ms }
  }

  async embedImage(
    path: string,
    opts: { dim?: number; backend?: string },
    timeoutMs: number,
  ): Promise<EmbedImageResult> {
    const body: Record<string, unknown> = { path }
    if (opts.dim !== undefined) body.dim = opts.dim
    if (opts.backend !== undefined) body.backend = opts.backend
    const out = await this.request(SIDECAR_ROUTES.embedImage, { method: 'POST', body, timeoutMs }) as Record<string, unknown>
    const { vector, fingerprint, dim, ms } = out
    if (!Array.isArray(vector) || !vector.every((x) => typeof x === 'number' && Number.isFinite(x))) {
      throw new EmbedderResponseError("/embed/image malformed 'vector' (expected finite number[])")
    }
    if (typeof fingerprint !== 'string' || typeof dim !== 'number' || typeof ms !== 'number') {
      throw new EmbedderResponseError('/embed/image malformed envelope {fingerprint, dim, ms}')
    }
    return { vector: vector as number[], fingerprint, dim, ms }
  }
}
