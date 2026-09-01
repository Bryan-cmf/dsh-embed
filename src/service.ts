/**
 * dsh-embed · `embedder` 服務實現(SPEC §3 契約)。
 *
 * 職責:
 * - 後端路由:backend 名 → sidecar(靜態表,不依賴網絡);未知後端即拋
 * - 輸入驗證:texts ≤64、圖像存在且 ≤30MB、instruct 僅 Qwen3 家族
 * - 失敗語義:目標後端不可用 → 立即拋 EmbedderUnavailableError;
 *   **禁止自動跨後端替補**(指紋不可互換,cos 漂移 0.968 實測)
 * - 回應衛士:指紋必須等於 `{backend}@{dim}`,否則 EmbedderResponseError;
 *   最終向量再套 mrlTruncate(冪等)保證「長 dim、單位範數」恆成立
 * - backends():live /backends 優先;sidecar 未起時以靜態目錄 alive:false 描述
 * - health():{mlx, tf} → up/down/starting(降級監測,不觸發 spawn)
 */
import { EmbedderResponseError, EmbedderUnavailableError, EmbedderValidationError } from './errors.ts'
import type { FetchLike } from './http-client.ts'
import { BACKEND_CATALOG, catalogBackendInfos, sidecarForBackend } from './catalog.ts'
import { fingerprint } from './fingerprint.ts'
import type { SidecarSupervisor } from './supervisor.ts'
import { mrlTruncate } from './vector.ts'
import {
  MAX_IMAGE_BYTES,
  MAX_TEXTS_PER_REQUEST,
  type BackendInfo,
  type EmbedImageOptions,
  type EmbedTextsOptions,
  type EmbedderService,
  type SidecarId,
} from './types.ts'
import fsp from 'node:fs/promises'

export interface EmbedderServiceOptions {
  supervisors: Record<SidecarId, SidecarSupervisor>
  defaults: { textBackend: string; visualBackend: string; textDim: number; visualDim: number }
  /** 單次 embed 請求超時;默認 120s(圖像頁批量場景留餘量)。 */
  requestTimeoutMs?: number
  fetchImpl?: FetchLike
  /** 注入替代 fs.stat(測試)。 */
  statFile?: (path: string) => Promise<{ size: number }>
  /** 額外後端→sidecar 路由覆寫(測試/擴展)。 */
  routes?: Record<string, SidecarId>
}

const PROBE_TIMEOUT_MS = 2_000

export class EmbedderServiceImpl implements EmbedderService {
  private readonly supervisors: Record<SidecarId, SidecarSupervisor>
  private readonly defaults: { textBackend: string; visualBackend: string; textDim: number; visualDim: number }
  private readonly routes: Record<string, SidecarId>
  private readonly statFile: (path: string) => Promise<{ size: number }>
  private readonly requestTimeoutMs: number

  constructor(options: EmbedderServiceOptions) {
    this.supervisors = options.supervisors
    this.defaults = options.defaults
    this.routes = options.routes ?? {}
    this.statFile = options.statFile ?? ((p: string) => fsp.stat(p))
    this.requestTimeoutMs = options.requestTimeoutMs ?? 120_000
  }

  private route(backend: string): SidecarId {
    const known = this.routes[backend] ?? sidecarForBackend(backend)
    if (known === null) {
      // 未知後端是調用方/配置錯誤(永久性),非可用性事件——拋 validation,
      // 避免被 isEmbedderUnavailableError() 捕獲而靜默降級 keyword、遮蔽配置錯。
      throw new EmbedderValidationError(
        `unknown backend '${backend}' (known: ${[...new Set([...Object.keys(this.routes), 'wemm2b-mlx4b', 'qwen3-4b-fp16', 'wemm2b-fp16'])].join(', ')})`,
      )
    }
    return known
  }

  /**
   * dim 預校驗(F4):catalog 已知後端先按支持維度擋掉顯然非法的 dim,
   * 免一次無謂往返;sidecar /embed 端仍為最終權威(僅 routes 註冊、
   * 不在 catalog 的擴展後端交由 sidecar 裁決)。
   */
  private verifyDimSupported(backend: string, dim: number, api: string): void {
    const entry = BACKEND_CATALOG[backend]
    if (entry === undefined) return
    if (!entry.dims.includes(dim)) {
      throw new EmbedderValidationError(
        `${api}: dim ${dim} not in supported dims of '${backend}' [${entry.dims.join(', ')}] (sidecar is the final authority)`,
        fingerprint(backend, dim),
      )
    }
  }

  private supervisor(backend: string): SidecarSupervisor {
    const id = this.route(backend)
    const supervisor = this.supervisors[id]
    if (supervisor === undefined) throw new EmbedderUnavailableError(`sidecar ${id} not configured`)
    return supervisor
  }

  async backends(): Promise<BackendInfo[]> {
    const out: BackendInfo[] = []
    for (const id of ['mlx', 'tf'] as const) {
      const supervisor = this.supervisors[id]
      if (supervisor === undefined || !supervisor.enabled) continue
      const client = supervisor.client()
      if (client !== null) {
        const live = await client.backends(PROBE_TIMEOUT_MS)
        if (live !== null && live.length > 0) {
          out.push(...live)
          continue
        }
      }
      // R2F1:catalog 兜底按「後端角色」選維度(視覺後端夾各自梯最大),由
      // catalogBackendInfos 內部裁決,杜絕 wemm2b-fp16@2560 這類非法指紋。
      out.push(...catalogBackendInfos(id, { text: this.defaults.textDim, visual: this.defaults.visualDim }, false))
    }
    return out
  }

  async embedTexts(texts: string[], opts: EmbedTextsOptions = {}): Promise<Float32Array[]> {
    if (!Array.isArray(texts) || texts.length === 0) {
      throw new EmbedderValidationError('embedTexts: texts must be a non-empty array')
    }
    if (texts.length > MAX_TEXTS_PER_REQUEST) {
      throw new EmbedderValidationError(`embedTexts: ${texts.length} texts exceeds limit ${MAX_TEXTS_PER_REQUEST}; batch callers must chunk`)
    }
    for (const text of texts) {
      if (typeof text !== 'string' || text.trim() === '') {
        throw new EmbedderValidationError('embedTexts: every text must be a non-empty string')
      }
    }
    const backend = opts.backend ?? this.defaults.textBackend
    const dim = opts.dim ?? this.defaults.textDim
    if (!Number.isInteger(dim) || dim <= 0) {
      throw new EmbedderValidationError(`embedTexts: dim must be a positive integer, got ${dim}`)
    }
    this.route(backend)
    this.verifyDimSupported(backend, dim, 'embedTexts')
    if (opts.instruct !== undefined && !backend.startsWith('qwen3')) {
      throw new EmbedderValidationError(
        `embedTexts: instruct is only supported on qwen3* backends, not '${backend}'`,
        fingerprint(backend, dim),
      )
    }
    return this.callSidecar(backend, dim, async (client) => {
      const response = await client.embedTexts(texts, { dim, instruct: opts.instruct, backend }, this.requestTimeoutMs)
      this.verifyFingerprint(backend, dim, response.fingerprint, response.dim)
      return response.vectors.map((vector) => mrlTruncate(vector, dim))
    })
  }

  async embedImage(path: string, opts: EmbedImageOptions = {}): Promise<Float32Array> {
    if (typeof path !== 'string' || path.trim() === '') {
      throw new EmbedderValidationError('embedImage: path must be a non-empty string')
    }
    const backend = opts.backend ?? this.defaults.visualBackend
    this.route(backend) // 未知後端先於 fs I/O 快速失敗
    let size: number
    try {
      size = (await this.statFile(path)).size
    } catch (error) {
      throw new EmbedderValidationError(`embedImage: cannot stat '${path}': ${error instanceof Error ? error.message : String(error)}`)
    }
    if (size > MAX_IMAGE_BYTES) {
      throw new EmbedderValidationError(`embedImage: '${path}' is ${size} bytes, over limit ${MAX_IMAGE_BYTES}`)
    }
    const dim = opts.dim ?? this.defaults.visualDim
    if (!Number.isInteger(dim) || dim <= 0) {
      throw new EmbedderValidationError(`embedImage: dim must be a positive integer, got ${dim}`)
    }
    this.verifyDimSupported(backend, dim, 'embedImage')
    const catalogEntry = BACKEND_CATALOG[backend]
    if (catalogEntry !== undefined && !catalogEntry.modalities.includes('image')) {
      throw new EmbedderValidationError(`embedImage: backend '${backend}' does not support images`, fingerprint(backend, dim))
    }
    const vectors = await this.callSidecar(backend, dim, async (client) => {
      const response = await client.embedImage(path, { dim, backend }, this.requestTimeoutMs)
      this.verifyFingerprint(backend, dim, response.fingerprint, response.dim)
      return [mrlTruncate(response.vector, dim)]
    })
    return vectors[0]!
  }

  async health(): Promise<{ mlx: 'up' | 'down' | 'starting'; tf: 'up' | 'down' | 'starting' }> {
    const map = (id: SidecarId): 'up' | 'down' | 'starting' => {
      const supervisor = this.supervisors[id]
      if (supervisor === undefined) return 'down'
      const state = supervisor.state
      if (state === 'up') return 'up'
      if (state === 'starting') return 'starting'
      return 'down'
    }
    return { mlx: map('mlx'), tf: map('tf') }
  }

  // ── 內部 ─────────────────────────────────────────────────────────────────

  private verifyFingerprint(backend: string, dim: number, gotFp: string, gotDim: number): void {
    const expected = fingerprint(backend, dim)
    if (gotFp !== expected || gotDim !== dim) {
      throw new EmbedderResponseError(
        `fingerprint mismatch: expected ${expected}, got '${gotFp}' (dim ${gotDim}); cross-backend vectors are not interchangeable — rebuild the index`,
        expected,
      )
    }
  }

  /**
   * 統一調用路徑:懶啟動 → touch → 執行。EmbedderUnavailableError 原樣上拋,
   * 絕不跨後端替補(SPEC §3 失敗語義)。
   */
  private async callSidecar<T>(backend: string, dim: number, call: (client: NonNullable<ReturnType<SidecarSupervisor['client']>>) => Promise<T>): Promise<T> {
    const supervisor = this.supervisor(backend)
    await supervisor.ensureStarted()
    supervisor.touch()
    const client = supervisor.client()
    if (client === null) {
      throw new EmbedderUnavailableError(`sidecar for backend '${backend}' is not up`, fingerprint(backend, dim))
    }
    return call(client)
  }
}
