/**
 * @bryan-cmf/dsh-embed — Host half。
 *
 * DSH 本地語義嵌入基礎設施(SPEC dsh-embed-spec.md v1.0):
 * - provide('embedder'):embedTexts / embedImage / backends / health,
 *   多後端路由 {backend, dim},全本地推理(127.0.0.1 + token,零出網)
 * - 進程監管:雙 sidecar 懶啟動 / 握手文件發現 / 健康檢查 / 指數退避重啟 /
 *   keep-alive 空閒退出(supervisor.ts)
 * - 消費方:dsh-insights memory v2(inject 'embedder',optional,失敗降級 kw)
 *
 * 本插件發布 `embedder` 服務 → 必須掛 HOST composition(跨 session);
 * 見 cordis.patch.yml 註釋。Phase 1 不註冊任何模型工具(消費工具屬 Phase 3)。
 */
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import z from '@deepseek-ai/schemastery'
import type { Context } from '@deepseek-ai/cordis'
import { BACKEND_CATALOG, catalogBackendInfos, sidecarForBackend } from './catalog.ts'
import { EmbedderResponseError, EmbedderError, EmbedderUnavailableError, EmbedderValidationError, isEmbedderUnavailableError } from './errors.ts'
import { fingerprint, isFingerprintStale, parseFingerprint } from './fingerprint.ts'
import { SidecarClient } from './http-client.ts'
import { RuntimeDir, expandTilde, isPidAlive } from './runtime-files.ts'
import { EmbedderServiceImpl } from './service.ts'
import { SystemClock, nodeSpawn } from './spawn.ts'
import { SidecarSupervisor, defaultKillPid } from './supervisor.ts'
import type { Clock, Logger, SpawnedProcess, SpawnSpec, SidecarState, SupervisorDeps } from './supervisor.ts'
import type { BackendInfo, EmbedImageOptions, EmbedTextsOptions, EmbedderService, RuntimeHandshake, SidecarId } from './types.ts'
import { MAX_IMAGE_BYTES, MAX_TEXTS_PER_REQUEST, SIDECAR_ROUTES } from './types.ts'
import { cosine, l2norm, mrlTruncate, rrfFuse } from './vector.ts'

const name = 'dsh-embed'

// ── config(SPEC §7)──────────────────────────────────────────────────────────

function sidecarSchema(venvDefault: string) {
  return z.object({
    enabled: z.boolean().default(true),
    venv: z.string().default(venvDefault),
    /** 覆寫 python 直譯器絕對路徑;默認 <venv>/bin/python。 */
    python: z.string().default(''),
    /** 覆寫 sidecar 腳本絕對路徑;默認隨插件分發的 lib/sidecar/<id>_serve.py。 */
    script: z.string().default(''),
    /** 空閒退出窗口(秒);0 = 不自動退出。 */
    keepAliveSec: z.number().min(0).max(86400).default(900),
    /** 插件載入即預熱的後端(打破懶啟動;僅 tf 配 'qwen3-4b-fp16')。 */
    eagerBackends: z.array(z.string()).default([]),
  })
}

const Config = z.object({
  /** runtime 握手/日誌目錄(SPEC §2)。 */
  runtimeDir: z.string().default('~/.dsh/run/dsh-embed'),
  defaults: z.object({
    textBackend: z.string().default('qwen3-4b-fp16'),
    visualBackend: z.string().default('wemm2b-mlx4b'),
    dim: z.number().min(32).max(4096).default(512),
  }),
  mlxSidecar: sidecarSchema('~/.dsh/dsh-embed/venv-mlx'),
  tfSidecar: sidecarSchema('~/.dsh/dsh-embed/venv-tf'),
  /** 健康檢查間隔;連續 healthFailureLimit 失敗 → kill + 退避重啟。 */
  healthIntervalMs: z.number().min(1000).max(600000).default(30000),
  healthFailureLimit: z.number().min(1).max(10).default(3),
  /** 單次 embed 請求超時。 */
  requestTimeoutMs: z.number().min(1000).max(600000).default(120000),
  /** 握手+健康就緒窗口(tf 冷啟動實測 ~30s,留足餘量)。 */
  startupTimeoutMs: z.number().min(5000).max(600000).default(180000),
  /** 崩潰退避:base × factor^(n-1),連續 maxRestartAttempts 次 → 冷卻。 */
  backoffBaseMs: z.number().min(100).max(60000).default(1000),
  backoffFactor: z.number().min(1).max(16).default(4),
  maxRestartAttempts: z.number().min(1).max(10).default(3),
})

interface SidecarConfigSlice {
  enabled: boolean
  venv: string
  python: string
  script: string
  keepAliveSec: number
  eagerBackends: string[]
}

interface ConfigType {
  runtimeDir: string
  defaults: { textBackend: string; visualBackend: string; dim: number }
  mlxSidecar: SidecarConfigSlice
  tfSidecar: SidecarConfigSlice
  healthIntervalMs: number
  healthFailureLimit: number
  requestTimeoutMs: number
  startupTimeoutMs: number
  backoffBaseMs: number
  backoffFactor: number
  maxRestartAttempts: number
}

// ── 插件裝配 ───────────────────────────────────────────────────────────────

/** 隨插件分發的 sidecar 腳本解析:lib/sidecar(構建複製)→ src/sidecar(直跑)。 */
function resolveSidecarScript(pkgRoot: string, id: SidecarId, override: string): string {
  if (override !== '') return expandTilde(override)
  const candidates = [path.join(pkgRoot, 'lib', 'sidecar', `${id}_serve.py`), path.join(pkgRoot, 'src', 'sidecar', `${id}_serve.py`)]
  for (const candidate of candidates) {
    try {
      if (fs.existsSync(candidate)) return candidate
    } catch {
      /* fs 不可用時退回首選路徑,讓 spawn 報出可讀錯誤 */
    }
  }
  return candidates[0]!
}

function pluginRoot(): string {
  // lib/index.js → lib → 倉庫根;src/index.ts 直跑時 → src → 倉庫根。
  return path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
}

/** cordis logger → supervisor Logger(鬆耦合,缺方法不炸)。 */
function toLogger(ctx: Context): Logger | undefined {
  const raw = (ctx as unknown as { logger?: Partial<Logger> }).logger
  if (raw === undefined) return undefined
  return {
    info: (m: string) => raw.info?.(m),
    warn: (m: string) => raw.warn?.(m),
    error: (m: string) => raw.error?.(m),
  }
}

const inject: string[] = []

function apply(ctx: Context, config: ConfigType): void {
  const logger = toLogger(ctx)
  const clock = new SystemClock()
  const pkgRoot = pluginRoot()
  const runtime = new RuntimeDir(config.runtimeDir)

  const makeSupervisor = (id: SidecarId, slice: SidecarConfigSlice): SidecarSupervisor => {
    const python = slice.python !== '' ? expandTilde(slice.python) : path.join(expandTilde(slice.venv), 'bin', 'python')
    const script = resolveSidecarScript(pkgRoot, id, slice.script)
    // sidecar 自帶空閒看門狗作兜底:窗口設在 host keep-alive 之後 60s,
    // 保證 host 側狀態機先殺(狀態轉換乾淨);host 關閉 keep-alive 時退回默認。
    const sidecarIdleSec = slice.keepAliveSec > 0 ? slice.keepAliveSec + 60 : 900
    const scriptArgs = ['--idle-timeout-sec', String(sidecarIdleSec)]
    if (slice.eagerBackends.length > 0) scriptArgs.push('--eager', slice.eagerBackends.join(','))
    return new SidecarSupervisor({
      id,
      enabled: slice.enabled,
      python,
      script,
      scriptArgs,
      runtime,
      clock,
      spawnProcess: nodeSpawn,
      logger,
      healthIntervalMs: config.healthIntervalMs,
      healthFailureLimit: config.healthFailureLimit,
      keepAliveMs: slice.keepAliveSec * 1000,
      startupTimeoutMs: config.startupTimeoutMs,
      backoffBaseMs: config.backoffBaseMs,
      backoffFactor: config.backoffFactor,
      maxRestartAttempts: config.maxRestartAttempts,
      requestTimeoutMs: config.requestTimeoutMs,
    })
  }

  const supervisors = {
    mlx: makeSupervisor('mlx', config.mlxSidecar),
    tf: makeSupervisor('tf', config.tfSidecar),
  }

  const service = new EmbedderServiceImpl({
    supervisors,
    defaults: config.defaults,
    requestTimeoutMs: config.requestTimeoutMs,
  })

  // 預熱:eagerBackends 打破懶啟動(tf 默認 ['qwen3-4b-fp16'],SPEC §7)。
  for (const id of ['tf', 'mlx'] as const) {
    const slice = id === 'tf' ? config.tfSidecar : config.mlxSidecar
    if (slice.enabled && slice.eagerBackends.length > 0) {
      void supervisors[id].ensureStarted().catch((error: unknown) => {
        logger?.warn?.(`[dsh-embed/${id}] eager warm-up failed (will retry lazily): ${error instanceof Error ? error.message : String(error)}`)
      })
    }
  }

  // 服務註冊:自動隨 fiber 卸載。
  ctx.provide('embedder', service)

  // 可逆清理:殺 sidecar、清握手文件(資料向量表不受影響,降級讀 keyword)。
  ctx.effect(() => () => {
    void (async () => {
      await supervisors.mlx.dispose()
      await supervisors.tf.dispose()
    })()
  }, 'dsh-embed.sidecars')
}

export { Config, apply, inject, name }

// ── 公開面(消費方與測試導入;對照 dsh-cross-search 倉庫風格) ────────────────

export type { BackendInfo, EmbedImageOptions, EmbedTextsOptions, EmbedderService, RuntimeHandshake, SidecarId } from './types.ts'
export { MAX_IMAGE_BYTES, MAX_TEXTS_PER_REQUEST, SIDECAR_ROUTES } from './types.ts'
export type { Clock, Logger, SpawnedProcess, SpawnSpec, SidecarState, SupervisorDeps } from './supervisor.ts'
export { SidecarSupervisor, defaultKillPid } from './supervisor.ts'
export { SystemClock, nodeSpawn, buildSidecarArgs } from './spawn.ts'
export { RuntimeDir, expandTilde, isPidAlive } from './runtime-files.ts'
export { SidecarClient } from './http-client.ts'
export { EmbedderServiceImpl } from './service.ts'
export { BACKEND_CATALOG, catalogBackendInfos, sidecarForBackend } from './catalog.ts'
export { fingerprint, isFingerprintStale, parseFingerprint } from './fingerprint.ts'
export { cosine, l2norm, mrlTruncate, rrfFuse } from './vector.ts'
export { EmbedderError, EmbedderResponseError, EmbedderUnavailableError, EmbedderValidationError, isEmbedderUnavailableError } from './errors.ts'
