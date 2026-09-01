/**
 * dsh-embed · 進程監管狀態機(SPEC §2)。
 *
 * 規則(全部有對應單元測試):
 * - 懶啟動:首次 ensureStarted() 觸發 spawn(冷啟動實測 mlx ~8s / tf ~30s)
 * - 握手:sidecar 綁定 127.0.0.1:0 後把 {port, token, pid} 寫入 runtime
 *   文件;supervisor 輪詢該文件 + GET /health 直到 ok
 * - 孤兒收養:啟動時若發現存活且健康的同位 sidecar(上次 host 崩潰殘留,
 *   keep-alive 窗口內),直接收養,避免重複 spawn;不健康則殺掉重來
 * - 健康檢查:每 healthIntervalMs 探測 /health;連續 healthFailureLimit 次
 *   失敗 → kill + 重啟(指數退避 backoffBase×factor^n,上限 maxRestartAttempts 次)
 * - 崩潰:進程非計劃退出 → 同退避重啟;連續失敗達上限 → failed,
 *   冷卻 failedCooldownMs 後新一輪可再試(混亂測試 kill -9 ×3 → 5 分鐘內自愈)
 * - keep-alive:最後一次調用後 keepAliveMs 空閒 → SIGTERM(寬限後 SIGKILL)
 *   退出,狀態歸 down;keepAliveMs<=0 表示不自動空閒退出
 * - dispose:殺進程、清握手文件、撤全部定時器
 *
 * 可注入縫隙(clock/spawnProcess/pidAlive/killPid/fetchImpl)讓全部狀態
 * 轉移可在無 Python、無真實進程下用假時鐘精確測試。
 */
import { EmbedderUnavailableError } from './errors.ts'
import { SidecarClient, type FetchLike } from './http-client.ts'
import { RuntimeDir, isPidAlive } from './runtime-files.ts'
import type { RuntimeHandshake, SidecarId } from './types.ts'

export type SidecarState = 'down' | 'starting' | 'up' | 'backoff' | 'failed' | 'stopping'

export interface Logger {
  info?(message: string): void
  warn?(message: string): void
  error?(message: string): void
}

export interface Clock {
  now(): number
  setTimeout(callback: () => void, ms: number): unknown
  clearTimeout(handle: unknown): void
  setInterval(callback: () => void, ms: number): unknown
  clearInterval(handle: unknown): void
  sleep(ms: number): Promise<void>
}

export interface SpawnSpec {
  id: SidecarId
  python: string
  script: string
  args: string[]
  /** runtime 握手目錄(經 --runtime-dir 傳入;sidecar 寫 <dir>/<id>.json)。 */
  runtimeDir: string
  env: Record<string, string>
}

export interface SpawnedProcess {
  readonly pid: number | undefined
  onExit(callback: (code: number | null, signal: string | null) => void): void
  /** 優雅終止;grace 後升級 SIGKILL;resolve 時進程確已退出。 */
  kill(graceMs: number): Promise<void>
}

export interface SupervisorDeps {
  clock: Clock
  spawnProcess: (spec: SpawnSpec, logFile: string) => SpawnedProcess
  fetchImpl?: FetchLike
  /** pid 存活判定(測試注入)。 */
  pidAlive?: (pid: number) => boolean
  /** 按 pid 終止進程組(測試注入;用於孤兒/非 child 進程)。 */
  killPid?: (pid: number, graceMs: number) => Promise<void>
}

export interface SidecarSupervisorOptions extends SupervisorDeps {
  id: SidecarId
  enabled: boolean
  /** 絕對路徑:venv 內 python 直譯器。 */
  python: string
  /** 絕對路徑:sidecar 進程入口腳本(<id>_serve.py)。 */
  script: string
  /** 額外 CLI 參數(如 --eager)。 */
  scriptArgs: string[]
  runtime: RuntimeDir
  logger?: Logger
  healthIntervalMs?: number
  healthTimeoutMs?: number
  healthFailureLimit?: number
  keepAliveMs?: number
  keepAliveSweepMs?: number
  startupTimeoutMs?: number
  handshakePollMs?: number
  backoffBaseMs?: number
  backoffFactor?: number
  maxRestartAttempts?: number
  failedCooldownMs?: number
  killGraceMs?: number
  requestTimeoutMs?: number
}

const DEFAULTS = {
  healthIntervalMs: 30_000,
  healthTimeoutMs: 2_000,
  healthFailureLimit: 3,
  keepAliveMs: 900_000,
  keepAliveSweepMs: 5_000,
  startupTimeoutMs: 180_000,
  handshakePollMs: 200,
  backoffBaseMs: 1_000,
  backoffFactor: 4,
  maxRestartAttempts: 3,
  failedCooldownMs: 30_000,
  killGraceMs: 5_000,
  requestTimeoutMs: 120_000,
}

type EffectiveOptions = SidecarSupervisorOptions & typeof DEFAULTS

export class SidecarSupervisor {
  readonly id: SidecarId
  private readonly o: EffectiveOptions
  private readonly pidAlive: (pid: number) => boolean

  private _state: SidecarState = 'down'
  private handshake_: RuntimeHandshake | null = null
  private child_: SpawnedProcess | null = null
  private childExited = false
  private startPromise_: Promise<void> | null = null
  private healthTimer: unknown = null
  private sweepTimer: unknown = null
  private backoffTimer: unknown = null
  private backoffUntil = 0
  private attempts = 0
  private failedAt = 0
  private healthFailures = 0
  private lastUsed_ = 0
  private killIntent: 'none' | 'idle' | 'restart' = 'none'
  private disposed = false

  constructor(options: SidecarSupervisorOptions) {
    this.id = options.id
    this.o = { ...DEFAULTS, ...options }
    this.pidAlive = options.pidAlive ?? isPidAlive
    this.sweepTimer = this.o.clock.setInterval(() => this.sweepIdle(), this.o.keepAliveSweepMs)
  }

  // ── 觀察面 ───────────────────────────────────────────────────────────────

  get state(): SidecarState {
    return this._state
  }

  get enabled(): boolean {
    return this.o.enabled
  }

  get handshake(): RuntimeHandshake | null {
    return this.handshake_
  }

  get attemptsCount(): number {
    return this.attempts
  }

  get lastUsed(): number {
    return this.lastUsed_
  }

  status(): { state: SidecarState; pid: number | null; port: number | null; attempts: number; lastUsedAt: number } {
    return {
      state: this._state,
      pid: this.handshake_?.pid ?? this.child_?.pid ?? null,
      port: this.handshake_?.port ?? null,
      attempts: this.attempts,
      lastUsedAt: this.lastUsed_,
    }
  }

  /** up 態下的 sidecar 客戶端;否則 null。 */
  client(): SidecarClient | null {
    if (this._state !== 'up' || this.handshake_ === null) return null
    return new SidecarClient(this.handshake_, this.o.fetchImpl)
  }

  touch(): void {
    this.lastUsed_ = this.o.clock.now()
  }

  // ── 需求路徑 ─────────────────────────────────────────────────────────────

  /**
   * 確保 sidecar 服務可用(懶啟動)。
   * 不可用且短期內無法恢復時拋 EmbedderUnavailableError——調用方降級。
   */
  async ensureStarted(): Promise<void> {
    if (this.disposed) throw new EmbedderUnavailableError(`sidecar ${this.id}: supervisor disposed`)
    if (!this.o.enabled) throw new EmbedderUnavailableError(`sidecar ${this.id}: disabled in config`)
    this.touch()
    for (let hop = 0; hop < 12; hop++) {
      if (this.disposed) throw new EmbedderUnavailableError(`sidecar ${this.id}: supervisor disposed`)
      if (this._state === 'up') return
      if (this.startPromise_ !== null) {
        try {
          await this.startPromise_
        } catch (error) {
          if (this._state === 'failed') continue
          throw error
        }
        continue
      }
      if (this._state === 'failed') {
        if (this.o.clock.now() - this.failedAt < this.o.failedCooldownMs) {
          throw new EmbedderUnavailableError(
            `sidecar ${this.id}: failed after ${this.attempts} attempts; cooling down (${Math.ceil((this.failedAt + this.o.failedCooldownMs - this.o.clock.now()) / 1000)}s left)`,
          )
        }
        this.attempts = 0
      }
      if (this._state === 'backoff') {
        const wait = Math.max(0, this.backoffUntil - this.o.clock.now())
        await this.o.clock.sleep(wait)
        continue
      }
      await this.runStartCycle()
    }
    throw new EmbedderUnavailableError(`sidecar ${this.id}: ensureStarted exceeded retry hops`)
  }

  // ── 啟動週期 ─────────────────────────────────────────────────────────────

  private runStartCycle(): Promise<void> {
    if (this.startPromise_ !== null) return this.startPromise_
    this.cancelBackoffTimer()
    this._state = 'starting'
    const p = this.doStart().then(
      () => {
        this.startPromise_ = null
      },
      (error: unknown) => {
        this.startPromise_ = null
        this.onStartFailure(error)
        throw error
      },
    )
    this.startPromise_ = p
    // 背景觸發(退避定時器)不讓 promise 未處理化:由調用點附加 catch。
    return p
  }

  private async doStart(): Promise<void> {
    const deadline = this.o.clock.now() + this.o.startupTimeoutMs
    const fail = async (message: string): Promise<never> => {
      await this.killChildBestEffort()
      await this.o.runtime.remove(this.id)
      throw new EmbedderUnavailableError(`sidecar ${this.id}: ${message}`)
    }

    // 孤兒處理:存活且健康 → 收養;否則清理後全新 spawn。
    const orphan = await this.o.runtime.read(this.id)
    if (orphan !== null) {
      if (this.pidAlive(orphan.pid)) {
        const probe = new SidecarClient(orphan, this.o.fetchImpl)
        const health = await probe.health(this.o.healthTimeoutMs)
        if (health !== null && health.ok) {
          this.handshake_ = orphan
          this.childExited = false
          this.enterUp(`adopted orphan sidecar pid=${orphan.pid} port=${orphan.port}`)
          return
        }
        await this.killPidBestEffort(orphan.pid)
      }
      await this.o.runtime.remove(this.id)
    }

    this.childExited = false
    const child = this.o.spawnProcess(
      {
        id: this.id,
        python: this.o.python,
        script: this.o.script,
        args: this.o.scriptArgs,
        runtimeDir: this.o.runtime.root,
        env: {},
      },
      this.o.runtime.logFile(this.id),
    )
    this.child_ = child
    child.onExit((code, signal) => this.onChildExit(code, signal))

    while (this.o.clock.now() < deadline) {
      if (this.disposed) return await fail('disposed during startup')
      if (this.childExited) return await fail(`process exited during startup (code=${this.lastExitCode} signal=${this.lastExitSignal})`)
      const hs = await this.o.runtime.read(this.id)
      if (hs !== null && this.pidAlive(hs.pid)) {
        const probe = new SidecarClient(hs, this.o.fetchImpl)
        const health = await probe.health(this.o.healthTimeoutMs)
        if (health !== null && health.ok) {
          this.handshake_ = hs
          this.enterUp(`sidecar up pid=${hs.pid} port=${hs.port}`)
          return
        }
      }
      await this.o.clock.sleep(this.o.handshakePollMs)
    }
    return await fail(`not healthy within startupTimeoutMs=${this.o.startupTimeoutMs}`)
  }

  private lastExitCode: number | null = null
  private lastExitSignal: string | null = null

  private enterUp(reason: string): void {
    this._state = 'up'
    this.attempts = 0
    this.healthFailures = 0
    this.killIntent = 'none'
    this.touch()
    this.watchHealth()
    this.o.logger?.info?.(`[dsh-embed/${this.id}] ${reason}`)
  }

  private onStartFailure(error: unknown): void {
    if (this.disposed) {
      this._state = 'down'
      return
    }
    const message = error instanceof Error ? error.message : String(error)
    this.scheduleRestart(`start failed: ${message}`)
  }

  // ── 崩潰與退避 ───────────────────────────────────────────────────────────

  private onChildExit(code: number | null, signal: string | null): void {
    this.childExited = true
    this.lastExitCode = code
    this.lastExitSignal = signal
    this.child_ = null
    if (this.killIntent !== 'none') return // 殺進程的路徑自己收尾(idle/restart)
    if (this._state === 'starting') return // doStart 輪詢循環經 childExited 快速失敗
    if (this._state === 'up') {
      this.scheduleRestart(`process exited unexpectedly (code=${code} signal=${signal})`)
    }
  }

  private scheduleRestart(reason: string): void {
    this.handshake_ = null
    this.stopHealthWatch()
    this.healthFailures = 0
    this.attempts += 1
    this.o.logger?.warn?.(`[dsh-embed/${this.id}] restart #${this.attempts}: ${reason}`)
    if (this.attempts >= this.o.maxRestartAttempts) {
      this._state = 'failed'
      this.failedAt = this.o.clock.now()
      this.o.logger?.error?.(`[dsh-embed/${this.id}] ${this.attempts} consecutive failures; entering cooldown ${this.o.failedCooldownMs}ms`)
      return
    }
    const delay = this.o.backoffBaseMs * this.o.backoffFactor ** (this.attempts - 1)
    this.backoffUntil = this.o.clock.now() + delay
    this._state = 'backoff'
    this.backoffTimer = this.o.clock.setTimeout(() => {
      this.backoffTimer = null
      void this.runStartCycle().catch(() => {})
    }, delay)
  }

  // ── 健康檢查 ─────────────────────────────────────────────────────────────

  private watchHealth(): void {
    this.stopHealthWatch()
    this.healthTimer = this.o.clock.setInterval(() => {
      void this.probeHealth()
    }, this.o.healthIntervalMs)
  }

  private stopHealthWatch(): void {
    if (this.healthTimer !== null) {
      this.o.clock.clearInterval(this.healthTimer)
      this.healthTimer = null
    }
  }

  private async probeHealth(): Promise<void> {
    if (this._state !== 'up' || this.handshake_ === null || this.disposed) return
    const client = new SidecarClient(this.handshake_, this.o.fetchImpl)
    const health = await client.health(this.o.healthTimeoutMs)
    if (this._state !== 'up') return
    if (health !== null && health.ok) {
      this.healthFailures = 0
      return
    }
    this.healthFailures += 1
    if (this.healthFailures >= this.o.healthFailureLimit) {
      this.o.logger?.warn?.(`[dsh-embed/${this.id}] ${this.healthFailures} consecutive health failures; killing for restart`)
      this.healthFailures = 0
      await this.killAndRestart()
    }
  }

  private async killAndRestart(): Promise<void> {
    const pid = this.handshake_?.pid
    if (this._state === 'up' || this._state === 'starting') this._state = 'stopping'
    this.killIntent = 'restart'
    this.stopHealthWatch()
    await this.killChildBestEffort()
    await this.killPidBestEffort(pid)
    await this.o.runtime.remove(this.id, pid)
    this.killIntent = 'none'
    this.scheduleRestart('killed after health-check failures')
  }

  // ── keep-alive 與停止 ────────────────────────────────────────────────────

  private sweepIdle(): void {
    if (this.disposed) return
    if (this.o.keepAliveMs <= 0) return
    if (this._state !== 'up') return
    if (this.o.clock.now() - this.lastUsed_ < this.o.keepAliveMs) return
    void this.stopIdle().catch(() => {})
  }

  /** 空閒關閉:優雅殺進程,狀態歸 down,重置退避計數。 */
  async stopIdle(): Promise<void> {
    if (this._state !== 'up') return
    this._state = 'stopping'
    this.killIntent = 'idle'
    this.stopHealthWatch()
    await this.killChildBestEffort()
    await this.killPidBestEffort(this.handshake_?.pid)
    await this.o.runtime.remove(this.id, this.handshake_?.pid)
    this.handshake_ = null
    this.killIntent = 'none'
    this._state = 'down'
    this.attempts = 0
    this.o.logger?.info?.(`[dsh-embed/${this.id}] idle shutdown (keep-alive ${this.o.keepAliveMs}ms)`)
  }

  async dispose(): Promise<void> {
    this.disposed = true
    this.stopHealthWatch()
    this.cancelBackoffTimer()
    if (this.sweepTimer !== null) {
      this.o.clock.clearInterval(this.sweepTimer)
      this.sweepTimer = null
    }
    if (this.startPromise_ !== null) {
      await this.startPromise_.catch(() => {})
    }
    if (this.child_ !== null || this.handshake_ !== null) {
      this.killIntent = 'idle'
      await this.killChildBestEffort()
      await this.killPidBestEffort(this.handshake_?.pid)
    }
    this.killIntent = 'none'
    // pid 守衛:僅當文件仍屬本 supervisor 管轄的進程時才刪(防誤刪並行實例的握手)
    await this.o.runtime.remove(this.id, this.handshake_?.pid)
    this.handshake_ = null
    this.child_ = null
    this._state = 'down'
  }

  // ── 進程終止輔助 ─────────────────────────────────────────────────────────

  private async killChildBestEffort(): Promise<void> {
    const child = this.child_
    this.child_ = null
    if (child === null) return
    try {
      await child.kill(this.o.killGraceMs)
    } catch {
      // best-effort:SIGKILL 也殺不掉時僅記錄
      this.o.logger?.warn?.(`[dsh-embed/${this.id}] child kill raised; continuing`)
    }
  }

  private async killPidBestEffort(pid: number | undefined): Promise<void> {
    if (pid === undefined) return
    if (!this.pidAlive(pid)) return
    try {
      await (this.o.killPid ?? defaultKillPid)(pid, this.o.killGraceMs)
    } catch {
      this.o.logger?.warn?.(`[dsh-embed/${this.id}] killPid(${pid}) raised; continuing`)
    }
  }

  private cancelBackoffTimer(): void {
    if (this.backoffTimer !== null) {
      this.o.clock.clearTimeout(this.backoffTimer)
      this.backoffTimer = null
    }
  }
}

/** 生產默認:按進程組 SIGTERM → 寬限 → SIGKILL。測試注入替代。 */
export async function defaultKillPid(pid: number, graceMs: number): Promise<void> {
  const send = (signal: NodeJS.Signals) => {
    try {
      process.kill(-pid, signal)
    } catch {
      try {
        process.kill(pid, signal)
      } catch {
        /* already gone */
      }
    }
  }
  send('SIGTERM')
  const deadline = Date.now() + graceMs
  while (Date.now() < deadline) {
    if (!isPidAlive(pid)) return
    await new Promise((resolve) => setTimeout(resolve, 100))
  }
  send('SIGKILL')
  const hardDeadline = Date.now() + 2000
  while (Date.now() < hardDeadline) {
    if (!isPidAlive(pid)) return
    await new Promise((resolve) => setTimeout(resolve, 100))
  }
}
