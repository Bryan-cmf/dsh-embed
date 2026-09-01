/**
 * dsh-embed 單元測試基建:手動時鐘、假進程、契約 mock sidecar(真 HTTP)。
 *
 * 全部 supervisor 狀態轉移在「假時鐘 + 假進程 + 127.0.0.1 真 HTTP mock」
 * 下測試——不依賴 Python/權重/真實時間;fetch 走真迴環網絡,契約解析
 * 路徑與生產完全一致。
 */
import http from 'node:http'
import assert from 'node:assert/strict'

// ── ManualClock:虛擬時間 + 可驅動定時器 ────────────────────────────────────

export class ManualClock {
  constructor(start = 1_000_000) {
    this.t = start
    this.next = 1
    this.timers = new Map()
  }

  now() {
    return this.t
  }

  setTimeout(callback, ms) {
    const id = this.next++
    this.timers.set(id, { callback, at: this.t + ms, ms, interval: false })
    return id
  }

  clearTimeout(handle) {
    this.timers.delete(handle)
  }

  setInterval(callback, ms) {
    const id = this.next++
    this.timers.set(id, { callback, at: this.t + ms, ms, interval: true })
    return id
  }

  clearInterval(handle) {
    this.timers.delete(handle)
  }

  sleep(ms) {
    return new Promise((resolve) => {
      this.setTimeout(resolve, ms)
    })
  }

  /** 推進虛擬時間到 target,依到期順序觸發定時器;每次觸發後讓真 I/O 跑。 */
  async flush(target) {
    for (;;) {
      let dueId = null
      let dueAt = Infinity
      for (const [id, timer] of this.timers) {
        if (timer.at <= target && timer.at < dueAt) {
          dueId = id
          dueAt = timer.at
        }
      }
      if (dueId === null) break
      const timer = this.timers.get(dueId)
      this.t = Math.max(this.t, timer.at)
      if (timer.interval) timer.at += timer.ms
      else this.timers.delete(dueId)
      await timer.callback()
      // 每次觸發間讓真實事件循環完整轉一圈(check+poll 相),fs/fetch 得以收尾
      await new Promise((resolve) => setImmediate(resolve))
      await new Promise((resolve) => globalThis.setTimeout(resolve, 0))
    }
    this.t = Math.max(this.t, target)
  }

  async advance(ms) {
    await this.flush(this.t + ms)
  }
}

/**
 * 驅動一個依賴虛擬時鐘的 promise 到 settle:輪流泵時間與真 I/O;
 * 虛擬時間與真實時間雙重上界,防測試掛起。
 */
export async function drive(clock, promise, maxVirtualMs = 10_000_000, maxRealMs = 15_000) {
  let settled = false
  let failure = undefined
  const guarded = promise.then(
    (value) => { settled = true; return value },
    (error) => { settled = true; failure = error },
  )
  let pumped = 0
  const startedAt = Date.now()
  while (!settled) {
    const before = clock.t
    await clock.flush(before + 500)
    pumped += clock.t - before
    assert.ok(pumped <= maxVirtualMs, `drive: promise did not settle within ${maxVirtualMs}ms virtual time`)
    assert.ok(Date.now() - startedAt <= maxRealMs, `drive: promise did not settle within ${maxRealMs}ms real time (state machine starved)`)
    await new Promise((resolve) => setImmediate(resolve))
    await new Promise((resolve) => globalThis.setTimeout(resolve, 0))
  }
  const value = await guarded
  if (failure !== undefined) throw failure
  return value
}

/** 讓真實事件循環跑若干毫秒(虛擬時鐘外的 fs/fetch 收尾)。 */
export async function settleIo(realMs = 300) {
  const deadline = Date.now() + realMs
  while (Date.now() < deadline) {
    await new Promise((resolve) => globalThis.setTimeout(resolve, 5))
  }
}

/** 泵虛擬時間+真 I/O 直到條件成立(雙重上界防掛起)。 */
export async function until(clock, predicate, maxVirtualMs = 600_000, maxRealMs = 5_000) {
  const startReal = Date.now()
  const startVirt = clock.t
  while (!predicate()) {
    await clock.flush(clock.t + 200)
    assert.ok(clock.t - startVirt <= maxVirtualMs, `until: predicate not met within ${maxVirtualMs}ms virtual time`)
    assert.ok(Date.now() - startReal <= maxRealMs, `until: predicate not met within ${maxRealMs}ms real time`)
    await new Promise((resolve) => setImmediate(resolve))
    await new Promise((resolve) => globalThis.setTimeout(resolve, 0))
  }
}

// ── 假進程(pid 取 9_000_000+,不與真實進程衝突) ────────────────────────────

export function makeFakeProcess(spec, pid) {
  const state = { exited: false, code: null, signal: null, handlers: [], killSignals: [] }
  const proc = {
    spec,
    pid: pid ?? 9_000_000 + Math.floor(Math.random() * 99_999),
    get exited() {
      return state.exited
    },
    get killSignals() {
      return state.killSignals
    },
    onExit(callback) {
      if (state.exited) callback(state.code, state.signal)
      else state.handlers.push(callback)
    },
    /** 依 SpawnedProcess 契約:參數是寬限毫秒;假進程立即優雅退出。 */
    kill(_graceMs = 5_000) {
      state.killSignals.push('SIGTERM')
      proc.exit(null, 'SIGTERM')
      return Promise.resolve()
    },
    /** 模擬退出/崩潰(code+signal 二選一)。 */
    exit(code = 1, signal = null) {
      if (state.exited) return
      state.exited = true
      state.code = code
      state.signal = signal
      for (const callback of state.handlers) callback(code, signal)
      state.handlers = []
    },
  }
  return proc
}

// ── 契約 mock sidecar:127.0.0.1 真 HTTP,SPEC §4 形狀 ─────────────────────

export function unitVector(n) {
  const v = []
  let norm = 0
  for (let i = 0; i < n; i++) {
    const x = 1 + ((i * 2_654_435_761) % 97) / 97
    v.push(x)
    norm += x * x
  }
  const s = Math.sqrt(norm)
  return v.map((x) => x / s)
}

export async function startMockSidecar(options = {}) {
  const token = 'm'.repeat(64)
  const state = {
    healthy: options.healthy ?? true,
    failEmbeds: options.failEmbeds ?? false,
    fullDim: options.fullDim ?? false,
    /** 非空時嵌入回應用它替代正確指紋(測 mismatch 防禦)。 */
    fpOverride: '',
    requests: [],
    closed: false,
  }
  const backendsList = options.backends ?? [
    { name: 'qwen3-4b-fp16', model: 'Qwen/Qwen3-Embedding-4B', dims: [512, 2560], modalities: ['text'], fingerprint: 'qwen3-4b-fp16@512', alive: true },
  ]
  const send = (res, status, payload) => {
    const body = JSON.stringify(payload)
    res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(body) })
    res.end(body)
  }
  const server = http.createServer((req, res) => {
    let raw = ''
    req.on('data', (chunk) => { raw += chunk })
    req.on('end', () => {
      let body = null
      try {
        body = raw === '' ? null : JSON.parse(raw)
      } catch {
        send(res, 400, { error: { code: 'bad_json', message: 'unparseable body' } })
        return
      }
      state.requests.push({ method: req.method, url: req.url, token: req.headers['x-embed-token'], body })
      if (req.headers['x-embed-token'] !== token) {
        send(res, 401, { error: { code: 'unauthorized', message: 'missing or invalid X-Embed-Token' } })
        return
      }
      if (req.method === 'GET' && req.url === '/health') {
        send(res, 200, { ok: state.healthy, uptime_s: 1.5, backend: options.id ?? 'tf' })
        return
      }
      if (req.method === 'GET' && req.url === '/backends') {
        send(res, 200, backendsList)
        return
      }
      if (req.method === 'POST' && req.url === '/embed/texts') {
        const b = body ?? {}
        if (state.failEmbeds) {
          send(res, 500, { error: { code: 'internal', message: 'simulated failure', fingerprint: `${b.backend ?? 'x'}@${b.dim ?? 512}` } })
          return
        }
        if (!Array.isArray(b.texts)) {
          send(res, 400, { error: { code: 'invalid_texts', message: 'texts must be an array' } })
          return
        }
        const dim = b.dim ?? 512
        const width = state.fullDim ? 2560 : dim
        const fingerprintName = b.backend ?? backendsList[0].name
        send(res, 200, { vectors: b.texts.map(() => unitVector(width)), fingerprint: state.fpOverride !== '' ? state.fpOverride : `${fingerprintName}@${dim}`, dim, ms: 1.2 })
        return
      }
      if (req.method === 'POST' && req.url === '/embed/image') {
        const b = body ?? {}
        if (state.failEmbeds) {
          send(res, 500, { error: { code: 'internal', message: 'simulated failure' } })
          return
        }
        const dim = b.dim ?? 512
        const width = state.fullDim ? 2048 : dim
        send(res, 200, { vector: unitVector(width), fingerprint: `${b.backend ?? 'wemm2b-mlx4b'}@${dim}`, dim, ms: 2.5 })
        return
      }
      send(res, 404, { error: { code: 'not_found', message: req.url } })
    })
  })
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve))
  return {
    port: server.address().port,
    token,
    state,
    backendsList,
    stop: async () => {
      if (state.closed) return
      state.closed = true
      await new Promise((resolve) => server.close(resolve))
    },
  }
}

// ── supervisor 測試 rig ─────────────────────────────────────────────────────

import { mkdtemp } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { RuntimeDir, SidecarSupervisor } from '../lib/index.js'

/**
 * 一個被測 supervisor + 配套 mock 世界:
 * - runtime 目錄:臨時目錄
 * - spawn:假進程;進程「運行」時啟動真 mock HTTP 並寫握手文件,
 *   進程退出/被殺時關閉 HTTP 並清握手文件(模擬真 sidecar 行為)
 * - pidAlive:alivePids 集合(假進程 pid 自動進出)
 */
export async function makeRig(overrides = {}) {
  const dir = await mkdtemp(path.join(tmpdir(), 'dsh-embed-test-'))
  const runtime = new RuntimeDir(dir)
  await runtime.ensureDir()
  const clock = new ManualClock()
  const alivePids = new Set()
  const killedPids = []
  const procs = []
  const servers = []
  const spawnedSpecs = []
  const behavior = { failSpawn: false, spawnDelayMs: 0, healthy: true, ...overrides.behavior }
  const id = overrides.id ?? 'tf'

  const spawnProcess = (spec, logFile) => {
    spawnedSpecs.push(spec)
    const proc = makeFakeProcess(spec)
    procs.push(proc)
    alivePids.add(proc.pid)
    // 假進程「運行」:異步啟動 mock sidecar + 寫握手(真進程要數秒,這裡即時)
    void (async () => {
      await new Promise((resolve) => setImmediate(resolve))
      if (behavior.spawnDelayMs > 0) await clock.sleep(behavior.spawnDelayMs)
      if (proc.exited) return
      if (behavior.failSpawn) {
        proc.exit(3, null)
        return
      }
      const server = await startMockSidecar({ id: spec.id, healthy: behavior.healthy })
      servers.push(server)
      proc.server = server
      runtime.writeSyncForTest(spec.id, { port: server.port, token: server.token, pid: proc.pid })
    })()
    return proc
  }
  // 進程退出 → 關 HTTP、清握手文件、pid 出存活集(模擬真 sidecar 的信號清理)
  const hookProcessExit = (proc) => {
    proc.onExit(() => {
      alivePids.delete(proc.pid)
      void proc.server?.stop()
      void runtime.remove(proc.spec.id, proc.pid)
    })
  }
  const originalSpawn = spawnProcess
  const wrappedSpawn = (spec, logFile) => {
    const proc = originalSpawn(spec, logFile)
    hookProcessExit(proc)
    return proc
  }

  const sup = new SidecarSupervisor({
    id,
    enabled: overrides.enabled ?? true,
    python: '/fake/venv/bin/python',
    script: `/fake/src/sidecar/${id}_serve.py`,
    scriptArgs: overrides.scriptArgs ?? [],
    runtime,
    clock,
    spawnProcess: wrappedSpawn,
    pidAlive: (pid) => alivePids.has(pid),
    killPid: async (pid, _grace) => {
      killedPids.push(pid)
      alivePids.delete(pid)
      const proc = procs.find((p) => p.pid === pid)
      if (proc !== undefined && !proc.exited) proc.exit(null, 'SIGTERM')
    },
    logger: overrides.logger,
    handshakePollMs: 200,
    startupTimeoutMs: 60_000,
    healthIntervalMs: 30_000,
    healthFailureLimit: 3,
    keepAliveMs: overrides.keepAliveMs ?? 900_000,
    keepAliveSweepMs: 5_000,
    backoffBaseMs: 1_000,
    backoffFactor: 4,
    maxRestartAttempts: 3,
    failedCooldownMs: 30_000,
    killGraceMs: 100,
    ...(overrides.supervisor ?? {}),
  })

  return { sup, clock, runtime, procs, servers, spawnedSpecs, alivePids, killedPids, behavior, dir }
}
