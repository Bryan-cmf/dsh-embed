/**
 * dsh-embed · 生產時鐘與進程生成(node:timers / node:child_process)。
 * 測試以 ManualClock / 假 spawner 注入同介面(supervisor.ts 依賴倒置)。
 */
import fs from 'node:fs'
import { spawn } from 'node:child_process'
import type { Clock, SpawnSpec, SpawnedProcess } from './supervisor.ts'

export class SystemClock implements Clock {  now(): number {
    return Date.now()
  }

  setTimeout(callback: () => void, ms: number): unknown {
    return globalThis.setTimeout(callback, ms)
  }

  clearTimeout(handle: unknown): void {
    if (handle !== null && handle !== undefined) globalThis.clearTimeout(handle as Parameters<typeof globalThis.clearTimeout>[0])
  }

  setInterval(callback: () => void, ms: number): unknown {
    return globalThis.setInterval(callback, ms)
  }

  clearInterval(handle: unknown): void {
    if (handle !== null && handle !== undefined) globalThis.clearInterval(handle as Parameters<typeof globalThis.clearInterval>[0])
  }

  sleep(ms: number): Promise<void> {
    return new Promise((resolve) => globalThis.setTimeout(resolve, ms))
  }
}

/**
 * sidecar 命令行拼裝(唯一真相;測試用同一函數解析)。
 * 契約見 src/sidecar/README.md:`python <script> --runtime-dir <dir> --name <id> [...args]`。
 */
export function buildSidecarArgs(spec: SpawnSpec): string[] {
  return [spec.script, '--runtime-dir', spec.runtimeDir, '--name', spec.id, ...spec.args]
}

/**
 * spawn sidecar:detached 進程組(可組級 kill),stdio 重定向到 runtime 目錄
 * 下的 `<id>.log`(append),unref 不阻擋 host 退出。
 */
export function nodeSpawn(spec: SpawnSpec, logFile: string): SpawnedProcess {
  const fd = fs.openSync(logFile, 'a')
  const args = buildSidecarArgs(spec)
  const child = spawn(spec.python, args, {
    detached: true,
    stdio: ['ignore', fd, fd],
    env: { ...process.env, ...spec.env },
  })
  child.unref()
  const exitCallbacks: Array<(code: number | null, signal: string | null) => void> = []
  let exited = false
  let exitCode: number | null = null
  let exitSignal: string | null = null
  let exitWaiter: (() => void) | null = null
  const finish = (code: number | null, signal: string | null): void => {
    if (exited) return
    exited = true
    exitCode = code
    exitSignal = signal
    try {
      fs.closeSync(fd)
    } catch {
      /* already closed */
    }
    for (const cb of exitCallbacks) cb(code, signal)
    exitWaiter?.()
    exitWaiter = null
  }
  child.on('exit', (code, signal) => finish(code, signal))
  // spawn 失敗(如 venv/python 缺失)只發 'error' 不發 'exit'——
  // 映射為立即退出,讓 supervisor 的啟動輪詢快速失敗
  child.on('error', () => finish(null, 'SPAWN_ERROR'))
  const waitForExit = (timeoutMs: number): Promise<boolean> =>
    new Promise((resolve) => {
      if (exited) return resolve(true)
      const timer = globalThis.setTimeout(() => resolve(false), timeoutMs)
      exitWaiter = () => {
        globalThis.clearTimeout(timer)
        resolve(true)
      }
    })
  const send = (signal: NodeJS.Signals): void => {
    if (exited || child.pid === undefined) return
    try {
      process.kill(-child.pid, signal)
    } catch {
      try {
        process.kill(child.pid, signal)
      } catch {
        /* already gone */
      }
    }
  }
  return {
    pid: child.pid,
    onExit(callback) {
      if (exited) callback(exitCode, exitSignal)
      else exitCallbacks.push(callback)
    },
    async kill(graceMs: number): Promise<void> {
      send('SIGTERM')
      if (await waitForExit(graceMs)) return
      send('SIGKILL')
      await waitForExit(2000)
    },
  }
}
