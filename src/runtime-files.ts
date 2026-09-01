/**
 * dsh-embed · runtime 握手文件層(SPEC §2)。
 *
 * 目錄默認 `~/.dsh/run/dsh-embed/`,內含 `mlx.json` / `tf.json`,
 * 由 sidecar 進程在綁定 127.0.0.1:0 後寫入 `{port, token, pid}`;
 * 插件輪詢這些文件完成發現。陳舊文件(pid 已死)由 supervisor 清理。
 */
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import type { RuntimeHandshake, SidecarId } from './types.ts'

export function expandTilde(p: string): string {
  if (p === '~') return os.homedir()
  if (p.startsWith('~/')) return path.join(os.homedir(), p.slice(2))
  return p
}

/** 進程是否存活(pid 復用窗口極小,僅供握手文件陳舊性初篩)。 */
export function isPidAlive(pid: number | undefined): boolean {
  if (typeof pid !== 'number' || !Number.isInteger(pid) || pid <= 0) return false
  try {
    process.kill(pid, 0)
    return true
  } catch (error) {
    return (error as NodeJS.ErrnoException).code === 'EPERM'
  }
}

export interface RuntimeDirDeps {
  fs?: typeof fsp
  syncFs?: typeof fs
}

export class RuntimeDir {
  readonly root: string
  private readonly fs_: typeof fsp
  private readonly syncFs_: typeof fs

  constructor(root: string, deps: RuntimeDirDeps = {}) {
    this.root = expandTilde(root)
    this.fs_ = deps.fs ?? fsp
    this.syncFs_ = deps.syncFs ?? fs
  }

  file(id: SidecarId): string {
    return path.join(this.root, `${id}.json`)
  }

  logFile(id: SidecarId): string {
    return path.join(this.root, `${id}.log`)
  }

  async ensureDir(): Promise<void> {
    await this.fs_.mkdir(this.root, { recursive: true })
  }

  /** 讀握手文件;缺失/畸形/字段非法返回 null。 */
  async read(id: SidecarId): Promise<RuntimeHandshake | null> {
    let raw: string
    try {
      raw = await this.fs_.readFile(this.file(id), 'utf8')
    } catch {
      return null
    }
    try {
      const parsed = JSON.parse(raw) as Partial<RuntimeHandshake>
      const { port, token, pid } = parsed
      if (
        typeof port === 'number' && Number.isInteger(port) && port > 0 && port < 65536 &&
        typeof token === 'string' && token.length >= 16 &&
        typeof pid === 'number' && Number.isInteger(pid) && pid > 0
      ) {
        return { port, token, pid }
      }
      return null
    } catch {
      return null
    }
  }

  /** 刪除握手文件(守衛:僅當內容仍屬指定 pid,避免刪掉後繼進程的文件)。 */
  async remove(id: SidecarId, onlyIfPid?: number): Promise<void> {
    if (onlyIfPid !== undefined) {
      const current = await this.read(id)
      if (current !== null && current.pid !== onlyIfPid) return
    }
    await this.fs_.rm(this.file(id), { force: true })
  }

  /** 測試輔助:同步寫握手文件。 */
  writeSyncForTest(id: SidecarId, handshake: RuntimeHandshake): void {
    this.syncFs_.mkdirSync(this.root, { recursive: true })
    this.syncFs_.writeFileSync(this.file(id), JSON.stringify(handshake))
  }
}
