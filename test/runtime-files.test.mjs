/** runtime 握手文件層單元(SPEC §2):~展開、原子讀、畸形容忍、守衛刪除。 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtemp, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import os from 'node:os'
import { RuntimeDir, expandTilde, isPidAlive } from '../lib/index.js'

test('expandTilde: ~ / ~/x / 無波浪號', () => {
  assert.equal(expandTilde('~'), os.homedir())
  assert.equal(expandTilde('~/run/dsh-embed'), path.join(os.homedir(), 'run/dsh-embed'))
  assert.equal(expandTilde('/abs/path'), '/abs/path')
  assert.equal(expandTilde('~nothome/x'), '~nothome/x')
})

test('isPidAlive: 自身 pid 存活;亂數 pid 不存活;非法輸入 false', () => {
  assert.equal(isPidAlive(process.pid), true)
  assert.equal(isPidAlive(undefined), false)
  assert.equal(isPidAlive(0), false)
  assert.equal(isPidAlive(-1), false)
  assert.equal(isPidAlive(9_000_001), false)
})

test('RuntimeDir: 寫讀往返 + ensureDir + 畸形返回 null', async () => {
  const dir = new RuntimeDir(await mkdtemp(path.join(tmpdir(), 'rt-')))
  await dir.ensureDir()
  assert.equal(await dir.read('mlx'), null)

  dir.writeSyncForTest('mlx', { port: 12345, token: 'a'.repeat(64), pid: 4242 })
  assert.deepEqual(await dir.read('mlx'), { port: 12345, token: 'a'.repeat(64), pid: 4242 })

  await writeFile(dir.file('tf'), '{ not json', 'utf8')
  assert.equal(await dir.read('tf'), null)

  // 字段非法(port/token/pid 域外)→ null
  await writeFile(dir.file('tf'), JSON.stringify({ port: 0, token: 'short', pid: -2 }), 'utf8')
  assert.equal(await dir.read('tf'), null)

  // 額外欄位容忍(sidecar handshake 附帶 backends 列表)
  await writeFile(dir.file('tf'), JSON.stringify({ port: 80, token: 'b'.repeat(64), pid: 9, extra: [1, 2] }), 'utf8')
  assert.deepEqual(await dir.read('tf'), { port: 80, token: 'b'.repeat(64), pid: 9 })
})

test('RuntimeDir.remove: 守衛刪除(僅當文件仍屬指定 pid)', async () => {
  const dir = new RuntimeDir(await mkdtemp(path.join(tmpdir(), 'rt-')))
  await dir.ensureDir()
  dir.writeSyncForTest('mlx', { port: 1, token: 'c'.repeat(64), pid: 111 })
  await dir.remove('mlx', 222) // pid 不符 → 不刪
  assert.notEqual(await dir.read('mlx'), null)
  await dir.remove('mlx', 111) // pid 相符 → 刪
  assert.equal(await dir.read('mlx'), null)
  await dir.remove('mlx') // 無條件刪(冪等)
})
