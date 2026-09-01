/**
 * 進程監管狀態機單元(SPEC §2,全部規則逐條):
 * 懶啟動 / 握手發現 / 孤兒收養與清理 / 崩潰退避重啟 / 上限→failed→冷卻 /
 * 健康檢查 3 連敗→殺+重啟 / keep-alive 空閒退出 / dispose 可逆性。
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { EmbedderUnavailableError } from '../lib/index.js'
import { drive, makeRig, settleIo, startMockSidecar, until } from './helpers.mjs'

test('懶啟動:無需求不 spawn;首次 ensureStarted → 握手發現 → up', async () => {
  const rig = await makeRig()
  assert.equal(rig.sup.state, 'down')
  assert.equal(rig.spawnedSpecs.length, 0)

  await drive(rig.clock, rig.sup.ensureStarted())
  assert.equal(rig.sup.state, 'up')
  assert.equal(rig.spawnedSpecs.length, 1)
  assert.ok(rig.sup.handshake !== null)
  assert.ok(rig.sup.handshake.port > 0)
  // spawn 規格:runtime 目錄與腳本按配置傳入
  const spec = rig.spawnedSpecs[0]
  assert.equal(spec.python, '/fake/venv/bin/python')
  assert.equal(spec.script, '/fake/src/sidecar/tf_serve.py')
  assert.equal(spec.runtimeDir, rig.runtime.root)
  await rig.sup.dispose()
})

test('up 態重複需求不重複 spawn', async () => {
  const rig = await makeRig()
  await drive(rig.clock, rig.sup.ensureStarted())
  await drive(rig.clock, rig.sup.ensureStarted())
  await drive(rig.clock, rig.sup.ensureStarted())
  assert.equal(rig.spawnedSpecs.length, 1)
  await rig.sup.dispose()
})

test('併發需求:兩個同時 ensureStarted 只 spawn 一次', async () => {
  const rig = await makeRig()
  const p1 = rig.sup.ensureStarted()
  const p2 = rig.sup.ensureStarted()
  await drive(rig.clock, Promise.all([p1, p2]))
  assert.equal(rig.sup.state, 'up')
  assert.equal(rig.spawnedSpecs.length, 1)
  await rig.sup.dispose()
})

test('keep-alive:空閒窗口後退出(down、進程死、握手文件清、計數歸零);再需求重啟', async () => {
  const rig = await makeRig({ keepAliveMs: 900_000 })
  await drive(rig.clock, rig.sup.ensureStarted())
  const first = rig.procs[0]
  assert.equal(rig.sup.state, 'up')

  // 推進超過 keep-alive + sweep 週期;期間健康探測(每30s)不阻擋退出
  await rig.clock.advance(900_000 + 20_000)
  await settleIo()
  assert.equal(rig.sup.state, 'down')
  assert.equal(first.exited, true)
  assert.equal(first.killSignals.includes('SIGTERM'), true)
  assert.equal(await rig.runtime.read('tf'), null)

  // 再調用恢復服務(新進程)
  await drive(rig.clock, rig.sup.ensureStarted())
  assert.equal(rig.sup.state, 'up')
  assert.equal(rig.spawnedSpecs.length, 2)
  await rig.sup.dispose()
})

test('keep-alive 使用中不清:嵌入調用刷新 lastUsed', async () => {
  const rig = await makeRig({ keepAliveMs: 600_000 })
  await drive(rig.clock, rig.sup.ensureStarted())
  await rig.clock.advance(400_000)
  await settleIo()
  rig.sup.touch() // 模擬一次嵌入調用
  await rig.clock.advance(400_000) // 距 touch 僅 400s < 600s
  assert.equal(rig.sup.state, 'up')
  await rig.sup.dispose()
})

test('崩潰:up 態非計劃退出 → 指數退避自動重啟(attempts 歸零於成功)', async () => {
  const rig = await makeRig()
  await drive(rig.clock, rig.sup.ensureStarted())
  const first = rig.procs[0]
  first.exit(1, null) // kill -9 模擬(非計劃)

  await rig.clock.advance(3_000) // backoff 1s → 重啟
  await until(rig.clock, () => rig.sup.state === 'up')
  assert.equal(rig.spawnedSpecs.length, 2)
  assert.equal(rig.sup.attemptsCount, 0) // 成功啟動歸零
  await rig.sup.dispose()
})

test('連續 3 次啟動失敗 → failed;冷卻期內需求即拋;冷卻後自愈', async () => {
  const rig = await makeRig({ behavior: { failSpawn: true } })
  // 失敗語義:需求路徑首次失敗即拋(SPEC §3「立即拋」);背景退避自動跑完
  await assert.rejects(drive(rig.clock, rig.sup.ensureStarted()), EmbedderUnavailableError)
  await rig.clock.advance(10_000) // 退避 1s、4s → 第 3 次失敗 → failed
  await settleIo()
  assert.equal(rig.sup.state, 'failed')
  assert.equal(rig.spawnedSpecs.length, 3)

  // 冷卻期內:立即拋,不 spawn
  await assert.rejects(rig.sup.ensureStarted(), EmbedderUnavailableError)
  assert.equal(rig.spawnedSpecs.length, 3)

  // 冷卻結束 + 故障修復 → 需求驅動新一輪成功
  await rig.clock.advance(31_000)
  rig.behavior.failSpawn = false
  await drive(rig.clock, rig.sup.ensureStarted())
  assert.equal(rig.sup.state, 'up')
  assert.equal(rig.spawnedSpecs.length, 4)
  await rig.sup.dispose()
})

test('健康檢查:3 連敗 → 殺舊進程 + 重啟', async () => {
  const rig = await makeRig()
  await drive(rig.clock, rig.sup.ensureStarted())
  const first = rig.procs[0]
  rig.servers[0].state.healthy = false // 僅舊 server 病;新進程默認健康
  rig.behavior.healthy = true

  // 30/60/90s 三次探測失敗 → 90s 殺 + 1s 退避 → 91s 起 commit 新進程(默認健康)
  await rig.clock.advance(95_000)
  await settleIo()
  await until(rig.clock, () => rig.sup.state === 'up')
  assert.equal(rig.spawnedSpecs.length, 2)
  assert.equal(first.exited, true)
  await rig.sup.dispose()
})

test('孤兒收養:握手文件指向存活健康 sidecar → 不 spawn 直接 up', async () => {
  const rig = await makeRig()
  // 模擬上次 host 崩潰殘留的 sidecar(不在 procs 內,pid 在存活集)
  const orphan = await startMockSidecar({ id: 'tf' })
  const orphanPid = 9_100_001
  rig.alivePids.add(orphanPid)
  rig.runtime.writeSyncForTest('tf', { port: orphan.port, token: orphan.token, pid: orphanPid })

  await drive(rig.clock, rig.sup.ensureStarted())
  assert.equal(rig.sup.state, 'up')
  assert.equal(rig.spawnedSpecs.length, 0) // 收養,未 spawn
  assert.equal(rig.sup.handshake.pid, orphanPid)
  await rig.sup.dispose() // dispose 應連收養孤兒一起殺
  assert.equal(rig.killedPids.includes(orphanPid), true)
  await orphan.stop()
})

test('殘留不健康孤兒:殺掉 + 清文件 + 全新 spawn', async () => {
  const rig = await makeRig()
  const orphan = await startMockSidecar({ id: 'tf' })
  const orphanPid = 9_100_002
  rig.alivePids.add(orphanPid)
  rig.runtime.writeSyncForTest('tf', { port: orphan.port, token: 'wrong-token-0000000000000000000000000000000000000000000000000000000000', pid: orphanPid })

  await drive(rig.clock, rig.sup.ensureStarted())
  // 401/不健康 → 殺孤兒 → 全新 spawn 成功
  assert.equal(rig.sup.state, 'up')
  assert.equal(rig.killedPids.includes(orphanPid), true)
  assert.equal(rig.spawnedSpecs.length, 1)
  await rig.sup.dispose()
  await orphan.stop()
})

test('死 pid 殘留文件:直接清理(不殺進程)後全新 spawn', async () => {
  const rig = await makeRig()
  rig.runtime.writeSyncForTest('tf', { port: 59999, token: 'd'.repeat(64), pid: 9_100_003 }) // pid 不在 alivePids

  await drive(rig.clock, rig.sup.ensureStarted())
  assert.equal(rig.sup.state, 'up')
  assert.equal(rig.killedPids.length, 0) // 死 pid 無需 kill
  assert.equal(rig.spawnedSpecs.length, 1)
  await rig.sup.dispose()
})

test('配置禁用:ensureStarted 即拋 EmbedderUnavailableError,絕不 spawn', async () => {
  const rig = await makeRig({ enabled: false })
  await assert.rejects(rig.sup.ensureStarted(), EmbedderUnavailableError)
  assert.equal(rig.spawnedSpecs.length, 0)
  assert.equal(rig.sup.state, 'down')
})

test('dispose:殺進程、清文件、撤定時器(推進時鐘無殭屍重啟)', async () => {
  const rig = await makeRig()
  await drive(rig.clock, rig.sup.ensureStarted())
  await rig.sup.dispose()
  assert.equal(rig.sup.state, 'down')
  assert.equal(rig.procs[0].exited, true)
  assert.equal(await rig.runtime.read('tf'), null)

  await rig.clock.advance(3_600_000) // 一小時虛擬時間
  assert.equal(rig.sup.state, 'down')
  assert.equal(rig.spawnedSpecs.length, 1) // 無殭屍重啟
})

test('啟動超時:握手文件遲遲不現 → EmbedderUnavailableError + 退避', async () => {
  const rig = await makeRig({ behavior: { spawnDelayMs: 10_000_000 } }) // 永不完成握手
  await assert.rejects(drive(rig.clock, rig.sup.ensureStarted()), EmbedderUnavailableError)
  assert.notEqual(rig.sup.state, 'up') // backoff(第1次失敗)或之後
  await rig.sup.dispose()
})
