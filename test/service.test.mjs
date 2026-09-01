/**
 * `embedder` 服務單元(SPEC §3 契約):
 * 後端路由 / 失敗語義(無跨後端替補)/ 輸入驗證 / 指紋校驗 /
 * 客戶端 MRL 保證 / backends() 與 health() 投影。
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  EmbedderResponseError,
  EmbedderServiceImpl,
  EmbedderUnavailableError,
  EmbedderValidationError,
  l2norm,
} from '../lib/index.js'
import { drive, makeRig } from './helpers.mjs'

async function makeService(overrides = {}) {
  const tfRig = await makeRig({ id: 'tf', ...(overrides.tf ?? {}) })
  const mlxRig = await makeRig({ id: 'mlx', ...(overrides.mlx ?? {}) })
  const service = new EmbedderServiceImpl({
    supervisors: { tf: tfRig.sup, mlx: mlxRig.sup },
    defaults: { textBackend: 'qwen3-4b-fp16', visualBackend: 'wemm2b-mlx4b', dim: 512 },
    ...(overrides.service ?? {}),
  })
  return { service, tfRig, mlxRig }
}

test('默認文本路由 → tf sidecar 懶啟動;mlx 不動', async () => {
  const { service, tfRig, mlxRig } = await makeService()
  const vectors = await drive(tfRig.clock, service.embedTexts(['hello', 'world']))
  assert.equal(vectors.length, 2)
  assert.ok(vectors[0] instanceof Float32Array)
  assert.equal(vectors[0].length, 512)
  assert.ok(Math.abs(l2norm(vectors[0]) - 1) < 1e-6)
  assert.equal(tfRig.spawnedSpecs.length, 1)
  assert.equal(mlxRig.spawnedSpecs.length, 0) // 未觸碰
  // 請求體:顯式 backend + dim
  const request = tfRig.servers[0].state.requests.find((r) => r.url === '/embed/texts')
  assert.equal(request.body.backend, 'qwen3-4b-fp16')
  assert.equal(request.body.dim, 512)
  assert.equal(request.token, tfRig.servers[0].token)
  await tfRig.sup.dispose()
  await mlxRig.sup.dispose()
})

test('默認圖像路由 → mlx sidecar;tf 不動', async () => {
  const statCalls = []
  const { service, tfRig, mlxRig } = await makeService({
    service: { statFile: async (p) => { statCalls.push(p); return { size: 1024 } } },
  })
  const vector = await drive(mlxRig.clock, service.embedImage('/tmp/a.png'))
  assert.ok(vector instanceof Float32Array)
  assert.equal(vector.length, 512)
  assert.equal(mlxRig.spawnedSpecs.length, 1)
  assert.equal(tfRig.spawnedSpecs.length, 0)
  assert.deepEqual(statCalls, ['/tmp/a.png'])
  // 圖像請求體:默認 visualBackend 顯式標註
  const request = mlxRig.servers[0].state.requests.find((r) => r.url === '/embed/image')
  assert.equal(request.body.backend, 'wemm2b-mlx4b')
  await tfRig.sup.dispose()
  await mlxRig.sup.dispose()
})

test('未知後端 → EmbedderUnavailableError,不 spawn 任何 sidecar', async () => {
  const { service, tfRig, mlxRig } = await makeService()
  await assert.rejects(service.embedTexts(['x'], { backend: 'nope' }), EmbedderUnavailableError)
  await assert.rejects(service.embedImage('/x.png', { backend: 'nope' }), EmbedderUnavailableError)
  assert.equal(tfRig.spawnedSpecs.length + mlxRig.spawnedSpecs.length, 0)
  await tfRig.sup.dispose()
  await mlxRig.sup.dispose()
})

test('失敗語義:目標後端不可用 → 拋 EmbedderUnavailableError;絕不跨後端替補', async () => {
  // tf 永遠起不來 → embedTexts(默認 tf 後端)必須失敗,且 mlx 從未被 spawn
  const { service, tfRig, mlxRig } = await makeService({ tf: { behavior: { failSpawn: true } } })
  await assert.rejects(drive(tfRig.clock, service.embedTexts(['q'])), EmbedderUnavailableError)
  assert.equal(mlxRig.spawnedSpecs.length, 0)
  assert.equal(mlxRig.sup.state, 'down')
  await tfRig.sup.dispose()
  await mlxRig.sup.dispose()
})

test('instruct:qwen3 後端透傳;非 qwen3 後端前置驗證錯(不 spawn)', async () => {
  const { service, tfRig, mlxRig } = await makeService()
  await drive(tfRig.clock, service.embedTexts(['query'], { instruct: 'memories' }))
  const request = tfRig.servers[0].state.requests.find((r) => r.url === '/embed/texts')
  assert.equal(request.body.instruct, 'memories')

  await assert.rejects(
    service.embedTexts(['x'], { backend: 'wemm2b-mlx4b', instruct: 'memories' }),
    EmbedderValidationError,
  )
  await assert.rejects(
    service.embedTexts(['x'], { backend: 'wemm2b-fp16', instruct: 'memories' }),
    EmbedderValidationError,
  )
  assert.equal(mlxRig.spawnedSpecs.length, 0) // 前置驗證在路由 spawn 之前
  await tfRig.sup.dispose()
  await mlxRig.sup.dispose()
})

test('請求上限:texts > 64 / 空 texts / 空字符串 / 非法 dim → 驗證錯,不 spawn', async () => {
  const { service, tfRig } = await makeService()
  await assert.rejects(service.embedTexts(Array.from({ length: 65 }, () => 'x')), EmbedderValidationError)
  await assert.rejects(service.embedTexts([]), EmbedderValidationError)
  await assert.rejects(service.embedTexts(['ok', '  ']), EmbedderValidationError)
  await assert.rejects(service.embedTexts(['x'], { dim: 0 }), EmbedderValidationError)
  await assert.rejects(service.embedTexts(['x'], { dim: 3.5 }), EmbedderValidationError)
  assert.equal(tfRig.spawnedSpecs.length, 0)
  await tfRig.sup.dispose()
})

test('圖像衛士:路徑不存在 / 超過 30MB → 驗證錯', async () => {
  const statFileMissing = async () => { throw new Error('ENOENT') }
  const statFileHuge = async () => ({ size: 30 * 1024 * 1024 + 1 })
  const { service, mlxRig, tfRig } = await makeService({ service: { statFile: statFileMissing } })
  await assert.rejects(service.embedImage('/nope.png'), EmbedderValidationError)
  const huge = await makeService({ service: { statFile: statFileHuge } })
  await assert.rejects(huge.service.embedImage('/big.png'), EmbedderValidationError)
  assert.equal(mlxRig.spawnedSpecs.length + huge.mlxRig.spawnedSpecs.length, 0)
  await service.health()
  await tfRig.sup.dispose()
  await mlxRig.sup.dispose()
  await huge.tfRig.sup.dispose()
  await huge.mlxRig.sup.dispose()
})

test('圖像模態衛士:純文本後端拒絕 embedImage', async () => {
  const { service, tfRig, mlxRig } = await makeService()
  await assert.rejects(service.embedImage('/a.png', { backend: 'qwen3-4b-fp16' }), EmbedderValidationError)
  assert.equal(tfRig.spawnedSpecs.length, 0)
  assert.equal(mlxRig.spawnedSpecs.length, 0)
  await tfRig.sup.dispose()
  await mlxRig.sup.dispose()
})

test('指紋衛士:回應指紋不符 → EmbedderResponseError(帶期望指紋)', async () => {
  const { service, tfRig, mlxRig } = await makeService()
  const rig = tfRig
  // 先起來,再讓服務器回錯指紋
  await drive(rig.clock, service.embedTexts(['warm']))
  rig.servers[0].state.fpOverride = 'wemm2b-mlx4b@512' // 跨後端指紋!
  await assert.rejects(drive(rig.clock, service.embedTexts(['x'])), (error) => {
    assert.ok(error instanceof EmbedderResponseError)
    assert.match(error.message, /qwen3-4b-fp16@512/)
    assert.equal(error.fingerprint, 'qwen3-4b-fp16@512')
    return true
  })
  await tfRig.sup.dispose()
  await mlxRig.sup.dispose()
})

test('客戶端 MRL 保證:sidecar 回全維 2560 → 客戶端截斷重歸一到 512', async () => {
  const { service, tfRig, mlxRig } = await makeService()
  await drive(tfRig.clock, service.embedTexts(['a']))
  tfRig.servers[0].state.fullDim = true
  const vectors = await drive(tfRig.clock, service.embedTexts(['b']))
  assert.equal(vectors[0].length, 512)
  assert.ok(Math.abs(l2norm(vectors[0]) - 1) < 1e-6)
  // 截斷前綴一致:全維向量的前 512 維再歸一
  await tfRig.sup.dispose()
  await mlxRig.sup.dispose()
})

test('backends():down 時給目錄 alive:false;up 時給 live 清單', async () => {
  const { service, tfRig, mlxRig } = await makeService()
  let list = await service.backends()
  assert.equal(list.length, 3) // mlx 目錄 1(wemm2b-mlx4b) + tf 目錄 2(qwen3-4b-fp16、wemm2b-fp16),全 alive:false
  assert.ok(list.every((b) => b.alive === false))
  assert.ok(list.some((b) => b.name === 'wemm2b-mlx4b'))
  assert.ok(list.some((b) => b.name === 'qwen3-4b-fp16'))
  assert.ok(list.some((b) => b.name === 'wemm2b-fp16'))
  assert.ok(list.every((b) => b.fingerprint.endsWith('@512')))

  await drive(tfRig.clock, service.embedTexts(['warm']))
  list = await service.backends()
  const tfLive = list.filter((b) => b.name === 'qwen3-4b-fp16')
  assert.equal(tfLive[0].alive, true) // live 清單蓋過目錄
  const mlxEntries = list.filter((b) => b.name === 'wemm2b-mlx4b')
  assert.equal(mlxEntries.length, 1)
  assert.equal(mlxEntries[0].alive, false) // 未啟動的 sidecar 仍走目錄
  await tfRig.sup.dispose()
  await mlxRig.sup.dispose()
})

test('health():up/starting/down 投影(down 態不觸發 spawn)', async () => {
  const { service, tfRig, mlxRig } = await makeService()
  assert.deepEqual(await service.health(), { mlx: 'down', tf: 'down' })
  await drive(tfRig.clock, service.embedTexts(['warm']))
  assert.deepEqual(await service.health(), { mlx: 'down', tf: 'up' })
  assert.equal(mlxRig.spawnedSpecs.length, 0) // health() 純觀察
  await tfRig.sup.dispose()
  await mlxRig.sup.dispose()
})
