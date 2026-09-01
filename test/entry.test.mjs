/** 插件入口裝配測試:Config 默認值、provide('embedder')、可逆清理、eager 預熱。 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtemp } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { Config, apply, inject, name } from '../lib/index.js'

/** 隔離環境:臨時 runtimeDir + 不存在的 venv(絕不觸碰真實 sidecar/venv)。 */
async function isolatedConfig(overrides = {}) {
  const dir = await mkdtemp(path.join(tmpdir(), 'dsh-embed-entry-'))
  return Config({
    runtimeDir: dir,
    mlxSidecar: { venv: path.join(dir, 'no-venv-mlx') },
    tfSidecar: { venv: path.join(dir, 'no-venv-tf') },
    ...overrides,
  })
}

function makeCtx(logger) {
  const provided = []
  const disposers = []
  return {
    provided,
    disposers,
    ctx: {
      provide: (serviceName, implementation) => provided.push({ serviceName, implementation }),
      effect: (register, label) => {
        disposers.push({ dispose: register(), label })
      },
      ...(logger === undefined ? {} : { logger }),
    },
  }
}

test('Config:schemastery 默認值解析(SPEC §7)', () => {
  const config = Config({})
  assert.equal(config.runtimeDir, '~/.dsh/run/dsh-embed')
  assert.equal(config.defaults.textBackend, 'qwen3-4b-fp16')
  assert.equal(config.defaults.visualBackend, 'wemm2b-mlx4b')
  assert.equal(config.defaults.dim, 512)
  assert.equal(config.mlxSidecar.enabled, true)
  assert.equal(config.mlxSidecar.venv, '~/.dsh/dsh-embed/venv-mlx')
  assert.equal(config.mlxSidecar.keepAliveSec, 900)
  assert.equal(config.tfSidecar.venv, '~/.dsh/dsh-embed/venv-tf')
  assert.deepEqual(config.tfSidecar.eagerBackends, [])
  assert.equal(config.healthIntervalMs, 30000)
  assert.equal(config.healthFailureLimit, 3)
  assert.equal(config.startupTimeoutMs, 180000)
  assert.equal(config.maxRestartAttempts, 3)
})

test('apply():註冊 embedder 服務 + 可逆清理;默認全懶啟動', async () => {
  const { ctx, provided, disposers } = makeCtx()
  apply(ctx, await isolatedConfig())

  assert.equal(name, 'dsh-embed')
  assert.deepEqual(inject, [])
  assert.equal(provided.length, 1)
  assert.equal(provided[0].serviceName, 'embedder')
  const embedder = provided[0].implementation
  assert.equal(typeof embedder.embedTexts, 'function')
  assert.equal(typeof embedder.embedImage, 'function')
  assert.equal(typeof embedder.backends, 'function')
  assert.equal(typeof embedder.health, 'function')

  // 未觸發任何 embed 調用 → 未 spawn;health() 純觀察返回 down
  assert.deepEqual(await embedder.health(), { mlx: 'down', tf: 'down' })

  // 清理路徑可逆(殺 sidecar+撤定時器;此處無進程,僅驗證不拋)
  for (const d of disposers) await d.dispose()
  assert.deepEqual(await embedder.health(), { mlx: 'down', tf: 'down' })
})

test('apply():eagerBackends 打破懶啟動(venv 缺失 → 快速失敗告警,不阻塞裝配)', async () => {
  const warnings = []
  const { ctx, provided, disposers } = makeCtx({ warn: (m) => warnings.push(m), info: () => {}, error: () => {} })
  apply(ctx, await isolatedConfig({ tfSidecar: { venv: '/nonexistent/eager-venv', eagerBackends: ['qwen3-4b-fp16'] } }))
  assert.equal(provided.length, 1)
  const deadline = Date.now() + 5_000
  while (warnings.length === 0 && Date.now() < deadline) {
    await new Promise((resolve) => globalThis.setTimeout(resolve, 50))
  }
  for (const d of disposers) await d.dispose()
  assert.ok(warnings.length >= 1, `expected eager warm-up warning, got: ${JSON.stringify(warnings)}`)
})
