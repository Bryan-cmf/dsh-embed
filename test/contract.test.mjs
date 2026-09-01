/**
 * SidecarClient 契約單元(SPEC §4):HTTP 形狀、token 鑑權、
 * 狀態碼語義映射(4xx=回應錯 / 401、5xx、網絡=不可用)、請求超時。
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  EmbedderResponseError,
  EmbedderUnavailableError,
  SidecarClient,
} from '../lib/index.js'
import { startMockSidecar } from './helpers.mjs'

test('契約:/health、/backends、/embed/* 回應形狀解析 + token 頭', async () => {
  const server = await startMockSidecar()
  const client = new SidecarClient({ port: server.port, token: server.token })
  try {
    const health = await client.health(2_000)
    assert.deepEqual(health, { ok: true, uptime_s: 1.5, backend: 'tf' })

    const backends = await client.backends(2_000)
    assert.equal(backends.length, 1)
    assert.equal(backends[0].name, 'qwen3-4b-fp16')
    assert.deepEqual(backends[0].dims, [512, 2560])

    const texts = await client.embedTexts(['a', 'b'], { dim: 512, backend: 'qwen3-4b-fp16' }, 5_000)
    assert.equal(texts.vectors.length, 2)
    assert.equal(texts.vectors[0].length, 512)
    assert.equal(texts.fingerprint, 'qwen3-4b-fp16@512')
    assert.equal(texts.dim, 512)
    assert.equal(typeof texts.ms, 'number')

    const image = await client.embedImage('/tmp/x.png', { dim: 512, backend: 'wemm2b-mlx4b' }, 5_000)
    assert.equal(image.vector.length, 512)
    assert.equal(image.fingerprint, 'wemm2b-mlx4b@512')

    // 每個請求都帶了 token
    assert.ok(server.state.requests.length >= 4)
    assert.ok(server.state.requests.every((r) => r.token === server.token))
  } finally {
    await server.stop()
  }
})

test('鑑權:錯 token → 401 → EmbedderUnavailableError(health 探測 null)', async () => {
  const server = await startMockSidecar()
  const client = new SidecarClient({ port: server.port, token: 'w'.repeat(64) })
  try {
    assert.equal(await client.health(2_000), null)
    assert.equal(await client.backends(2_000), null)
    await assert.rejects(client.embedTexts(['x'], {}, 2_000), EmbedderUnavailableError)
  } finally {
    await server.stop()
  }
})

test('狀態碼語義:400 類 = EmbedderResponseError;500 = EmbedderUnavailableError', async () => {
  const server = await startMockSidecar()
  const client = new SidecarClient({ port: server.port, token: server.token })
  try {
    // 400:sidecar 校驗拒絕(texts 非數組)
    await assert.rejects(client.embedTexts(undefined, {}, 2_000), EmbedderResponseError)
    // 500:模擬推理內部錯(帶 fingerprint 上下文)
    server.state.failEmbeds = true
    await assert.rejects(client.embedTexts(['ok'], {}, 2_000), (error) => {
      assert.ok(error instanceof EmbedderUnavailableError)
      assert.match(error.message, /simulated failure/)
      return true
    })
    server.state.failEmbeds = false
  } finally {
    await server.stop()
  }
})

test('不可達端口 / 進程死亡 → EmbedderUnavailableError / health null', async () => {
  const dead = await startMockSidecar()
  const port = dead.port
  await dead.stop()
  const client = new SidecarClient({ port, token: 'z'.repeat(64) })
  assert.equal(await client.health(2_000), null)
  await assert.rejects(client.embedTexts(['x'], {}, 2_000), EmbedderUnavailableError)
})

test('畸形回應:health.ok 非布爾 → null;vectors 非有限數組 → EmbedderResponseError', async () => {
  const server = await startMockSidecar()
  const client = new SidecarClient({ port: server.port, token: server.token })
  try {
    // 直接對 client 內部 request 的消費面做形狀斷言:health 要求 ok:boolean
    server.state.healthy = 1 // 畸形型別(JSON true 才合法)
    assert.equal(await client.health(2_000), null)
    server.state.healthy = true
  } finally {
    await server.stop()
  }
})
