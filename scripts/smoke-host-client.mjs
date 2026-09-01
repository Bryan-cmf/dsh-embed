#!/usr/bin/env node
/**
 * dsh-embed host↔sidecar 契約煙霧測試(隔離環境,不需要模型權重)。
 *
 * 用 sidecar 的 --fake 確定性後端 + 臨時 runtime 目錄,以生產客戶端
 * (SidecarClient)走完整握手→健康→嵌入流程,驗證:
 *   1. CLI 契約:--runtime-dir/--name/--idle-timeout-sec/--fake
 *   2. 握手文件:{port, token, pid}
 *   3. HTTP 契約:/health /backends /embed/texts /embed/image
 *   4. 指紋格式:{backend}@{dim},向量單位範數
 * 退出碼 0 = 全過;非 0 = 契約破損(先查 src/sidecar/README.md 對齊)。
 *
 * 用法:node scripts/smoke-host-client.mjs [tf|mlx|both]
 */
import { spawn } from 'node:child_process'
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { SidecarClient, l2norm, fingerprint } from '../lib/index.js'

const pkgRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const which = process.argv[2] ?? 'both'
const results = []
let failures = 0

function check(name, ok, detail = '') {
  results.push(`${ok ? 'PASS' : 'FAIL'} ${name}${detail !== '' ? ` — ${detail}` : ''}`)
  if (!ok) failures++
}

async function waitFor(predicate, timeoutMs, what) {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    const value = await predicate()
    if (value !== null && value !== undefined && value !== false) return value
    await new Promise((r) => setTimeout(r, 100))
  }
  throw new Error(`timeout waiting for ${what}`)
}

const TINY_PNG = Buffer.from(
  '89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d4944415478da63fcffff3f030005fe02fea72d99480000000049454e44ae426082',
  'hex',
)

async function smoke(id) {
  const runtimeDir = await fsp.mkdtemp(path.join(os.tmpdir(), `dsh-embed-smoke-${id}-`))
  const script = path.join(pkgRoot, 'lib', 'sidecar', `${id}_serve.py`)
  const python = path.join(os.homedir(), '.dsh', 'dsh-embed', `venv-${id}`, 'bin', 'python')
  const logFile = path.join(runtimeDir, `${id}.log`)
  const child = spawn(python, [script, '--runtime-dir', runtimeDir, '--name', id, '--fake', '--idle-timeout-sec', '60'], {
    stdio: ['ignore', fs.openSync(logFile, 'a'), fs.openSync(logFile, 'a')],
  })
  try {
    check(`${id}: python venv 存在`, fs.existsSync(python), python)
    const handshake = await waitFor(async () => {
      try {
        const raw = await fsp.readFile(path.join(runtimeDir, `${id}.json`), 'utf8')
        const parsed = JSON.parse(raw)
        return parsed.port > 0 && typeof parsed.token === 'string' && parsed.pid > 0 ? parsed : null
      } catch {
        return null
      }
    }, 30_000, 'handshake file')
    check(`${id}: 握手 {port,token,pid}`, true, `port=${handshake.port} pid=${handshake.pid}`)

    const client = new SidecarClient({ port: handshake.port, token: handshake.token })
    const health = await waitFor(() => client.health(2000), 30_000, '/health ok')
    check(`${id}: /health ok`, health !== null && health.ok === true, JSON.stringify(health))

    const backends = await client.backends(5000)
    check(`${id}: /backends 非空且形狀正確`, Array.isArray(backends) && backends.length > 0
      && backends.every((b) => typeof b.name === 'string' && Array.isArray(b.dims)), (backends ?? []).map((b) => b.name).join(','))

    // 文本嵌入:各 sidecar 的主後端
    const textBackend = id === 'tf' ? 'qwen3-4b-fp16' : 'wemm2b-mlx4b'
    const texts = await client.embedTexts(['hello world', '你好世界'], { dim: 512, backend: textBackend }, 60_000)
    check(`${id}: /embed/texts 指紋 ${fingerprint(textBackend, 512)}`, texts.fingerprint === fingerprint(textBackend, 512), texts.fingerprint)
    check(`${id}: 向量數量/維度`, texts.vectors.length === 2 && texts.vectors[0].length === 512)
    check(`${id}: 向量單位範數`, Math.abs(l2norm(texts.vectors[0]) - 1) < 1e-3, `norm=${l2norm(texts.vectors[0]).toFixed(4)}`)

    // 圖像嵌入(僅 mlx 視覺後端;fake 模式無需真解碼,但路徑必須存在)
    if (id === 'mlx') {
      const imagePath = path.join(runtimeDir, 'tiny.png')
      await fsp.writeFile(imagePath, TINY_PNG)
      const image = await client.embedImage(imagePath, { dim: 512, backend: 'wemm2b-mlx4b' }, 60_000)
      check('mlx: /embed/image 指紋', image.fingerprint === fingerprint('wemm2b-mlx4b', 512), image.fingerprint)
      check('mlx: 圖像向量維度/範數', image.vector.length === 512 && Math.abs(l2norm(image.vector) - 1) < 1e-3)
    }

    // 鑑權:錯 token → 401 → health null
    const badClient = new SidecarClient({ port: handshake.port, token: '0'.repeat(64) })
    check(`${id}: 錯 token 被拒`, (await badClient.health(2000)) === null)
  } finally {
    try {
      process.kill(-child.pid, 'SIGTERM')
    } catch {
      try {
        process.kill(child.pid, 'SIGTERM')
      } catch { /* already gone */ }
    }
    await new Promise((r) => setTimeout(r, 300))
    await fsp.rm(runtimeDir, { recursive: true, force: true }).catch(() => {})
  }
}

const targets = which === 'both' ? ['tf', 'mlx'] : [which]
for (const id of targets) {
  try {
    await smoke(id)
  } catch (error) {
    check(`${id}: 煙霧流程`, false, error instanceof Error ? error.message : String(error))
  }
}
console.log(results.join('\n'))
console.log(failures === 0 ? `SMOKE_OK (${targets.join('+')})` : `SMOKE_FAILED (${failures} failures)`)
process.exit(failures === 0 ? 0 : 1)
