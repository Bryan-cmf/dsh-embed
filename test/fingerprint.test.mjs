/** 指紋規則單元(SPEC §5):fp={backend}@{dim},跨後端不可互換。 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { fingerprint, isFingerprintStale, parseFingerprint } from '../lib/index.js'

test('fingerprint: 構造', () => {
  assert.equal(fingerprint('wemm2b-mlx4b', 512), 'wemm2b-mlx4b@512')
  assert.equal(fingerprint('qwen3-4b-fp16', 2560), 'qwen3-4b-fp16@2560')
})

test('parseFingerprint: 往返;畸形返回 null', () => {
  assert.deepEqual(parseFingerprint('wemm2b-mlx4b@512'), { backend: 'wemm2b-mlx4b', dim: 512 })
  // 後端名含 @ 時以最後一個 @ 為界
  assert.deepEqual(parseFingerprint('a@b@64'), { backend: 'a@b', dim: 64 })
  assert.equal(parseFingerprint('nobody'), null)
  assert.equal(parseFingerprint('@512'), null)
  assert.equal(parseFingerprint('x@'), null)
  assert.equal(parseFingerprint('x@abc'), null)
  assert.equal(parseFingerprint('x@0'), null)
  assert.equal(parseFingerprint('x@-5'), null)
})

test('isFingerprintStale: 當前指紋外一律過期(含缺失/畸形)', () => {
  assert.equal(isFingerprintStale('qwen3-4b-fp16@512', 'qwen3-4b-fp16', 512), false)
  // 換後端 = 過期(重建索引)
  assert.equal(isFingerprintStale('wemm2b-fp16@512', 'qwen3-4b-fp16', 512), true)
  // 換 dim = 過期
  assert.equal(isFingerprintStale('qwen3-4b-fp16@2560', 'qwen3-4b-fp16', 512), true)
  assert.equal(isFingerprintStale(undefined, 'qwen3-4b-fp16', 512), true)
  assert.equal(isFingerprintStale('', 'qwen3-4b-fp16', 512), true)
  assert.equal(isFingerprintStale('garbage', 'qwen3-4b-fp16', 512), true)
})
