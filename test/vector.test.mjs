/** 純函數單元:MRL 截斷重歸一、cosine、RRF 融合(SPEC §6、§10)。 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { cosine, l2norm, mrlTruncate, rrfFuse } from '../lib/index.js'

test('mrlTruncate: 截斷到 dim 並重歸一(norm=1)', () => {
  const v = new Float32Array([3, 4, 0, 0])
  const out = mrlTruncate(v, 2)
  assert.equal(out.length, 2)
  assert.ok(Math.abs(l2norm(out) - 1) < 1e-6)
  assert.ok(Math.abs(out[0] - 0.6) < 1e-6)
  assert.ok(Math.abs(out[1] - 0.8) < 1e-6)
})

test('mrlTruncate: 冪等(已截斷已歸一的向量不變)', () => {
  const once = mrlTruncate(new Float32Array([1, 2, 3, 4]), 2)
  const twice = mrlTruncate(once, 2)
  assert.equal(twice.length, once.length)
  for (let i = 0; i < once.length; i++) {
    assert.ok(Math.abs(twice[i] - once[i]) < 1e-7)
  }
})

test('mrlTruncate: dim 等於全長時仍歸一', () => {
  const out = mrlTruncate(new Float32Array([1, 1]), 2)
  assert.ok(Math.abs(l2norm(out) - 1) < 1e-6)
})

test('mrlTruncate: 零向量守衛(不產生 NaN)', () => {
  const out = mrlTruncate(new Float32Array([0, 0, 0]), 2)
  assert.ok(Number.isFinite(out[0]))
  assert.equal(out[0], 0)
})

test('mrlTruncate: 向量短於 dim 拋 RangeError(協議錯)', () => {
  assert.throws(() => mrlTruncate(new Float32Array([1, 2]), 4), RangeError)
  assert.throws(() => mrlTruncate([1, 2], 0), RangeError)
})

test('cosine: 正交=0、同向=1、零向量=0、長度不等拋', () => {
  assert.ok(Math.abs(cosine(new Float32Array([1, 0]), new Float32Array([0, 1]))) < 1e-7)
  assert.ok(Math.abs(cosine(new Float32Array([2, 0]), new Float32Array([5, 0])) - 1) < 1e-7)
  assert.equal(cosine(new Float32Array([0, 0]), new Float32Array([1, 1])), 0)
  assert.throws(() => cosine(new Float32Array([1]), new Float32Array([1, 2])), RangeError)
})

test('rrfFuse: k=60 公式 score=Σ 1/(60+rank),rank 1-based', () => {
  const fused = rrfFuse([['a', 'b'], ['b', 'c']])
  const byId = Object.fromEntries(fused.map((item) => [item.id, item.score]))
  // 'a': 只在列表0 rank1 → 1/61;'b': 列表0 rank2 + 列表1 rank1 → 1/62+1/61
  assert.ok(Math.abs(byId.a - 1 / 61) < 1e-12)
  assert.ok(Math.abs(byId.b - (1 / 62 + 1 / 61)) < 1e-12)
  assert.ok(Math.abs(byId.c - 1 / 62) < 1e-12)
  // 排序:b > a > c
  assert.deepEqual(fused.map((item) => item.id), ['b', 'a', 'c'])
})

test('rrfFuse: 來源標記 sources(kw/sem 標記)', () => {
  const fused = rrfFuse([['x', 'y'], ['y', 'z']])
  const byId = Object.fromEntries(fused.map((item) => [item.id, item]))
  assert.deepEqual(byId.x.sources, [0])
  assert.deepEqual(byId.y.sources, [0, 1])
  assert.deepEqual(byId.z.sources, [1])
})

test('rrfFuse: 同列表重複 id 只計一次(最高排名)', () => {
  const fused = rrfFuse([['a', 'a', 'a', 'b']])
  assert.equal(fused.length, 2)
  assert.ok(Math.abs(fused[0].score - 1 / 61) < 1e-12)
})

test('rrfFuse: 平手按最早出現穩定排序(確定性)', () => {
  const fused = rrfFuse([['p', 'q'], ['q', 'p']])
  // p 與 q 分數相同(1/61+1/62);p 先在列表0 rank0 出現 → 排前
  assert.deepEqual(fused.map((item) => item.id), ['p', 'q'])
  assert.ok(Math.abs(fused[0].score - fused[1].score) < 1e-12)
})

test('rrfFuse: limit 截取與空輸入', () => {
  assert.deepEqual(rrfFuse([[], []]), [])
  assert.equal(rrfFuse([['a', 'b', 'c']], { limit: 2 }).length, 2)
  assert.equal(rrfFuse([['a']], { k: 60 }).length, 1)
  assert.throws(() => rrfFuse([['a']], { k: 0 }), RangeError)
})
