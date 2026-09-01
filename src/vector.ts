/**
 * dsh-embed · 向量純函數:MRL 截斷+重歸一、cosine、RRF 融合。
 *
 * MRL 語義與 poc `embedders.py::WeMM.mrl` 一致:截斷後逐行重 L2 歸一,
 * 零向量守衛(範數 0 時原樣返回,避免除零 NaN)。
 * 客戶端在最終回傳前再套一次 mrlTruncate——對已截斷已歸一的向量是
 * 冪等操作,但保證無論 sidecar 版本如何,`{backend}@{dim}` 指紋對應的
 * 向量恆為「長 dim、單位範數」。
 */

/** MRL 截斷 + L2 重歸一。`vec.length < dim` 時拋(回應短於請求維度屬協議錯)。 */
export function mrlTruncate(vec: Float32Array | number[], dim: number): Float32Array {
  if (!Number.isInteger(dim) || dim <= 0) {
    throw new RangeError(`mrlTruncate: dim must be a positive integer, got ${dim}`)
  }
  if (vec.length < dim) {
    throw new RangeError(`mrlTruncate: vector length ${vec.length} < requested dim ${dim}`)
  }
  const src = vec instanceof Float32Array ? vec.subarray(0, dim) : vec.slice(0, dim)
  const out = new Float32Array(src)
  let sum = 0
  for (let i = 0; i < dim; i++) sum += out[i] * out[i]
  const norm = Math.sqrt(sum)
  if (norm > 0) {
    for (let i = 0; i < dim; i++) out[i] /= norm
  }
  return out
}

/** L2 範數(測試與診斷用)。 */
export function l2norm(vec: Float32Array | number[]): number {
  let sum = 0
  for (let i = 0; i < vec.length; i++) sum += vec[i] * vec[i]
  return Math.sqrt(sum)
}

/** cosine 相似度;長度不等拋 RangeError;零向量返回 0。 */
export function cosine(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) {
    throw new RangeError(`cosine: length mismatch ${a.length} vs ${b.length}`)
  }
  let dot = 0
  let na = 0
  let nb = 0
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]
    na += a[i] * a[i]
    nb += b[i] * b[i]
  }
  if (na === 0 || nb === 0) return 0
  return dot / (Math.sqrt(na) * Math.sqrt(nb))
}

export interface RrfFusedItem {
  id: string
  /** score = Σ_lists 1/(k + rank),rank 為 1-based。 */
  score: number
  /** 出現的來源列表索引(升序),供「kw/sem 各自來源標記」。 */
  sources: number[]
}

export interface RrfFuseOptions {
  /** RRF 常數,默認 60(SPEC §6)。 */
  k?: number
  /** 截取前 N 條;默認不截。 */
  limit?: number
}

/**
 * Reciprocal Rank Fusion over id 列表列表。
 *
 * SPEC §6:`fused = RRF(kwHits, semHits, k=60)`,`score = Σ 1/(60+rank)`。
 * 平手時按「最早出現位置」(來源列表索引,再 rank)穩定排序——純函數,
 * 對相同輸入恆有相同輸出,供 dsh-insights memory v2 與回歸測試複用。
 */
export function rrfFuse(lists: (readonly string[])[], opts: RrfFuseOptions = {}): RrfFusedItem[] {
  const k = opts.k ?? 60
  if (!Number.isFinite(k) || k <= 0) throw new RangeError(`rrfFuse: k must be positive, got ${k}`)
  const map = new Map<string, { score: number; sources: number[]; firstList: number; firstRank: number }>()
  for (let listIndex = 0; listIndex < lists.length; listIndex++) {
    const list = lists[listIndex]
    if (list === undefined) continue
    const seenInList = new Set<string>()
    for (let rank = 0; rank < list.length; rank++) {
      const id = list[rank]
      if (id === undefined || seenInList.has(id)) continue
      seenInList.add(id)
      let entry = map.get(id)
      if (entry === undefined) {
        entry = { score: 0, sources: [], firstList: listIndex, firstRank: rank }
        map.set(id, entry)
      }
      entry.score += 1 / (k + rank + 1)
      if (!entry.sources.includes(listIndex)) entry.sources.push(listIndex)
    }
  }
  const items: RrfFusedItem[] = [...map.entries()].map(([id, entry]) => ({
    id,
    score: entry.score,
    sources: entry.sources,
  }))
  items.sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score
    const fa = map.get(a.id)!
    const fb = map.get(b.id)!
    if (fa.firstList !== fb.firstList) return fa.firstList - fb.firstList
    return fa.firstRank - fb.firstRank
  })
  return opts.limit === undefined ? items : items.slice(0, opts.limit)
}
