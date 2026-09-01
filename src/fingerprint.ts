/**
 * dsh-embed · 向量指紋(SPEC §5)。
 *
 * 指紋 = `{backend}@{dim}`,如 'wemm2b-mlx4b@512'。跨後端不可互換
 * (MLX vs bf16 cos 漂移 0.968 實測)→ 換後端必須重建索引。
 * 任何 `vec.fp ≠ 當前配置指紋` 的行視為過期。
 */

export function fingerprint(backend: string, dim: number): string {
  return `${backend}@${dim}`
}

export interface ParsedFingerprint {
  backend: string
  dim: number
}

/** 解析指紋;格式非法返回 null(調用方按過期處理)。 */
export function parseFingerprint(fp: string): ParsedFingerprint | null {
  if (typeof fp !== 'string') return null
  const at = fp.lastIndexOf('@')
  if (at <= 0 || at === fp.length - 1) return null
  const backend = fp.slice(0, at)
  const dim = Number(fp.slice(at + 1))
  if (!Number.isInteger(dim) || dim <= 0) return null
  return { backend, dim }
}

/**
 * 行級指紋過期判定:`rowFp` 為當前 `{backend}@{dim}` 之外的任何值
 * (含缺失/畸形)都算過期,觸發重嵌(記憶表)或整表重建(資產/技能表)。
 */
export function isFingerprintStale(rowFp: string | undefined, currentBackend: string, currentDim: number): boolean {
  if (typeof rowFp !== 'string' || rowFp === '') return true
  return rowFp !== fingerprint(currentBackend, currentDim)
}
