/**
 * dsh-embed · 後端靜態目錄與路由表。
 *
 * 三個後端(Phase 0 選型鎖定,Plan §2):
 * - wemm2b-mlx4b  :WeMM-2B @MLX-4bit,視覺/技能後端,sidecar-mlx
 * - qwen3-4b-fp16 :Qwen3-Embedding-4B @fp16,在線文本,sidecar-tf
 * - wemm2b-fp16   :WeMM-2B @fp16(transformers 官方路徑 fallback),sidecar-tf
 *
 * dims 為 sidecar 實際支持的完整 MRL 梯(t6 F1 對齊;與 mlx_embed.py /
 * tf_embed.py 的 dims 聲明一致,運行時以 /backends 返回為準):WeMM-2B
 * 全維 2048、Qwen3-4B 全維 2560(梯含 2048)。本目錄供 sidecar 未起時的
 * backends() 描述、後端→sidecar 路由(不依賴網絡)與 dim 預校驗。
 */
import type { BackendInfo, SidecarId } from './types.ts'

export interface CatalogEntry {
  name: string
  model: string
  dims: number[]
  modalities: ('text' | 'image')[]
  sidecar: SidecarId
}

/** WeMM-2B MRL 梯;Qwen3-4B MRL 梯(2560 全維,梯含 2048)。 */
export const WEMM2B_MRL_DIMS = [64, 128, 256, 512, 1024, 2048] as const
export const QWEN3_4B_MRL_DIMS = [64, 128, 256, 512, 1024, 2048, 2560] as const

export const BACKEND_CATALOG: Record<string, CatalogEntry> = {
  'wemm2b-mlx4b': {
    name: 'wemm2b-mlx4b',
    model: 'hfadam/WeMM-Embedding-2B-MLX-4bit',
    dims: [...WEMM2B_MRL_DIMS],
    modalities: ['text', 'image'],
    sidecar: 'mlx',
  },
  'qwen3-4b-fp16': {
    name: 'qwen3-4b-fp16',
    model: 'Qwen/Qwen3-Embedding-4B',
    dims: [...QWEN3_4B_MRL_DIMS],
    modalities: ['text'],
    sidecar: 'tf',
  },
  'wemm2b-fp16': {
    name: 'wemm2b-fp16',
    model: 'tencent/WeMM-Embedding-2B',
    dims: [...WEMM2B_MRL_DIMS],
    modalities: ['text', 'image'],
    sidecar: 'tf',
  },
}

export function catalogBackendInfos(sidecar: SidecarId, defaultDim: number, alive: boolean): BackendInfo[] {
  return Object.values(BACKEND_CATALOG)
    .filter((entry) => entry.sidecar === sidecar)
    .map((entry) => ({
      name: entry.name,
      model: entry.model,
      dims: [...entry.dims],
      modalities: [...entry.modalities],
      fingerprint: `${entry.name}@${defaultDim}`,
      alive,
    }))
}

/** 後端名 → sidecar 路由;未知後端返回 null。 */
export function sidecarForBackend(backend: string): SidecarId | null {
  return BACKEND_CATALOG[backend]?.sidecar ?? null
}
