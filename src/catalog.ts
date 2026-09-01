/**
 * dsh-embed · 後端靜態目錄與路由表。
 *
 * 三個後端(Phase 0 選型鎖定,Plan §2):
 * - wemm2b-mlx4b  :WeMM-2B @MLX-4bit,視覺/技能後端,sidecar-mlx
 * - qwen3-4b-fp16 :Qwen3-Embedding-4B @fp16,在線文本,sidecar-tf
 * - wemm2b-fp16   :WeMM-2B @fp16(transformers 官方路徑 fallback),sidecar-tf
 *
 * dims 為 poc 實測可用 MRL 維度(存儲統一 512;全維 WeMM-2B=2048、
 * Qwen3-4B=2560)。運行時真相以 sidecar /backends 為準;本目錄僅供
 * sidecar 未起時的 backends() 描述與後端→sidecar 路由(路由不依賴網絡)。
 */
import type { BackendInfo, SidecarId } from './types.ts'

export interface CatalogEntry {
  name: string
  model: string
  dims: number[]
  modalities: ('text' | 'image')[]
  sidecar: SidecarId
}

export const BACKEND_CATALOG: Record<string, CatalogEntry> = {
  'wemm2b-mlx4b': {
    name: 'wemm2b-mlx4b',
    model: 'hfadam/WeMM-Embedding-2B-MLX-4bit',
    dims: [512, 2048],
    modalities: ['text', 'image'],
    sidecar: 'mlx',
  },
  'qwen3-4b-fp16': {
    name: 'qwen3-4b-fp16',
    model: 'Qwen/Qwen3-Embedding-4B',
    dims: [512, 2560],
    modalities: ['text'],
    sidecar: 'tf',
  },
  'wemm2b-fp16': {
    name: 'wemm2b-fp16',
    model: 'tencent/WeMM-Embedding-2B',
    dims: [512, 2048],
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
