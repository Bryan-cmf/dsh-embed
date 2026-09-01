/**
 * dsh-embed · 錯誤類型。
 *
 * SPEC §3 失敗語義:目標後端不可用 → 立即拋 EmbedderUnavailableError,
 * 調用方必須捕獲並降級 keyword;禁止自動跨後端替補(指紋不可互換)。
 * SPEC §9:錯誤消息帶 fingerprint 上下文。
 */

/** 嵌入基礎設施錯誤基類。 */
export class EmbedderError extends Error {
  /** 機器可讀錯誤碼,如 'unavailable' / 'response' / 'validation'。 */
  readonly code: string
  /** 已知時的向量指紋上下文(`{backend}@{dim}`),未知為空串。 */
  readonly fingerprint: string

  constructor(message: string, code: string, fingerprint = '') {
    super(fingerprint !== '' ? `${message} (fingerprint: ${fingerprint})` : message)
    this.name = 'EmbedderError'
    this.code = code
    this.fingerprint = fingerprint
  }
}

/**
 * 目標後端不可用(sidecar 停機/禁用/啟動失敗/網絡錯)。
 * 調用方(dsh-insights memory v2)捕獲後降級純 keyword 檢索。
 */
export class EmbedderUnavailableError extends EmbedderError {
  constructor(message: string, fingerprint = '') {
    super(message, 'unavailable', fingerprint)
    this.name = 'EmbedderUnavailableError'
  }
}

/** sidecar 回應畸形(形狀/指紋不符契約)。 */
export class EmbedderResponseError extends EmbedderError {
  constructor(message: string, fingerprint = '') {
    super(message, 'response', fingerprint)
    this.name = 'EmbedderResponseError'
  }
}

/** 調用方輸入非法(texts 超上限、instruct 用於非 Qwen3 後端等)。 */
export class EmbedderValidationError extends EmbedderError {
  constructor(message: string, fingerprint = '') {
    super(message, 'validation', fingerprint)
    this.name = 'EmbedderValidationError'
  }
}

export function isEmbedderUnavailableError(error: unknown): error is EmbedderUnavailableError {
  return error instanceof EmbedderError && error.code === 'unavailable'
}
