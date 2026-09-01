#!/usr/bin/env node
/**
 * 構建輔助:把 src/sidecar/ 下的 Python sidecar 腳本複製到 lib/sidecar/
 * (tsx 直讀不了 .py;SPEC §8「隨插件分發的 Python,構建時複製」)。
 * src/sidecar 缺失或為空時靜默跳過(Phase 1 骨架期僅有 README)。
 */
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const src = path.join(root, 'src', 'sidecar')
const dest = path.join(root, 'lib', 'sidecar')

if (!fs.existsSync(src)) {
  process.exit(0)
}
fs.mkdirSync(dest, { recursive: true })
for (const entry of fs.readdirSync(src)) {
  if (entry.endsWith('.py')) {
    fs.copyFileSync(path.join(src, entry), path.join(dest, entry))
    console.log(`[copy-sidecar] ${entry} -> lib/sidecar/`)
  }
}
