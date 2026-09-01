import { defineConfig } from 'tsdown'

const ID = '@bryan-cmf/dsh-embed'

// Host-only plugin: single ESM entry. Node builtins (child_process, fs, http
// via global fetch) stay external; cordis/schemastery are peer deps.
export default defineConfig([
  {
    name: ID,
    entry: { index: 'src/index.ts' },
    outDir: 'lib',
    format: ['esm'],
    platform: 'node',
    target: 'es2024',
    fixedExtension: false,
    dts: false,
    clean: false,
  },
])
