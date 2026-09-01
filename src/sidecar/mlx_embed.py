#!/usr/bin/env python3
"""WeMM-2B MLX-4bit 嵌入後端（SPEC §4.1）。

前向算法逐行移植自已驗證的 dsh-wemm-poc/scripts/run_eval_mlx.py（Phase 0
results_mlx.md：D2 92% / D3 93%，與 bf16 零差異）。任何語義改動都必須先過
test/contract/parity_mlx.py 的 cos≥0.999 移植正確性測試。

要點：
  - prompt = '<|im_start|>user\\n{text}<|im_end|>\\n'；tokenizer 之後保證末位是
    <embedding>（id=248077，缺失則手工補）
  - get_input_embeddings → language_model.model(inputs_embeds=...) →
    取序列末位（單條輸入即最後非 pad 位）→ float32 → L2 歸一
  - MRL：dim < full_dim 時截斷 + 重歸一（common.mrl）
  - 圖像：prepare_inputs(padding=False, add_special_tokens=False) 後走
    get_input_embeddings(pixel_values, image_grid_thw) 同路徑
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from common import Backend, log, resolve_hf_snapshot

EMB_TOKEN = 248077            # <embedding>（WeMM/Qwen3-VL 詞表）
REPO = 'hfadam/WeMM-Embedding-2B-MLX-4bit'

TEXT_PROMPT = '<|im_start|>user\n{text}<|im_end|>\n'
IMAGE_PROMPT = ('<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>'
                '<|im_end|>\n')


def resolve_mlx_model_path(override=None):
    """權重解析：--model 參數 > HF 緩存快照（離線，不觸網）。"""
    if override:
        p = Path(override).expanduser()
        if not p.exists():
            raise FileNotFoundError(f'model path not found: {p}')
        return p
    cache = Path(os.environ.get('HF_HUB_CACHE', str(Path.home() / '.cache/huggingface/hub')))
    base = cache / f'models--{REPO.replace("/", "--")}' / 'snapshots'
    snaps = sorted((d for d in base.glob('*') if d.is_dir()),
                   key=lambda p: p.stat().st_mtime)
    if not snaps:
        raise FileNotFoundError(
            f'{REPO} not found in HF cache ({base}); pre-download weights first')
    return snaps[-1]


class WemmMlx4bBackend(Backend):
    name = 'wemm2b-mlx4b'
    repo = REPO
    full_dim = 2048
    dims = [64, 128, 256, 512, 1024, 2048]
    modalities = ['text', 'image']

    def __init__(self, model_override=None):
        Backend.__init__(self)
        self._model_override = model_override
        try:
            self._path = resolve_mlx_model_path(model_override)
            sha = self._path.name
            if len(sha) == 40 and all(c in '0123456789abcdef' for c in sha):
                self._sha = sha
            else:
                self._sha = resolve_hf_snapshot(REPO)
        except FileNotFoundError:
            self._path = None
            self._sha = resolve_hf_snapshot(REPO)
            self.weights_available = False

    def _load(self) -> None:
        import mlx.core as mx
        from mlx_vlm import load
        from mlx_vlm.utils import load_image, prepare_inputs   # noqa: F401 註冊到實例
        if self._path is None:
            raise FileNotFoundError(f'{REPO} weights unavailable')
        log(f'loading {REPO} from {self._path} ...')
        self._mx = mx
        self._load_image_fn = load_image
        self._prepare_inputs_fn = prepare_inputs
        self._net, self._processor = load(str(self._path))
        mx.eval(self._net.parameters())

    # -- 文本前向（run_eval_mlx.embed_text 移植；勿改語義）--------
    def _embed_text_raw(self, text: str):
        mx = self._mx
        tok = self._processor.tokenizer
        ids = tok.encode(TEXT_PROMPT.format(text=text))
        if ids[-1] != EMB_TOKEN:
            ids = ids + [EMB_TOKEN]
        x = mx.array([ids])
        feats = self._net.get_input_embeddings(input_ids=x, mask=None)
        h = self._net.language_model.model(
            x, inputs_embeds=feats.inputs_embeds, mask=None,
            cache=None, position_ids=feats.position_ids)
        v = h[0, -1].astype(mx.float32)
        v = v / mx.sqrt(mx.sum(v * v))
        return np.array(v)

    # -- 批量：逐條推理（與已驗證腳本逐位一致；不做 padding 批）--
    def _embed_texts_raw(self, texts, instruct=None):
        return [self._embed_text_raw(t) for t in texts]

    # -- 圖像前向（run_eval_mlx.embed_image 移植）----------------
    def _embed_image_raw(self, path: str):
        mx = self._mx
        img = self._load_image_fn(str(path))
        inputs = self._prepare_inputs_fn(
            self._processor, images=[img], prompts=[IMAGE_PROMPT],
            padding=False, add_special_tokens=False)
        ids = inputs['input_ids']
        feats = self._net.get_input_embeddings(
            input_ids=ids,
            pixel_values=inputs.get('pixel_values'),
            image_grid_thw=inputs.get('image_grid_thw'),
            mask=None)
        lm = self._net.language_model
        h = lm.model(
            ids, inputs_embeds=feats.inputs_embeds, mask=None,
            cache=None, position_ids=feats.position_ids)
        v = h[0, -1].astype(mx.float32)
        v = v / mx.sqrt(mx.sum(v * v))
        return np.array(v)
