#!/usr/bin/env python3
"""transformers 嵌入後端（SPEC §4.2）：Qwen3-Embedding-4B fp16（文本）+
WeMM-Embedding-2B fp16（文本/圖像，官方路徑 fallback，惰性加載）。

移植自已驗證的 dsh-wemm-poc/scripts/embedders.py（Phase 0：D1 95% / MRR 0.82；
精度實驗結論 fp16 質量零差異）。

Qwen3 家族官方配方：
  - tokenizer padding_side='left'；query 側拼 instruct 前綴
    （'Instruct: {instruct}\\nQuery: {text}'），文檔側裸文本（對稱檢索按官方配方）
  - last-token pooling（attention_mask 最後非 pad 位）+ L2 歸一
  - truncation max_length=512（Phase 0 驗證配方）

WeMM 官方配方：processor.apply_chat_template + qwen_vl_utils.process_vision_info
(image_patch_size=16) → model.embedding(**inputs)（輸出已 L2 歸一）。
"""

from __future__ import annotations

import numpy as np

from common import Backend, log, resolve_hf_snapshot

MAX_LENGTH_TOKENS = 512   # Phase 0 驗證配方（run_eval.py / embedders.py）
TEXT_BATCH = 16           # qwen3 批推理塊大小（對齊 dsh-insights 隊列批量）


def _torch_device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class Qwen3Fp16Backend(Backend):
    name = 'qwen3-4b-fp16'
    repo = 'Qwen/Qwen3-Embedding-4B'
    full_dim = 2560
    dims = [64, 128, 256, 512, 1024, 2048, 2560]
    modalities = ['text']

    def __init__(self):
        Backend.__init__(self)
        self._sha = resolve_hf_snapshot(self.repo) or None
        # 權重可用性：緩存快照存在即可（離線啟動）
        self.weights_available = self._sha is not None

    def _load(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        self._torch = torch
        self._device = _torch_device()
        log(f'loading {self.repo} fp16 on {self._device} ...')
        kwargs = {'revision': self._sha} if self._sha else {}
        self.tok = AutoTokenizer.from_pretrained(self.repo, padding_side='left', **kwargs)
        self._net = AutoModel.from_pretrained(
            self.repo, dtype=torch.float16, **kwargs
        ).to(self._device).eval()

    def _last_token_pool(self, h, mask):
        # Qwen3-Embedding 官方 last_token_pool（模型卡原版，支持批量 left padding）：
        # left padding 時序列末位 h[:, -1] 即每行最後真實 token；batch=1 時與
        # dsh-wemm-poc 單條路徑（h[0, len-1]）逐位一致。
        torch = self._torch
        left_padding = (mask[:, -1].sum() == mask.shape[0])
        if left_padding:
            return h[:, -1]
        sequence_lengths = torch.eq(mask, 0).int().argmax(dim=-1) - 1
        return h[torch.arange(h.size(0)), sequence_lengths]

    def _embed_texts_raw(self, texts, instruct=None):
        torch = self._torch
        if instruct:
            texts = [f'Instruct: {instruct}\nQuery: {t}' for t in texts]
        out_vecs = []
        for i in range(0, len(texts), TEXT_BATCH):
            chunk = texts[i:i + TEXT_BATCH]
            inputs = self.tok(chunk, padding=True, truncation=True,
                              max_length=MAX_LENGTH_TOKENS, return_tensors='pt').to(self._device)
            with torch.inference_mode():
                hidden = self._net(**inputs).last_hidden_state
            emb = self._last_token_pool(hidden, inputs.attention_mask)
            emb = torch.nn.functional.normalize(emb, dim=-1)
            out_vecs.extend(emb.float().cpu().numpy())
        return out_vecs

    def _embed_image_raw(self, path: str):
        raise NotImplementedError('qwen3 backend is text-only')


class Wemm2bFp16Backend(Backend):
    """官方 transformers 路徑 fallback（SPEC §11 回滾：visualBackend 一鍵切換用）。"""

    name = 'wemm2b-fp16'
    repo = 'tencent/WeMM-Embedding-2B'
    full_dim = 2048
    dims = [64, 128, 256, 512, 1024, 2048]
    modalities = ['text', 'image']

    def __init__(self):
        Backend.__init__(self)
        self._sha = resolve_hf_snapshot(self.repo) or None
        self.weights_available = self._sha is not None

    def _load(self) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor
        self._torch = torch
        self._device = _torch_device()
        log(f'loading {self.repo} fp16 on {self._device} ...')
        kwargs = {'revision': self._sha} if self._sha else {}
        self.processor = AutoProcessor.from_pretrained(
            self.repo, trust_remote_code=True, **kwargs)
        self._net = AutoModel.from_pretrained(
            self.repo, trust_remote_code=True, dtype=torch.float16, **kwargs
        ).to(self._device).eval()

    def _run(self, text=None, images=None):
        """embedders.WeMM._run 逐行移植（官方 model.embedding 語義）。"""
        torch = self._torch
        from qwen_vl_utils import process_vision_info
        content = []
        if images:
            content.append({'type': 'image', 'image': images[0]})
        if text is not None:
            content.append({'type': 'text', 'text': text})
        if not content:
            raise ValueError('empty input')
        messages = [{'role': 'user', 'content': content}]
        prompt = None
        try:
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)
            imgs, vids = None, None
            if images:
                imgs, vids = process_vision_info(messages, image_patch_size=16)
        except Exception:
            # 模板不接受純圖像消息時，補一段佔位文本重試（embedders.py 行為）
            content.append({'type': 'text', 'text': 'image'})
            messages = [{'role': 'user', 'content': content}]
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)
            imgs, vids = None, None
            if images:
                imgs, vids = process_vision_info(messages, image_patch_size=16)
        inputs = self.processor(
            text=prompt, images=imgs, videos=vids,
            return_tensors='pt',
        ).to(self._device)
        with torch.inference_mode():
            emb = self._net.embedding(**inputs)
        return emb[0].float().cpu().numpy()   # (D,) 已 L2 歸一化

    def _embed_texts_raw(self, texts, instruct=None):
        # 與已驗證腳本一致：逐條推理（官方路徑單條語義）
        return [self._run(text=t) for t in texts]

    def _embed_image_raw(self, path: str):
        return self._run(images=[str(path)])
