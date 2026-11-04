"""Native LM output rows with the same option protocol as Decision Head."""
import hashlib
import json
from pathlib import Path

from decision_index.engines.decision_head_engine import DecisionHeadEngine
from decision_index.engines.sglang_engine import SGLangDecisionEngine


class UnifiedLMHeadEngine(DecisionHeadEngine):
    name = 'unified_lm_head'
    scoring_version = 'unified-lm-option-softmax-v1'

    def __init__(self, model, device='cuda', dtype=None, **options):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

        # Variable-length ReadTwice prompts exhausted cuDNN's shape-specialized
        # kernel workspace outside the PyTorch allocator. Keep native flash /
        # memory-efficient SDPA, without cuDNN's per-shape JIT workspace.
        if device.startswith('cuda'):
            torch.backends.cuda.enable_cudnn_sdp(False)
        if options.get('enable_thinking', False) is not False:
            raise ValueError('Unified LM decisions require non-thinking prompts')
        if any(key in options for key in ('head_checkpoint', 'chat_template', 'candidate_cache', 'candidate_batch_size')):
            raise ValueError('Unified LM uses its own fixed label scoring protocol')
        self.torch, self.device = torch, device
        self.options = options
        self.protocol = SGLangDecisionEngine(base_url='http://localhost:1', model=model,
                                             tokenizer=model, label_mode='unified_letters',
                                             read_twice=options.get('read_twice', False))
        self.tok = self.protocol.tok
        labels, tokens, _ = self.protocol._labels([str(i) for i in range(256)])
        config = AutoConfig.from_pretrained(model, local_files_only=True)
        loader = AutoModelForImageTextToText if hasattr(config, 'vision_config') else AutoModelForCausalLM
        model_dtype = getattr(torch, dtype or ('bfloat16' if device.startswith('cuda') else 'float32'))
        self.model = loader.from_pretrained(model, dtype=model_dtype, local_files_only=True,
                                           attn_implementation='sdpa').to(device).eval()
        self.model.requires_grad_(False)
        self.backbone = getattr(self.model.model, 'language_model', self.model.model)
        output = self.model.get_output_embeddings()
        self.mapping = {'capacity': 256, 'hidden_size': output.weight.shape[1],
                        'labels': [{'label': label, 'token_id': token} for label, token in zip(labels, tokens)]}
        self.head = torch.nn.Linear(output.weight.shape[1], 256, bias=output.bias is not None,
                                    dtype=model_dtype, device=device)
        ids = torch.tensor(tokens, device=output.weight.device)
        with torch.no_grad():
            self.head.weight.copy_(output.weight.index_select(0, ids))
            if output.bias is not None:
                self.head.bias.copy_(output.bias.index_select(0, ids))
        self.head.requires_grad_(False)
        self.provenance = {'kind': self.name, 'base_model': str(Path(model).resolve()),
                           'base_config_sha256': hashlib.sha256((Path(model) / 'config.json').read_bytes()).hexdigest(),
                           'head_sha256': hashlib.sha256(self.head.weight.detach().cpu().view(torch.uint8).numpy().tobytes()).hexdigest(),
                           'scoring_version': self.scoring_version, 'label_map': self.mapping,
                           'prompt_version': self.protocol.prompt_version, 'enable_thinking': False,
                           'read_twice': self.protocol.read_twice,
                           'sdpa_cudnn_enabled': False,
                           'device': device, 'dtype': str(model_dtype)}
