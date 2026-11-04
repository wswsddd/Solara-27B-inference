"""Train-aligned Qwen3.5 Decision Head, with optional Megatron LoRA."""
import hashlib
import json
from pathlib import Path

from decision_index.engines.base import Engine, Unsupported
from decision_index.engines.sglang_engine import SGLangDecisionEngine


class DecisionHeadEngine(Engine):
    name = 'decision_head'
    scoring_version = 'decision-head-option-softmax-v2'

    def __init__(self, model, head_checkpoint, device='cuda', dtype=None, **options):
        if options.get('enable_thinking', False) is not False or any(
                key in options for key in ('chat_template', 'candidate_cache', 'candidate_batch_size')):
            raise ValueError('Decision Head requires the fixed non-thinking training protocol')
        super().__init__(**options)
        import torch
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

        self.torch = torch
        self.device = device
        path = Path(head_checkpoint)
        if not (path / 'COMPLETE').is_file():
            raise ValueError('Incomplete decision-head checkpoint')
        meta = json.loads((path / 'decision_config.json').read_text())
        versions = {f'v{v}-masked-option-head-v1' for v in (61, 62, 63, 65, 66, 67, 69)}
        if meta['version'] not in versions:
            raise ValueError('Unsupported decision-head version')
        if (path / 'COMPLETE').read_text().strip() != meta['version']:
            raise ValueError('Checkpoint completion/version mismatch')
        self.mapping = json.loads((path / 'label_map.json').read_text())
        if self.mapping != meta['label_map']:
            raise ValueError('Checkpoint label map mismatch')
        full_backbone = meta.get('backbone_format') == 'huggingface'
        expected_model = path.resolve() if full_backbone else Path(meta['base_model']).resolve()
        if full_backbone and (meta['version'] != 'v69-masked-option-head-v1' or meta.get('backbone_checkpoint') != '.'):
            raise ValueError('Unsupported full-backbone Decision Head checkpoint')
        if Path(model).resolve() != expected_model:
            raise ValueError('Must evaluate with the same training base model')
        if not full_backbone and hashlib.sha256((Path(model) / 'config.json').read_bytes()).hexdigest() != meta['base_config_sha256']:
            raise ValueError('Frozen base config changed')
        # Reuse the exact train-aligned DI prompt preparation, without HTTP calls.
        self.protocol = SGLangDecisionEngine(base_url='http://localhost:1', model=model,
                                             tokenizer=model, label_mode='unified_letters')
        self.tok = self.protocol.tok
        template_hash = self.protocol.provenance['chat_template_sha256']
        if template_hash != self.mapping['chat_template_sha256']:
            raise ValueError('Chat template differs from training')
        for item in self.mapping['labels']:
            if self.tok.encode(item['label'], add_special_tokens=False) != [item['token_id']]:
                raise ValueError('Tokenizer differs from training')
        model_dtype = getattr(torch, dtype or ('bfloat16' if device.startswith('cuda') else 'float32'))
        config = AutoConfig.from_pretrained(model, local_files_only=True)
        loader = AutoModelForImageTextToText if hasattr(config, 'vision_config') else AutoModelForCausalLM
        self.model = loader.from_pretrained(
            model, dtype=model_dtype, local_files_only=True, attn_implementation='sdpa').to(device).eval()
        self.model.requires_grad_(False)
        self.backbone = getattr(self.model.model, 'language_model', self.model.model)
        adapter_info = None
        adapter_sha = None
        if 'adapter_config' in meta:
            from decision_index.engines.decision_lora import load_decision_lora
            adapter_info = load_decision_lora(self.backbone, path, meta)
            adapter_sha = hashlib.sha256((path / 'lora_parameters.safetensors').read_bytes()).hexdigest()
        elif (path / 'lora_parameters.safetensors').exists() or (not full_backbone and meta['version'] != 'v61-masked-option-head-v1'):
            raise ValueError('LoRA checkpoint is missing adapter metadata')
        self.model.requires_grad_(False).eval()
        self.head = torch.nn.Linear(self.mapping['hidden_size'], self.mapping['capacity'],
                                    bias=False, dtype=model_dtype, device=device)
        self.head.load_state_dict(load_file(str(path / 'decision_head.safetensors')))
        self.head.requires_grad_(False)
        self.provenance = {'kind': self.name, 'base_model': model,
                           'head_checkpoint': str(path.resolve()),
                           'head_sha256': hashlib.sha256((path / 'decision_head.safetensors').read_bytes()).hexdigest(),
                           'scoring_version': self.scoring_version, 'label_map': self.mapping,
                           'checkpoint_version': meta['version'], 'brier_lambda': meta.get('brier_lambda', 0),
                           'lora_sha256': adapter_sha, 'lora': adapter_info,
                           'base_config_sha256': meta['base_config_sha256'],
                           'backbone_format': meta.get('backbone_format', 'frozen-base'),
                           'prompt_version': self.protocol.prompt_version,
                           'device': device, 'dtype': str(model_dtype), 'enable_thinking': False}

    def __call__(self, state, questions):
        torch = self.torch
        answers, traces = {}, {}
        for name, question in questions.items():
            prepared = self.protocol.prepare_question(state, question)
            n = len(prepared['keys'])
            if n > self.mapping['capacity']:
                raise Unsupported(f'{n} candidates exceed head capacity')
            if prepared['labels'] != [x['label'] for x in self.mapping['labels'][:n]]:
                raise ValueError('Evaluation label order differs from training')
            ids = self.tok.apply_chat_template(prepared['messages'], tokenize=True,
                add_generation_prompt=True, enable_thinking=False, return_tensors='pt', return_dict=False).to(self.device)
            limit = getattr(self.model.config, 'text_config', self.model.config).max_position_embeddings
            if ids.shape[-1] > limit:
                raise Unsupported('Prompt exceeds model context; no truncation')
            with torch.inference_mode():
                hidden = self.backbone(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state[:, -1]
                logits = self.head(hidden)[0, :n].float()
                probs = logits.softmax(-1).cpu().tolist()
            if not torch.isfinite(logits).all().item():
                raise ValueError('Nonfinite decision logits')
            probabilities = dict(zip(prepared['keys'], probs))
            if question['type'] == 'choice':
                answers[name] = {'type': 'choice', 'choice': max(probabilities, key=probabilities.get),
                                 'probabilities': probabilities}
            else:
                answers[name] = {'type': 'noul', 'noul': probabilities['true']}
            traces[name] = {'scoring_version': self.scoring_version,
                            'key_to_label': dict(zip(prepared['keys'], prepared['labels'])),
                            'option_logits': dict(zip(prepared['keys'], logits.cpu().tolist()))}
        return {'model': self.name, 'answers': answers}, {'questions': traces}

    def synchronize(self):
        if self.device.startswith('cuda'):
            self.torch.cuda.synchronize()

    def close(self):
        self.protocol.close()
