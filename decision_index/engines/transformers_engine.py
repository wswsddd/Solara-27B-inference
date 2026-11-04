import copy
import math
import os

from decision_index.engines.base import Engine, Unsupported, text

SYSTEM_PROMPT = "You are a decision engine. Read the state, then answer the question by replying with exactly one option key from the list. Do not explain."


def render_prompt(state, question):
    lines = ["State:", text(state) if state not in ("", None, {}, []) else "(empty)", "", "Question:", text(question.get("instructions", "")), "", "Options:"]
    criteria = question["criteria"] if question["type"] == "choice" else {"false": "False", "true": "True"}
    for key, description in criteria.items():
        lines.append(f"{key}: {key if description is None else text(description)}")
    lines += ["", "Reply with exactly one option key."]
    return "\n".join(lines)


class TransformersEngine(Engine):
    name = "transformers"
    scoring_version = "independent-key-eos-v1"
    latency = "Device-synchronized in-process request wall time including prompt construction and independent candidate scoring; excludes model loading. First shapes may include kernel warmup."

    def __init__(self, model="Qwen/Qwen2.5-0.5B-Instruct", revision=None, device=None, dtype=None, attn="sdpa", max_tokens=None, cache_prefix=True, chat_template=True, enable_thinking=False, candidate_batch_size=16, candidate_cache=True, **options):
        super().__init__(**options)
        if not isinstance(candidate_batch_size, int) or candidate_batch_size < 1:
            raise ValueError("candidate_batch_size must be a positive integer")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        if dtype is None:
            dtype = "bfloat16" if self.device == "cuda" else "float32"
        self.dtype = getattr(torch, dtype)
        self.tok = AutoTokenizer.from_pretrained(model, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(model, revision=revision, dtype=self.dtype, attn_implementation=attn).to(self.device).eval()
        self.attn = attn
        text_config = getattr(self.model.config, "text_config", self.model.config)
        limit = getattr(text_config, "max_position_embeddings", None) or getattr(self.tok, "model_max_length", None)
        self.limit = min(limit, max_tokens) if max_tokens else limit
        # Hybrid linear-attn (e.g. Qwen3.5) DynamicCache.crop cannot roll back like DI expects.
        layer_types = getattr(text_config, "layer_types", None) or []
        self.prefix_cache_ok = not any("linear" in str(t).lower() for t in layer_types)
        self.cache_prefix = bool(cache_prefix) and self.prefix_cache_ok
        self.chat_template = chat_template and getattr(self.tok, "chat_template", None) is not None
        self.enable_thinking = enable_thinking
        self.candidate_batch_size = candidate_batch_size
        self.candidate_cache = bool(candidate_cache)
        self.eos_token_id = self.tok.eos_token_id
        self.pad_token_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else (self.eos_token_id or 0)
        self.model_id = model
        self.cache = None
        self.cached_ids = []
        self.provenance = {
            "kind": "transformers",
            "repo": model,
            "revision": revision or getattr(self.model.config, "_commit_hash", None),
            "device": self.device,
            "dtype": dtype,
            "attn_implementation": attn,
            "context_limit_tokens": self.limit,
            "chat_template": self.chat_template,
            "enable_thinking": self.enable_thinking,
            "prefix_cache": self.cache_prefix,
            "scoring_version": self.scoring_version,
            "candidate_batch_size": self.candidate_batch_size,
            "candidate_cache": self.candidate_cache,
            "answer_eos_token_id": self.eos_token_id,
            "policy": "Single-token options use next-token log-probabilities at one shared prompt position. Otherwise, every option key plus the tokenizer EOS is scored with an independent causal history; batch rows or separately cloned native caches isolate candidates. Scores are summed log-probabilities, normalized over all supplied options. No candidate packing, filtering, truncation, or length normalization. Native prompt-cache rollback is disabled for linear-attention models.",
        }

    def runtime(self):
        torch = self.torch
        info = {"torch": torch.__version__, "device": self.device}
        if self.device == "cuda":
            info.update(cuda=torch.version.cuda, gpu=torch.cuda.get_device_name())
        try:
            import transformers

            info["transformers"] = transformers.__version__
        except Exception:
            pass
        return info

    def synchronize(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        elif self.device == "mps":
            self.torch.mps.synchronize()

    def _prompt_ids(self, state, question):
        body = render_prompt(state, question)
        if self.chat_template:
            template_options = {}
            if self.enable_thinking is not None:
                template_options["enable_thinking"] = self.enable_thinking
            prompt = self.tok.apply_chat_template([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": body}], tokenize=False, add_generation_prompt=True, **template_options)
            return self.tok(prompt, add_special_tokens=False)["input_ids"]
        return self.tok(SYSTEM_PROMPT + "\n\n" + body + "\nAnswer:", add_special_tokens=True)["input_ids"]

    def _option_ids(self, keys):
        out = []
        for key in keys:
            ids = self.tok((" " if not self.chat_template else "") + key, add_special_tokens=False)["input_ids"]
            if not ids:
                raise Unsupported(f"option key {key!r} tokenizes to nothing")
            out.append(ids)
        return out

    def _common_prefix(self, ids):
        n = min(len(ids), len(self.cached_ids), len(ids) - 1)
        p = 0
        while p < n and ids[p] == self.cached_ids[p]:
            p += 1
        return p

    def _prefill(self, ids, need_candidate_cache):
        """Return the native prompt output; never put candidate suffixes in this cache."""
        torch = self.torch
        p = self._common_prefix(ids) if self.cache_prefix and self.cache is not None else 0
        cache = self.cache if p else None
        if cache is not None and cache.get_seq_length() > p:
            cache.crop(p - cache.get_seq_length())
        with torch.inference_mode():
            out = self.model(
                input_ids=torch.tensor([ids[p:]], device=self.device),
                attention_mask=torch.ones((1, len(ids)), dtype=torch.long, device=self.device),
                past_key_values=cache,
                use_cache=self.cache_prefix or need_candidate_cache,
                logits_to_keep=1,
            )
        if self.cache_prefix:
            self.cache = out.past_key_values
            self.cached_ids = list(ids)
        return out, p

    def _forward(self, prompt_ids, options):
        """Uncached reference: each complete candidate occupies its own batch row."""
        torch = self.torch
        q_len = len(prompt_ids)
        width = max(map(len, options))
        ids = torch.full((len(options), q_len + width), self.pad_token_id, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids)
        for row, option in enumerate(options):
            sequence = prompt_ids + option
            ids[row, :len(sequence)] = torch.tensor(sequence, device=self.device)
            mask[row, :len(sequence)] = 1
        positions = torch.arange(q_len - 1, q_len + width, device=self.device)
        with torch.inference_mode():
            out = self.model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=positions)
            logp = out.logits.float().log_softmax(-1)
            return [sum(logp[row, pos, token].item() for pos, token in enumerate(option + [self.eos_token_id])) for row, option in enumerate(options)]

    def _forward_cached(self, prompt_ids, options, cache, first_logprobs):
        """Clone both KV and recurrent state, then expand to independent batch rows."""
        torch = self.torch
        branch_cache = copy.deepcopy(cache)
        branch_cache.reorder_cache(torch.zeros(len(options), dtype=torch.long, device=self.device))
        width = max(map(len, options))
        ids = torch.full((len(options), width), self.pad_token_id, dtype=torch.long, device=self.device)
        mask = torch.ones((len(options), len(prompt_ids) + width), dtype=torch.long, device=self.device)
        for row, option in enumerate(options):
            ids[row, :len(option)] = torch.tensor(option, device=self.device)
            mask[row, len(prompt_ids) + len(option):] = 0
        with torch.inference_mode():
            out = self.model(input_ids=ids, attention_mask=mask, past_key_values=branch_cache, use_cache=True)
            logp = out.logits.float().log_softmax(-1)
            return [first_logprobs[option[0]].item() + sum(logp[row, pos - 1, token].item() for pos, token in enumerate(option[1:] + [self.eos_token_id], start=1)) for row, option in enumerate(options)]

    def _score_options(self, prompt_ids, options):
        torch = self.torch
        single_token = all(len(option) == 1 for option in options)
        if not single_token and self.eos_token_id is None:
            raise Unsupported("Multi-token option keys require a tokenizer EOS to score complete answers")
        if single_token or self.candidate_cache or self.cache_prefix:
            try:
                prefill, p = self._prefill(prompt_ids, not single_token and self.candidate_cache)
            except torch.OutOfMemoryError as exc:
                self.reset_cache()
                raise Unsupported(f"out of memory prefilling a {len(prompt_ids)}-token prompt") from exc
            first = prefill.logits[0, -1].float().log_softmax(-1)
            if single_token:
                return [first[option[0]].item() for option in options], p, False
            cache = prefill.past_key_values if self.candidate_cache else None
            if cache is not None and not hasattr(cache, "reorder_cache"):
                cache = None
            del prefill
        else:
            cache, first, p = None, None, 0
        chunk = min(self.candidate_batch_size, len(options))
        scores, start = [], 0
        while start < len(options):
            try:
                batch = options[start:start + chunk]
                part = self._forward_cached(prompt_ids, batch, cache, first) if cache is not None else self._forward(prompt_ids, batch)
                scores.extend(part)
                start += len(batch)
            except torch.OutOfMemoryError:
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                if chunk <= 1:
                    raise Unsupported(f"out of memory scoring one complete option after a {len(prompt_ids)}-token prompt")
                chunk = max(1, chunk // 2)
        return scores, p, cache is not None

    def __call__(self, state, questions):
        answers, raw = {}, {}
        for k, q in questions.items():
            if q["type"] not in ("choice", "noul"):
                raise Unsupported("Unsupported question type " + str(q["type"]))
            keys = list(q["criteria"]) if q["type"] == "choice" else ["false", "true"]
            ids = self._prompt_ids(state, q)
            options = self._option_ids(keys)
            single_token = all(len(option) == 1 for option in options)
            suffix_length = max(map(len, options)) + (0 if single_token else 1)
            if len(ids) + suffix_length > self.limit:
                raise Unsupported(f"prompt of {len(ids)} tokens plus complete option exceeds the {self.limit}-token context limit")
            scores, p, candidate_cache_used = self._score_options(ids, options)
            top = max(scores)
            weights = [math.exp(s - top) for s in scores]
            z = sum(weights)
            probs = {key: w / z for key, w in zip(keys, weights)}
            choice = max(keys, key=lambda x: probs[x])
            raw[k] = {"option_logprobs": dict(zip(keys, scores)), "prompt_tokens": len(ids), "cached_prefix_tokens": p, "scoring_mode": "next_token" if single_token else "complete_key_eos", "answer_eos_token_id": None if single_token else self.eos_token_id, "candidate_cache_used": candidate_cache_used, "scoring_version": self.scoring_version}
            if q["type"] == "choice":
                answers[k] = {"type": "choice", "choice": choice, "probabilities": probs}
            else:
                answers[k] = {"type": "noul", "noul": probs["true"]}
        return {"model": self.model_id, "answers": answers, "usage": {"input_tokens": sum(v["prompt_tokens"] for v in raw.values())}}, raw

    def reset_cache(self):
        self.cache = None
        self.cached_ids = []
