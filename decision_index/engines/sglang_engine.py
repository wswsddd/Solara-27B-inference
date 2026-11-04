"""Remote first-token decisions through an existing SGLang deployment.

Like llm_client.get_noul_probability, the primary interface is OpenAI-compatible
chat completions with max_tokens=1 and logprobs=True. Missing top-k candidates
are recovered with SGLang's native selected-token logprobs, never fabricated.
Only the tokenizer is loaded in this process; model weights stay on the server.
"""

import hashlib
import itertools
import json
import math
import os
import string
import time
from urllib.parse import urlsplit

import httpx

from decision_index.engines.base import Engine, Unsupported
from decision_index.engines.http import CAPACITY_MARKERS
from decision_index.engines.transformers_engine import SYSTEM_PROMPT, render_prompt


# Keep A-Z unchanged, then use common, single Unicode characters instead of
# AA/AB/... . Admission still depends on the deployed tokenizer's exact round
# trip: a character is not assumed to be a token. No new vocabulary is added.
CHARACTER_LABELS = "".join(dict.fromkeys(string.ascii_uppercase + (
    "的一是不了在人有我他这中大来上国个到说们为子和你地出道也时年得就那要下以生会自着去之过家学对可她里后小么心多天而能好都然没日于起还发成事只作当想看文无开手十用主行方又如前所本见经头面公同三已老从动两长知民样现分将外但身些与高意进把法此实回二理美点月明其种声全工己话儿者向情部正名定女问力机给等几很业最间新什打便位因重被走电四第门相次东政海口使教西再平真听世气信北少关并内加化由却代军产入先山五太水万市眼体别处总才场师书比住员九笑性通目华报立马命张活难神数件安表原车白应路期叫死常提感金何更反合放做系计或司利受光王果亲界及今取六至流战品式结度拉非风百究论元花区容决备南石克术告共台象支院保认达接设且指运深满争低强许各更解量象众平名"
)))


class IncompleteLogprobs(ValueError):
    """The server did not return a finite logprob for every requested token."""


class SGLangDecisionEngine(Engine):
    name = "sglang"
    scoring_version = "sglang-next-token-v1"
    prompt_version = "di-clean-label-options-v2"
    latency = "HTTP wall time for all questions in a suite request, including tokenization, candidate mapping and any selected-token fallback; excludes server/model startup."

    def __init__(self, base_url=None, model="default", tokenizer=None,
                 label_mode="unified_letters", logprob_backend="auto", top_logprobs=20,
                 timeout=600, max_retries=2, token_env="SGLANG_API_KEY",
                 enable_thinking=False, label_alphabet=None, read_twice=False, **options):
        super().__init__(**options)
        from transformers import AutoTokenizer

        base_url = (base_url or os.environ.get("SGLANG_BASE_URL", "")).rstrip("/")
        if not base_url:
            raise ValueError("sglang requires base_url (or SGLANG_BASE_URL)")
        parsed = urlsplit(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base_url must be an HTTP(S) service URL without credentials/query/fragment")
        if label_mode not in ("auto", "keys", "letters", "characters", "unified_letters"):
            raise ValueError("label_mode must be auto, keys, letters, characters or unified_letters")
        if label_alphabet is not None and (label_mode != "characters" or label_alphabet not in ("mixed", "cjk", "alnum", "latin")):
            raise ValueError("label_alphabet requires label_mode=characters and must be mixed, cjk, alnum or latin")
        if logprob_backend not in ("auto", "chat", "native"):
            raise ValueError("logprob_backend must be auto, chat or native")
        if enable_thinking is not False:
            raise ValueError("Next-token decisions require enable_thinking=false")
        if not isinstance(top_logprobs, int) or top_logprobs < 1:
            raise ValueError("top_logprobs must be a positive integer")
        if not isinstance(max_retries, int) or max_retries < 0 or timeout <= 0:
            raise ValueError("max_retries must be nonnegative and timeout must be positive")

        if not isinstance(read_twice, bool):
            raise ValueError("read_twice must be a boolean")
        self.read_twice = read_twice
        self.model_id = model
        self.label_mode = label_mode
        if label_mode == "unified_letters":
            self.prompt_version = "di-unified-letter-options-v4-ragtruth-train-aligned"
        if read_twice:
            self.prompt_version += "-read-twice-user-v1"
        self.label_alphabet = (label_alphabet or "mixed") if label_mode == "characters" else None
        self.logprob_backend = logprob_backend
        self.top_logprobs = top_logprobs
        self.max_retries = max_retries
        self.tokenizer_path = str(tokenizer or model)
        self.tok = AutoTokenizer.from_pretrained(self.tokenizer_path)
        if not getattr(self.tok, "chat_template", None):
            raise ValueError("sglang next-token decisions require the deployed model's chat tokenizer")
        self.template_options = {"thinking": False} if "deepseek" in model.lower() else {"enable_thinking": False}
        self._key_tokens = {}
        self._letter_pool = None
        self._character_pool = None
        # Accept either http://host:8000 or its OpenAI base http://host:8000/v1.
        root = base_url[:-3] if base_url.endswith("/v1") else base_url
        headers = {}
        token = os.environ.get(token_env)
        if token:
            headers["Authorization"] = "Bearer " + token
        self.client = httpx.Client(base_url=root + "/", headers=headers, timeout=timeout, trust_env=False)
        self.provenance = {
            "kind": "sglang", "base_url": root, "model": model,
            "tokenizer": self.tokenizer_path,
            "chat_template_sha256": hashlib.sha256(json.dumps(self.tok.chat_template, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            "enable_thinking": False, "scoring_version": self.scoring_version,
            "prompt_version": self.prompt_version, "read_twice": self.read_twice,
            "label_mode": label_mode,
            "label_alphabet": self.label_alphabet,
            "label_policy": "single-token-uppercase-unified-v1" if label_mode == "unified_letters" else (("single-token-character-v1" if self.label_alphabet == "mixed" else f"single-token-character-{self.label_alphabet}-v1") if label_mode == "characters" else "single-token-uppercase-v1"),
            "logprob_backend": logprob_backend, "top_logprobs": top_logprobs,
            "policy": "One next-token distribution per question, normalized only over all candidate tokens. Auto and characters retain distinct single-token keys. Otherwise every key is mapped in insertion order to a distinct single-token label: auto/letters use A-Z then admitted two-letter labels; characters uses the recorded single-character alphabet, never multi-character aliases or new vocabulary. Character alphabets: mixed=A-Z+CJK; cjk=CJK only; alnum=A-Z+a-z+digits+CJK; latin=alnum+ASCII punctuation excluding colon+Latin letters U+00C0..U+017F+CJK. Remapped options display only the label and original description, without prepending the old key; a null description retains the old key as its only semantic content. Original keys are restored in the returned answers. No candidate suffix scoring or EOS. Incomplete chat top-k falls back to native selected-token logprobs in auto mode; incomplete probabilities are errors, never zero-filled. HTTP proxies from the environment are disabled.",
        }
        if label_mode == "unified_letters":
            self.provenance["policy"] += " Unified letters overrides all direct-key shortcuts: every choice and noul uses the same verified uppercase label pool. Noul order is false then true (A=false, B=true); the returned probability is still P(true)."
            self.provenance["policy"] += " RAGTruth uses its training layout: empty State, context prompt and model response in Question, and complete false/true option descriptions. Other questions retain their existing layout."

    def _token_id(self, key):
        if key not in self._key_tokens:
            ids = self.tok.encode(key, add_special_tokens=False)
            self._key_tokens[key] = ids[0] if len(ids) == 1 and self.tok.decode(ids, clean_up_tokenization_spaces=False) == key else None
        return self._key_tokens[key]

    def _labels(self, keys):
        ids = [self._token_id(key) for key in keys]
        direct = all(token is not None for token in ids) and len(set(ids)) == len(ids)
        if self.label_mode not in ("letters", "unified_letters") and direct:
            return keys, ids, False
        if self.label_mode == "keys":
            raise Unsupported("next-token keys mode needs distinct single-token keys; use label_mode=auto or letters")
        if self.label_mode == "characters":
            if self._character_pool is None:
                pool, seen = [], set()
                candidates = CHARACTER_LABELS
                if self.label_alphabet != "mixed":
                    cjk = "".join(label for label in CHARACTER_LABELS if not label.isascii())
                    if self.label_alphabet == "cjk":
                        candidates = cjk
                    else:
                        latin = "".join(chr(n) for n in range(0xC0, 0x180) if chr(n).isalpha())
                        extra = string.punctuation.replace(":", "") + latin if self.label_alphabet == "latin" else ""
                        candidates = string.ascii_uppercase + string.ascii_lowercase + string.digits + extra + cjk
                for label in candidates:
                    token = self._token_id(label)
                    if token is not None and token not in seen:
                        seen.add(token)
                        pool.append((label, token))
                self._character_pool = pool
            if len(keys) > len(self._character_pool):
                raise Unsupported(f"{len(keys)} candidates exceed this tokenizer's {len(self._character_pool)} single-character token labels")
            selected = self._character_pool[:len(keys)]
            return [s[0] for s in selected], [s[1] for s in selected], True
        if self._letter_pool is None:
            # AA, AB, ... are admitted only when this tokenizer really encodes
            # them as one token that decodes back to the exact displayed label.
            pool, seen = [], set()
            for width in (1, 2):
                for chars in itertools.product(string.ascii_uppercase, repeat=width):
                    label = "".join(chars)
                    token = self._token_id(label)
                    if token is not None and token not in seen:
                        seen.add(token)
                        pool.append((label, token))
            self._letter_pool = pool
        if len(keys) > len(self._letter_pool):
            raise Unsupported(f"{len(keys)} candidates exceed this tokenizer's {len(self._letter_pool)} single-token letter labels")
        selected = self._letter_pool[:len(keys)]
        return [s[0] for s in selected], [s[1] for s in selected], True

    def _post(self, path, body):
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.post(path, json=body)
                if response.status_code in (400, 413, 422) and any(marker in response.text.lower() for marker in CAPACITY_MARKERS):
                    raise Unsupported(response.text)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in (429, 500, 502, 503, 504) or attempt == self.max_retries:
                    raise
            except httpx.TransportError:
                if attempt == self.max_retries:
                    raise
            time.sleep(min(2 ** attempt, 4))

    @staticmethod
    def _require_scores(by_token, token_ids):
        missing = [token for token in token_ids if token not in by_token or not math.isfinite(by_token[token])]
        if missing:
            raise IncompleteLogprobs(f"Missing/nonfinite next-token logprobs for {len(missing)} candidate token IDs: {missing[:8]}")
        return [by_token[token] for token in token_ids]

    def _chat_scores(self, messages, token_ids):
        raw = self._post("v1/chat/completions", {
            "model": self.model_id, "messages": messages,
            "max_tokens": 1, "temperature": 0, "logprobs": True,
            "top_logprobs": self.top_logprobs,
            "chat_template_kwargs": self.template_options,
        })
        choices = raw.get("choices") or []
        content = ((choices[0].get("logprobs") or {}).get("content") or []) if choices else []
        if not content:
            raise IncompleteLogprobs("Chat response has no first-token logprobs")
        first = content[0]
        entries = [first] + (first.get("top_logprobs") or [])
        by_text = {}
        for entry in entries:
            token_text = entry.get("token")
            # SGLang may encode non-ASCII tokens using bytes. Match the complete
            # token, never strip whitespace or merge similar-looking tokens.
            if entry.get("bytes") is not None:
                try:
                    token_text = bytes(entry["bytes"]).decode("utf-8")
                except (UnicodeDecodeError, ValueError, TypeError):
                    continue
            value = entry.get("logprob")
            if token_text is not None and isinstance(value, (int, float)):
                by_text[token_text] = float(value)
        by_id = {}
        for token in token_ids:
            decoded = self.tok.decode([token], clean_up_tokenization_spaces=False)
            if decoded in by_text:
                by_id[token] = by_text[decoded]
        return self._require_scores(by_id, token_ids), raw

    def _native_scores(self, messages, token_ids):
        prompt_ids = self.tok.apply_chat_template(messages, tokenize=True, return_dict=False, add_generation_prompt=True, **self.template_options)
        raw = self._post("generate", {
            "input_ids": prompt_ids,
            "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            "return_logprob": True, "logprob_start_len": -1,
            "token_ids_logprob": token_ids, "return_text_in_logprobs": False,
        })
        positions = raw.get("meta_info", {}).get("output_token_ids_logprobs") or []
        if not positions:
            raise IncompleteLogprobs("Native response has no output_token_ids_logprobs for the first token")
        by_id = {}
        for entry in positions[0]:
            if len(entry) >= 2 and isinstance(entry[0], (int, float)):
                by_id[int(entry[1])] = float(entry[0])
        return self._require_scores(by_id, token_ids), raw

    def prepare_question(self, state, question):
        """Shared, network-free train/eval protocol: labels, text and token IDs."""
        kind = question["type"]
        if kind not in ("choice", "noul"):
            raise Unsupported("Unsupported question type " + kind)
        criteria = question["criteria"] if kind == "choice" else {"false": "False", "true": "True"}
        instructions = question.get("instructions", "")
        # RAGTruth SFT rows were converted from a choice task: their context
        # and response live in Question, and the options spell out the two
        # meanings. Match that input without changing noul's P(true) API or
        # mutating the frozen suite. Match the task schema, never its answers.
        if (self.label_mode == "unified_letters" and kind == "noul"
                and instructions == "The response contains content that is not supported by the context in the prompt."
                and isinstance(state, dict) and set(state) == {"prompt", "response"}):
            descriptions = question.get("criteria") or {}
            if set(descriptions) != {"false", "true"} or not all(isinstance(state[k], str) for k in ("prompt", "response")):
                raise Unsupported("Training-aligned RAGTruth requires prompt/response strings and complete false/true descriptions")
            criteria = {key: descriptions[key] for key in ("false", "true")}
            instructions = f"Context prompt:\n{state['prompt']}\n\nModel response:\n{state['response']}\n\n{instructions}"
            state = None
        keys = list(criteria)
        if len(keys) < 2:
            raise Unsupported("A next-token decision needs at least two candidates")
        if kind == "noul" and self.label_mode != "unified_letters":
            labels = keys
            token_ids = [self._token_id(key) for key in keys]
            if any(token is None for token in token_ids) or len(set(token_ids)) != 2:
                raise Unsupported("noul requires distinct single-token true and false labels")
            mapped = False
        else:
            labels, token_ids, mapped = self._labels(keys)
        displayed = {label: criteria[key] if criteria[key] is not None else key for label, key in zip(labels, keys)} if mapped else dict(criteria)
        body = render_prompt(state, {"type": "choice", "instructions": instructions, "criteria": displayed})
        if self.read_twice:
            body = body + "\n\n" + body
        return {"messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": body}],
                "keys": keys, "labels": labels, "token_ids": token_ids, "remapped": mapped}

    def __call__(self, state, questions):
        answers, traces = {}, {}
        for name, question in questions.items():
            kind = question["type"]
            prepared = self.prepare_question(state, question)
            messages, keys, labels, token_ids, mapped = (prepared[k] for k in ("messages", "keys", "labels", "token_ids", "remapped"))
            backend = self.logprob_backend
            fallback_reason = None
            if backend == "native":
                scores, raw = self._native_scores(messages, token_ids)
            else:
                try:
                    scores, raw = self._chat_scores(messages, token_ids)
                    backend = "chat"
                except IncompleteLogprobs as exc:
                    if backend != "auto":
                        raise
                    fallback_reason = str(exc)
                    scores, raw = self._native_scores(messages, token_ids)
                    backend = "native"
            peak = max(scores)
            weights = [math.exp(score - peak) for score in scores]
            normalizer = sum(weights)
            probabilities = dict(zip(keys, (weight / normalizer for weight in weights)))
            if kind == "choice":
                answers[name] = {"type": kind, "choice": max(probabilities, key=probabilities.get), "probabilities": probabilities}
            else:
                answers[name] = {"type": kind, "noul": probabilities["true"]}
            traces[name] = {
                "scoring_version": self.scoring_version, "backend": backend,
                "prompt_version": self.prompt_version,
                "read_twice": self.read_twice,
                "label_mode": self.label_mode, "label_policy": self.provenance["label_policy"],
                "label_alphabet": self.label_alphabet,
                "remapped": mapped, "key_to_label": dict(zip(keys, labels)),
                "key_to_token_id": dict(zip(keys, token_ids)),
                "option_logprobs": dict(zip(keys, scores)),
                "fallback_reason": fallback_reason, "response": raw,
            }
        return {"model": self.model_id, "answers": answers}, traces

    def close(self):
        self.client.close()
