#!/usr/bin/env python3
"""Run decision_index.runner on one shard of the suite (data parallel across nodes)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decision_index.runner import run  # noqa: E402
from decision_index.suite.io import Suite  # noqa: E402


def shard_keep(shard: int, num_shards: int):
    if not 0 <= shard < num_shards:
        raise ValueError(f"shard must be in [0, {num_shards}), got {shard}")

    def keep(ev: dict) -> bool:
        digest = hashlib.sha256(ev["run_id"].encode()).hexdigest()
        return int(digest, 16) % num_shards == shard

    return keep


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", help="HF/local checkpoint for transformers; tokenizer default for sglang")
    p.add_argument("--model-tag", help="label in logs; default: basename of model-path")
    p.add_argument("--suite-dir", type=Path, required=True)
    p.add_argument("--edition", default="0.3")
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rows", type=Path, help="optional rows file instead of full suite")
    p.add_argument("--limit", type=int)
    p.add_argument("--compact", action="store_true")
    p.add_argument("--fresh", action="store_true", help="ignore existing results in this shard out dir")
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--engine", default="transformers")
    p.add_argument("--head-checkpoint", help="decision_head: v61 frozen head or v62/v63/v65-v67 LoRA + head checkpoint")
    p.add_argument("--device", help="transformers engine option, e.g. cuda")
    p.add_argument("--chat-template", choices=("true", "false"), help="transformers --option chat_template=...")
    p.add_argument("--enable-thinking", choices=("true", "false"), help="chat template thinking mode; default false")
    p.add_argument("--candidate-batch-size", type=int, help="independent multi-token candidates per batch; default 16")
    p.add_argument("--candidate-cache", choices=("true", "false"), help="clone native prompt caches for independent candidates; default true")
    p.add_argument("--base-url", help="sglang service URL, e.g. http://host:8000/v1")
    p.add_argument("--model-name", help="model name served by sglang")
    p.add_argument("--tokenizer-path", help="sglang tokenizer path/ID; defaults to --model-path")
    p.add_argument("--label-mode", choices=("unified_letters", "auto", "keys", "letters", "characters"), help="sglang: default unified_letters maps every choice and noul to verified letter tokens; other modes retain historical behavior")
    p.add_argument("--label-alphabet", choices=("mixed", "cjk", "alnum", "latin"), help="sglang characters mode: choose the verified single-character alphabet; default mixed")
    p.add_argument("--logprob-backend", choices=("auto", "chat", "native"), help="sglang: auto falls back from incomplete chat top-k to native selected-token scores")
    p.add_argument("--top-logprobs", type=int, help="sglang chat top-k logprobs; default 20")
    p.add_argument("--request-timeout", type=float, help="sglang HTTP timeout in seconds; default 600")
    p.add_argument("--api-key-env", help="sglang API-key environment variable name; default SGLANG_API_KEY")
    p.add_argument("--read-twice", action="store_true", help="sglang/unified_lm_head: repeat the complete user prompt twice; system remains once")
    args = p.parse_args()
    if args.read_twice and args.engine not in ("sglang", "unified_lm_head"):
        p.error("--read-twice requires --engine sglang or unified_lm_head")
    if args.engine == 'decision_head':
        if not args.head_checkpoint:
            p.error('--engine decision_head requires --head-checkpoint')
        if (args.enable_thinking == 'true' or args.chat_template is not None
                or args.candidate_batch_size is not None or args.candidate_cache is not None):
            p.error('decision_head uses the fixed non-thinking protocol; candidate/chat-template options are unsupported')
    elif args.head_checkpoint:
        p.error('--head-checkpoint requires --engine decision_head')

    if args.engine == "sglang":
        if not args.model_name:
            p.error("--engine sglang requires --model-name (the deployed service's model name)")
        tokenizer = args.tokenizer_path or args.model_path
        if not tokenizer:
            p.error("--engine sglang requires --tokenizer-path or --model-path")
        if args.enable_thinking == "true":
            p.error("sglang next-token decisions require --enable-thinking false")
        if args.label_alphabet is not None and args.label_mode != "characters":
            p.error("--label-alphabet requires --label-mode characters")
        if args.device or args.chat_template is not None or args.candidate_batch_size is not None or args.candidate_cache is not None:
            p.error("--device, --chat-template and --candidate-* are local transformers options")
        engine_opts = {"model": args.model_name, "tokenizer": tokenizer, "read_twice": args.read_twice}
        for attr, key in (("base_url", "base_url"), ("label_mode", "label_mode"),
                          ("label_alphabet", "label_alphabet"),
                          ("logprob_backend", "logprob_backend"), ("top_logprobs", "top_logprobs"),
                          ("request_timeout", "timeout"), ("api_key_env", "token_env")):
            value = getattr(args, attr)
            if value is not None:
                engine_opts[key] = value
    else:
        if not args.model_path:
            p.error("--model-path is required for this engine")
        if any(value is not None for value in (args.base_url, args.model_name, args.tokenizer_path, args.label_mode, args.label_alphabet, args.logprob_backend, args.top_logprobs, args.request_timeout, args.api_key_env)):
            p.error("SGLang service options require --engine sglang")
        engine_opts = {"model": args.model_path}
        if args.engine == 'unified_lm_head':
            engine_opts['read_twice'] = args.read_twice
        if args.engine == 'decision_head':
            engine_opts['head_checkpoint'] = args.head_checkpoint
        if args.device:
            engine_opts["device"] = args.device
        if args.chat_template is not None:
            engine_opts["chat_template"] = args.chat_template == "true"
        if args.enable_thinking is not None:
            engine_opts["enable_thinking"] = args.enable_thinking == "true"
        if args.candidate_batch_size is not None:
            if args.candidate_batch_size < 1:
                p.error("--candidate-batch-size must be positive")
            engine_opts["candidate_batch_size"] = args.candidate_batch_size
        if args.candidate_cache is not None:
            engine_opts["candidate_cache"] = args.candidate_cache == "true"
    tag = args.model_tag or Path(args.model_path or args.model_name).name

    keep_fn = shard_keep(args.shard, args.num_shards)
    if args.rows:
        rows = args.rows
        corpus = None
        keep = keep_fn
    else:
        suite = Suite(args.suite_dir, args.edition)
        report = suite.verify(strict=not args.no_verify)
        rows = suite.row_paths

        def keep(ev):
            return suite.in_edition(ev) and keep_fn(ev)

        corpus = report["sha256"]

    args.out.mkdir(parents=True, exist_ok=True)
    meta = {
        "model_path": args.model_path,
        "model_tag": tag,
        "engine": args.engine,
        "head_checkpoint": args.head_checkpoint,
        "model_name": args.model_name,
        "tokenizer_path": args.tokenizer_path,
        "base_url": args.base_url,
        "read_twice": args.read_twice,
        "shard": args.shard,
        "num_shards": args.num_shards,
        "edition": args.edition,
        "suite_dir": str(args.suite_dir),
        "rows": str(args.rows) if args.rows else None,
    }
    (args.out / "shard_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    result = run(
        args.engine,
        engine_opts,
        rows,
        args.out,
        limit=args.limit,
        compact=args.compact,
        resume=not args.fresh,
        corpus_sha256=corpus,
        keep=keep,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
