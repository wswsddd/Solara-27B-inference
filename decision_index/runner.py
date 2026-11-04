import collections
import hashlib
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from decision_index import constants as C
from decision_index.engines import NativeAbstention, Unsupported, load_engine, validate
from decision_index.suite.io import atomic_json, dumps, read_jsonl


def stamp():
    return datetime.now(timezone.utc).isoformat()


def is_device_error(result):
    message = result.get('error', '').lower()
    return result.get('exception') in ('OutOfMemoryError', 'AcceleratorError') or any(
        marker in message for marker in ('device-side assert', 'cuda ran out of memory',
            'cuda out of memory', 'cudnn_status_execution_failed',
            'cublas_status_execution_failed', 'illegal memory access'))


def iter_rows(rows_path, keep=None):
    paths = rows_path if isinstance(rows_path, (list, tuple)) else [rows_path]
    for path in paths:
        for row in read_jsonl(path):
            if keep is None or keep(row["_evaluation"]):
                yield row


def validate_resume_provenance(previous, engine_name, provenance):
    """Do not mix predictions from different model/scoring contracts."""
    kind = provenance.get("kind")
    if kind not in ("transformers", "sglang", "decision_head", "unified_lm_head"):
        return
    old = previous.get("model_source", {})
    if kind == "transformers":
        fields = ("repo", "revision", "chat_template", "enable_thinking", "scoring_version")
    elif kind in ('decision_head', 'unified_lm_head'):
        fields = ('base_model', 'base_config_sha256', 'head_sha256', 'lora_sha256',
                  'checkpoint_version', 'label_map', 'scoring_version', 'prompt_version', 'enable_thinking')
    else:
        fields = ("base_url", "model", "tokenizer", "chat_template_sha256", "enable_thinking", "scoring_version", "prompt_version", "label_mode", "label_policy", "logprob_backend", "top_logprobs")
    changed = [key for key in fields if old.get(key) != provenance.get(key)]
    if previous.get("engine") != engine_name:
        changed.append("engine")
    if changed:
        raise ValueError("Cannot resume results with different model/scoring settings (" + ", ".join(changed) + "). Use a new output directory or --fresh.")


def run(engine_name, engine_options, rows_path, out_dir, limit=None, compact=False, resume=True, warm=True, seed=C.RUN_SEED, corpus_sha256=None, halt_on_device_error=True, log=print, keep=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.jsonl"
    previous_environment = None
    if resume and results_path.exists() and results_path.stat().st_size:
        environment_path = out / "environment.json"
        if environment_path.exists():
            previous_environment = json.loads(environment_path.read_text(encoding="utf-8"))

    def event(**kw):
        record = {"time": stamp(), **kw}
        log(dumps(record))
        atomic_json(out / "status.json", record, indent=None)

    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
    t = time.perf_counter()
    event(event="loading", engine=engine_name)
    engine = load_engine(engine_name, **engine_options)
    engine.synchronize()
    if resume and results_path.exists() and results_path.stat().st_size and engine.provenance.get("kind") in ("transformers", "sglang", "decision_head", "unified_lm_head"):
        if previous_environment is None:
            engine.close()
            raise ValueError("Cannot verify model/scoring settings without environment.json; use a new output directory or --fresh.")
        try:
            validate_resume_provenance(previous_environment, engine_name, engine.provenance)
        except ValueError:
            engine.close()
            raise
    atomic_json(out / "environment.json", {"engine": engine_name, "engine_options": engine_options, "model_source": engine.provenance, **engine.runtime(), "loaded_seconds": time.perf_counter() - t, "frozen_corpus_sha256": corpus_sha256, "rows_path": [str(p) for p in rows_path] if isinstance(rows_path, (list, tuple)) else str(rows_path), "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "latency": engine.latency})
    if warm:
        engine.warmup()
        engine.synchronize()
    event(event="ready", engine=engine_name)
    previous = {}
    if resume and results_path.exists():
        for r in read_jsonl(results_path, complete_lines_only=True):
            previous[r["run_id"]] = r["status"]
        if previous:
            text = results_path.read_text(encoding="utf-8")
            if not text.endswith("\n"):
                results_path.write_text(text[: text.rfind("\n") + 1], encoding="utf-8")
    counts = collections.Counter()
    start = time.perf_counter()
    finished = 0
    completed = set(previous)
    with results_path.open("a" if resume else "w", encoding="utf-8") as logf:
        for row in iter_rows(rows_path, keep):
            e = row["_evaluation"]
            rid = e["run_id"]
            if rid in previous and previous[rid] != "error":
                continue
            payload = {"state": row["state"], "questions": row["questions"]}
            t = time.perf_counter()
            engine.synchronize()
            result = {**e, "started_utc": stamp(), "engine": engine_name}
            if not compact:
                result["payload"] = payload
            try:
                response, raw = engine(**payload)
                engine.synchronize()
                validate(payload["questions"], response)
                result.update(status="ok", response=response)
                if not compact:
                    result["raw_output"] = raw
            except NativeAbstention as exc:
                result.update(status="abstained", error=str(exc))
                if not compact:
                    result["raw_output"] = exc.raw
            except Unsupported as exc:
                result.update(status="unsupported", error=str(exc))
            except Exception as exc:
                result.update(status="error", error=str(exc), exception=type(exc).__name__, traceback=traceback.format_exc())
            elapsed = (time.perf_counter() - t) * 1000
            result.update(completed_utc=stamp(), total_wall_ms=elapsed, model_request_wall_ms=elapsed)
            logf.write(dumps(result) + "\n")
            logf.flush()
            counts[result["status"]] += 1
            finished += 1
            completed.add(rid)
            if finished % 10 == 0:
                event(event="progress", engine=engine_name, completed=len(completed), counts=dict(counts), elapsed_seconds=round(time.perf_counter() - start, 1))
            if halt_on_device_error and is_device_error(result):
                event(event="failed", engine=engine_name, completed=len(completed), counts=dict(counts), error=result.get("error"), failed_run_id=rid, elapsed_seconds=round(time.perf_counter() - start, 1))
                raise RuntimeError("Device error; stopping before the accelerator context is reused")
            if counts["error"] >= 5 and counts["ok"] == 0:
                event(event="failed", engine=engine_name, completed=len(completed), counts=dict(counts), error="five failed requests without a success")
                raise RuntimeError("Five failed requests without a success; stop instead of filling the benchmark with runtime errors")
            if limit and finished >= limit:
                break
    engine.close()
    final = dict(event="complete", engine=engine_name, completed=len(completed), counts=dict(counts), elapsed_seconds=round(time.perf_counter() - start, 1))
    event(**final)
    return final
