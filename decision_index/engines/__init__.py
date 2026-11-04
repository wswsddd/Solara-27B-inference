from decision_index.engines.base import Engine, NativeAbstention, RandomEngine, Unsupported, validate

REGISTRY = {
    "http": "decision_index.engines.http:HttpSystemOne",
    "transformers": "decision_index.engines.transformers_engine:TransformersEngine",
    "sglang": "decision_index.engines.sglang_engine:SGLangDecisionEngine",
    "unified_lm_head": "decision_index.engines.unified_lm_head_engine:UnifiedLMHeadEngine",
    "random": "decision_index.engines.base:RandomEngine",
}


def load_engine(name, **options):
    import importlib

    target = REGISTRY.get(name, name)
    if ":" not in target:
        raise KeyError(f"unknown engine {name!r}; use one of {sorted(REGISTRY)} or a dotted module:Class path")
    module, cls = target.split(":")
    return getattr(importlib.import_module(module), cls)(**options)


__all__ = ["Engine", "RandomEngine", "Unsupported", "NativeAbstention", "validate", "load_engine", "REGISTRY"]
