# Solara-27B

Solara-27B is a full-parameter fine-tune of Qwen/Qwen3.8-27B. The public
Decision Index 0.3 run uses the unified LM-head option softmax, Read Twice on,
and thinking off. Model weights are Apache-2.0; inference code is MIT.

- [Model weights](https://huggingface.co/WangSenQWQ/Solara-27B)
- [Public scores.json](https://huggingface.co/datasets/WangSenQWQ/Solara-27B-decision-index-results/blob/main/runs/solara_27b_readtwice/scores.json)
- Public Decision Index 0.3: **72.31**, complete=true, 140178/140178 scoreable
  requests successful.

The uploaded repository includes `release.json` with exact model/results
revisions. Use the exact inference commit linked in the submission entry.

## 1. Install and download

Use Linux x86_64, Python 3.12 and an NVIDIA driver compatible with the CUDA 13.0
Torch wheel. Single-GPU target: one NVIDIA H100 80GB. From this repository:

```bash
bash scripts/install_env.sh .venv
source .venv/bin/activate
python -m pip check
MODEL_REVISION=$(python -c 'import json; print(json.load(open("release.json"))["model_revision"])')
hf download WangSenQWQ/Solara-27B --revision "$MODEL_REVISION" --local-dir ./model
```

The server loads the checkpoint in-process. Core pins are in `requirements.txt`:
Torch 2.14.1 and Transformers 5.19.0, matching the public evaluation environment.
These are core version pins, not a complete transitive lockfile. Model and
results repositories require no login when public.

## 2. Start inference

From the repository root:

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 bash scripts/serve_model.sh ./model
```

Wait for readiness:

```bash
curl --noproxy '*' -f http://127.0.0.1:8008/health
python scripts/smoke_test.py --url http://127.0.0.1:8008
```

The health response includes `"read_twice": true` and `"enable_thinking": false`.
Read Twice is fixed on by the server. The smoke test uses synthetic data.

The service binds to localhost and has no authentication. Run the evaluator on
the same host, or use an SSH tunnel. Ensure localhost bypasses HTTP proxies:

```bash
export no_proxy=localhost,127.0.0.1,::1 NO_PROXY=localhost,127.0.0.1,::1
```

## 3. Run private evaluation

Keep the maintainers' private evaluator and scoring unchanged. Configure its
System One HTTP adapter with:

```text
--engine http --option base_url=http://127.0.0.1:8008 --model Solara-27B
```

Pass each private case's original `state` and `questions` to
`POST http://127.0.0.1:8008/v1/systemone`. Do not send gold answers,
`expected`, metadata or `_evaluation`. No public suite directory is required.

```python
import httpx

payload = {
    "model": "Solara-27B",
    "state": "The kettle is red.",
    "questions": {
        "color": {
            "type": "choice",
            "instructions": "What color is the kettle?",
            "criteria": {"red": "Red", "blue": "Blue"},
        },
        "is_red": {"type": "noul", "instructions": "The kettle is red."},
    },
}
with httpx.Client(timeout=600, trust_env=False) as client:
    response = client.post("http://127.0.0.1:8008/v1/systemone", json=payload)
    response.raise_for_status()
    answers = response.json()["answers"]
    print(answers["color"]["choice"], answers["color"]["probabilities"])
    print(answers["is_red"]["noul"])  # P(true)
```

Each answer is indexed by the original question key. A choice returns the
original option key and a probability for every supplied option. A noul returns
P(true). Use those values with the official private scorers.

## 4. Exact inference settings and limits

- Read Twice: repeat the complete rendered user input twice, separated by
  `\n\n`, within one user message. Do not repeat the system message.
  One forward pass per question, not two model calls.
- Thinking is fixed off. Scores are the softmax over the unified-letter rows of
  the model's own LM head (`unified-lm-option-softmax-v1`). cuDNN SDPA is
  disabled; PyTorch SDPA is used.
- Single GPU, one BF16 model, serialized frontend decisions. No quantization.
- Context limit: 262144 tokens, measured after Read Twice and chat-template
  rendering. No truncation or option filtering.
- Choice and noul types only; text/JSON state, no image input. Up to 256
  unified-letter candidates. HTTP body limit: 16 MiB. Each choice requires at
  least two candidates.
- HTTP 413: declared capacity failure; 422: invalid/unsupported input;
  502: decision failure. GET /health returns 200 after the model has loaded.
- Prompt version:
  `di-unified-letter-options-v4-ragtruth-train-aligned-read-twice-user-v1`.
  Scoring version: `unified-lm-option-softmax-v1`.

For official latency measurement, use one process and one request at a time on
one H100 80GB. Public-run timings come from 16 nodes × 8 GPUs and are separate.

## 5. Public evaluation record

Public suite 0.3: index 72.31; complete=true; all 140178 scoreable requests
successful. This is not the private Full score. Original model/run identifiers
are preserved in results.

Public reproduction uses this same server:

```bash
python -m decision_index pipeline --engine http \
  --option base_url=http://127.0.0.1:8008 --model Solara-27B \
  --suite-dir suite-0.3 --edition 0.3 --compact \
  --out runs/solara_27b_reproduction
```

Tests without loading weights:

```bash
python -m unittest test_systemone_server -v
```
