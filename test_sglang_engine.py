"""Protocol/score regression checks without deploying a server or loading weights."""

import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from decision_index.engines.base import Unsupported, validate
from decision_index.engines.sglang_engine import IncompleteLogprobs, SGLangDecisionEngine
from decision_index.runner import run, validate_resume_provenance


class TinyTokenizer:
    chat_template = "test-chat-template"

    def __init__(self):
        self.ids = {key: i + 20 for i, key in enumerate(("false", "true", "red", "blue"))}
        self.render_calls = []

    def encode(self, value, add_special_tokens=False):
        if len(value) == 1 and "\u4e00" <= value <= "\u9fff":
            self.ids[value] = 50000 + ord(value)
        if len(value) == 1 and value in "abcdefghijklmnopqrstuvwxyz0123456789":
            self.ids[value] = 4000 + ord(value)
        if value.isascii() and value.isupper() and value.isalpha() and len(value) <= 2:
            if value not in self.ids:
                self.ids[value] = 1000 + (ord(value[0]) - 64) * 27 + (ord(value[1]) - 64 if len(value) == 2 else 0)
        return [self.ids[value]] if value in self.ids else [1, 2, 3]

    def decode(self, ids, **kwargs):
        return next((key for key, token in self.ids.items() if ids == [token]), "<multi>")

    def apply_chat_template(self, messages, **kwargs):
        self.render_calls.append((messages, kwargs))
        return [7, 8, 9]


def chat_response(scores):
    items = [{"token": key, "bytes": list(key.encode()), "logprob": value} for key, value in scores.items()]
    content = [{**items[0], "top_logprobs": items}] if items else []
    return {"choices": [{"logprobs": {"content": content}}]}


class SGLangEngineTests(unittest.TestCase):
    def engine(self, handler, **options):
        options.setdefault("label_mode", "auto")  # Historical tests stay explicit.
        tokenizer = TinyTokenizer()
        module = types.ModuleType("transformers")
        module.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer)
        with patch.dict(sys.modules, {"transformers": module}):
            engine = SGLangDecisionEngine(base_url="http://unit:8000/v1", model="served-qwen", tokenizer="tokenizer", max_retries=0, **options)
        engine.client.close()
        engine.client = httpx.Client(base_url="http://unit:8000/", transport=httpx.MockTransport(handler))
        self.addCleanup(engine.close)
        return engine

    def test_read_twice_repeats_only_user_and_blocks_mixed_resume(self):
        from decision_index.suite.build.adapters_added import RAGTRUTH_INSTRUCTIONS, RAGTRUTH_CRITERIA
        base = self.engine(lambda r: self.fail("no network"), label_mode="unified_letters")
        twice = self.engine(lambda r: self.fail("no network"), label_mode="unified_letters", read_twice=True)
        cases = [
            ({"context": "evidence"}, {"type": "choice", "instructions": "pick", "criteria": {"x": "first", "y": "second"}}),
            (None, {"type": "noul", "instructions": "Is it correct?"}),
            ({"prompt": "evidence", "response": "answer"}, {"type": "noul", "instructions": RAGTRUTH_INSTRUCTIONS, "criteria": RAGTRUTH_CRITERIA}),
        ]
        for state, question in cases:
            before = json.dumps([state, question])
            single = base.prepare_question(state, question)
            doubled = twice.prepare_question(state, question)
            self.assertEqual(doubled["messages"][0], single["messages"][0])
            self.assertEqual(len(doubled["messages"]), 2)
            body = single["messages"][1]["content"]
            self.assertEqual(doubled["messages"][1]["content"], body + "\n\n" + body)
            for key in ("keys", "labels", "token_ids", "remapped"):
                self.assertEqual(single[key], doubled[key])
            self.assertEqual(json.dumps([state, question]), before)
        self.assertFalse(base.provenance["read_twice"])
        self.assertTrue(twice.provenance["read_twice"])
        self.assertEqual(twice.prompt_version, base.prompt_version + "-read-twice-user-v1")
        with self.assertRaisesRegex(ValueError, "prompt_version"):
            validate_resume_provenance({"engine": "sglang", "model_source": base.provenance}, "sglang", twice.provenance)

    def test_read_twice_native_request_and_boolean_validation(self):
        sent = []
        def handler(request):
            sent.append(json.loads(request.content))
            tokens = engine.prepare_question(None, question)["token_ids"]
            return httpx.Response(200, json={"meta_info": {"output_token_ids_logprobs": [[[value, token] for value, token in zip([-3., -1.], tokens)]]}})
        engine = self.engine(handler, label_mode="unified_letters", read_twice=True, logprob_backend="native")
        question = {"type": "choice", "instructions": "pick", "criteria": {"x": "first", "y": "second"}}
        expected = engine.prepare_question(None, question)
        response, traces = engine(None, {"q": question})
        validate({"q": question}, response)
        self.assertEqual(engine.tok.render_calls[-1][0], expected["messages"])
        self.assertFalse(engine.tok.render_calls[-1][1]["enable_thinking"])
        self.assertEqual(sent[0]["sampling_params"]["max_new_tokens"], 1)
        self.assertEqual(response["answers"]["q"]["choice"], "y")
        self.assertTrue(traces["q"]["read_twice"])
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.engine(lambda r: self.fail("no network"), read_twice="false")

    def test_unified_letters_remaps_all_keys_and_noul_but_restores_api_semantics(self):
        engine = self.engine(lambda r: httpx.Response(200, json=chat_response({"A": -3., "B": -1.})), label_mode="unified_letters")
        questions = {"pick": {"type": "choice", "criteria": {"red": "red", "blue": "blue"}}, "flag": {"type": "noul"}}
        answer, raw = engine({}, questions)
        validate(questions, answer)
        self.assertEqual(answer["answers"]["pick"]["choice"], "blue")
        self.assertEqual(raw["pick"]["key_to_label"], {"red": "A", "blue": "B"})
        self.assertEqual(raw["flag"]["key_to_label"], {"false": "A", "true": "B"})
        self.assertAlmostEqual(answer["answers"]["flag"]["noul"], 1 / (1 + math.exp(-2)))
        self.assertEqual(raw["flag"]["prompt_version"], "di-unified-letter-options-v4-ragtruth-train-aligned")
        self.assertEqual(engine.provenance["label_policy"], "single-token-uppercase-unified-v1")

    def test_shared_preparation_is_network_free_and_matches_sent_messages(self):
        sent = []
        def handler(request):
            sent.append(json.loads(request.content))
            return httpx.Response(200, json=chat_response({"A": -3., "B": -1.}))
        engine = self.engine(handler, label_mode="unified_letters")
        question = {"type": "choice", "instructions": "select", "criteria": {"true": "Yes", "false": "No"}}
        prepared = engine.prepare_question({}, question)
        self.assertFalse(sent)
        engine({}, {"q": question})
        self.assertEqual(sent[0]["messages"], prepared["messages"])
        self.assertEqual(prepared["keys"], ["true", "false"])

    def test_ragtruth_matches_training_layout_and_keeps_positive_probability(self):
        from decision_index.engines.transformers_engine import SYSTEM_PROMPT
        from decision_index.suite.build.adapters_added import RAGTRUTH_INSTRUCTIONS, RAGTRUTH_CRITERIA

        sent = []
        def handler(request):
            sent.append(json.loads(request.content))
            return httpx.Response(200, json=chat_response({"A": -3., "B": -1.}))
        engine = self.engine(handler, label_mode="unified_letters")
        state = {"prompt": "Evidence:\nA kettle is red.", "response": "The kettle is blue."}
        question = {"type": "noul", "instructions": RAGTRUTH_INSTRUCTIONS, "criteria": dict(RAGTRUTH_CRITERIA)}
        before = json.dumps([state, question])
        response, traces = engine(state, {"q": question})
        expected = ("State:\n(empty)\n\nQuestion:\nContext prompt:\nEvidence:\nA kettle is red."
                    "\n\nModel response:\nThe kettle is blue.\n\n"
                    "The response contains content that is not supported by the context in the prompt."
                    "\n\nOptions:\nA: All content of the response is supported by the context in the prompt."
                    "\nB: The response contains content that is not supported by the context in the prompt."
                    "\n\nReply with exactly one option key.")
        self.assertEqual(sent[0]["messages"], [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": expected}])
        self.assertEqual(json.dumps([state, question]), before)
        validate({"q": question}, response)
        self.assertAlmostEqual(response["answers"]["q"]["noul"], 1 / (1 + math.exp(-2)))
        self.assertEqual(traces["q"]["key_to_label"], {"false": "A", "true": "B"})

    def test_ragtruth_alignment_is_scoped_and_requires_complete_input(self):
        from decision_index.engines.transformers_engine import render_prompt
        from decision_index.suite.build.adapters_added import RAGTRUTH_INSTRUCTIONS, RAGTRUTH_CRITERIA

        state = {"prompt": "evidence", "response": "answer"}
        question = {"type": "noul", "instructions": RAGTRUTH_INSTRUCTIONS, "criteria": RAGTRUTH_CRITERIA}
        historical = self.engine(lambda r: self.fail("no network"), label_mode="auto")
        self.assertEqual(historical.prepare_question(state, question)["messages"][1]["content"], render_prompt(state, question))
        engine = self.engine(lambda r: self.fail("no network"), label_mode="unified_letters")
        generic = {**question, "instructions": "Is this response correct?"}
        expected = render_prompt(state, {"type": "choice", "instructions": generic["instructions"], "criteria": {"A": "False", "B": "True"}})
        self.assertEqual(engine.prepare_question(state, generic)["messages"][1]["content"], expected)
        with self.assertRaises(Unsupported):
            engine.prepare_question(state, {**question, "criteria": {"true": "unsupported"}})

    def test_resume_rejects_previous_unified_letter_prompt_protocol(self):
        engine = self.engine(lambda r: self.fail("no network"), label_mode="unified_letters")
        previous = {"engine": "sglang", "model_source": {**engine.provenance, "prompt_version": "di-unified-letter-options-v3"}}
        with self.assertRaisesRegex(ValueError, "prompt_version"):
            validate_resume_provenance(previous, "sglang", engine.provenance)

    def test_chat_matches_llm_client_and_normalizes_complete_scores(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json=chat_response({"A": -2., "B": -3.}))

        engine = self.engine(handler)
        q = {"q": {"type": "choice", "instructions": "pick", "criteria": {"A": "first", "B": "second"}}}
        answer, raw = engine({}, q)
        validate(q, answer)
        self.assertEqual(answer["answers"]["q"]["choice"], "A")
        self.assertAlmostEqual(answer["answers"]["q"]["probabilities"]["A"], 1 / (1 + math.exp(-1)))
        body = json.loads(requests[0].content)
        self.assertEqual(requests[0].url.path, "/v1/chat/completions")
        self.assertEqual((body["max_tokens"], body["temperature"], body["logprobs"]), (1, 0, True))
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertFalse(raw["q"]["remapped"])

    def test_151_options_are_all_scored_and_mapped_back_after_native_fallback(self):
        requests = []

        def handler(request):
            requests.append(request)
            body = json.loads(request.content)
            if request.url.path == "/v1/chat/completions":
                return httpx.Response(200, json=chat_response({"A": -4.}))
            ids = body["token_ids_logprob"]
            scores = [[-0.1 if i == 77 else -10., token, None] for i, token in enumerate(ids)]
            return httpx.Response(200, json={"meta_info": {"output_token_ids_logprobs": [scores]}})

        engine = self.engine(handler)
        criteria = {f"option_{i}": f"class {i}" for i in range(151)}
        q = {"intent": {"type": "choice", "instructions": "classify", "criteria": criteria}}
        answer, raw = engine("utterance", q)
        validate(q, answer)
        self.assertEqual(answer["answers"]["intent"]["choice"], "option_77")
        self.assertEqual(len(answer["answers"]["intent"]["probabilities"]), 151)
        self.assertEqual(len(set(raw["intent"]["key_to_token_id"].values())), 151)
        self.assertEqual(raw["intent"]["backend"], "native")
        self.assertIsNotNone(raw["intent"]["fallback_reason"])
        body = json.loads(requests[1].content)
        self.assertEqual(body["input_ids"], [7, 8, 9])
        self.assertEqual(body["logprob_start_len"], -1)
        self.assertEqual(body["sampling_params"]["max_new_tokens"], 1)
        self.assertEqual(len(body["token_ids_logprob"]), 151)
        rendered, kwargs = engine.tok.render_calls[0]
        self.assertIn("class 132", rendered[1]["content"])
        self.assertNotIn("option_132", rendered[1]["content"])
        self.assertFalse(kwargs["enable_thinking"])
        self.assertFalse(kwargs["return_dict"])

    def test_noul_keeps_true_false_even_when_letters_are_forced(self):
        calls = []

        def handler(request):
            calls.append(json.loads(request.content))
            return httpx.Response(200, json=chat_response({"true": -1., "false": -3.}))

        engine = self.engine(handler, label_mode="letters")
        q = {"relevant": {"type": "noul", "instructions": "relevant?"}}
        answer, raw = engine(None, q)
        validate(q, answer)
        self.assertAlmostEqual(answer["answers"]["relevant"]["noul"], 1 / (1 + math.exp(-2)))
        self.assertEqual(raw["relevant"]["key_to_label"], {"false": "false", "true": "true"})

    def test_characters_maps_151_options_to_prefix_free_single_character_tokens(self):
        def handler(request):
            ids = json.loads(request.content)["token_ids_logprob"]
            scores = [[-0.1 if i == 132 else -10., token, None] for i, token in enumerate(ids)]
            return httpx.Response(200, json={"meta_info": {"output_token_ids_logprobs": [scores]}})

        engine = self.engine(handler, label_mode="characters", logprob_backend="native")
        q = {"intent": {"type": "choice", "criteria": {f"option_{i}": f"class {i}" for i in range(151)}}}
        answer, raw = engine({}, q)
        validate(q, answer)
        labels = list(raw["intent"]["key_to_label"].values())
        self.assertEqual(answer["answers"]["intent"]["choice"], "option_132")
        self.assertEqual(labels[:26], list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        self.assertTrue(all(len(label) == 1 and len(engine.tok.encode(label)) == 1 for label in labels))
        self.assertEqual(len(set(labels)), 151)
        self.assertEqual(len(set(raw["intent"]["key_to_token_id"].values())), 151)
        self.assertEqual(raw["intent"]["label_policy"], "single-token-character-v1")
        self.assertEqual(engine.provenance["label_mode"], "characters")

    def test_characters_preserves_atomic_keys_and_noul(self):
        engine = self.engine(lambda r: httpx.Response(200, json=chat_response({"A": -2., "B": -1., "true": -1., "false": -2.})), label_mode="characters")
        q = {"pick": {"type": "choice", "criteria": {"A": "first", "B": "second"}}, "flag": {"type": "noul"}}
        answer, raw = engine({}, q)
        validate(q, answer)
        self.assertFalse(raw["pick"]["remapped"])
        self.assertFalse(raw["flag"]["remapped"])

    def test_character_alphabets_are_explicit_and_provenance_distinguishes_them(self):
        sources = []
        for alphabet in ("mixed", "cjk", "alnum", "latin"):
            engine = self.engine(lambda r: self.fail("no network"), label_mode="characters", label_alphabet=alphabet)
            labels, ids, mapped = engine._labels([f"option_{i}" for i in range(151)])
            self.assertTrue(mapped)
            self.assertTrue(all(len(label) == 1 for label in labels))
            self.assertEqual(len(set(ids)), 151)
            if alphabet == "cjk":
                self.assertTrue(all(not label.isascii() for label in labels))
            if alphabet in ("alnum", "latin"):
                self.assertEqual(labels[:62], list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"))
            sources.append(engine.provenance)
        self.assertEqual(len({s["label_policy"] for s in sources}), 4)
        previous = {"engine": "sglang", "model_source": sources[0]}
        for source in sources[1:]:
            with self.assertRaises(ValueError):
                validate_resume_provenance(previous, "sglang", source)

    def test_alphabet_requires_characters_mode(self):
        with self.assertRaisesRegex(ValueError, "label_alphabet"):
            self.engine(lambda r: self.fail("no network"), label_alphabet="cjk")
        with self.assertRaisesRegex(ValueError, "label_alphabet"):
            self.engine(lambda r: self.fail("no network"), label_mode="characters", label_alphabet="invalid")

    def test_characters_rejects_insufficient_pool_without_falling_back_to_double_letters(self):
        engine = self.engine(lambda r: self.fail("must not call upstream"), label_mode="characters")
        with patch("decision_index.engines.sglang_engine.CHARACTER_LABELS", "AB"):
            with self.assertRaises(Unsupported):
                engine._labels(["option_0", "option_1", "option_2"])

    def test_incomplete_chat_is_error_in_strict_chat_mode(self):
        calls = []

        def handler(request):
            calls.append(request)
            return httpx.Response(200, json=chat_response({"A": -1.}))

        engine = self.engine(handler, logprob_backend="chat")
        with self.assertRaises(IncompleteLogprobs):
            engine({}, {"q": {"type": "choice", "criteria": {"A": "a", "B": "b"}}})
        self.assertEqual(len(calls), 1)

    def test_incomplete_native_is_error_not_a_zero_probability(self):
        def handler(request):
            token = json.loads(request.content)["token_ids_logprob"][0]
            return httpx.Response(200, json={"meta_info": {"output_token_ids_logprobs": [[[-1., token, None]]]}})

        engine = self.engine(handler, logprob_backend="native")
        with self.assertRaises(IncompleteLogprobs):
            engine({}, {"q": {"type": "choice", "criteria": {"A": "a", "B": "b"}}})

    def test_nonfinite_scores_are_rejected(self):
        engine = self.engine(lambda r: httpx.Response(200, json=chat_response({"A": -1., "B": -2.})))
        with self.assertRaises(IncompleteLogprobs):
            engine._require_scores({1: -1, 2: float("nan")}, [1, 2])

    def test_keys_mode_rejects_shared_multitoken_prefix_without_requesting(self):
        engine = self.engine(lambda r: self.fail("must not call upstream"), label_mode="keys")
        with self.assertRaises(Unsupported):
            engine({}, {"q": {"type": "choice", "criteria": {"option_1": "a", "option_132": "b"}}})

    def test_letters_mode_remaps_already_single_token_keys(self):
        engine = self.engine(lambda r: httpx.Response(200, json=chat_response({"A": -2., "B": -1.})), label_mode="letters")
        q = {"q": {"type": "choice", "criteria": {"red": "red", "blue": "blue"}}}
        answer, raw = engine({}, q)
        validate(q, answer)
        self.assertEqual(answer["answers"]["q"]["choice"], "blue")
        self.assertEqual(raw["q"]["key_to_label"], {"red": "A", "blue": "B"})

    def test_prompt_keeps_only_new_labels_and_descriptions_but_restores_original_keys(self):
        requests = []

        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json=chat_response({"A": -2., "B": -1.}))

        engine = self.engine(handler)
        q = {"q": {"type": "choice", "criteria": {"option_1": "account blocked", "option_132": "translate"}}}
        answer, trace = engine({}, q)
        body = requests[0]["messages"][1]["content"]
        self.assertIn("A: account blocked", body)
        self.assertIn("B: translate", body)
        self.assertNotIn("option_1", body)
        self.assertNotIn("option_132", body)
        self.assertEqual(answer["answers"]["q"]["choice"], "option_132")
        self.assertEqual(trace["q"]["key_to_label"], {"option_1": "A", "option_132": "B"})
        self.assertEqual(trace["q"]["prompt_version"], "di-clean-label-options-v2")

    def test_null_descriptions_keep_original_key_as_semantic_option_content(self):
        requests = []

        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json=chat_response({"A": -1., "B": -2.}))

        engine = self.engine(handler)
        engine({}, {"q": {"type": "choice", "criteria": {"translate_text": None, "bank_account": None}}})
        body = requests[0]["messages"][1]["content"]
        self.assertIn("A: translate_text", body)
        self.assertIn("B: bank_account", body)

    def test_multiple_questions_preserve_question_keys_and_original_choices(self):
        engine = self.engine(lambda r: httpx.Response(200, json=chat_response({"A": -2., "B": -1., "true": -1., "false": -2.})))
        q = {"intent": {"type": "choice", "criteria": {"option_1": "a", "option_132": "b"}}, "flag": {"type": "noul"}}
        response, trace = engine({}, q)
        validate(q, response)
        self.assertEqual(set(response["answers"]), {"intent", "flag"})
        self.assertEqual(response["answers"]["intent"]["choice"], "option_132")

    def test_capacity_rejection_is_unsupported(self):
        engine = self.engine(lambda r: httpx.Response(400, json={"error": "Maximum context length exceeded"}))
        with self.assertRaises(Unsupported):
            engine({}, {"q": {"type": "choice", "criteria": {"A": "a", "B": "b"}}})

    def test_resume_rejects_changed_endpoint_labels_template_and_backend(self):
        engine = self.engine(lambda r: self.fail("no network"))
        source = engine.provenance
        previous = {"engine": "sglang", "model_source": dict(source)}
        validate_resume_provenance(previous, "sglang", source)
        for field in ("base_url", "label_mode", "label_policy", "scoring_version", "prompt_version", "model", "chat_template_sha256", "logprob_backend"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_resume_provenance(previous, "sglang", {**source, field: "changed"})

    def test_runner_keeps_standard_result_and_records_protocol_errors(self):
        engine = self.engine(lambda r: httpx.Response(200, json=chat_response({"A": -1.})), logprob_backend="chat")
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            rows = directory / "rows.jsonl"
            row = {"state": {}, "questions": {"q": {"type": "choice", "criteria": {"A": "a", "B": "b"}}}, "_evaluation": {"run_id": "case", "dataset": "unit"}}
            rows.write_text(json.dumps(row) + "\n")
            with patch("decision_index.runner.load_engine", return_value=engine):
                run("sglang", {}, rows, directory / "out", warm=False, log=lambda r: None)
            result = json.loads((directory / "out/results.jsonl").read_text())
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["exception"], "IncompleteLogprobs")
            env = json.loads((directory / "out/environment.json").read_text())
            self.assertEqual(env["model_source"]["scoring_version"], "sglang-next-token-v1")


if __name__ == "__main__":
    unittest.main()
