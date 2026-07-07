"""Tests for training evaluation features."""

from unittest.mock import MagicMock

import pytest
import torch

from enigma_engine.core.chat_format import IM_END, TOOL_CALL, TOOL_CALL_END, attach_chat_tokens


@pytest.fixture(scope="module")
def chat_tok():
    from enigma_engine.core.tokenizer import get_tokenizer

    tok = get_tokenizer("bpe")
    attach_chat_tokens(tok)
    return tok


class _StubGenModel:
    """Model stand-in: generate() appends canned ids to the prompt and
    records the prompt it was given."""

    def __init__(self, gen_ids: list[int]):
        self._gen = gen_ids
        self.training = False
        self.seen_prompt_ids: list[int] | None = None

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def generate(self, x, **kwargs):
        self.seen_prompt_ids = x[0].tolist()
        return torch.cat([x, torch.tensor([self._gen], dtype=torch.long, device=x.device)], dim=1)


def _tool_call_ids(tok, payload: str) -> list[int]:
    return [TOOL_CALL] + tok.encode(payload, add_special_tokens=False) + [TOOL_CALL_END, IM_END]


class TestTrainingEvaluation:
    """Test before/after training evaluation."""

    def test_evaluate_model_returns_dict(self):
        """evaluate_model returns dict with expected keys."""
        from enigma_engine.training.training_evaluation import evaluate_model

        result = evaluate_model(None, None, [], device="cpu")
        assert isinstance(result, dict)
        assert "perplexity" in result
        assert "loss" in result
        assert "num_prompts" in result

    def test_default_test_prompts_exist(self):
        """DEFAULT_TEST_PROMPTS is available and populated."""
        from enigma_engine.training.training_evaluation import DEFAULT_TEST_PROMPTS

        assert isinstance(DEFAULT_TEST_PROMPTS, list)
        assert len(DEFAULT_TEST_PROMPTS) > 0
        assert all(isinstance(p, str) for p in DEFAULT_TEST_PROMPTS)

    def test_training_config_evaluation_defaults(self):
        """TrainingConfig has evaluation fields with correct defaults."""
        from enigma_engine.training.training import TrainingConfig

        config = TrainingConfig()
        assert isinstance(config.run_evaluation, bool)
        assert hasattr(config, "eval_test_prompts")

    def test_evaluate_tool_usage_returns_dict(self):
        """evaluate_tool_usage returns dict with success metrics."""
        from enigma_engine.training.training_evaluation import evaluate_tool_usage

        result = evaluate_tool_usage(None, None, [], device="cpu")
        assert isinstance(result, dict)
        assert "success_rate" in result
        assert "total_tests" in result
        assert "successes" in result
        assert "failures" in result


class TestToolUsageLiveFormat:
    """evaluate_tool_usage renders the live chat pathway and grades the
    <|tool_call|>{json}<|/tool_call|> ids."""

    def test_matching_tool_name_passes(self, chat_tok):
        """A live tool call whose name matches expected_tool counts as success."""
        from enigma_engine.training.training_evaluation import evaluate_tool_usage

        model = _StubGenModel(
            _tool_call_ids(chat_tok, '{"name": "calculate", "arguments": {"expression": "7 * 8"}}')
        )
        cases = [{"prompt": "What is 7 times 8?", "expected_tool": "calculate"}]
        result = evaluate_tool_usage(model, chat_tok, cases, device="cpu")
        assert result["successes"] == 1
        assert result["success_rate"] == 1.0

    def test_prompt_is_rendered_with_tool_offer(self, chat_tok):
        """The graded pathway is the served one: chat-rendered prompt with the
        tools system block, not the bare test-case text."""
        from enigma_engine.core.chat_format import IM_START
        from enigma_engine.training.training_evaluation import evaluate_tool_usage

        model = _StubGenModel(_tool_call_ids(chat_tok, '{"name": "calculate", "arguments": {}}'))
        cases = [{"prompt": "What is 7 times 8?", "expected_tool": "calculate"}]
        evaluate_tool_usage(model, chat_tok, cases, device="cpu")
        assert model.seen_prompt_ids is not None
        assert IM_START in model.seen_prompt_ids
        text = chat_tok.decode(model.seen_prompt_ids, skip_special_tokens=True)
        assert "Available tools:" in text
        assert "What is 7 times 8?" in text

    def test_name_match_is_case_insensitive(self, chat_tok):
        """Tool-name matching ignores case."""
        from enigma_engine.training.training_evaluation import evaluate_tool_usage

        model = _StubGenModel(
            _tool_call_ids(chat_tok, '{"name": "Get_Weather", "arguments": {"city": "Tokyo"}}')
        )
        cases = [{"prompt": "Weather in Tokyo?", "expected_tool": "get_weather"}]
        result = evaluate_tool_usage(model, chat_tok, cases, device="cpu")
        assert result["successes"] == 1

    def test_wrong_tool_name_fails(self, chat_tok):
        """A live tool call with a different name is not a success."""
        from enigma_engine.training.training_evaluation import evaluate_tool_usage

        model = _StubGenModel(
            _tool_call_ids(chat_tok, '{"name": "get_weather", "arguments": {"city": "Tokyo"}}')
        )
        cases = [{"prompt": "What is 7 times 8?", "expected_tool": "calculate"}]
        result = evaluate_tool_usage(model, chat_tok, cases, device="cpu")
        assert result["successes"] == 0

    def test_plain_text_reply_not_counted(self, chat_tok):
        """Text mentioning a tool without the tool-call ids is not a call."""
        from enigma_engine.training.training_evaluation import evaluate_tool_usage

        model = _StubGenModel(chat_tok.encode("[CMD]calculate[/CMD]", add_special_tokens=False) + [IM_END])
        cases = [{"prompt": "What is 7 times 8?", "expected_tool": "calculate"}]
        result = evaluate_tool_usage(model, chat_tok, cases, device="cpu")
        assert result["successes"] == 0

    def test_default_tool_cases_use_live_tool_names(self):
        """DEFAULT_TOOL_TEST_CASES reference tools that exist in the live pipeline."""
        from enigma_engine.training.training_evaluation import DEFAULT_TOOL_TEST_CASES

        assert len(DEFAULT_TOOL_TEST_CASES) > 0
        names = {c["expected_tool"] for c in DEFAULT_TOOL_TEST_CASES}
        assert names <= {"calculate", "remember", "get_weather"}
        assert all("expected_command" not in c for c in DEFAULT_TOOL_TEST_CASES)


class TestEvalModelModeRestore:
    """Evaluation functions must restore model.training mode."""

    def _make_model_and_tokenizer(self):
        """Create a tiny model + tokenizer for testing."""
        import torch.nn as nn

        model = nn.Linear(4, 4)

        tok = MagicMock()
        tok.encode.return_value = [1, 2, 3, 4, 5]
        tok.eos_token_id = 0
        return model, tok

    def test_evaluate_model_restores_train_mode(self):
        """evaluate_model restores model.train() after running."""
        from enigma_engine.training.training_evaluation import evaluate_model

        model, tok = self._make_model_and_tokenizer()
        model.train()
        assert model.training

        # Non-empty prompts so model.eval() is actually called
        evaluate_model(model, tok, ["hello world"], device="cpu")
        assert model.training, "model should be back in training mode"

    def test_evaluate_model_preserves_eval_mode(self):
        """evaluate_model leaves model in eval mode if it started that way."""
        from enigma_engine.training.training_evaluation import evaluate_model

        model, tok = self._make_model_and_tokenizer()
        model.eval()
        assert not model.training

        evaluate_model(model, tok, ["hello world"], device="cpu")
        assert not model.training, "model should remain in eval mode"

    def test_evaluate_tool_usage_restores_train_mode(self, chat_tok):
        """evaluate_tool_usage restores model.train() after running."""
        from enigma_engine.training.training_evaluation import evaluate_tool_usage

        model = _StubGenModel(_tool_call_ids(chat_tok, '{"name": "calculate", "arguments": {}}'))
        model.train()
        assert model.training

        cases = [{"prompt": "test", "expected_tool": "calculate"}]
        evaluate_tool_usage(model, chat_tok, cases, device="cpu")
        assert model.training, "model should be back in training mode"

    def test_run_golden_eval_restores_train_mode(self, tmp_path):
        """run_golden_eval restores model.train() after running."""
        import json
        from enigma_engine.training.training_evaluation import run_golden_eval

        model, tok = self._make_model_and_tokenizer()
        model.train()
        assert model.training

        # Provide a real golden file with cases so model.eval() is reached
        golden = tmp_path / "golden.json"
        golden.write_text(json.dumps([{"prompt": "test", "expected": ["x"]}]), encoding="utf-8")
        run_golden_eval(model, tok, golden, device="cpu")
        assert model.training, "model should be back in training mode"


class TestEvalBugFixes:
    """Tests for S730 and S731 bug fixes."""

    def test_num_prompts_excludes_skipped(self):
        """S730: num_prompts should only count actually evaluated prompts."""
        import torch
        from enigma_engine.training.training_evaluation import evaluate_model

        # Model that returns valid logits for any input
        model = MagicMock()
        model.training = False
        model.eval = MagicMock()
        model.train = MagicMock()
        # Return logits shaped (1, seq_len-1, vocab_size=10)
        model.return_value = torch.randn(1, 4, 10)

        tok = MagicMock()
        # First prompt: 1 token → skipped (need ≥ 2)
        # Second prompt: 5 tokens → evaluated
        tok.encode.side_effect = [[1], [1, 2, 3, 4, 5]]

        result = evaluate_model(model, tok, ["a", "hello world"], device="cpu")
        assert result["num_prompts"] == 1  # Only the 5-token prompt

    def test_golden_eval_skips_empty_expected(self, tmp_path):
        """S731: Cases with empty 'expected' should be skipped entirely."""
        import json
        from enigma_engine.training.training_evaluation import run_golden_eval
        import torch.nn as nn

        model = nn.Linear(4, 4)
        tok = MagicMock()
        tok.encode.return_value = [1, 2, 3]
        tok.decode.return_value = "anything"
        tok.eos_token_id = 0  # stops immediately

        golden = tmp_path / "golden.json"
        golden.write_text(json.dumps([{"prompt": "test", "expected": []}]), encoding="utf-8")

        result = run_golden_eval(model, tok, golden, device="cpu")
        # Empty expected = no constraint = meaningless test → skip
        assert result["total"] == 0


class TestGSM8KBenchmark:
    """Eval-1: GSM8K reasoning benchmark."""

    # ---- parse_final_number ----

    def test_parse_final_number_gold_format(self):
        """`#### N` is the canonical GSM8K answer marker."""
        from enigma_engine.training.training_evaluation import parse_final_number

        assert parse_final_number("Janet has 3 ducks ... #### 18") == 18.0

    def test_parse_final_number_with_commas(self):
        """Numbers with thousand separators must parse."""
        from enigma_engine.training.training_evaluation import parse_final_number

        assert parse_final_number("total #### 1,234") == 1234.0

    def test_parse_final_number_negative(self):
        """Negative numbers must parse."""
        from enigma_engine.training.training_evaluation import parse_final_number

        assert parse_final_number("loss #### -42") == -42.0

    def test_parse_final_number_decimal(self):
        """Decimal numbers must parse."""
        from enigma_engine.training.training_evaluation import parse_final_number

        assert parse_final_number("price #### 3.5") == 3.5

    def test_parse_final_number_fallback_last_number(self):
        """No `####` → fall back to last number in text."""
        from enigma_engine.training.training_evaluation import parse_final_number

        assert parse_final_number("the answer is 42") == 42.0

    def test_parse_final_number_returns_none_no_number(self):
        """No number anywhere → None."""
        from enigma_engine.training.training_evaluation import parse_final_number

        assert parse_final_number("I don't know") is None

    def test_parse_final_number_empty_string(self):
        """Empty input → None."""
        from enigma_engine.training.training_evaluation import parse_final_number

        assert parse_final_number("") is None

    # ---- load_gsm8k ----

    def test_load_gsm8k_missing_file_raises_clear_error(self, tmp_path):
        """Missing JSONL → FileNotFoundError mentioning recovery command."""
        from enigma_engine.training.training_evaluation import load_gsm8k

        missing = tmp_path / "no_such.jsonl"
        try:
            load_gsm8k(missing)
        except FileNotFoundError as exc:
            msg = str(exc)
            assert "gsm8k" in msg.lower() or "openai/gsm8k" in msg
            assert "load_dataset" in msg or "datasets" in msg
        else:
            raise AssertionError("expected FileNotFoundError")

    def test_load_gsm8k_reads_jsonl(self, tmp_path):
        """Read items from a JSONL file."""
        import json
        from enigma_engine.training.training_evaluation import load_gsm8k

        path = tmp_path / "g.jsonl"
        items = [
            {"question": "q1", "answer": "step\n#### 1"},
            {"question": "q2", "answer": "step\n#### 2"},
            {"question": "q3", "answer": "step\n#### 3"},
        ]
        path.write_text("\n".join(json.dumps(it) for it in items) + "\n", encoding="utf-8")

        loaded = load_gsm8k(path)
        assert len(loaded) == 3
        assert loaded[0]["question"] == "q1"
        assert loaded[2]["answer"].endswith("#### 3")

    def test_load_gsm8k_caps_n(self, tmp_path):
        """Positive `n` caps the number of returned items."""
        import json
        from enigma_engine.training.training_evaluation import load_gsm8k

        path = tmp_path / "g.jsonl"
        items = [{"question": f"q{i}", "answer": f"#### {i}"} for i in range(5)]
        path.write_text("\n".join(json.dumps(it) for it in items) + "\n", encoding="utf-8")

        loaded = load_gsm8k(path, n=2)
        assert len(loaded) == 2

    # ---- run_gsm8k_benchmark ----

    def test_run_gsm8k_benchmark_with_mock_engine(self):
        """Engine that always answers 18 → correct on items whose gold is 18."""
        from enigma_engine.training.training_evaluation import run_gsm8k_benchmark

        engine = MagicMock()
        engine.generate.return_value = "thinking ... #### 18"

        examples = [
            {"question": "q1", "answer": "step\n#### 18"},  # correct
            {"question": "q2", "answer": "step\n#### 18"},  # correct
            {"question": "q3", "answer": "step\n#### 99"},  # wrong
        ]

        result = run_gsm8k_benchmark(engine, examples, num_shots=0, max_gen=32)
        assert result["total"] == 3
        assert result["correct"] == 2
        assert abs(result["accuracy"] - 2 / 3) < 1e-6

    def test_run_gsm8k_benchmark_handles_unparseable_output(self):
        """Engine output with no number → counted incorrect, no crash."""
        from enigma_engine.training.training_evaluation import run_gsm8k_benchmark

        engine = MagicMock()
        engine.generate.return_value = "i dunno"

        examples = [{"question": "q1", "answer": "#### 7"}]
        result = run_gsm8k_benchmark(engine, examples, num_shots=0, max_gen=32)
        assert result["total"] == 1
        assert result["correct"] == 0
        assert result["accuracy"] == 0.0

    def test_run_gsm8k_benchmark_floating_point_tolerance(self):
        """3.0 vs 3.00 must compare equal (numeric, not string)."""
        from enigma_engine.training.training_evaluation import run_gsm8k_benchmark

        engine = MagicMock()
        engine.generate.return_value = "#### 3.00"

        examples = [{"question": "q1", "answer": "#### 3"}]
        result = run_gsm8k_benchmark(engine, examples, num_shots=0, max_gen=32)
        assert result["correct"] == 1

    def test_run_gsm8k_benchmark_empty_returns_zero_total(self):
        """No examples → total=0, accuracy=0.0, no crash."""
        from enigma_engine.training.training_evaluation import run_gsm8k_benchmark

        engine = MagicMock()
        result = run_gsm8k_benchmark(engine, [], num_shots=0, max_gen=32)
        assert result["total"] == 0
        assert result["correct"] == 0
        assert result["accuracy"] == 0.0
