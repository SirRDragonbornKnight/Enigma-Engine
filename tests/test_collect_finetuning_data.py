"""Tests for collect_finetuning_data.py reasoning-data collectors.

Pass 155: D-4 OpenThoughts3 + D-11 SmolTalk2 fetchers. We mock the
`datasets` library via fake module injection so tests do not require
HuggingFace network access. See AA learned principle:
"Optional dependency tests: inject fake module via types.ModuleType
+ monkeypatch.setitem(sys.modules, ...)".
"""

import importlib
import sys
import types
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Fake datasets injection ─────────────────────────────────────────────────


def _install_fake_datasets(monkeypatch, rows_by_path, splits_by_path=None):
    """Inject a fake `datasets` module whose `load_dataset(path, ...)`
    returns the rows registered for that dataset path.

    `rows_by_path` keys are HF repo paths; values are lists of dict rows.
    `splits_by_path` optionally maps repo path → list of split names for
    `get_dataset_split_names`. Defaults to ["train"] for known paths.
    Streaming and non-streaming both yield the same row list.
    """
    fake = types.ModuleType("datasets")

    class _FakeDataset:
        def __init__(self, rows):
            self._rows = rows

        def __iter__(self):
            return iter(self._rows)

        def __len__(self):
            return len(self._rows)

    def _load_dataset(path, *args, **kwargs):
        if path not in rows_by_path:
            raise ValueError("BuilderConfig 'unknown' not found. Available: ['default', 'subset_a', 'subset_b']")
        rows = rows_by_path[path]
        # Per-split rows: a dict value maps split name -> row list, and the
        # requested split MUST exist -- a collector that ignores its split
        # argument (or hardcodes "train") fails loudly instead of silently
        # receiving the same rows for every split (test-suite audit
        # 2026-07-17: identical-rows-per-split + dedup made the split tests
        # unfalsifiable).
        if isinstance(rows, dict):
            split = kwargs.get("split")
            if split not in rows:
                raise ValueError(f"Unknown split {split!r}. Available: {sorted(rows)}")
            rows = rows[split]
        return _FakeDataset(rows)

    def _get_dataset_split_names(path, config=None, *args, **kwargs):
        if splits_by_path is not None and path in splits_by_path:
            return list(splits_by_path[path])
        if path in rows_by_path:
            return ["train"]
        raise ValueError("BuilderConfig 'unknown' not found. Available: ['default', 'subset_a', 'subset_b']")

    fake.load_dataset = _load_dataset
    fake.get_dataset_split_names = _get_dataset_split_names
    monkeypatch.setitem(sys.modules, "datasets", fake)


@pytest.fixture
def cf_module(monkeypatch):
    """Reload collect_finetuning_data after fake `datasets` is installed."""
    if "collect_finetuning_data" in sys.modules:
        monkeypatch.delitem(sys.modules, "collect_finetuning_data")
    return importlib.import_module("collect_finetuning_data")


# ── OpenThoughts3 (D-4) ─────────────────────────────────────────────────────


class TestCollectOpenThoughts3:
    """Pass 139 spec verified the row schema:
        {"difficulty": int, "source": str, "domain": str,
         "conversations": [{"from": "human"/"gpt", "value": str}]}
    The gpt value contains <think>...</think> reasoning followed by the
    final answer. Tags MUST be preserved verbatim (no whitespace collapse).
    """

    _SAMPLE_ROW = {
        "difficulty": 5,
        "source": "competition_math",
        "domain": "math",
        "conversations": [
            {"from": "human", "value": "What is 2+2?"},
            {
                "from": "gpt",
                "value": ("<think>\nOkay, I need to add two and two.\n2 + 2 = 4.\n</think>\n\nThe answer is 4."),
            },
        ],
    }

    def test_preserves_think_tags_verbatim(self, monkeypatch, cf_module):
        """<think>...</think> must survive — no whitespace collapse."""
        _install_fake_datasets(
            monkeypatch,
            {"open-thoughts/OpenThoughts3-1.2M": [self._SAMPLE_ROW]},
        )
        # Re-import after injection so the fake module is picked up
        cf = importlib.reload(cf_module)
        pairs = cf.collect_openthoughts3(max_samples=10)
        assert len(pairs) == 1
        completion = pairs[0]["completion"]
        assert "<think>" in completion
        assert "</think>" in completion
        # Verbatim — newlines inside <think> survived
        assert "\nOkay, I need to add" in completion
        assert "</think>\n\nThe answer is 4." in completion

    def test_skips_rows_with_no_human_gpt_pair(self, monkeypatch, cf_module):
        """Rows missing either turn are dropped, not crashed on."""
        rows = [
            {"conversations": [{"from": "human", "value": "Q"}]},  # no gpt
            {"conversations": [{"from": "gpt", "value": "A"}]},  # no human
            self._SAMPLE_ROW,
        ]
        _install_fake_datasets(monkeypatch, {"open-thoughts/OpenThoughts3-1.2M": rows})
        cf = importlib.reload(cf_module)
        pairs = cf.collect_openthoughts3(max_samples=10)
        assert len(pairs) == 1

    def test_respects_max_samples(self, monkeypatch, cf_module):
        """`max_samples` caps streaming early.

        DISTINCT rows on purpose: with 50 identical rows, dedup collapsed
        the result to 1 whether or not the cap existed, so deleting the
        `seen >= max_samples` break stayed green (audit 2026-07-17).
        """
        rows = [
            {
                "conversations": [
                    {"from": "human", "value": f"What is {i} + {i}?"},
                    {"from": "gpt", "value": f"<think>\nadding.\n</think>\n\nThe answer is {2 * i}."},
                ]
            }
            for i in range(50)
        ]
        _install_fake_datasets(monkeypatch, {"open-thoughts/OpenThoughts3-1.2M": rows})
        cf = importlib.reload(cf_module)
        pairs = cf.collect_openthoughts3(max_samples=3)
        assert len(pairs) == 3  # exactly the cap: iteration stopped there


# ── OASST1 + Dolly: a gated repo must not kill the run (audit 2026-08-22) ────


class TestTheFirstTwoLoadersSurviveAGatedRepo:
    """collect_oasst and collect_dolly called load_dataset BARE while every
    later sibling wrapped it and returned []. OASST runs FIRST under --all, so
    one gated, moved or renamed repo raised out of main() and took the whole
    collection with it -- including the sources that were still reachable."""

    _OASST_ROWS = [
        {"message_id": "a", "parent_id": None, "role": "prompter", "lang": "en",
         "text": "What is a neutron star?"},
        {"message_id": "b", "parent_id": "a", "role": "assistant", "lang": "en",
         "rank": 0, "text": "The collapsed core left behind by a supernova."},
    ]
    _DOLLY_ROW = {"instruction": "Name a primary color.", "context": "",
                  "response": "Blue is a primary color."}

    def test_oasst_returns_empty_on_a_raising_loader(self, monkeypatch, cf_module, caplog):
        _install_fake_datasets(monkeypatch, {})
        cf = importlib.reload(cf_module)
        with caplog.at_level("ERROR"):
            assert cf.collect_oasst() == []
        assert "Failed to load OASST1" in caplog.text

    def test_dolly_returns_empty_on_a_raising_loader(self, monkeypatch, cf_module, caplog):
        _install_fake_datasets(monkeypatch, {})
        cf = importlib.reload(cf_module)
        with caplog.at_level("ERROR"):
            assert cf.collect_dolly() == []
        assert "Failed to load Dolly 15k" in caplog.text

    def test_a_dead_oasst_no_longer_takes_the_whole_all_run_down(self, monkeypatch, cf_module, tmp_path):
        """The receipt that matters: --all with OASST unreachable still walks
        on and writes the sources that ARE reachable."""
        _install_fake_datasets(monkeypatch, {"databricks/databricks-dolly-15k": [self._DOLLY_ROW]})
        cf = importlib.reload(cf_module)
        monkeypatch.setattr(sys, "argv", ["collect_finetuning_data.py", "--all",
                                          "--output-dir", str(tmp_path)])

        cf.main()

        assert not (tmp_path / "oasst1.jsonl").exists()
        assert (tmp_path / "dolly_15k.jsonl").exists(), "the run died on the first dead source"
        assert "Blue is a primary color." in (tmp_path / "dolly_15k.jsonl").read_text(encoding="utf-8")

    def test_the_healthy_paths_still_extract(self, monkeypatch, cf_module):
        """Falsifier for the two above: with the repos present, both loaders
        still return their pairs -- the wrap did not swallow a live fetch."""
        _install_fake_datasets(monkeypatch, {
            "OpenAssistant/oasst1": self._OASST_ROWS,
            "databricks/databricks-dolly-15k": [self._DOLLY_ROW],
        })
        cf = importlib.reload(cf_module)
        assert cf.collect_oasst() == [{"prompt": "What is a neutron star?",
                                       "completion": "The collapsed core left behind by a supernova."}]
        assert cf.collect_dolly() == [{"prompt": "Name a primary color.",
                                       "completion": "Blue is a primary color."}]


# ── SmolTalk2 (D-11) ────────────────────────────────────────────────────────


class TestCollectSmolTalk2:
    """SmolTalk2 (`HuggingFaceTB/smoltalk2`) ships many configs, not one
    'SFT'. Standard ChatML schema: `messages: [{role, content}]`.
    """

    _SAMPLE_ROW = {
        "messages": [
            {"role": "user", "content": "Explain gravity briefly."},
            {
                "role": "assistant",
                "content": (
                    "Gravity is the force pulling masses together. On Earth it accelerates objects at ~9.8 m/s^2."
                ),
            },
        ],
    }

    def test_extracts_user_assistant_pair(self, monkeypatch, cf_module):
        _install_fake_datasets(monkeypatch, {"HuggingFaceTB/smoltalk2": [self._SAMPLE_ROW]})
        cf = importlib.reload(cf_module)
        pairs = cf.collect_smoltalk2(max_samples=10, config="default")
        assert len(pairs) == 1
        assert "gravity" in pairs[0]["prompt"].lower()
        assert "9.8" in pairs[0]["completion"]

    def test_unknown_config_returns_empty_with_log(self, monkeypatch, cf_module, caplog):
        """Missing config → log available configs, return [], do not crash.
        Per learned principle: detect gated/missing on first attempt, no loop.
        """
        _install_fake_datasets(monkeypatch, {})  # any path raises
        cf = importlib.reload(cf_module)
        with caplog.at_level("ERROR"):
            pairs = cf.collect_smoltalk2(max_samples=10, config="unknown")
        assert pairs == []
        # Error message should help user pick a real config
        assert any("config" in rec.message.lower() or "available" in rec.message.lower() for rec in caplog.records)

    def test_skips_short_or_empty_messages(self, monkeypatch, cf_module):
        rows = [
            {
                "messages": [
                    {"role": "user", "content": ""},
                    {"role": "assistant", "content": "hi"},
                ]
            },  # empty user
            {
                "messages": [
                    {"role": "user", "content": "ok?"},
                    {"role": "assistant", "content": "y"},  # too short
                ]
            },
            self._SAMPLE_ROW,
        ]
        _install_fake_datasets(monkeypatch, {"HuggingFaceTB/smoltalk2": rows})
        cf = importlib.reload(cf_module)
        pairs = cf.collect_smoltalk2(max_samples=10, config="default")
        assert len(pairs) == 1

    def test_split_none_iterates_all_splits(self, monkeypatch, cf_module):
        """When split=None, all splits in the config are concatenated.

        Pass 155b: real SmolTalk2 SFT config has 25 named splits, none
        called "train". Default behaviour is auto-iterate-all so users
        do not need to pick one of 25 names.
        """
        def _row(topic: str, answer: str) -> dict:
            return {
                "messages": [
                    {"role": "user", "content": f"Explain {topic} briefly."},
                    {"role": "assistant", "content": answer},
                ]
            }

        # DISTINCT rows per split: with one identical row for every split,
        # dedup collapsed to 1 whether the collector iterated all splits,
        # only the first, or hardcoded "train" (audit 2026-07-17).
        _install_fake_datasets(
            monkeypatch,
            {
                "HuggingFaceTB/smoltalk2": {
                    "split_a_think": [_row("gravity", "Gravity pulls masses together, about 9.8 m/s^2 here.")],
                    "split_b_no_think": [_row("rain", "Water condenses in clouds and falls when drops grow heavy.")],
                    "split_c": [_row("tides", "The moon's gravity drags the oceans into two daily bulges.")],
                }
            },
            splits_by_path={"HuggingFaceTB/smoltalk2": ["split_a_think", "split_b_no_think", "split_c"]},
        )
        cf = importlib.reload(cf_module)
        pairs = cf.collect_smoltalk2(max_samples=10, config="SFT", split=None)
        assert len(pairs) == 3  # one from EVERY split, including the last
        completions = " ".join(p["completion"] for p in pairs)
        assert "9.8" in completions and "condenses" in completions and "bulges" in completions

    def test_explicit_split_used_directly(self, monkeypatch, cf_module):
        """When split is provided, function uses it without enumerating.
        The row lives ONLY under the requested split name, so passing any
        other split (or enumerating) fails the load."""
        _install_fake_datasets(
            monkeypatch,
            {"HuggingFaceTB/smoltalk2": {"my_exact_split": [self._SAMPLE_ROW]}},
        )
        cf = importlib.reload(cf_module)
        pairs = cf.collect_smoltalk2(max_samples=10, config="SFT", split="my_exact_split")
        assert len(pairs) == 1


# ── 2026-07-15 diet: short-record collectors ────────────────────────────────


class TestCollectNoRobots:
    """No Robots (`HuggingFaceH4/no_robots`): human-written ChatML pairs;
    completions above the cap are DROPPED, not truncated."""

    _SAMPLE_ROW = {
        "messages": [
            {"role": "user", "content": "Write a two-line poem about rain."},
            {"role": "assistant", "content": "Rain taps the glass in gray refrain,\nthe streetlights bloom in every pane."},
        ],
        "category": "creative",
    }

    def test_extracts_pair_and_caps_completion(self, monkeypatch, cf_module):
        long_row = {
            "messages": [
                {"role": "user", "content": "Tell me everything about Rome."},
                {"role": "assistant", "content": "x" * 2000},
            ]
        }
        _install_fake_datasets(monkeypatch, {"HuggingFaceH4/no_robots": [self._SAMPLE_ROW, long_row]})
        cf = importlib.reload(cf_module)
        pairs = cf.collect_no_robots(max_samples=10, max_completion_chars=600)
        assert len(pairs) == 1
        assert "poem" in pairs[0]["prompt"]

    def test_missing_dataset_returns_empty(self, monkeypatch, cf_module, caplog):
        _install_fake_datasets(monkeypatch, {})
        cf = importlib.reload(cf_module)
        with caplog.at_level("ERROR"):
            assert cf.collect_no_robots(max_samples=10) == []


class TestCollectEverydayConversations:
    def test_skips_scripted_greeting_opener(self, monkeypatch, cf_module):
        """The dataset opens every conversation with 'Hi' -> canned greeting;
        the substantive exchange is the second pair (measured: literal
        first-pair extraction yielded 4 of 2,260 records)."""
        row = {
            "messages": [
                {"role": "user", "content": "Hey!"},
                {"role": "assistant", "content": "Hello! How can I help you today?"},
                {"role": "user", "content": "I'm interested in learning the violin."},
                {"role": "assistant", "content": "Great choice -- start with a rental and a teacher."},
            ],
        }
        _install_fake_datasets(monkeypatch, {"HuggingFaceTB/everyday-conversations-llama3.1-2k": [row]})
        cf = importlib.reload(cf_module)
        pairs = cf.collect_everyday_conversations(max_samples=10)
        assert len(pairs) == 1
        assert pairs[0]["prompt"].startswith("I'm interested")
        assert pairs[0]["completion"].startswith("Great choice")

    def test_takes_first_exchange_only(self, monkeypatch, cf_module):
        row = {
            "full_topic": "Hobbies",
            "messages": [
                {"role": "user", "content": "Do you have any hobby ideas?"},
                {"role": "assistant", "content": "Drawing is a good start -- cheap and portable."},
                {"role": "user", "content": "What about the second one?"},
                {"role": "assistant", "content": "Photography, if you have a phone camera."},
            ],
        }
        _install_fake_datasets(monkeypatch, {"HuggingFaceTB/everyday-conversations-llama3.1-2k": [row]})
        cf = importlib.reload(cf_module)
        pairs = cf.collect_everyday_conversations(max_samples=10)
        assert len(pairs) == 1
        assert pairs[0]["completion"].startswith("Drawing")


class TestCollectTriviaQA:
    _SAMPLE_ROW = {
        "question": "Which planet is the largest in our solar system?",
        "answer": {"value": "Jupiter", "aliases": ["Jupiter"]},
    }

    def test_short_answer_becomes_sentence(self, monkeypatch, cf_module):
        _install_fake_datasets(monkeypatch, {"mandarjoshi/trivia_qa": [self._SAMPLE_ROW]})
        cf = importlib.reload(cf_module)
        pairs = cf.collect_triviaqa(max_samples=10)
        assert pairs == [{"prompt": "Which planet is the largest in our solar system?", "completion": "Jupiter."}]

    def test_long_answers_dropped(self, monkeypatch, cf_module):
        rows = [
            {"question": "A long-winded question about history?", "answer": {"value": "y" * 300}},
            self._SAMPLE_ROW,
        ]
        _install_fake_datasets(monkeypatch, {"mandarjoshi/trivia_qa": rows})
        cf = importlib.reload(cf_module)
        assert len(cf.collect_triviaqa(max_samples=10)) == 1


class TestCollectNQOpen:
    def test_normalizes_search_query_shape(self, monkeypatch, cf_module):
        row = {"question": "who wrote the origin of species", "answer": ["Charles Darwin"]}
        _install_fake_datasets(monkeypatch, {"google-research-datasets/nq_open": [row]})
        cf = importlib.reload(cf_module)
        pairs = cf.collect_nq_open(max_samples=10)
        assert pairs == [{"prompt": "Who wrote the origin of species?", "completion": "Charles Darwin."}]


class TestSmolTalk2CompletionCap:
    def test_cap_drops_long_and_skips_think_splits(self, monkeypatch, cf_module):
        long_row = {
            "messages": [
                {"role": "user", "content": "Explain the universe."},
                {"role": "assistant", "content": "z" * 5000},
            ]
        }
        short_row = {
            "messages": [
                {"role": "user", "content": "Explain gravity briefly."},
                {"role": "assistant", "content": "Gravity pulls masses together; drop a cup and the floor wins."},
            ]
        }
        # The think split holds a KEEPABLE short row with a marker answer:
        # if the skip guard vanishes, the marker leaks into the result. The
        # old identical-rows fake couldn't tell skipping from dedup
        # (audit 2026-07-17).
        think_row = {
            "messages": [
                {"role": "user", "content": "Explain thinking briefly."},
                {"role": "assistant", "content": "THINK-SPLIT-MARKER: this row must never be collected."},
            ]
        }
        _install_fake_datasets(
            monkeypatch,
            {
                "HuggingFaceTB/smoltalk2": {
                    "big_corpus_think": [think_row],
                    "magpie_no_think": [long_row, short_row],
                }
            },
            splits_by_path={"HuggingFaceTB/smoltalk2": ["big_corpus_think", "magpie_no_think"]},
        )
        cf = importlib.reload(cf_module)
        pairs = cf.collect_smoltalk2(max_samples=10, config="SFT", max_completion_chars=600)
        assert len(pairs) == 1
        assert pairs[0]["completion"].startswith("Gravity")
        assert not any("THINK-SPLIT-MARKER" in p["completion"] for p in pairs)

    def test_cap_also_drops_giant_prompts(self, monkeypatch, cf_module):
        """LongAlign-style rows pair a 64k-char context with a short answer;
        the diet cap must drop them (a 1024 block would keep the prompt only
        as truncated garbage)."""
        giant_prompt_row = {
            "messages": [
                {"role": "user", "content": "ctx " * 5000 + "Summarize."},
                {"role": "assistant", "content": "A short summary sentence."},
            ]
        }
        _install_fake_datasets(monkeypatch, {"HuggingFaceTB/smoltalk2": [giant_prompt_row]})
        cf = importlib.reload(cf_module)
        assert cf.collect_smoltalk2(max_samples=10, config="SFT", max_completion_chars=600) == []

    def test_no_cap_keeps_old_behavior(self, monkeypatch, cf_module):
        long_row = {
            "messages": [
                {"role": "user", "content": "Explain the universe."},
                {"role": "assistant", "content": "z" * 5000},
            ]
        }
        _install_fake_datasets(monkeypatch, {"HuggingFaceTB/smoltalk2": [long_row]})
        cf = importlib.reload(cf_module)
        pairs = cf.collect_smoltalk2(max_samples=10, config="default")
        assert len(pairs) == 1  # uncapped: long completions still collected


# ── D-11 wiring withdrawn: combine_all emits the JSONL and nothing else ─────


def test_combine_all_emits_no_text_file(tmp_path):
    """The D-11 .txt had no reader: `make_sft_data.py` and
    `make_pretrain_curated.py` both open `combined_finetune.jsonl` and
    `finetune_enigma.py` has no plain-text branch, so the second output
    only ever cost a write. Pinned as a NEGATIVE because the old shape
    reads like a missing feature: a well-meant re-add would resurrect a
    file that nothing consumes."""
    import collect_finetuning_data as cf
    import json

    src = tmp_path / "smoltalk2.jsonl"
    with src.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"prompt": "What is 2+2?", "completion": "4."}) + "\n")
        f.write(json.dumps({"prompt": "Capital of France?", "completion": "Paris."}) + "\n")
    cf.combine_all(tmp_path)
    assert (tmp_path / "combined_finetune.jsonl").exists()
    assert not (tmp_path / "combined_finetune.txt").exists(), \
        "combine_all emitted the withdrawn combined_finetune.txt again"


# ── Round-2 review (2026-08-13): the diet is the default, combine leaves a receipt ──


def test_the_recorded_diet_is_the_default(cf_module):
    """--all must reproduce the recorded 2026-07-15 diet: the smoltalk2 cap
    defaults to 600 (the one knob driving completion cap + prompt cap +
    think-split skip -- None streamed exactly the long-trace class
    OpenThoughts3 is excluded for), and SlimOrca -- never in the 105,203-row
    live build, no length filter at all -- is NOT swept in by --all."""
    assert cf_module._DIET_SMOLTALK2_CAP == 600
    src = Path(cf_module.__file__).read_text(encoding="utf-8")
    assert "default=_DIET_SMOLTALK2_CAP" in src, "the smoltalk2 cap default left the diet"
    assert "if args.slimorca is not None or args.all" not in src, \
        "SlimOrca is back in --all"


def test_combine_is_atomic_and_leaves_a_manifest(cf_module, tmp_path):
    """combine_all truncated the LIVE combined corpus on open, and nothing
    recorded which files fed it (synthetic_search_seed.jsonl once rode into
    the mix that way, silently)."""
    import json as _json

    (tmp_path / "a.jsonl").write_text('{"prompt": "p1", "completion": "c1"}\n',
                                      encoding="utf-8")
    (tmp_path / "b.jsonl").write_text('{"prompt": "p2", "completion": "c2"}\n'
                                      '{"prompt": "p2", "completion": "c2"}\n',
                                      encoding="utf-8")
    cf_module.combine_all(tmp_path)

    man = _json.loads((tmp_path / "combined_finetune.manifest.json")
                      .read_text(encoding="utf-8"))
    assert man["sources"] == {"a.jsonl": 1, "b.jsonl": 2}
    assert man["combined_records"] == 2  # the duplicate deduped
    assert not list(tmp_path.glob("*.jsonl.tmp")), "atomic write left droppings"
