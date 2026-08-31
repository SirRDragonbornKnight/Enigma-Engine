"""finetune_enigma.py helpers — data loading, packing/mask alignment, the
chat-row re-init, and the masked loss path through the real model class."""

import inspect
import json
from dataclasses import replace

import pytest
import torch

import finetune_enigma as ft
from enigma_engine.core.chat_format import CHAT_TOKENS, attach_chat_tokens, render_training
from enigma_engine.core.tokenizer import get_tokenizer


@pytest.fixture(scope="module")
def tok():
    return attach_chat_tokens(get_tokenizer("bpe"))


def test_quality_gate_keeps_urls_and_drops_junk():
    """A cited link is supervision, not junk -- the old gate dropped every
    record whose ANSWER contained a URL, so she could never learn to cite a
    source. The real junk classes (raw HTML, mojibake, source loops) stay."""
    from make_sft_data import _is_low_quality

    def rec(answer):
        return {"messages": [{"role": "user", "content": "q"},
                             {"role": "assistant", "content": answer}]}

    assert not _is_low_quality(rec("The paper is at https://arxiv.org/abs/2404.05405 if you want the details."))
    assert _is_low_quality(rec('See <div class="x">this</div>'))          # raw HTML
    assert _is_low_quality(rec("it broke � here"))                   # mojibake
    assert _is_low_quality(rec("no no no no no no way"))                  # source loop


def _nano4718():
    from enigma_engine.core.model import Enigma
    from enigma_engine.core.model_presets import MODEL_PRESETS

    return Enigma(replace(MODEL_PRESETS["nano"], vocab_size=4718))


def test_load_examples_accepts_both_schemas_and_skips_overlong(tok, tmp_path):
    p = tmp_path / "d.jsonl"
    recs = [
        {"prompt": "Hi", "completion": "Hello!"},
        {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]},
        {"prompt": "long", "completion": "w " * 3000},  # > block -> skipped, counted
        {"prompt": "no reply"},  # malformed -> dropped
    ]
    p.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    ex = ft.load_examples(p, tok, block=128)
    assert len(ex) == 2
    for ids, mask in ex:
        assert len(ids) == len(mask) and any(mask)


def test_pack_blocks_target_alignment_and_pad_ignore(tok):
    msgs = [{"role": "user", "content": "Say hi"}, {"role": "assistant", "content": "hi"}]
    ids, mask = render_training(tok, msgs)
    X, Y = ft.pack_blocks([(ids, mask)], block=64)
    assert X.shape == (1, 64) and Y.shape == (1, 64)
    padded = ids + [0] * 64
    for i in range(64):
        if Y[0, i].item() != ft.IGNORE:
            assert Y[0, i].item() == padded[i + 1]  # teacher-forced next token
    assert (Y[0, len(ids) :] == ft.IGNORE).all()  # padding never trains
    assert int((Y[0] != ft.IGNORE).sum()) > 0  # but the assistant span does


def test_pack_blocks_trains_only_the_assistant_span(tok):
    """WHICH positions train, not just what values they carry. The alignment
    test above verifies Y[i] == padded[i+1] at non-IGNORE positions -- but a
    mask regression that trains USER-turn tokens (mask[i] instead of
    mask[i+1], or mask=all-True) still yields correct VALUES everywhere and
    passed it (test-suite audit 2026-07-17). This is the module's headline
    property: she learns to ANSWER, never to imitate the user."""
    msgs = [
        {"role": "user", "content": "Please greet the visitors warmly."},
        {"role": "assistant", "content": "hi"},
    ]
    ids, mask = render_training(tok, msgs)
    assert any(mask) and not all(mask)  # sanity: both spans exist
    X, Y = ft.pack_blocks([(ids, mask)], block=64)

    trained = [i for i in range(64) if Y[0, i].item() != ft.IGNORE]
    # position i trains iff its TARGET (token i+1) is assistant content
    expected = [i for i in range(len(ids) - 1) if mask[i + 1]]
    assert trained, "no position trains at all"
    assert trained == expected
    # the learned targets are exactly the assistant-span ids...
    assert [Y[0, i].item() for i in trained] == [ids[i + 1] for i in expected]
    # ...and the user's words are not among them
    target_text = tok.decode([Y[0, i].item() for i in trained], skip_special_tokens=True)
    for word in ("Please", "greet", "visitors", "warmly"):
        assert word not in target_text


def test_reinit_chat_rows_touches_only_chat_rows(tok):
    m = _nano4718()
    before = m.tok_embeddings.weight.detach().clone()
    rows = ft.reinit_chat_rows(m, tok)
    emb = m.tok_embeddings.weight
    assert rows == sorted(CHAT_TOKENS.values())
    assert not torch.allclose(emb[rows], before[rows])
    keep = [i for i in range(emb.shape[0]) if i not in set(rows)]
    assert torch.allclose(emb[keep], before[keep])


def test_masked_loss_is_finite_and_backprops(tok):
    m = _nano4718().train()
    msgs = [{"role": "user", "content": "Who are you?"}, {"role": "assistant", "content": "Enigma."}]
    ids, mask = render_training(tok, msgs)
    X, Y = ft.pack_blocks([(ids, mask)], block=96)
    _, loss = m(X, targets=Y, pad_token_id=ft.IGNORE)
    assert torch.isfinite(loss)
    loss.backward()


def test_data_sha_recorded_and_checked(tmp_path):
    p = tmp_path / "mix.jsonl"
    p.write_text('{"messages": []}\n', encoding="utf-8")
    sha1 = ft._data_sha256(p)
    p.write_text('{"messages": [{"role": "user", "content": "x"}]}\n', encoding="utf-8")
    assert ft._data_sha256(p) != sha1


def test_resume_refuses_changed_data(tmp_path):
    p = tmp_path / "mix.jsonl"
    p.write_text("row1\n", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        ft._refuse_changed_data(p, "0" * 64)
    assert "regenerated" in str(e.value) or "changed" in str(e.value)
    # And the pass case: recorded sha matches the live file -> no raise.
    ft._refuse_changed_data(p, ft._data_sha256(p))


def test_accum_scales_weight_by_tokens():
    """Dividing each micro-batch loss by grad_accum averages MEANS, which gives
    a short batch the same pull as a full one. The step is supposed to be a
    per-TOKEN mean, so each draw is weighted by its share of the step's
    supervised tokens (D-6, measured 9.23% adversarial skew)."""
    from finetune_enigma import _accum_scales

    assert _accum_scales([100, 100, 100, 100]) == [0.25, 0.25, 0.25, 0.25]
    s = _accum_scales([1000, 10])
    assert abs(s[0] - 1000 / 1010) < 1e-12 and abs(s[1] - 10 / 1010) < 1e-12
    # A step whose every draw is pure padding would divide by zero otherwise;
    # its gradient is zero either way.
    assert _accum_scales([0, 0]) == [0.0, 0.0]


def test_train_loop_uses_token_scales():
    src = inspect.getsource(ft)
    assert "_accum_scales(" in src
    assert "loss / args.grad_accum" not in src


def test_builtin_block_meta_stamp(tmp_path):
    """What a checkpoint records about its corpus is READ from the builder's
    own manifest, never guessed: no manifest (or no key) leaves the key off, so
    a lineage never claims a property nobody measured."""
    data = tmp_path / "mix.jsonl"
    data.write_text('{"messages": []}\n', encoding="utf-8")

    assert ft._data_meta(data) == {}, "no manifest -> no claim"

    manifest = tmp_path / "mix.manifest.json"
    manifest.write_text(json.dumps({"records": 1}), encoding="utf-8")
    assert ft._data_meta(data) == {}, "manifest without the key -> no claim"

    manifest.write_text(json.dumps({"records": 1, "builtin_block": True}), encoding="utf-8")
    assert ft._data_meta(data)["builtin_block"] is True

    manifest.write_text(json.dumps({"builtin_block": False}), encoding="utf-8")
    assert ft._data_meta(data)["builtin_block"] is False

    # A corrupt manifest must not take a training run down over a receipt --
    # and "corrupt" includes bytes that are not UTF-8 at all. read_text raises
    # UnicodeDecodeError there, which is a ValueError and NOT a
    # JSONDecodeError, so a net catching only the latter let it escape and kill
    # the trainer at startup against this function's own contract.
    manifest.write_text("{not json", encoding="utf-8")
    assert ft._data_meta(data) == {}
    manifest.write_bytes(b'{"builtin_block": "\xff\xfe not utf-8"}')
    assert ft._data_meta(data) == {}

    # ...and the stamp lands ONCE, after both meta branches: the chat-format
    # re-init branch REPLACES meta wholesale, so a stamp written inside either
    # branch alone is lost on the other path.
    assert inspect.getsource(ft.main).count("_data_meta(") == 1


def test_resume_check_reads_the_checkpoints_schedule_not_the_new_one():
    # Presence pin against the dead-guard wiring: the comparison must consume
    # saved_sched (the checkpoint's recorded sha); wired to the fresh schedule
    # it would compare the live file to itself and never fire.
    src = inspect.getsource(ft.main)
    assert 'saved_sched.get("data_sha256")' in src
    assert '_refuse_changed_data(Path(args.data), saved_sched["data_sha256"])' in src
