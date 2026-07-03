"""finetune_enigma.py helpers — data loading, packing/mask alignment, the
chat-row re-init, and the masked loss path through the real model class."""

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


def test_reinit_chat_rows_touches_only_chat_rows():
    m = _nano4718()
    before = m.tok_embeddings.weight.detach().clone()
    rows = ft.reinit_chat_rows(m)
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
