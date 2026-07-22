"""The vocabulary belongs to the WEIGHTS, not the checkout.

A v1 checkpoint (vocab 4718) and a v2 checkpoint (vocab 16366) must each load
the vocab file they were trained against from the same repo. Serving one with
the other's table decodes to garbage without raising, so these pin the mapping.
"""

import pytest

from enigma_engine.core.tokenizer import VOCAB_DIR, get_tokenizer, vocab_file_for_size


def _table_size(path):
    import json

    return len(json.loads(path.read_text(encoding="utf-8"))["token_to_id"])


def test_v1_and_v2_resolve_to_different_files():
    v1 = vocab_file_for_size(4718)
    v2 = vocab_file_for_size(16366)
    assert v1 != v2
    assert _table_size(v1) == 4718
    assert _table_size(v2) == 16366


@pytest.mark.parametrize("vocab_size, table", [(4718, 4718), (16366, 16366)])
def test_selected_tokenizer_reports_the_matching_vocab(vocab_size, table):
    tok = get_tokenizer("bpe", vocab_path=vocab_file_for_size(vocab_size))
    assert tok.vocab_size == table
    assert tok.decode(tok.encode("hello enigma")) == "hello enigma"


@pytest.mark.parametrize("padded, table", [(4784, 4718), (16384, 16366)])
def test_padded_checkpoint_vocab_resolves_to_its_real_table(padded, table):
    """Models round embedding rows up to a multiple of 64, so a checkpoint's
    vocab_size can exceed the real table -- the largest table that fits wins."""
    assert _table_size(vocab_file_for_size(padded)) == table


def test_size_smaller_than_every_table_raises():
    # silently falling back would serve a wider table than the model has rows for
    with pytest.raises(ValueError):
        vocab_file_for_size(100)


def test_gap_size_between_tables_raises():
    """A vocab_size far above a table is not padding -- it means the trained
    table is missing from the directory. Falling back to the nearest smaller
    table would decode the missing ids as garbage, silently."""
    with pytest.raises(ValueError):
        vocab_file_for_size(8000)


def test_default_directory_load_is_still_v1():
    """The bare call must not change behavior for the live lineage."""
    assert get_tokenizer("bpe").vocab_size == 4718
    assert (VOCAB_DIR / "bpe_vocab.json").exists()


@pytest.mark.parametrize("tok_vocab", [4718, 20000])
def test_encoder_align_refuses_a_mismatched_tokenizer(tok_vocab):
    """A v1 table against a v2 head trains ids that mean something else and
    never crashes on its own -- the trainer has to refuse it up front."""
    import torch.nn as nn

    from enigma_engine.training.encoder_align import Trainer

    class _Cfg:
        vocab_size = 16384

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _Cfg()
            self.w = nn.Linear(2, 2)

    class _Tok:
        vocab_size = tok_vocab

    with pytest.raises(ValueError, match="vocab"):
        Trainer(_Model(), _Tok())


def test_encoder_align_accepts_the_matching_tokenizer():
    import torch.nn as nn

    from enigma_engine.training.encoder_align import Trainer

    class _Cfg:
        vocab_size = 16384

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _Cfg()
            self.w = nn.Linear(2, 2)

    class _Tok:
        vocab_size = 16366  # real table under a padded head

    assert Trainer(_Model(), _Tok()) is not None
