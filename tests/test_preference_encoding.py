"""Preference-encoding regression: the prompt/completion boundary.

``tokenizer.encode`` appends a trailing EOS to the prompt-only text, but that
EOS is not present in the combined ``prompt + completion`` encoding. Counting
the prompt-only length directly over-counted by one and masked the FIRST
completion token out of every DPO/SimPO/KTO/ORPO gradient (pairs that differ
mainly in their first token -- 'Yes.' vs 'No.' -- then trained on almost
nothing). All four encoders now share ``_encoded_prompt_len``; these tests lock
the boundary in place.
"""

import torch

from enigma_engine.core.model import Enigma
from enigma_engine.core.model_presets import ForgeConfig
from enigma_engine.training.training import Trainer, TrainingConfig


class _EosTokenizer:
    """Mimics the real AdvancedTokenizer contract: BOS prepended and EOS
    appended when ``add_special_tokens`` is set, char-level body otherwise. The
    char-level body keeps the prompt encoding a true prefix of the combined
    encoding, exactly as the real BPE tokenizer does on this fixed format."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.bos_token_id = 1
        self.eos_token_id = 2

    def encode(self, text, add_special_tokens=True):
        body = [3 + (ord(c) % (self.vocab_size - 4)) for c in text]
        if add_special_tokens:
            return [self.bos_token_id] + body + [self.eos_token_id]
        return body

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(65 + (int(i) % 26)) for i in ids)


def _trainer():
    torch.manual_seed(0)
    model = Enigma(
        ForgeConfig(
            vocab_size=64, dim=32, n_layers=2, n_heads=2,
            max_seq_len=64, dropout=0.0, use_gradient_checkpointing=False,
        )
    )
    tok = _EosTokenizer(model.config.vocab_size)
    return Trainer(model, tok, TrainingConfig(epochs=1, batch_size=2))


def test_encoded_prompt_len_excludes_trailing_eos():
    """The prompt-prefix length must drop the trailing EOS that encode()
    appends to the prompt-only text."""
    tr = _trainer()
    prompt = "hi there"
    full_prompt_ids = tr.tokenizer.encode(f"User: {prompt}\nAssistant: ")
    assert full_prompt_ids[-1] == tr.tokenizer.eos_token_id  # sanity: EOS is appended
    assert tr._encoded_prompt_len(prompt) == len(full_prompt_ids) - 1


def test_dpo_pair_keeps_first_completion_token():
    """The first completion token must survive as a real (non -100) label --
    the off-by-one used to mask it out."""
    tr = _trainer()
    prompt = "hi there"
    chosen = "Yes."
    pairs: list = []
    ok = tr._encode_dpo_pair(
        {"prompt": prompt, "chosen": chosen, "rejected": "No."}, pairs, max_len_dpo=64
    )
    assert ok and len(pairs) == 1

    input_ids, chosen_labels, _rejected_labels, _mask = pairs[0]
    chosen_ids = input_ids[0].tolist()
    labels = chosen_labels[0].tolist()

    first_completion_id = tr.tokenizer.encode(chosen, add_special_tokens=False)[0]
    unmasked = [tok for tok in labels if tok != -100]
    # First trained token is the first completion token, not the second.
    assert unmasked[0] == first_completion_id
    # It sits at the position immediately after the prompt prefix in chosen_ids.
    prompt_len = tr._encoded_prompt_len(prompt)
    assert chosen_ids[prompt_len] == first_completion_id
    # Every completion token plus the terminal EOS is trained (nothing dropped).
    assert len(unmasked) == len(chosen_ids) - prompt_len
