"""Guards on pretrain_enigma.py --seed: a sweep compares runs, so a seeded run
must fix BOTH random streams (torch for weight init, numpy for batch sampling)
and an unseeded run must leave both untouched. No GPU or corpus needed.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "pretrain_enigma.py"

_spec = importlib.util.spec_from_file_location("pretrain_enigma_mod", SCRIPT)
pretrain = importlib.util.module_from_spec(_spec)
sys.modules["pretrain_enigma_mod"] = pretrain
_spec.loader.exec_module(pretrain)


def _draw():
    """One sample from each stream a run depends on."""
    return float(np.random.rand()), float(torch.rand(1).item())


def test_same_seed_reproduces_both_streams():
    assert pretrain.apply_seed(4242) is True
    first = _draw()
    assert pretrain.apply_seed(4242) is True
    assert _draw() == first


def test_different_seed_changes_both_streams():
    pretrain.apply_seed(1)
    numpy_first, torch_first = _draw()
    pretrain.apply_seed(2)
    numpy_second, torch_second = _draw()
    # both streams must move -- seeding only torch was the wiring bug this pins
    assert numpy_first != numpy_second
    assert torch_first != torch_second


def test_none_seed_does_not_seed():
    assert pretrain.apply_seed(None) is False
    # an unseeded call leaves the streams advancing freely
    pretrain.apply_seed(7)
    baseline = _draw()
    pretrain.apply_seed(7)
    assert pretrain.apply_seed(None) is False
    assert _draw() == baseline  # None neither reseeds nor disturbs the stream


def test_seed_flag_defaults_to_none():
    parser_src = SCRIPT.read_text(encoding="utf-8")
    assert '"--seed"' in parser_src
    # the live lineage ran unseeded; the default must not change that
    seed_decl = parser_src.split('"--seed"', 1)[1][:200]
    assert "default=None" in seed_decl
