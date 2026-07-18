"""LoRA offload device-placement regression.

``LoraTrainer.train`` used to pass ``cpu=self.offload_config.offload_optimizer``
to ``Accelerator``. With the default config (offload_optimizer=True) that forced
the ENTIRE run onto the CPU while the batch loop still moved inputs to 'cuda',
crashing on the first forward with a device mismatch -- and "offload optimizer
states" never meant "run everything on CPU" in the first place. ``cpu`` must
instead track whether there is genuinely no GPU.

The trainer is built via ``object.__new__`` so the test needs neither PEFT nor a
real GPU-training pass: it drives ``train`` only up to the accelerator setup and
records the ``cpu`` argument.
"""

import pytest
import torch
import torch.nn as nn

import enigma_engine.core.lora_utils as lu
from enigma_engine.core.lora_utils import LoraConfig, LoraTrainer, OffloadConfig


class _RecordingAccelerator:
    last_cpu = None

    def __init__(self, gradient_accumulation_steps=1, cpu=False):
        _RecordingAccelerator.last_cpu = cpu
        self.device = torch.device("cuda" if (not cpu and torch.cuda.is_available()) else "cpu")

    def prepare(self, model, optimizer):
        return model, optimizer


def _bare_trainer(offload_config: OffloadConfig) -> LoraTrainer:
    """A LoraTrainer with just the attributes train() touches before it
    early-returns -- no PEFT wrapping, no adapters."""
    tr = object.__new__(LoraTrainer)
    tr.model = nn.Linear(4, 4)
    tr.lora_config = LoraConfig()  # lora_plus_lambda == 1.0 -> single AdamW group
    tr.offload_config = offload_config
    tr.gradient_accumulation_steps = 1
    tr.learning_rate = 1e-4
    tr.weight_decay = 0.01
    tr.adam_beta1 = 0.9
    tr.adam_beta2 = 0.95
    tr.adam_eps = 1e-8
    tr.on_progress = None
    tr.on_loss = None
    return tr


@pytest.mark.parametrize("gpu_present", [True, False])
def test_offload_optimizer_does_not_force_cpu_only(monkeypatch, gpu_present):
    """cpu= must reflect "no GPU", NOT offload_optimizer. Parametrized over
    BOTH hardware states by masking torch.cuda.is_available: the original
    single-assert version only discriminated on a CUDA machine -- on a
    CPU-only box the regression (cpu=offload_optimizer=True) produced the
    same value as the expectation and passed (test-suite audit 2026-07-17).
    The gpu_present=True leg is the one the bug breaks."""
    monkeypatch.setattr(lu, "ACCELERATE_AVAILABLE", True)
    # raising=False: without accelerate installed the module never defines
    # the name; the recorder stands in either way.
    monkeypatch.setattr(lu, "Accelerator", _RecordingAccelerator, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: gpu_present)
    # get_memory_info would query real CUDA state, which the mask can lie
    # about -- stub the two keys train() logs. Same for clear_vram: with
    # is_available masked True on a CPU-only box, torch.cuda.synchronize()
    # would _lazy_init() and raise before the assertion under test is ever
    # reached (re-audit 2026-07-18).
    monkeypatch.setattr(
        lu, "get_memory_info", lambda: {"vram_available_gb": 0.0, "ram_available_gb": 8.0}
    )
    monkeypatch.setattr(lu, "clear_vram", lambda: None)
    _RecordingAccelerator.last_cpu = None

    # Default-ish config that used to trigger the crash: offload both on.
    tr = _bare_trainer(OffloadConfig(cpu_offload=True, offload_optimizer=True))
    # No valid batches -> train() returns right after the accelerator setup.
    monkeypatch.setattr(tr, "_create_batches", lambda data, max_length: [])

    tr.train([{"prompt": "a", "completion": "b"}])

    assert _RecordingAccelerator.last_cpu == (not gpu_present)
