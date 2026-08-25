"""Direct tests for hardware_detection -- the module the load path reads the
machine through: ``Enigma.auto_configure`` (model.py:947) detects then asks for
an optimal config, ``model_utils.detect_hardware`` hands the same profile out
as a dict (model_utils.py:228-287), and the tokenizer caches size themselves
off ``InferenceMemoryBudget`` (bpe_tokenizer.py:142, advanced_tokenizer.py:151).
None of it had a test of its own before (2026-08-23 review: the negative-space
list).

Every test drives ``torch.cuda`` through stubs in BOTH states, so this file
makes no GPU call of its own and reads identically on a GPU-less machine -- no
skips, and no assertion that depends on what card is in the box.
``TrainingMemoryBudget`` is left uncovered: nothing outside this module
constructs one.
"""

from __future__ import annotations

import types

import pytest
import torch

from enigma_engine.core import hardware_detection as hw


def _fake_cuda(monkeypatch, *, available, vram_gb=32.0, name="FakeGPU 9000",
               cuda_version="12.8", bf16=True):
    """Point every torch entry point the module reaches at a stub.

    ``device_count`` is stubbed for coherence only -- hardware_detection never
    reads it; ``is_available`` is the switch it actually branches on. The MPS
    backend is stubbed off because it is the other branch detect_hardware can
    take, and a Mac has to read the same as this box."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1 if available else 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index=0: name)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index=0: types.SimpleNamespace(total_memory=int(vram_gb * 1024**3)),
    )
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: bf16)
    monkeypatch.setattr(torch.version, "cuda", cuda_version if available else None)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)


def _cpu_profile(ram_gb: float) -> hw.HardwareProfile:
    return hw.HardwareProfile(ram_gb=ram_gb, total_ram_gb=ram_gb)


def _gpu_profile(vram_gb: float, ram_gb: float = 64.0) -> hw.HardwareProfile:
    return hw.HardwareProfile(
        device="cuda", gpu_available=True, has_cuda=True,
        gpu_vram_gb=vram_gb, ram_gb=ram_gb, total_ram_gb=ram_gb,
    )


@pytest.fixture(autouse=True)
def _no_cached_profile():
    """detect_hardware memoizes into a module global, so a test that left a
    stubbed profile behind would answer for every later caller in the
    session -- and a real profile cached by an earlier test would make these
    tests read the box instead of the stubs."""
    hw.clear_cached_profile()
    yield
    hw.clear_cached_profile()


def test_a_machine_without_cuda_detects_as_cpu(monkeypatch):
    _fake_cuda(monkeypatch, available=False)

    profile = hw.detect_hardware()

    assert profile.device == "cpu"
    assert profile.gpu_available is False and profile.has_cuda is False
    assert profile.gpu_vram_gb == 0.0
    assert profile.gpu_name == "" and profile.cuda_version == ""
    assert profile.hardware_type != "desktop_gpu"
    assert profile.cpu_cores >= 1 and profile.cpu_threads >= 1
    assert profile.ram_gb > 0 and profile.total_ram_gb == profile.ram_gb


def test_a_machine_with_cuda_detects_the_device_it_was_told_about(monkeypatch):
    _fake_cuda(monkeypatch, available=True, vram_gb=32.0, name="FakeGPU 9000",
               cuda_version="12.8")

    profile = hw.detect_hardware()

    assert profile.device == "cuda"
    assert profile.gpu_available is True and profile.has_cuda is True
    assert profile.gpu_name == "FakeGPU 9000"
    assert profile.gpu_vram_gb == pytest.approx(32.0)
    assert profile.cuda_version == "12.8"
    assert profile.hardware_type == "desktop_gpu"
    assert profile.has_mps is False


def test_the_profile_is_cached_until_it_is_cleared(monkeypatch):
    """auto_configure detects on every construction; the memo is what keeps
    that from re-probing the driver each time."""
    _fake_cuda(monkeypatch, available=False)
    first = hw.detect_hardware()

    assert hw.detect_hardware() is first
    assert hw.get_cached_profile() is first

    _fake_cuda(monkeypatch, available=True, vram_gb=8.0)
    assert hw.detect_hardware() is first

    hw.clear_cached_profile()
    assert hw.get_cached_profile() is None
    assert hw.detect_hardware().device == "cuda"


def test_get_hardware_is_the_same_detection(monkeypatch):
    """The alias `enigma_engine.core.__init__` exports."""
    _fake_cuda(monkeypatch, available=False)

    assert hw.get_hardware() is hw.detect_hardware()


def test_model_size_walks_the_ram_and_vram_tiers():
    """The sizing auto_configure asks for by name; the tiers are the contract,
    so each one gets a profile. Profiles are explicit here -- no detection
    runs, so no stub is needed."""
    assert hw.recommend_model_size(hw.HardwareProfile(is_raspberry_pi=True, ram_gb=8.0)) == "pi_zero"
    assert hw.recommend_model_size(_cpu_profile(1.0)) == "pi_zero"
    assert hw.recommend_model_size(_cpu_profile(3.0)) == "nano"
    assert hw.recommend_model_size(_cpu_profile(6.0)) == "tiny"
    assert hw.recommend_model_size(_cpu_profile(12.0)) == "small"
    assert hw.recommend_model_size(_cpu_profile(64.0)) == "medium"
    assert hw.recommend_model_size(_gpu_profile(24.0)) == "large"
    assert hw.recommend_model_size(_gpu_profile(12.0)) == "medium"
    assert hw.recommend_model_size(_gpu_profile(6.0)) == "small"
    assert hw.recommend_model_size(_gpu_profile(4.0)) == "tiny"
    assert hw.recommend_model_size(_gpu_profile(2.0)) == "nano"


def test_training_batch_size_walks_the_vram_tiers():
    """encoder_align.py:223 sends the user here when auto-estimation refuses,
    so the tiers are user-facing advice, not an internal detail."""
    assert hw.recommend_training_batch_size(_gpu_profile(48.0)) == 16
    assert hw.recommend_training_batch_size(_gpu_profile(24.0)) == 8
    assert hw.recommend_training_batch_size(_gpu_profile(12.0)) == 4
    assert hw.recommend_training_batch_size(_gpu_profile(6.0)) == 2
    assert hw.recommend_training_batch_size(_gpu_profile(2.0)) == 1
    assert hw.recommend_training_batch_size(_cpu_profile(64.0)) == 2
    assert hw.recommend_training_batch_size(_cpu_profile(8.0)) == 1


def test_optimal_config_precision_follows_bf16_support(monkeypatch):
    gpu = _gpu_profile(24.0)

    _fake_cuda(monkeypatch, available=True, vram_gb=24.0, bf16=True)
    assert hw.get_optimal_config(gpu)["precision"] == "bfloat16"

    _fake_cuda(monkeypatch, available=True, vram_gb=24.0, bf16=False)
    on_gpu = hw.get_optimal_config(gpu)
    assert on_gpu["precision"] == "float16"
    assert on_gpu["use_half"] is True

    _fake_cuda(monkeypatch, available=False)
    on_cpu = hw.get_optimal_config(_cpu_profile(64.0))
    assert on_cpu["precision"] == "float32"
    assert on_cpu["use_half"] is False
    assert on_cpu["device"] == "cpu"


def test_optimal_config_sizes_batch_and_context_from_the_budget(monkeypatch):
    """The two numbers auto_configure passes on to the model."""
    _fake_cuda(monkeypatch, available=True, vram_gb=24.0)
    on_gpu = hw.get_optimal_config(_gpu_profile(24.0))
    assert (on_gpu["batch_size"], on_gpu["max_seq_len"]) == (8, 2048)
    assert on_gpu["model_size"] == "large"

    _fake_cuda(monkeypatch, available=False)
    on_cpu = hw.get_optimal_config(_cpu_profile(64.0))
    assert (on_cpu["batch_size"], on_cpu["max_seq_len"]) == (2, 512)
    assert on_cpu["model_size"] == "medium"


def test_a_gpu_less_profile_still_probes_the_live_driver_for_vram(monkeypatch):
    """RECORDED, not endorsed. ``vram_gb`` defaults to 0 and 0 is the
    auto-detect sentinel, so ``from_profile`` on a profile that reported NO GPU
    re-probes torch.cuda and adopts whatever the machine has -- on a CUDA box
    that makes get_optimal_config's batch/context tiers read the real card even
    for a CPU profile. Pinned so that changing it is a deliberate edit here."""
    _fake_cuda(monkeypatch, available=True, vram_gb=32.0)

    budget = hw.InferenceMemoryBudget.from_profile(_cpu_profile(8.0))

    assert budget.vram_gb == pytest.approx(32.0)


def test_tokenizer_cache_caps_scale_with_ram_and_clamp():
    """bpe_tokenizer.py:142 and advanced_tokenizer.py:151 size their caches off
    this budget; the clamps are what keep a Pi and a workstation both sane."""
    small = hw.InferenceMemoryBudget(ram_gb=1.0, vram_gb=1.0)
    big = hw.InferenceMemoryBudget(ram_gb=1024.0, vram_gb=1.0)

    assert small.bpe_cache_cap == 2000
    assert big.bpe_cache_cap == 100000
    assert small.advanced_tok_cache_cap == 2000
    assert big.advanced_tok_cache_cap == 100000
    assert small.token_count_cache_cap == 1024
    assert big.token_count_cache_cap == 65536


@pytest.mark.parametrize("cuda_up", [False, True])
def test_the_dict_form_the_consumers_read_matches_the_profile(monkeypatch, cuda_up):
    """model_utils.detect_hardware hands callers ``dataclasses.asdict(profile)``
    and its own recommend_model_size rebuilds a profile out of that dict
    (model_utils.py:228-287) -- the alias keys have to agree with the fields
    they alias, or the rebuild sizes off stale values."""
    from enigma_engine.core import model_utils as mu

    _fake_cuda(monkeypatch, available=cuda_up, vram_gb=24.0)

    info = mu.detect_hardware()
    profile = hw.get_cached_profile()

    assert profile is not None
    assert info["ram_gb"] == info["total_ram_gb"] == profile.ram_gb
    assert info["has_cuda"] == info["gpu_available"] == cuda_up
    assert (info["gpu_vram_gb"] > 0) is cuda_up
    assert mu.recommend_model_size(info) == hw.recommend_model_size(profile)


def test_memory_estimate_scales_with_precision_and_batch():
    """model_utils.estimate_memory_usage (model_utils.py:287) scales this
    result by the quantization factor, so the half/full base has to be an
    honest 2x."""
    full = hw.estimate_memory_usage("small", batch_size=1, seq_len=512, use_half=False)
    half = hw.estimate_memory_usage("small", batch_size=1, seq_len=512, use_half=True)
    wider = hw.estimate_memory_usage("small", batch_size=4, seq_len=512, use_half=False)

    assert half["model_memory"] == pytest.approx(full["model_memory"] / 2)
    assert half["kv_cache"] == pytest.approx(full["kv_cache"] / 2)
    assert full["total"] == pytest.approx(full["model_memory"] + full["kv_cache"])
    assert wider["kv_cache"] == pytest.approx(full["kv_cache"] * 4)
