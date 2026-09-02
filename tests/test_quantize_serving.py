"""int8 weight-only quantization of a serving checkpoint, and the LOAD side
that makes it honest.

These tests pin MECHANICS, not quality: an untrained tiny model's top-1 margins
are order-statistic noise, so a top-1 agreement gate here would be decided at
authoring time rather than by the quantizer. Closeness is asserted the one way
that is scale-relative and meaningful on random weights -- cosine on the logits.
The real-checkpoint quality gate lives on the real checkpoint.
"""

from __future__ import annotations

import pathlib
import sys

import pytest
import torch

from enigma_engine.core.model import Enigma
from enigma_engine.core.model_presets import ForgeConfig

pytest.importorskip("torchao")

from quantize_serving_ckpt import load_serving_ckpt, quantize  # noqa: E402

CFG = dict(vocab_size=64, dim=64, n_layers=2, n_heads=2, max_seq_len=64,
           dropout=0.0, use_gradient_checkpointing=False)


def _fp32_ckpt(path, step=11):
    torch.manual_seed(7)
    cfg = ForgeConfig(**CFG)
    model = Enigma(cfg).eval()
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg.to_dict(),
        "step": step,
        "meta": {"chat_format": "enigma-chat-v1"},
    }, path)
    return model


def _is_subclass_weight(w) -> bool:
    """A torchao tensor subclass, not a plain fp32 Parameter/Tensor."""
    return type(w).__name__ not in ("Parameter", "Tensor")


def _quantized_linears(model):
    return [m for m in model.modules() if isinstance(m, torch.nn.Linear) and _is_subclass_weight(m.weight)]


def test_quantize_writes_the_marker_and_the_four_keys(tmp_path):
    src, out = tmp_path / "fp32.pth", tmp_path / "int8.pth"
    _fp32_ckpt(src)
    quantize(src, out)
    ck = torch.load(out, map_location="cpu", weights_only=True)
    assert set(ck) == {"model_state_dict", "config", "step", "meta"}
    assert ck["meta"]["quant"] == "int8_wo"
    assert ck["step"] == 11 and ck["meta"]["chat_format"] == "enigma-chat-v1"


def test_loaded_int8_model_has_subclass_linear_weights(tmp_path):
    src, out = tmp_path / "fp32.pth", tmp_path / "int8.pth"
    _fp32_ckpt(src)
    quantize(src, out)
    model, ck = load_serving_ckpt(out)
    assert _quantized_linears(model), "no linear came back as a torchao subclass tensor"


def test_int8_forward_runs(tmp_path):
    src, out = tmp_path / "fp32.pth", tmp_path / "int8.pth"
    _fp32_ckpt(src)
    quantize(src, out)
    model, _ = load_serving_ckpt(out)
    ids = torch.randint(1, 64, (1, 8))
    with torch.no_grad():
        logits = model(ids)
    assert logits.shape[-1] >= 64 and torch.isfinite(logits).all()


def test_int8_generates_sixteen_tokens(tmp_path):
    src, out = tmp_path / "fp32.pth", tmp_path / "int8.pth"
    _fp32_ckpt(src)
    quantize(src, out)
    model, _ = load_serving_ckpt(out)
    ids = torch.randint(1, 64, (1, 4))
    with torch.no_grad():
        got = [int(t) for t in model.generate_stream(ids, max_new_tokens=16, temperature=0.0, stop_tokens=[-1])]
    assert len(got) == 16


def test_int8_logits_track_fp32(tmp_path):
    """Scale-relative closeness: cosine, on a fixed seed-7 input."""
    src, out = tmp_path / "fp32.pth", tmp_path / "int8.pth"
    fp32 = _fp32_ckpt(src).eval()
    quantize(src, out)
    model, _ = load_serving_ckpt(out)
    torch.manual_seed(7)
    ids = torch.randint(1, 64, (1, 16))
    with torch.no_grad():
        a = fp32(ids).flatten()
        b = model(ids).flatten()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    assert cos > 0.99, f"int8 logits diverged from fp32: cosine={cos:.6f}"


def test_plain_fp32_checkpoint_takes_the_old_path(tmp_path):
    """No marker -> no torchao anywhere near it, weights bit-identical."""
    src = tmp_path / "fp32.pth"
    ref = _fp32_ckpt(src)
    model, ck = load_serving_ckpt(src)
    assert "quant" not in ck.get("meta", {})
    assert not _quantized_linears(model)
    for k, v in ref.state_dict().items():
        assert torch.equal(v, model.state_dict()[k]), k


def test_loader_refuses_a_mismarked_checkpoint(tmp_path):
    """The marker says int8, the tensors are fp32. Loading must FAIL, not
    quietly serve a model whose weights never went through the quantizer."""
    src = tmp_path / "lying.pth"
    torch.manual_seed(7)
    cfg = ForgeConfig(**CFG)
    torch.save({
        "model_state_dict": Enigma(cfg).state_dict(),
        "config": cfg.to_dict(),
        "step": 1,
        "meta": {"quant": "int8_wo"},
    }, src)
    with pytest.raises((SystemExit, RuntimeError)):
        load_serving_ckpt(src)


def test_int8_loads_in_an_interpreter_that_has_not_imported_torchao(tmp_path):
    """The one this file could not catch in-process: torchao registers its
    tensor subclasses as weights_only safe globals ON IMPORT, so a bare
    torch.load of an int8 checkpoint in a FRESH interpreter raises
    UnpicklingError. Every test above passes either way, because importing
    this module already imported torchao -- so the regression only shows up
    from a clean process, which is how serve and bench actually start.
    """
    import subprocess
    import sys

    src, out = tmp_path / "fp32.pth", tmp_path / "int8.pth"
    _fp32_ckpt(src)
    quantize(src, out)

    root = str(pathlib.Path(__file__).resolve().parent.parent)
    script = (
        "import sys; sys.path.insert(0, %r)\n"
        "import torch\n"
        "from quantize_serving_ckpt import load_checkpoint\n"
        "assert 'torchao' not in sys.modules, 'torchao imported too early to prove anything'\n"
        "ck = load_checkpoint(%r)\n"
        "assert ck['meta']['quant'] == 'int8_wo'\n"
        "print('OK')\n"
    ) % (root, str(out))
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert proc.returncode == 0, f"clean-interpreter load failed:\n{proc.stdout}\n{proc.stderr[-1500:]}"
    assert "OK" in proc.stdout


@pytest.fixture
def no_torchao(monkeypatch):
    """Make `import torchao` fail the way it does under the SYSTEM python.

    Without this the misdiagnosis is invisible: with torchao installed the
    retry simply re-raises, so a plain test passes against the broken code and
    proves nothing (the receipt was taken under system python).
    """
    monkeypatch.setitem(sys.modules, "torchao", None)


def test_a_corrupt_file_is_not_blamed_on_torchao(tmp_path, no_torchao):
    """Receipt: garbage bytes reported "needs torchao to load (it carries
    quantized tensors)". The retry exists for ONE diagnosis; every other
    unpickling failure must reach the caller as itself."""
    from quantize_serving_ckpt import load_checkpoint

    bad = tmp_path / "garbage.pth"
    bad.write_bytes(b"not a torch file at all")
    with pytest.raises(Exception) as exc:
        load_checkpoint(bad)
    assert "torchao" not in str(exc.value), str(exc.value)[:200]


def test_an_unsafe_global_is_not_blamed_on_torchao(tmp_path, no_torchao):
    """Same receipt, second shape: a checkpoint carrying argparse.Namespace is
    refused by weights_only for its OWN reason, which the caller needs to see."""
    import argparse

    from quantize_serving_ckpt import load_checkpoint

    ck = tmp_path / "ns.pth"
    torch.save({"model_state_dict": {}, "config": argparse.Namespace(x=1)}, ck)
    with pytest.raises(Exception) as exc:
        load_checkpoint(ck)
    assert "torchao" not in str(exc.value), str(exc.value)[:200]
    assert "Namespace" in str(exc.value) or "argparse" in str(exc.value)


def test_int8_weights_disable_the_eyes_instead_of_killing_the_boot(tmp_path):
    """Receipt 2026-09-02: booting an int8 checkpoint with --eyes raised a bare
    RuntimeError at the projection graft ('Tensor' object has no attribute
    'qdata') OUTSIDE boot's degrade catch, taking the whole server down. An
    organ that cannot work steps aside; text serving never dies for an eye."""
    import serve_enigma as serve

    src, out = tmp_path / "fp32.pth", tmp_path / "int8.pth"
    _fp32_ckpt(src)
    quantize(src, out)
    int8_ck = torch.load(out, map_location="cpu", weights_only=True)
    reason = serve._eyes_blocked_by(int8_ck)
    assert reason and "int8" in reason, reason

    plain_ck = torch.load(src, map_location="cpu", weights_only=True)
    assert serve._eyes_blocked_by(plain_ck) is None, "fp32 must still get its eyes"


def test_quantize_refuses_existing_out(tmp_path):
    src, out = tmp_path / "fp32.pth", tmp_path / "int8.pth"
    _fp32_ckpt(src)
    out.write_bytes(b"x")
    with pytest.raises(SystemExit):
        quantize(src, out)


def test_quantize_refuses_an_out_inside_models(tmp_path, monkeypatch):
    """The served sft2 and the v8 rollback stay untouchable by CONTRACT."""
    import quantize_serving_ckpt as q

    src = tmp_path / "fp32.pth"
    _fp32_ckpt(src)
    models = tmp_path / "models"
    (models / "enigma_v2_sft2").mkdir(parents=True)
    monkeypatch.setattr(q, "MODELS_DIR", models.resolve())
    with pytest.raises(SystemExit) as exc:
        quantize(src, models / "enigma_v2_sft2" / "model.pth")
    assert "models" in str(exc.value)
