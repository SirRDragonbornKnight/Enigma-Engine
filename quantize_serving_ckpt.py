"""int8 weight-only quantization for a serving checkpoint -- and the load side
that keeps the resulting number honest.

Quantizing is the easy half. The trap is loading: serve and bench both build a
fresh fp32 Enigma and call load_state_dict, and an int8 tensor-subclass state
dict handed to that either crashes or -- worse, on some version pairs -- lands
as ordinary fp32 tensors. A "quantized" server that silently dequantized would
pass every identity check and every port probe while measuring nothing.

So the marker travels IN the checkpoint (meta["quant"] = "int8_wo"), the loader
quantizes the fresh model with the SAME config BEFORE load_state_dict, and it
ASSERTS afterwards that the linear weights really are torchao subclass tensors.
A checkpoint whose marker and tensors disagree fails loudly instead of serving.

    python quantize_serving_ckpt.py --in "...\\v2_serving_only.pth" \\
                                    --out "...\\v2_serving_int8.pth"

The eager, core-ATen path is the only one used: torchao's compiled kernels do
not build here (Windows, and this torch pair skips the cpp extensions), which
is exactly why the portable lane picked weight-only int8 over anything needing
a compiler.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch

from enigma_engine.core.model import Enigma
from enigma_engine.core.model_presets import ForgeConfig
from enigma_engine.core.safe_save import atomic_torch_save

ROOT = Path(__file__).resolve().parent
# The served weights and the v8 rollback live here. --out may never land inside
# it: this stays a CONTRACT, not a matter of the operator's care.
MODELS_DIR = ROOT / "models"
QUANT_MARKER = "int8_wo"
SERVING_KEYS = ("model_state_dict", "config", "step", "meta")


def _int8_config():
    from torchao.quantization import Int8WeightOnlyConfig

    return Int8WeightOnlyConfig()


def _is_quantized_weight(weight) -> bool:
    """True when *weight* is a torchao tensor subclass rather than plain fp32."""
    return type(weight).__name__ not in ("Parameter", "Tensor")


def quantized_linear_count(model) -> int:
    return sum(
        1 for m in model.modules()
        if isinstance(m, torch.nn.Linear) and _is_quantized_weight(m.weight)
    )


def apply_int8(model) -> None:
    """Quantize *model*'s linears in place, eager, weight-only."""
    from torchao.quantization import quantize_

    quantize_(model, _int8_config())


def load_checkpoint(path: str | Path) -> dict:
    """torch.load a serving checkpoint under weights_only, int8 included.

    An int8 state dict carries torchao tensor subclasses, and weights_only
    refuses globals it was never told about. torchao registers its own on
    IMPORT -- so a process that has not imported torchao cannot read a
    checkpoint this tool wrote, and fails with UnpicklingError (measured: a
    plain torch.load of an int8 file in a fresh interpreter). Importing torchao
    unconditionally would tax every fp32 boot, so the import happens only after
    a load actually fails, and only once.
    """
    path = Path(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        try:
            import torchao  # noqa: F401  -- registers torchao's safe globals
        except ImportError:
            raise SystemExit(
                f"REFUSED: {path} needs torchao to load (it carries quantized tensors) "
                "and torchao is not installed in this interpreter."
            ) from None
        return torch.load(path, map_location="cpu", weights_only=True)


def load_serving_ckpt(path: str | Path):
    """Load a serving checkpoint into a ready model. Returns (model, ck).

    A plain fp32 checkpoint takes exactly the path it always took. A checkpoint
    marked int8_wo gets its fresh model quantized BEFORE load_state_dict, then
    verified: weights that came back plain mean the state dict silently
    dequantized, and that is a hard failure, never a quiet fp32 server.
    """
    path = Path(path)
    ck = load_checkpoint(path)
    if not (isinstance(ck, dict) and "model_state_dict" in ck and "config" in ck):
        raise SystemExit(f"{path} is not an Enigma checkpoint (need model_state_dict + config)")
    model = Enigma(ForgeConfig.from_dict(ck["config"]))
    prepare_for_state_dict(model, ck)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    assert_quant_applied(model, ck, path)
    return model.eval(), ck


def prepare_for_state_dict(model, ck: dict) -> bool:
    """Quantize *model* when *ck* is marked int8, BEFORE its state dict lands.

    Serve calls this instead of load_serving_ckpt because its eyes graft mutates
    the config between construction and load; sharing this primitive is what
    keeps ONE int8 rule instead of two that drift.
    """
    if (ck.get("meta") or {}).get("quant") != QUANT_MARKER:
        return False
    apply_int8(model)
    return True


def assert_quant_applied(model, ck: dict, path=None) -> None:
    """Fail loudly when a checkpoint marked int8 loaded as plain fp32."""
    if (ck.get("meta") or {}).get("quant") != QUANT_MARKER:
        return
    if quantized_linear_count(model) == 0:
        raise SystemExit(
            f"REFUSED: {path or 'checkpoint'} is marked quant={QUANT_MARKER} but its linear "
            "weights loaded as plain fp32 -- the state dict dequantized silently. "
            "Serving this would report an int8 number measured on fp32 weights."
        )


def quantize(src: str | Path, out: str | Path) -> dict:
    """Write an int8 weight-only copy of the serving checkpoint at *src*."""
    src, out = Path(src), Path(out)
    if not src.is_file():
        raise SystemExit(f"REFUSED: --in {src} does not exist")
    if out.exists():
        raise SystemExit(
            f"REFUSED: {out} already exists -- model artifacts are versioned, "
            f"never rebuilt in place. Name a NEW --out."
        )
    try:
        inside_models = out.resolve().is_relative_to(Path(MODELS_DIR).resolve())
    except OSError:
        inside_models = False
    if inside_models:
        raise SystemExit(
            f"REFUSED: --out {out} is inside {MODELS_DIR} -- the served checkpoint and "
            "the rollback are not overwritable by this tool. Write elsewhere."
        )

    ck = load_checkpoint(src)
    if not (isinstance(ck, dict) and "model_state_dict" in ck and "config" in ck):
        raise SystemExit(f"REFUSED: {src} is not an Enigma checkpoint")
    if (ck.get("meta") or {}).get("quant"):
        raise SystemExit(f"REFUSED: {src} is already quantized (meta quant={ck['meta']['quant']})")

    model = Enigma(ForgeConfig.from_dict(ck["config"]))
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()
    apply_int8(model)
    n = quantized_linear_count(model)
    if n == 0:
        raise SystemExit("REFUSED: quantize_() changed no linear weights -- nothing was quantized")

    meta = dict(ck.get("meta") or {})
    meta["quant"] = QUANT_MARKER
    payload = {"model_state_dict": model.state_dict(), "config": ck["config"], "meta": meta}
    if "step" in ck:
        payload["step"] = ck["step"]
    atomic_torch_save({k: payload[k] for k in SERVING_KEYS if k in payload}, out)
    return {"quantized_linears": n}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--in", dest="src", required=True, help="serving-only .pth to read (never modified)")
    p.add_argument("--out", dest="out", required=True, help="int8 .pth to write; must not exist, must be outside models/")
    args = p.parse_args()

    info = quantize(args.src, args.out)
    src_mb = Path(args.src).stat().st_size / (1024 * 1024)
    out_mb = Path(args.out).stat().st_size / (1024 * 1024)
    print(f"read  {args.src}  ({src_mb:,.1f} MB)")
    print(f"  quantized linears: {info['quantized_linears']} (int8 weight-only, eager)")
    print(f"wrote {args.out}  ({out_mb:,.1f} MB)")
    print(f"  saved {src_mb - out_mb:,.1f} MB ({100 * (1 - out_mb / src_mb):.1f}% of the input)")


if __name__ == "__main__":
    main()
