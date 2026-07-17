"""The training arsenal: LR schedule + optimizers shared by the pretrain and
SFT passes.

This lives in the package (not in ``pretrain_enigma.py``) so both
``pretrain_enigma.py`` and ``finetune_enigma.py`` import it as a peer —
a training SCRIPT should not import from another training script.

Defaults reproduce the live 182M lineage EXACTLY: ``get_lr`` cosine math is
byte-identical to the recorded schedule, and ``build_optimizer("adamw", ...)``
keeps the same param grouping and parameters()-iteration order so its
state_dict still fits the lineage's checkpoints. ``muon``/``wsd`` are
future-run knobs — never switch either on an existing lineage.
"""

from __future__ import annotations

import math

import torch


def get_lr(
    step: int,
    warmup: int,
    total: int,
    peak: float,
    min_ratio: float = 0.1,
    schedule: str = "cosine",
    decay_frac: float = 0.1,
) -> float:
    """Linear warmup, then either cosine decay to ``min_ratio * peak`` (default —
    the live run's recorded schedule; its math is byte-identical to the original)
    or ``wsd``: hold at peak, then LINEAR decay to ZERO over the last
    ``decay_frac`` of the run. WSD/D2Z beats cosine at high tokens-per-param and,
    unlike cosine, lets a "finished" run keep training from the stable phase —
    the multi-epoch lever."""
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    if schedule == "wsd":
        decay_start = int(total * (1.0 - decay_frac))
        if step < decay_start:
            return peak
        if step >= total:
            return 0.0
        return peak * (total - step) / max(1, total - decay_start)
    if step >= total:
        return peak * min_ratio
    prog = (step - warmup) / max(1, total - warmup)
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))


def _newton_schulz5(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Quintic Newton-Schulz iteration that approximately orthogonalizes a 2-D
    update (drives all singular values toward 1). Runs in bfloat16 — the
    iteration is stable there and it is the fast path on the GPU. Coefficients
    are the modded-nanogpt/Moonlight standard."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.to(torch.bfloat16)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.mT
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        s = x @ x.mT
        x = a * x + (b * s + c * (s @ s)) @ x
    if transposed:
        x = x.mT
    return x.to(g.dtype)


class Muon(torch.optim.Optimizer):
    """Muon for 2-D weight matrices (Moonlight/Kimi-K2 variant, arXiv:2502.16982):
    SGD-momentum whose update is orthogonalized by Newton-Schulz, with decoupled
    weight decay and the 0.2*sqrt(max(shape)) RMS match so one --lr serves both
    Muon and the aux AdamW. Embeddings/heads/1-D gains do NOT belong here —
    route them to AdamW (see build_optimizer)."""

    def __init__(self, params, lr: float = 6e-4, momentum: float = 0.95, weight_decay: float = 0.1, ns_steps: int = 5):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim > 2:  # safety: fold trailing dims (none in our model)
                    g = g.reshape(g.size(0), -1)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(g)
                u = g.add(buf, alpha=group["momentum"])  # nesterov blend
                u = _newton_schulz5(u, group["ns_steps"])
                p.mul_(1.0 - group["lr"] * group["weight_decay"])
                p.add_(u.reshape(p.shape), alpha=-group["lr"] * 0.2 * math.sqrt(max(p.shape)))
        return loss


class CompositeOptimizer:
    """Muon(body) + AdamW(embeddings/1-D) behind one optimizer face. Duck-typed
    where the loop needs it: param_groups (lr updates + grad clip + GradScaler
    unscale), step/zero_grad, state_dict/load_state_dict. The state format is
    tagged so loading it into a plain-AdamW run (or vice versa) fails loudly."""

    def __init__(self, opts):
        self.opts = list(opts)

    @property
    def param_groups(self):
        return [g for o in self.opts for g in o.param_groups]

    def step(self, closure=None):
        for o in self.opts:
            o.step()

    def zero_grad(self, set_to_none: bool = True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"composite": [o.state_dict() for o in self.opts]}

    def load_state_dict(self, sd):
        if "composite" not in sd:
            raise ValueError("optimizer state is not composite -- this checkpoint was saved by a different --optimizer")
        for o, s in zip(self.opts, sd["composite"]):
            o.load_state_dict(s)


def build_optimizer(model: torch.nn.Module, kind: str, lr: float, weight_decay: float):
    """``adamw`` (default) reproduces the live run's optimizer EXACTLY — same
    groups, same parameters()-iteration order, so its state_dict keeps fitting
    the 51k lineage. ``muon`` routes the 2-D body matrices to Muon and keeps
    embeddings (tied head) + 1-D gains on AdamW — for FUTURE runs only.
    Weight decay stays on >=2-D tensors only (norm gains are never decayed)."""
    decay, no_decay, body = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2:
            no_decay.append(p)
        elif kind == "muon" and "tok_embeddings" not in name and "output" not in name:
            body.append(p)
        else:
            decay.append(p)
    if kind == "muon":
        optim = CompositeOptimizer(
            [
                Muon(body, lr=lr, weight_decay=weight_decay),
                torch.optim.AdamW(
                    [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
                    lr=lr,
                    betas=(0.9, 0.95),
                ),
            ]
        )
        print(
            f"optim: Muon on {len(body)} body matrices (NS5, rms-matched) + AdamW on "
            f"{len(decay)} embedding + {len(no_decay)} 1-D tensors",
            flush=True,
        )
        return optim
    optim = torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
        betas=(0.9, 0.95),
    )
    print(f"optim: weight-decay on {len(decay)} tensors, none on {len(no_decay)}", flush=True)
    return optim
