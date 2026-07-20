"""Training package.

The Forge-era dispatcher (schema/registry/dispatch + the monolithic
Trainer) was retired in the 2026-07-18 compression pass. What remains
is the vision-align trainer used by align_vision.py; the live text
paths are the bespoke root scripts (pretrain_enigma.py,
finetune_enigma.py, dpo_enigma.py).

No package-level re-exports: importing this package must not pull in
torch (same rule as ``enigma_engine/core/__init__.py``), and every
caller already imports the module directly (the lazy ``__getattr__``
shim that used to live here had zero consumers and was removed
2026-07-19 -- it would also have become ambiguous once the audio-align
twin lands). Import the trainer as:

    from enigma_engine.training.vision_align import Trainer, TrainingConfig
"""
