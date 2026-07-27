# Model Sizes

How the architecture presets work in Enigma Engine.

---

## Picking a Size (CLI)

Pretraining takes a preset name via `--size`:

```
python pretrain_enigma.py --size large
```

The production Enigma (182M params, the model behind
`models/enigma_dpo/model.pth`) was pretrained with the `large` preset.
SFT and DPO inherit the architecture from the checkpoint passed via
`--init`, so size is only chosen at pretrain time.

---

## Available Presets

Defined in `enigma_engine/core/model_presets.py` (`MODEL_PRESETS`).
Parameter counts are approximate and depend on vocab size.

| Preset | ~Params | Hardware note (from the preset descriptions) |
|--------|---------|----------------------------------------------|
| pi_zero | ~500K | Raspberry Pi Zero 2W -- needs <1 GB RAM |
| nano | ~1M | Microcontrollers -- needs <1 GB |
| tiny | ~5M | Edge devices -- needs <1 GB |
| small | ~27M | Entry GPU -- needs ~1 GB VRAM |
| medium | ~85M | Mid-range GPU -- needs ~2.5 GB VRAM |
| base | ~120M | Mid-range GPU -- needs ~3 GB VRAM |
| large | ~182M (the production Enigma) | Good GPU -- needs ~6 GB VRAM |
| v2_deep_186m | ~186M (28L x 768d) | The v2 deep-thin lineage; 5090-class for pretrain |
| v2_deep_238m | ~238M (20L x 1024d) | v2 candidate, wall-clock-optimal on the 5090 |
| v2_deep_542m | ~542M (30L x 1280d) | v2 candidate, largest the 5090 sanely trains (needs grad-ckpt) |
| xl | ~600M | RTX 4090 / 16 GB+ GPU -- needs ~12 GB VRAM |
| xxl | ~1.5B | RTX 4090 / 32 GB+ GPU -- needs ~34 GB VRAM |
| huge | ~3B | Multi-GPU / 48 GB+ -- needs ~50 GB VRAM |
| giant | ~7B | Multi-GPU / A100 -- needs ~77 GB VRAM |
| colossal | ~13B | Multi-node -- needs ~153 GB VRAM |
| titan | ~30B | Datacenter -- needs ~280 GB VRAM |
| omega | ~70B+ | Large cluster -- needs ~904 GB VRAM |

RoPE head dimensions are always even (required for rotary embeddings).

---

## Rules of Thumb

- Training needs roughly 3-4x the VRAM of inference at the same size.
- Bigger models learn more but train slower; at local scale, data
  quality and token count matter as much as parameter count.
- Use `--sanity` to confirm a preset fits in memory before a long run.
