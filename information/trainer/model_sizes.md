# Model Sizes

How the model size system works in Enigma Engine.

---

## Creating a Model (GUI)

In the **MODELS** tab, type a number in the **Memory (GB)** field
and click CREATE. The engine auto-picks the largest architecture
preset that fits your memory budget.

| Memory (GB) | Auto-picks | ~Params | Train VRAM |
|-------------|-----------|---------|-----------|
| 0.5 | tiny | ~5M | ~0.5 GB |
| 1 | small | ~27M | ~1 GB |
| 4 | base | ~120M | ~3 GB |
| 8 | large | ~200M | ~6 GB |
| 12+ | xl | ~600M | ~12 GB |

The Memory field auto-detects your GPU VRAM (or system RAM).
You can change the number before clicking CREATE.

---

## CLI Model Size

From the command line, use `--model-size` with a preset name:

```
python run.py --train data/training.txt --model-size large
```

Available CLI presets: `pi_zero`, `nano`, `tiny`, `small`,
`medium`, `large`.

---

## Available Presets (internal)

These are the architecture presets the engine selects from
automatically. You don't need to pick these — the Memory (GB)
input handles it.

| Preset | Parameters | Best For |
|--------|-----------|----------|
| pi_zero | ~500K | Testing, Raspberry Pi Zero |
| nano | ~1M | Microcontrollers, basic tasks |
| tiny | ~5M | Edge devices, simple chat |
| small | ~27M | Entry GPU, learning |
| medium | ~85M | Mid-range GPU |
| large | ~200M | RTX 3080+ |
| xl | ~600M | RTX 4090/5090 |
| xxl | ~1.5B | Multi-GPU |
| huge | ~3B | Server GPU |
| giant | ~7B | Multi-node datacenter |
| colossal | ~13B | Distributed cloud |
| titan | ~30B | Full datacenter |
| omega | ~70B+ | Research frontier |

---

## How It Works

1. You type a Memory (GB) value (e.g. `16`)
2. The engine estimates training VRAM for each preset
3. It picks the largest preset that fits your budget
4. The model is created and shown on the MODELS tab

---

## Hardware Requirements

| Size | Minimum VRAM |
|------|-------------|
| Under 100M | CPU or any GPU |
| 100M - 500M | 4 GB VRAM |
| 500M - 2B | 8 GB VRAM |
| 2B - 7B | 16 GB VRAM |
| 7B - 13B | 24 GB+ VRAM |
| 13B+ | Multi-GPU / 48 GB+ |
