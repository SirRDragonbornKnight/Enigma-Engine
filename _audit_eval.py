"""Independent audit evaluator — trusts NO logged number.
Loads the live checkpoint, rebuilds the model from the checkpoint's OWN config,
and recomputes validation loss/ppl directly on the held-out tail of tokens.bin,
exactly mirroring pretrain_enigma.py's data path. Prints ground-truth step.
"""

import sys, json, math, time, statistics
from pathlib import Path
import numpy as np
import torch

ROOT = Path(r"C:\Users\SirKn\Enigma Engine")
sys.path.insert(0, str(ROOT))
TOKENS_BIN = ROOT / "data" / "pretrain" / "tokens.bin"
TOKENS_META = ROOT / "data" / "pretrain" / "tokens.json"
CKPT = ROOT / "models" / "enigma_pretrain_large" / "latest.pth"
HEADER_BYTES = 256
BLOCK = 1024
MICRO_BATCH = 12
GRAD_ACCUM = 16
TARGET_TOKENS = 5.66e10
TOKENS_PER_STEP = MICRO_BATCH * GRAD_ACCUM * BLOCK
TOTAL_STEPS = int(TARGET_TOKENS / TOKENS_PER_STEP)
ITERS = 200

torch.manual_seed(1234)
np.random.seed(1234)
device = "cuda" if torch.cuda.is_available() else "cpu"

meta = json.loads(TOKENS_META.read_text(encoding="utf-8"))
vocab = meta["vocab_size"]
total = meta["total_tokens"]

# load checkpoint (atomic_torch_save => always a complete file; retry on transient lock)
ck = None
for _ in range(5):
    try:
        ck = torch.load(CKPT, map_location="cpu")
        break
    except Exception as e:
        print(f"  retry load: {e}", flush=True)
        time.sleep(3)
assert ck is not None
step = int(ck.get("step", -1))

from enigma_engine.core.model import Enigma
from enigma_engine.core.model_presets import ForgeConfig

config = ForgeConfig.from_dict(ck["config"])
if BLOCK > config.max_seq_len:
    config.max_seq_len = BLOCK
config.dropout = 0.0
model = Enigma(config).to(device).eval()
missing, unexpected = model.load_state_dict(ck["model_state_dict"], strict=False)
real_missing = [k for k in missing if "freqs_cis" not in k and "causal_mask" not in k]
assert not unexpected and not real_missing, f"LOAD MISMATCH missing={real_missing} unexpected={unexpected}"
n_params = sum(p.numel() for p in model.parameters())

data = np.memmap(TOKENS_BIN, dtype=np.uint32, mode="r", offset=HEADER_BYTES)
n = len(data)
val_n = min(10_000_000, n // 100)
train_end = n - val_n


def get_val_batch():
    ix = np.random.randint(train_end, n - BLOCK - 1, size=MICRO_BATCH, dtype=np.int64)
    x = np.stack([np.asarray(data[i : i + BLOCK], dtype=np.int64) for i in ix])
    y = np.stack([np.asarray(data[i + 1 : i + 1 + BLOCK], dtype=np.int64) for i in ix])
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


losses = []
t0 = time.time()
with torch.no_grad():
    for _ in range(ITERS):
        X, Y = get_val_batch()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            _, loss = model(X, targets=Y)
        losses.append(float(loss.item()))
mean = sum(losses) / len(losses)
sd = statistics.pstdev(losses)
sem = sd / math.sqrt(len(losses))

print("=== INDEPENDENT AUDIT (recomputed, no log trusted) ===", flush=True)
print(f"PARAMS={n_params:,} ({n_params / 1e6:.3f}M)")
print(
    f"CONFIG dim={config.dim} layers={config.n_layers} heads={config.n_heads} "
    f"kv={config.n_kv_heads} diff_attn={config.use_differential_attn} "
    f"neftune={config.neftune_alpha} dropout={config.dropout}"
)
print(f"CKPT_STEP={step}")
print(
    f"PROGRESS step {step}/{TOTAL_STEPS:,} = {100 * step / TOTAL_STEPS:.2f}%  "
    f"tokens={step * TOKENS_PER_STEP / 1e9:.3f}B / {TARGET_TOKENS / 1e9:.1f}B"
)
print(f"VAL over {ITERS} iters = {ITERS * MICRO_BATCH * BLOCK:,} tokens of {val_n:,} held-out")
print(f"VAL_LOSS={mean:.4f}  SEM={sem:.4f}  PPL={math.exp(min(20, mean)):.3f}")
print(
    f"VAL_95CI loss=[{mean - 1.96 * sem:.4f},{mean + 1.96 * sem:.4f}] "
    f"ppl=[{math.exp(mean - 1.96 * sem):.3f},{math.exp(mean + 1.96 * sem):.3f}]"
)
print(f"RANDOM_BASELINE ln(V={vocab})={math.log(vocab):.4f}  ppl={vocab}")
print(f"CORPUS total={total:,} memmap_n={n:,} MATCH={total == n}")
print(f"eval_wall={time.time() - t0:.1f}s device={device}")
