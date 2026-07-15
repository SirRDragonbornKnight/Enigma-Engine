# Ultra review - 2026-07-12 (branch engine-refactor @ c1f827e1)

> **RESOLUTION 2026-07-13:** all 3 CRITICALS below are FIXED (full suite 449 green, ruff clean).
> #1 prompt-mask off-by-one -> all four preference encoders (DPO/SimPO/KTO/ORPO) now share
>    `Trainer._encoded_prompt_len`, which strips the trailing EOS; regression test in
>    `tests/test_preference_encoding.py`.
> #2 extend_length.ps1 clobber -> the script now resumes an existing block-2048 run; and
>    `pretrain_enigma.py` resolves a bare-`--resume` `--out` to the checkpoint's own directory
>    (also closes finding #4's wrong-lineage write).
> #3 LoRA offload crash -> `core/lora_utils.py` passes `cpu=(device == "cpu")`, syncs the batch
>    device to `accelerator.device`, and warns that optimizer-state offload is a no-op on the
>    plain-AdamW path; regression test in `tests/test_lora_offload.py`.
> Majors/minors below are UNTRIAGED and still open.

Whole-repo multi-agent review ("/code-review ultra", local max-effort fallback).
Method: 19 finder angles (12 subsystem + removed-behavior/reuse/simplification/
efficiency/altitude/conventions + 2 gap sweeps), one adversarial verifier per
deduped candidate (CONFIRMED / PLAUSIBLE / REFUTED). Finders were instructed NOT
to re-report the CODE_REVIEW.md closed backlog or known-deferred items.

Stats: 94 candidates -> 93 after dedup -> 74 survived verification + 6 sweep
survivors = 80 findings; 1 refuted; 18 candidates lost their verifier to the
session usage limit (listed unverified in the appendix).

## Coverage gaps (session limit killed these finders - re-run to close)

- training/training.py lines 1-2600 (core train loop first half)
- kv_cache.py, chat_format.py, memory_store.py, safe_save.py, model_registry.py, commands.py, plugin_loader.py, mod_tools.py, calculator.py (dedicated subsystem pass; partial coverage via other angles)
- rl_training.py + reward_functions.py (dedicated pass)
- collect_finetuning/distill/vision/search + create_smoke_test_data (dedicated pass)

## Findings (severity-ranked; top 15 hand-verified line-by-line)

### 1. enigma_engine/training/training.py:3929 - [critical/CONFIRMED/correctness] (found by: training-loop-2)

All four preference encoders treat len(prompt_ids) as the prompt-prefix length inside the full encoding, but encode() appends a trailing EOS to the prompt-only encoding, so prompt_len over-counts by one and the FIRST completion token is always masked out of the preference log-probability.

**Failure scenario:** prompt encode gives [BOS, p1..pk, EOS] (len k+2) while chosen encode gives [BOS, p1..pk, c1..cm, EOS]; masking the first k+2 positions labels c1 as -100. For a 1-token completion the only scored token is the terminal EOS, so DPO compares p(EOS)-vs-p(EOS) and the pair carries zero preference signal; pairs like chosen='Yes.' vs rejected='No.' (differing mainly in the first token) train on almost nothing. Clones: train_simpo line 4356, train_kto _encode_sample line 4540, train_orpo resp_mask lines 4785/4801 (mask starts at prompt_len-1 = first completion token, which predicts the second, skipping c1 the same way). SimPO's length normalization also divides by the wrong count (m-1 tokens).

```
        prompt_len = len(prompt_ids)
```

### 2. extend_length.ps1:59 - [critical/CONFIRMED/correctness] (found by: train-scripts)

Re-running extend_length.ps1 after a pause always warm-starts from step 0 via --init-from and progressively overwrites the partially-trained 2048 run's checkpoints, despite the header advertising the same stop/resume on/off workflow as resume_training.ps1; there is no resume path and no guard against an existing models/enigma_pretrain_2048/latest.pth.

**Failure scenario:** Day 1: user runs extend_length.ps1, trains to step 20,000, asks Claude to stop at a safe checkpoint (the documented workflow). Day 2: user runs extend_length.ps1 again to continue. The script's only guards are already-running and source-exists; it launches pretrain_enigma.py with --init-from (fresh step 0). At step 250 the first save rotates the step-20,000 latest.pth to prev.pth and replaces it; at step 500 the second save destroys prev.pth too. All prior 2048-run progress is silently gone, ~2.5 GPU-days lost per the script's own estimate. pretrain_enigma.py's --init-from guard (line 191) only refuses when --out equals the SOURCE checkpoint's directory, so nothing stops this.

```
$trainArgs = '--init-from models/enigma_pretrain_large/model.pth --out models/enigma_pretrain_2048 --block 2048 --micro-batch 6 --grad-accum 16 --tokens 10e9 --lr 3e-4 --warmup 300 --weight-decay 0.1 --grad-clip 1.0 --optimizer adamw --schedule wsd --wsd-decay-frac 0.1 --dropout 0.0 --no-diff-attn --no-grad-ckpt --no-compile --archive-every 25000'
```

### 3. enigma_engine/core/lora_utils.py:930 - [critical/CONFIRMED/correctness] (found by: adapt-merge)

LoraTrainer.train() passes offload_optimizer as Accelerator(cpu=...), forcing the whole model onto CPU while every batch is unconditionally moved to 'cuda', so the default config crashes on the first forward when accelerate and CUDA are both present.

**Failure scenario:** OffloadConfig defaults are cpu_offload=True and offload_optimizer=True (training/dispatch.py mode 'lora' passes no offload_config, so it gets these defaults). With accelerate installed and a GPU present, train() builds Accelerator(cpu=True) which prepares the model on CPU, then line 987 does input_ids.to('cuda') because device = 'cuda' if torch.cuda.is_available(); the first fwd_model(inputs, ...) raises RuntimeError: expected all tensors on the same device. Semantically, 'offload optimizer states to CPU' is also implemented as 'run all training on CPU', which is never what the flag documents.

```
                cpu=self.offload_config.offload_optimizer,
```

### 4. pretrain_enigma.py:380 - [major/CONFIRMED/correctness] (found by: removed-behavior)

A bare --resume of a run whose directory does not match models/enigma_pretrain_<size> (e.g. the new --init-from warm-start run in models/enigma_pretrain_2048) silently writes its checkpoints into the wrong lineage's directory, because --out is neither recorded in the checkpoint schedule nor derivable from the checkpoint path.

**Failure scenario:** extend_length.ps1 launches the block-2048 warm-start into models/enigma_pretrain_2048 (commit 7d3fcc6 forces --out at warm-start time). The user later pauses and resumes it with the documented bare pattern from KNOWN_ISSUES #1 ('a bare --resume models/.../latest.pth restores tokens/lr/warmup/batch/etc.'): `python pretrain_enigma.py --resume models/enigma_pretrain_2048/latest.pth`. The recorded schedule restores block/lr/tokens, but out falls back to ROOT/models/enigma_pretrain_base (args.size default 'base'). models/enigma_pretrain_base EXISTS and holds latest.pth + model.pth of a different lineage: the first --save-every save rotates that lineage's latest.pth to prev.pth and overwrites it with 2048-run weights, while models/enigma_pretrain_2048 silently stops receiving checkpoints. The immutable-lineage guard added for --init-from (requires explicit --out, refuses the source dir) is not re-established on the resume path, even though the deleted comment in commit 7d3fcc6 named exactly this hazard class ('the old behavior silently started a FRESH run into the same --out directory').

```
    out = Path(args.out) if args.out else ROOT / "models" / f"enigma_pretrain_{args.size}"
```

### 5. enigma_engine/training/training_queue.py:270 - [major/CONFIRMED/correctness] (found by: removed-behavior)

start() called while a previous loop thread is still inside _execute_job (after stop()) spawns a superseding loop that claims the NEXT pending job immediately, so two training jobs run concurrently — the exact proven failure commit c6b93e4 claimed to close is narrowed to a window but not eliminated.

**Failure scenario:** Queue holds jobs #1 and #2. start() runs job #1 (a long training run). User calls stop() — docstring says 'Stop the queue after the current job finishes', and it only sets flags, it does not join the thread. User calls start() again while job #1 is still training: the guard at line 270 checks only `self._running` (now False), so a new loop thread is created and `self._thread` is reassigned. The old thread is blocked inside `self._execute_job(job1)` and only evaluates the `self._thread is not me` supersession check (line 317) BETWEEN jobs. The new loop's `_claim_next_pending()` skips job #1 (status 'running') and claims job #2 — jobs #1 and #2 now train concurrently on the same GPU (VRAM OOM, interleaved checkpoint writes). The original CODE_REVIEW backlog entry was 'stop() then start() spawns a second loop thread ... two jobs run concurrently (proven)'; the fix added supersession and atomic claim but start() still neither joins nor refuses a live predecessor thread.

```
            if self._running:
```

### 6. make_sft_data.py:655 - [major/CONFIRMED/correctness] (found by: collectors-2)

Training data never contains the memory block ('Things you remember:') combined with a tool-spec block in one system message, but serve's _with_context joins them with '\n\n' whenever a memory hit coincides with offered tools, producing a system-message shape the model never saw -- the exact bug class this dataset was built to fix.

**Failure scenario:** An OpenAI client sends tools on every request (client tools are always honored, serve_enigma.py:492-493), or _looks_memorable/_looks_arithmetic fires alongside a BM25 memory hit; serve renders system = 'Things you remember:\n- ...' + '\n\n' + 'You are Enigma...Available tools:...'. gen_memory_read_examples emits the memory block ALONE as system (no preamble, no tools) and gen_tool_examples emits preamble+tools ALONE, so the combined shape is 0% of training. The repo's own measured history (2026-07-06 eval: she ignored the untrained memory-block shape; serve comment: 'a system message that OPENS with "Available tools:" is a shape the model never saw') says this 182M model fails on unseen block shapes -- so memory recall regresses precisely when a tool is also offered.

```
            block = "Things you remember:\n" + "\n".join(f"- {ln}" for ln in lines)
```

### 7. enigma_engine/core/lora_utils.py:855 - [major/CONFIRMED/correctness] (found by: removed-behavior)

LoraTrainer.__init__ calls model.gradient_checkpointing_enable() whenever OffloadConfig.gradient_checkpointing is set (its DEFAULT is True), without the kv-share guard, so after commit e7c50dc made that method raise ValueError on kv_share_groups models, constructing a LoraTrainer for such a model hard-crashes by default — the graceful skip-with-warning invariant was re-established only in training.py.

**Failure scenario:** Build an Enigma model with config.kv_share_groups > 0 and instantiate LoraTrainer(model, ...) with the default OffloadConfig (gradient_checkpointing=True, line 362). Line 855 calls `self.model.gradient_checkpointing_enable()`, which since e7c50dc raises `ValueError("gradient checkpointing cannot be combined with kv_share_groups...")` (model.py lines 341-345). LoraTrainer.__init__ dies before training starts, with no way to LoRA-train a kv-shared model except knowing to pass a non-default OffloadConfig. The same commit taught Trainer (training/training.py lines 1253-1259, _kv_share_blocks_checkpointing) to log a warning and continue without checkpointing; LoraTrainer — repaired in the same dormant-code backlog pass — was not given the equivalent guard.

```
self.model.gradient_checkpointing_enable()
```

### 8. enigma_engine/training/training.py:3979 - [major/CONFIRMED/correctness] (found by: training-loop-2)

_generate_online_dpo_pairs generates from a prompt encoded with encode()'s trailing EOS un-stripped, so the model sees a finished document -- the exact documented serve_enigma EOS bug, un-fixed in this sibling call site (used by online DPO and train_rest).

**Failure scenario:** KNOWN_ISSUES.md item 8 and the closed 'serve_enigma.py EOS bug' state that encode() brackets text as [BOS]...[EOS] and generation callers MUST strip the trailing EOS or the model 'sees a finished document and replies with EOS/new-document'. Here the prompt ids feed generation unmodified, so with the project tokenizer the very first sampled token is conditioned on EOS: responses are empty/new-document garbage, reward_fn scores noise, and every online-DPO / ReST preference pair is degenerate (empty responses are then silently dropped by _encode_dpo_pair, so ReST rounds train on nothing or on junk). Additionally `ids` grows by up to max_new_tokens with no cap at model max_seq_len, so a near-max-length prompt overruns the RoPE table mid-generation.

```
                encoded = self.tokenizer.encode(f"User: {prompt_text}\nAssistant: ")
```

### 9. serve_enigma.py:235 - [major/CONFIRMED/correctness] (found by: serve)

serve never passes top_k or repetition_penalty to model.generate_stream, so every request silently gets top_k=50 truncation and repetition_penalty=1.1 — sampling modifiers the OpenAI-compatible API neither exposes nor documents, and the penalty applies to PROMPT tokens too.

**Failure scenario:** generate_stream defaults are top_k=50, repetition_penalty=1.1 (model.py:839-841), and sample_next_token penalizes every token in `generated`, which is seeded with the full prompt. A client requesting temperature=1.0, top_p=1.0 expecting the raw distribution gets a top-50-truncated, prompt-penalized one; in instruct mode the digits of the user's "12345 * 678" and the JSON punctuation saturating the tool-spec system prompt are all penalized 1.1x exactly when the model must echo them into a <|tool_call|> payload, biasing it toward malformed arguments. serve clamps temperature to >=1e-3 (line 314) so the greedy bypass in sample_next_token never engages and the penalty is always live, including for eval_behavior's near-greedy scoring runs.

```
                for t in model.generate_stream(
```

### 10. dpo_enigma.py:148 - [major/CONFIRMED/correctness] (found by: train-scripts)

The fp16 autocast fallback path has no GradScaler — pretrain_enigma.py and finetune_enigma.py both create one for exactly this case — so on a CUDA GPU without bf16 support DPO backpropagates unscaled fp16 gradients that underflow to zero.

**Failure scenario:** Run dpo_enigma.py on any pre-Ampere CUDA GPU (torch.cuda.is_bf16_supported() False): amp_dtype becomes torch.float16 and loss.backward() at line 200 runs with no scaler (grep confirms zero GradScaler/scaler references in this file). DPO gradients at lr 2e-6 are tiny; fp16's ~6e-5 normal floor flushes most of them to zero in the backward through the fp16-autocast regions, so the 'alignment polish' trains partially or not at all while loss/accuracy print plausibly (the loss itself is fp32 via the .float() log_softmax). Silent — no warning, no crash. Its siblings guard this: pretrain_enigma.py:351-352 and finetune_enigma.py:290-291 both build torch.amp.GradScaler(enabled=use_scaler).

```
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
```

### 11. enigma_engine/core/lora_utils.py:1412 - [major/CONFIRMED/correctness] (found by: adapt-merge)

LoRAAdapterManager.merge_into_base() feeds raw saved parameter VALUES (what save()/create() wrote: param.data of every requires_grad param) into merge_lora_weights(), which treats them as pre-multiplied DELTAS and adds them onto the live parameters, doubling every trainable weight instead of merging.

**Failure scenario:** mgr.save('coding', model) on a non-PEFT model (e.g. DoRA-wrapped via apply_dora, or a plain model) stores the current values of lora_a/lora_b/m (or all params). mgr.merge_into_base('coding', model): model has no merge_and_unload, so the manual path runs state_dict[key] = base + lora_weight, where base and lora_weight are the SAME value -> every matched parameter becomes 2x its trained value (DoRA magnitude vector m doubled distorts all outputs), and nothing is ever folded into the base weights (no W += scale*B@A ever happens). The shape check passes because the keys name the adapter params themselves, so the guard added for the earlier no-op bug does not catch this inversion.

```
        merge_lora_weights(model, weights)
```

### 12. enigma_engine/training/training_evaluation.py:357 - [major/CONFIRMED/correctness] (found by: training-aux)

run_golden_eval greedily generates from tokenizer.encode(prompt) output without stripping the trailing EOS, so the model sees a finished document and emits EOS on the first step, producing empty responses and false regressions.

**Failure scenario:** Trainer with config.golden_eval_path set calls run_golden_eval (training.py lines 2372/2587/3178). AdvancedBPETokenizer.encode(prompt) defaults add_special_tokens=True and appends eos_token_id (advanced_tokenizer.py lines 401-403). run_golden_eval sets generated=list(token_ids) (line 366) with that trailing EOS, then the greedy loop's first forward predicts EOS (the natural post-EOS token) and breaks immediately (lines 374-377), yielding an empty response. Every golden case with a real answer (e.g. expected ['Paris']) scores as failed, reporting a catastrophic false regression. sample_enigma.py line 64 strips this EOS; run_golden_eval does not.

```
                        tokens = tokenizer.encode(prompt)
```

### 13. enigma_engine/core/lora_utils.py:1331 - [major/CONFIRMED/correctness] (found by: adapt-merge)

LoRAAdapterManager.create() never attaches LoRA to the model (config rank/alpha only go into meta.json), so on an unwrapped model 'the adapter' it saves is a full snapshot of every parameter — a full-model copy masquerading as a rank-8 adapter.

**Failure scenario:** mgr = LoRAAdapterManager(); mgr.create('coding', model, LoraConfig(rank=16)) on a plain Enigma model (the docstring's exact usage): all params have requires_grad=True, so adapter.pth = the entire model (~hundreds of MB for the 182M model instead of a few MB), meta.json records rank=16 that corresponds to nothing, and a later switch() 'adapter swap' silently overwrites every base weight with the snapshot. The class's stated purpose ('base model stays frozen and each skill lives in a separate adapter file') is only true if the caller separately remembers to wrap with create_lora_model/apply_dora, which create() neither does nor verifies.

```
            if param.requires_grad:
```

### 14. enigma_engine/core/model_components.py:624 - [major/CONFIRMED/correctness] (found by: core-model)

The non-SDPA attention fallback (CPU/MPS, and any CUDA model with use_differential_attn) builds a top-left (T,T) causal mask for a rectangular cached decode where scores are (B,H,T,Tk), Tk>T — it never received the bottom-right-alignment fix the SDPA path got.

**Failure scenario:** Model on CPU (SDPA path is gated on x.is_cuda at line 575), KV cache warm, then a multi-token continuation — e.g. forward(new_turn_ids, use_cache=True, start_pos=8) after an 8-token prefill. Reproduced live on a tiny Enigma: `RuntimeError: The size of tensor a (12) must match the size of tensor b (4) at non-singleton dimension 3`. Same break for differential-attention models on CUDA, which are forced onto this branch. generate()/generate_stream avoid it only because they decode one token at a time; the public forward(start_pos=...) contract (chunked prefill, prefix-cache continuation) is broken on this path.

```
torch.full((T, T), float("-inf"), device=scores.device, dtype=scores.dtype),
```

### 15. serve_enigma.py:684 - [major/CONFIRMED/correctness] (found by: serve)

A model tool call whose JSON parses to a valid object but lacks a 'name' key (e.g. {"tool": "calculate", ...}) vanishes silently: it is neither executed, nor surfaced as a tool_call, nor collected into raw_all, because parse_assistant_ids only attaches 'raw' when JSON parsing fails outright.

**Failure scenario:** Model emits <|tool_call|>{"tool": "calculate", "arguments": {"expression": "7*8"}}<|/tool_call|>. parse_assistant_ids (chat_format.py:306) returns {"name": None, "arguments": {...}} with NO 'raw' key. In serve, raw_all filter `not c.get("name") and c.get("raw")` skips it, _openai_tool_calls filters it out, _loop_on_builtins sees no named calls — the model's action disappears entirely from the response (finish_reason "stop", empty content), the exact failure class the raw_all mechanism was built to prevent (same defect on the stream path at line 610).

```
        raw_all += [c["raw"] for c in parsed if not c.get("name") and c.get("raw")]
```

### 16. enigma_engine/core/adaptive_trainer.py:159 - [major/CONFIRMED/correctness] (found by: adapt-merge)

TrainingPlan.decide_action() escalates difficulty when the student FAILS a stage (score < 7) and never raises it on success, inverting the adaptive heuristic the module documents ('adjusts difficulty ... based on current ability').

**Failure scenario:** Student scores 4/10 on 'basics' at 'simple' difficulty -> decide_action returns 'retry' AND bumps current_difficulty to 'medium', so the failing student's next lesson set is generated from HARDER prompts (build_adaptive_prompt uses current_difficulty), compounding failure until max_retries exhausts and it advances anyway. Conversely a student scoring 9/10 advances immediately and advance_stage() resets difficulty to 'simple', so the 'medium'/'advanced' prompt banks are only ever reachable through failure — the opposite of curriculum adaptation.

```
self.current_difficulty = DIFFICULTY_LEVELS[idx + 1]
```

### 17. enigma_engine/core/model.py:729 - [major/CONFIRMED/correctness] (found by: core-model)

forward_multimodal never drops the intra-forward `_shared_kv` after its layer loop, so with kv_share_groups>0 leader layers keep graph-attached K/V activations pinned on the module — the exact regression the cleanup at lines 508-510 of the main forward() was added to fix.

**Failure scenario:** kv-shared model + vision training (training/training.py:5217 calls forward_multimodal every step): after each step the last batch's K/V graph stays alive on every leader Attention (VRAM held across steps), and deepcopy(model) — used to build DPO/KTO reference models — fails. Reproduced live: after forward_multimodal, leaders show _shared_kv pinned and `copy.deepcopy(m)` raises `RuntimeError: Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol`; after plain forward() the same model deep-copies fine.

```
        for layer in self.layers:
```

### 18. enigma_engine/training/training.py:5753 - [major/CONFIRMED/correctness] (found by: training-loop-2)

train_audio ignores config.max_grad_accumulation entirely -- it zero-grads and steps the optimizer on every single sample -- the exact sibling of the V-1 vision gradient-accumulation fix that the fix pass missed.

**Failure scenario:** CODE_REVIEW's V-1 fix comment in train_vision (line 5168) says 'Other train_* methods honor config.max_grad_accumulation; vision must too', yet train_audio still does optimizer.zero_grad() at the top of the per-sample loop and optimizer.step()+scheduler.step() after every backward. A user who sets max_grad_accumulation=8 to emulate batch-8 audio training silently gets batch-1 updates with 8x the intended optimizer/scheduler step rate; loss scaling and effective LR schedule diverge from every other train_* path with no warning.

```
optimizer.zero_grad()
```

### 19. enigma_engine/core/lora_utils.py:417 - [major/CONFIRMED/correctness] (found by: adapt-merge)

DoRALinear creates lora_a/lora_b on the default device and dtype (torch.zeros with no device=/dtype=), instead of the wrapped layer's, so applying DoRA to a model already on GPU (or in bf16/fp16) crashes or silently upcasts the forward.

**Failure scenario:** model.cuda(); apply_dora(model) — DoRALinear.__init__ keeps self.weight on cuda but self.lora_a/self.lora_b are CPU fp32; the first forward computes direction = self.weight + lora_update and raises RuntimeError: Expected all tensors to be on the same device (cuda vs cpu). On a bf16 model moved correctly, fp32 lora params promote the whole weight_dora computation to fp32 each forward (extra memory + dtype churn). m is derived from the weight so it inherits the right device, making the mismatch specific to the two LoRA matrices.

```
        self.lora_a = nn.Parameter(torch.zeros(rank, in_features))
```

### 20. enigma_engine/core/model_components.py:322 - [major/CONFIRMED/correctness] (found by: core-model)

Attention hard-caps the KV cache at MAX_CACHE_SEQ_LEN=4096 regardless of config.max_seq_len, so long-context presets (xxl/huge/giant/colossal/titan/omega with max_seq_len 8192-32768) silently lose their earliest context past 4096 during cached generation.

**Failure scenario:** create_model('xxl') (max_seq_len=8192, RoPE table sized 16384) then generate past 4096 total tokens with use_cache: KVCache._begin_update hits `end_pos > self.max_seq_len` and torch.rolls the cache left with only a log warning, evicting the oldest tokens — the model attends to a truncated, position-shifted history while the caller believes the full 8192-token window is live. The bottom-right-aligned decode mask and RoPE offsets assume the cache holds the true prefix, so outputs degrade silently rather than erroring.

```
            config.max_seq_len if hasattr(config, "max_seq_len") else self.MAX_CACHE_SEQ_LEN, self.MAX_CACHE_SEQ_LEN
```

### 21. enigma_engine/core/advanced_tokenizer.py:266 - [major/CONFIRMED/correctness] (found by: tokenizer-data)

AdvancedBPETokenizer.save() writes the Enigma "encoder" format without the merges, and load()'s "encoder" branch never reads merges, so a save->load round-trip loses all BPE merges and encode() silently degrades to character-level.

**Failure scenario:** t = AdvancedBPETokenizer(bpe_vocab_with_merges); t.save('x.json') -> file has "encoder"/"special_tokens"/"use_utf8_bytes" but no "merges". t2 = AdvancedBPETokenizer('x.json') enters the `if "encoder" in data` branch (line 186) which sets token_to_id but never assigns self.merges (stays []). t2.encode('information') then hits `if self.merges:` == False and falls to the char/byte fallback, emitting ['i','n','f','o',...] instead of the merged subwords t.encode produced -- a broken, silent vocab load/save asymmetry.

```
        data = {
```

### 22. enigma_engine/training/training.py:4830 - [major/CONFIRMED/correctness] (found by: training-loop-2)

train_orpo's scheduler.step() is dead code (self._setup_optimizer() at line 4751 just set self.scheduler = None and ORPO never builds one) and ORPO never increments self.state.step, so warmup/min_lr_ratio config is silently ignored and ORPO checkpoints record a stale step counter.

**Failure scenario:** An ORPO run trains at constant peak learning_rate with zero warmup (unlike train()/train_dpo which both build SequentialLR warmup+decay), risking early-step instability the guarded step() call was clearly meant to prevent; and because no code path in train_orpo does self.state.step += 1 (SimPO line 4437 and KTO line 4636 both do), `{stem}_best.pt` saved from ORPO stores step=0 (or a leftover count from a prior train() call on the same Trainer), corrupting the resume/schedule math of any later run that loads it.

```
if self.scheduler:
```

### 23. enigma_engine/core/progressive_growing.py:171 - [major/CONFIRMED/correctness] (found by: adapt-merge)

validate_growth() never checks vocab_size (or max_seq_len), and expand_model_weights() silently truncates embedding/predict-head rows via min() when the target vocab is smaller, shipping a 'grown' checkpoint that lost token embeddings.

**Failure scenario:** Call expand_model_weights(sd, src_cfg(vocab_size=8000), tgt_cfg(vocab_size=4718, everything else >=)). validate_growth passes (it checks dim/n_layers/n_heads/n_kv_heads/hidden_dim only). copy_rows = min(src_padded, tgt_padded) = tgt_padded, so trained embedding rows [tgt_padded:src_padded] are dropped with no warning, unlike _expand_2d/_expand_1d which raise on any shrink. Any token id >= tgt vocab now maps to a zero row; generation for those tokens is garbage and nothing ever flagged the shrink.

```
    copy_rows = min(src_padded_vocab, tgt_padded_vocab)
```

### 24. enigma_engine/training/training.py:5370 - [major/CONFIRMED/correctness] (found by: training-loop-2)

train_vision/train_audio checkpoints never contain the trained vision_encoder/audio_encoder weights and record the wrong optimizer: _save_checkpoint saves only self.model.state_dict() and self.optimizer (the stale __init__ AdamW), while these methods train a separately-passed encoder module with a LOCAL optimizer.

**Failure scenario:** Train a vision encoder for hours; `{stem}_vision_best.pt` holds the model (projection layer included) but not one parameter of vision_encoder, and its 'optimizer_state_dict' is the untouched __init__ optimizer that never saw a gradient. Reloading the best checkpoint pairs the tuned projection with an untrained encoder -- the multimodal capability the run was for is silently lost. Identical for train_audio at line 5860 (`_audio_best.pt`) and the periodic `_vision{N}.pt`/`_audio{N}.pt` saves.

```
self._save_checkpoint(checkpoint_dir / f"{self._checkpoint_stem}_vision_best.pt")
```

### 25. make_sft_data.py:704 - [major/CONFIRMED/correctness] (found by: collectors-2)

gen_teaching_examples silently explodes a string-valued "questions" (or "answers") field into per-character training records because a str is iterable and each character passes the isinstance(q, str) filter, violating the documented 'skipped LOUDLY, never silently' contract for the user-authored teachings.jsonl.

**Failure scenario:** User hand-edits teachings.jsonl and writes {"questions": "Who is Rex?", "answers": ["My dog."]} (string instead of list -- the exact near-miss of the documented shape). Confirmed by execution: 9 garbage records are emitted ('W' -> 'My dog.', 'h' -> 'My dog.', ..., '?' -> 'My dog.'), no warning printed; these ride the x8 TEACHINGS_REPEAT oversample into mix.jsonl and train single-character prompts. tests/test_teachings.py covers empty lists and bad JSON but not this shape.

```
        qs = rec.get("questions") or ([rec["q"]] if rec.get("q") else [])
```

### 26. enigma_engine/core/model_components.py:356 - [major/CONFIRMED/correctness] (found by: core-model)

Differential-attention lambda is initialized to parameter 0.05 but consumed through sigmoid, so the effective lambda at step 0 is sigmoid(0.05) ≈ 0.512 — not "near zero" as the code states, and nowhere close to standard attention at init.

**Failure scenario:** Enable use_differential_attn: from the first forward the layer computes softmax(s1) - 0.512*softmax(s2), i.e. half of the second head group's attention mass is subtracted immediately, producing large negative attention weights on fresh random heads. The stated design ("initialized near zero so early training behaves close to standard attention") requires the effective lambda near 0, i.e. a raw init around -3 (sigmoid(-3)≈0.047); 0.05 is near-zero only pre-sigmoid.

```
            self._diff_lambda = nn.Parameter(torch.full((self.n_heads // 2,), 0.05))
```

### 27. enigma_engine/core/model_components.py:1016 - [major/CONFIRMED/correctness] (found by: core-model)

use_moe is a silent no-op: TransformerBlock unconditionally builds a dense FeedForward and no MoE/expert implementation exists anywhere in the package, yet the flag, docs, get_moe_aux_loss plumbing, and trainer hook all advertise MoE support.

**Failure scenario:** ForgeConfig(use_moe=True) constructs, trains, and serves without error but with a plain dense FFN; get_moe_aux_loss() always returns 0.0 because the hasattr(feed_forward, 'get_aux_loss') check at line 1170 can never pass (grep: no class in enigma_engine defines get_aux_loss or an expert/router FFN). The trainer's aux-loss hook (training/training.py:3308) silently adds zero. Users believing they trained an MoE model actually trained a dense one, with no warning.

```
        self.feed_forward = FeedForward(config)
```

### 28. enigma_engine/training/training.py:4235 - [major/CONFIRMED/correctness] (found by: training-loop-2)

Online-DPO pair generation samples k=min(4, len(preference_data)) prompts from a list FILTERED to items that have a prompt, so mixed-validity preference data crashes random.sample with ValueError mid-training.

**Failure scenario:** preference_data has 10 items but only 2 contain a 'prompt' key (the other 8 were already skipped by _encode_dpo_pair, so training starts fine with 2 pairs). At the first step where state.step % dpo_online_interval == 0, random.sample(population_of_2, k=4) raises 'ValueError: Sample larger than population', aborting the run after arbitrary training progress instead of at setup.

```
                            [d["prompt"] for d in preference_data if d.get("prompt")], k=min(4, len(preference_data))
```

### 29. enigma_engine/core/dataset.py:63 - [major/CONFIRMED/correctness] (found by: tokenizer-data)

The "tinystories-instruct" dataset entry points at the plain TinyStories repo, so requesting the instruct corpus silently downloads the non-instruct data.

**Failure scenario:** download_dataset("tinystories-instruct", dest) calls snapshot_download(repo_id="roneneldan/TinyStories") -- the same repo as the plain "tinystories" entry. The instruct data lives in a different HF repo (roneneldan/TinyStoriesInstruct), so the caller receives ordinary stories with no instruction/prompt structure, and no error is raised.

```
        "repo_id": "roneneldan/TinyStories",
```

### 30. eval_behavior.py:118 - [major/CONFIRMED/correctness] (found by: sweep-scripts)

The memory-category probes never clear or namespace the server's persistent MemoryStore, so facts written by a PREVIOUS eval run (or a previous checkpoint's eval against the same --memory-dir) satisfy the BM25 recall and the memory gate passes even when the current model never calls the remember tool.

**Failure scenario:** Run the eval against checkpoint A with serve --memory-dir data/memory_eval: A calls remember, 'My cat's name is Biscuit.' is appended to the JSONL-backed MemoryStore on disk. Retrain; checkpoint B has regressed and never emits the remember tool call. Re-run the eval against B with the same throwaway dir: the teach requests do nothing, but render_context still BM25-injects the facts persisted by run A, B answers 'Biscuit', and the memory category prints 4/4 PASS at its 0.75 gate -- the 'end-to-end: tool call -> MemoryStore write -> BM25 recall' contract (docstring line 25) is only measured on the very first run against a fresh directory. MemoryStore.remember also returns the existing record on exact duplicates, so nothing about a re-run distinguishes a real save from a stale one.

```
            for fact in c.get("teach", []):
```

### 31. serve_enigma.py:614 - [major/CONFIRMED/correctness] (found by: sweep-scripts)

In the streaming instruct path, content deltas from an intermediate built-in-tool hop are yielded to the client BEFORE the loop decides to discard that hop, so a streamed response contains hop-0 spoken text that the non-streaming path (which overwrites `out` each hop and returns only the final hop's content) never returns -- the same request yields different content depending on the stream flag.

**Failure scenario:** Model answers 'What is 17% of 200?' by generating hop-0 content 'Let me work that out.' followed by a <|tool_call|>calculate call. Stream=true: the events() generator has already emitted 'Let me work that out.' as SSE deltas (lines 587-604 stream content as it decodes) before parse_assistant_ids sees the calculate call and _loop_on_builtins triggers `continue`; hop 1's final answer then streams immediately after, concatenated with no separator ('Let me work that out.17 percent of 200 is 34.'). Stream=false on the identical request: `out` is overwritten by the hop-1 parse and only '17 percent of 200 is 34.' is returned. Clients comparing or caching across modes see divergent transcripts, and streamed UIs show the stitched double-answer artifact.

```
                if _loop_on_builtins(parsed, hop):
```

### 32. collect_search_data.py:63 - [major/CONFIRMED/correctness] (found by: sweep-scripts)

The synthetic <search> corpus is emitted into data/finetune (which combine_all sweeps into combined_finetune.jsonl and thence into the live SFT mix -- 31 <search> rows are in the current data/sft/mix.jsonl), but no code anywhere in the repo intercepts <search>/</search> at inference: the 'B-3 RAG splice' the docstring says the engine performs does not exist, so the trained behavior is un-serveable.

**Failure scenario:** The SFT'd model, asked a factual question like 'Who won the 2024 US presidential election?', emits '<search>2024 US presidential election winner</search>' and stops (the gold completion it was trained on, 31 rows in mix.jsonl). serve_enigma's parse_assistant_ids handles only THINK(10/11) and TOOL_CALL(4720/21); ids 6/7 land in content_ids and decode(skip_special_tokens=True) strips them, so the user receives the bare query text '2024 US presidential election winner' as the entire answer -- no lookup runs, no tag is even visible. grep for search_start/<search> across the repo hits only the tokenizer files, this collector, and its test: there is no interceptor in serve_enigma.py, model.generate*, or anywhere else.

```
OUTPUT_DIR = Path("data/finetune")
```

### 33. enigma_engine/training/training.py:3922 - [major/CONFIRMED/altitude] (found by: altitude)

The FORGE Trainer carries a full parallel DPO implementation whose pair rendering uses a legacy 'User:/Assistant:' transcript template instead of chat_format.render_training, duplicating both the DPO math and the template truth that dpo_enigma.py owns on the live path.

**Failure scenario:** Two implementations of one formula must be edited in lockstep -- and demonstrably were not: CODE_REVIEW records that _get_sequence_logps sat length-averaged (wrong beta scale) until 2026-07-06, fixed only 'matching the documented DPO/APO formulas and dpo_enigma.py'. The template half is still divergent today: anyone reconnecting the FORGE trainer for a preference pass on an SFT checkpoint (its stated purpose per CLEANUP_TRACKER: 'reconnect it or replace') trains preferences on 'User: ...\nAssistant: ...' bytes that serve_enigma never renders, silently violating the repo's one-renderer rule (CLAUDE.md: 'train + serve share one chat renderer ... so the prompt format can't drift'). This is not the deferred except-Exception hygiene item; it is a distinct duplicated-truth/parallel-implementation problem. The generalization: train_dpo's pair encoding should go through chat_format.render_training (as dpo_enigma._render does), or the FORGE DPO path should be deleted now that its bespoke replacement exists (git is the archive, per repo rule 3).

```
        prompt_ids = self.tokenizer.encode(f"User: {prompt}\nAssistant: ")
```

### 34. enigma_engine/config/defaults.py:152 - [major/CONFIRMED/simplification] (found by: simplification)

Roughly 28 CONFIG knobs (the entire ARCHITECTURE, TRAINING, and INFERENCE sections plus most path keys) are read by no code anywhere in the repo -- survivors of the already-removed dead-knob batch -- including hand-synced duplicate pairs depth/num_layers and heads/num_heads.

**Failure scenario:** A user sets embed_dim, learning_rate, temperature, device, precision, max_gen, etc. in forge_config.json or FORGE_DEVICE expecting them to take effect; nothing ever reads them (the live pretrain/SFT/serve scripts take everything from argparse and ForgeConfig), so the setting silently does nothing. The duplicate depth/num_layers and heads/num_heads pairs invite setting one and not the other. The prior fix (CODE_REVIEW.md closed item) removed require_api_key/blocked_paths but left these sections: repo-wide grep shows the only CONFIG readers anywhere are plugin_loader.py:206 (trusted_plugins) and the four mkdir dir keys (data_dir/models_dir/memory_dir/logs_dir) inside _ensure_initialized; the file header still advertises a 'SECURITY - What must never be touched (blocked paths)' section (line 31) that no longer exists.

```
        "embed_dim": 256,
```

### 35. enigma_engine/training/training.py:2723 - [major/CONFIRMED/efficiency] (found by: training-loop-2)

The eager (non-streaming) path materializes the ENTIRE tokenized dataset plus attention masks directly in GPU memory, because _create_batches builds tensors with device=self.device -- making _train_one_batch's documented 'lazy device transfer' a no-op and shrinking the VRAM headroom the auto-batch estimator assumed.

**Failure scenario:** Any dataset up to streaming_threshold (~50K sequences by default) is list()-ed as batch tensors created at device=self.device (line 2252-2253), so e.g. 50K seqs x 1024 tokens x 8 bytes x 2 tensors (ids+mask) = ~0.8 GB, and ~3.2 GB at max_seq_len 4096, sits in VRAM for the whole run alongside activations -- VRAM that _estimate_batch_size() budgeted as free (it only accounts weights+optimizer+grads+activations). Result: OOM on runs the auto-batch sizing certified, or forced batch-size collapse; val_batches add the same permanent cost. The packing path returns CPU tensors and _train_one_batch's comment 'Lazy device transfer: tensors stay on CPU until needed' documents the intended contract that the standard path violates.

```
                    batches = list(self._create_batches(sequences, max_length=max_seq_len))
```

### 36. serve_enigma.py:93 - [major/CONFIRMED/efficiency] (found by: efficiency)

serve runs the model in full fp32 with neither autocast nor TF32 enabled (unlike pretrain/finetune which run bf16 autocast and set allow_tf32), doubling the memory-bandwidth cost of every decoded token and leaving prefill matmuls off the tensor cores.

**Failure scenario:** Batch-1 decode of the 182M model is weight-read bound: fp32 weights are ~728 MB of traffic per token vs ~364 MB under bf16, roughly halving the achievable tokens/s ceiling for every request; the fp32 KV cache (lazy-init dtype=k.dtype at model_components.py:497) doubles cache traffic the same way, and prefill GEMMs run in slow non-TF32 fp32. Cheaper: wrap generation in torch.autocast('cuda', dtype=torch.bfloat16) -- exactly the mixed-precision route the model's own RoPE guard in apply_rotary_embedding recommends ('keep the model fp32 and use torch.autocast'), and numerics the entire training stack already validated -- plus torch.backends.cuda.matmul.allow_tf32 = True as pretrain_enigma.py:174 does.

```
model.to(DEVICE).eval()
```

### 37. enigma_engine/core/model_components.py:603 - [major/CONFIRMED/efficiency] (found by: efficiency)

The KV-cache incremental-decode SDPA branch builds a fresh (Tq, Tk) boolean mask per layer per token even though for Tq==1 (every decode step after prefill) the mask is all-True, i.e. mathematically a no-op that only forces SDPA off its fastest no-mask path.

**Failure scenario:** Every served token after prefill: 16 layers x (torch.ones + tril allocation kernels) per token = ~32-48 extra kernel launches and allocations per token, and passing attn_mask disqualifies the flash-decode backend (flash SDPA only supports is_causal, not arbitrary masks), pushing single-query attention to the mem-efficient/math backend for the whole generation. Cheaper: special-case Tq == 1 (the code's own comment notes the mask is all-True there) and call F.scaled_dot_product_attention with no mask at all; the rectangular mask is only needed for multi-query cached decode (Tq > 1), which the single-sequence generate loops never produce after prefill.

```
                attn_causal = torch.ones(Tq, Tk, dtype=torch.bool, device=q_s.device).tril(diagonal=Tk - Tq)
```

### 38. finetune_enigma.py:222 - [major/CONFIRMED/altitude] (found by: altitude)

The checkpoint resume discipline (latest->prev.pth missing/corrupt fallback + schedule-lock restore with --override-schedule) is copy-pasted from pretrain_enigma.py into finetune_enigma.py and has already drifted between the two copies.

**Failure scenario:** A fix to the resume path must land twice or the scripts diverge -- it already happened once (the corrupt-latest fallback exists in both pretrain 207-218 and finetune 205-216 because the same fix was applied to each). The copies already disagree: pretrain restores every recorded schedule key via unconditional setattr and prints a 'checkpoint predates schedule recording' notice (pretrain 231-239); finetune filters with hasattr and has no such notice (finetune 222, 227-231), so a schedule key recorded by one convention and resumed under the other is silently ignored and the resumed SFT run trains on CLI values while the operator believes the lock held.

```
        diffs = {k: (v, getattr(args, k)) for k, v in saved_sched.items() if hasattr(args, k) and getattr(args, k) != v}
```

### 39. pretrain_enigma.py:208 - [major/CONFIRMED/efficiency] (found by: train-scripts)

The resume/init-from checkpoint dict `ck` is loaded with map_location=cuda and never freed, pinning roughly 2.2 GB of VRAM (fp32 weights plus both AdamW moment tensors for the 182M model) for the entire multi-day training run.

**Failure scenario:** Every resumed session of the live run: ck = torch.load(rp, map_location=device) puts model_state_dict (~730 MB) and optimizer exp_avg/exp_avg_sq (~1.5 GB) on the GPU; load_state_dict and optim.load_state_dict COPY that data into the model/optimizer, but ck stays referenced in main()'s frame until the process exits, so ~2.2 GB of duplicate tensors sit in the CUDA caching allocator for days. That headroom is what an OOM-free bump of --micro-batch (or the block-2048 warm start, whose ck is equally retained) would use. finetune_enigma.py:286 and dpo_enigma.py:123 both do `del ck` at this exact point; pretrain has no `del ck` (grep confirms).

```
            ck = torch.load(rp, map_location=device)
```

### 40. enigma_engine/training/training.py:3403 - [major/CONFIRMED/efficiency] (found by: efficiency)

The FORGE trainer's inner loop forces two host-device syncs per micro-batch: _train_one_batch returns loss.item() and the caller immediately does another (batch[0] != pad).sum().item() for token counting, stalling the pipeline twice per batch.

**Failure scenario:** Any run through Trainer.train (the FORGE stack's own hot loop): every micro-batch blocks the CPU on the loss sync (needed only for the NaN/max_loss guard, which could test a detached GPU accumulator every log_every steps) and then again on the non-pad count, which is knowable on the CPU side before the H2D copy (the batch tensor originates from CPU tokenization). On small models where a step is a few ms, two syncs plus the queue drain can add double-digit-percent overhead. Cheaper: return loss.detach(), count non-pad tokens on the CPU tensor before transfer (or accumulate a GPU counter), and call .item() only at logging/guard boundaries.

```
return loss.item() * self.config.max_grad_accumulation
```

### 41. enigma_engine/core/model_components.py:515 - [major/CONFIRMED/efficiency] (found by: efficiency)

GQA expansion uses k.repeat_interleave(n_rep, dim=2) on the FULL cached K/V every decode step, materializing an n_rep-times copy of the entire history per layer per token instead of letting SDPA handle GQA natively.

**Failure scenario:** Live 'large' config (n_heads=16, n_kv_heads=4 -> n_rep=4, 16 layers, fp32 serve): at context ~1024 each decode step copies 2 tensors x 1024 x 16 x 64 x 4B = 8 MB per layer -> ~128 MB of read+write traffic per generated token, all thrown away immediately; at 1024-token generations that is ~128 GB of wasted bandwidth per request, plausibly 10-25% of per-token decode time on top of the weight reads. Cheaper: pass enable_gqa=True to F.scaled_dot_product_attention (PyTorch 2.5+) and feed the un-expanded [B, n_kv_heads, T, D] K/V, or restrict the expansion to the single new K/V row and keep the cache expanded once.

```
            k = k.repeat_interleave(self.n_rep, dim=2)
```

### 42. make_sft_data.py:775 - [major/CONFIRMED/altitude] (found by: altitude)

fit_mix_to_block re-implements finetune_enigma.load_examples' record normalization (prompt/completion key-fallback chains) and its acceptance rule (len(ids) <= block+1) as a second copy that must be kept in sync by hand.

**Failure scenario:** finetune_enigma.py later accepts another key (e.g. 'input') or changes its length rule; make_sft_data's copy is not updated, so fit_mix_to_block blesses records as 'fits' that the trainer then silently skips at load time -- resurrecting the exact bug the function exists to prevent (its own header: 'Previously long records passed through untouched and 76% of the mix was silently skipped at train time'). The key chains are already triplicated inside make_sft_data itself (_norm_q line 59, fit_mix_to_block lines 782-783, _assistant_text line 875).

```
    limit = block + 1  # the trainer keeps examples with len(ids) <= block+1
```

### 43. collect_pretraining_data.py:847 - [major/CONFIRMED/reuse] (found by: reuse)

save_progress() hand-rolls an atomic tmp-write-then-rename with .bak rotation, re-implementing enigma_engine.core.safe_save.atomic_write_json but WITHOUT the fsync-before-rename the shared helper provides.

**Failure scenario:** Power loss (the exact scenario this multi-day collector runs through -- the 06-10 Windows Update force-reboot) can commit the rename metadata ahead of the tmp file's data blocks, leaving a complete-looking but truncated/empty progress.json AND a .bak that was replaced away in the same window; load_progress then 'starts fresh' and the collector re-downloads gigabytes / re-fetches completed Gutenberg IDs. safe_save closed this identical window for checkpoints (CODE_REVIEW: atomic_safetensors_save fsync fix); this copy silently keeps it open.

```
        tmp.write_text(json.dumps(progress, indent=2), encoding="utf-8")
```

### 44. enigma_engine/training/training.py:5623 - [major/CONFIRMED/efficiency] (found by: training-loop-2)

train_audio eagerly preprocesses EVERY mel tensor onto the GPU during data prep (train and val), the sibling of the V-2 vision lazy-preprocess OOM fix that was not applied to audio.

**Failure scenario:** The prep loop calls mel_tensor.to(self.device) for all items and keeps them alive in `pairs` for the whole run, so the entire dataset resides in VRAM before the first training step. At the scale the V-2 vision comment warns about (hundreds of thousands of clips; an 80-mel x ~3000-frame fp32 spectrogram is ~1 MB each), the GPU OOMs during preparation, before any checkpoint exists. val_pairs at line 5651 duplicates the pattern and its comment even acknowledges 'matching the training preprocess pattern' that vision deliberately abandoned.

```
mel_tensor = mel_tensor.to(self.device)
```

### 45. enigma_engine/core/memory_store.py:197 - [major/CONFIRMED/reuse] (found by: reuse)

MemoryStore._rewrite() hand-rolls tmp-write-then-replace with no fsync, re-implementing safe_save.atomic_write_text which lives in the same package and adds the fsync durability plus .bak rotation.

**Failure scenario:** User memories are the store's whole point ('runtime-learning layer'; weights are frozen between passes). A power loss after `tmp.replace(self.file)` commits the rename metadata but before the data blocks flush leaves memories.jsonl truncated or empty -- every remembered fact is gone with no .bak to fall back on. The repo treats missing fsync-before-rename as a defect class (safe_save.py:48-49 comment; CODE_REVIEW closed the same gap in atomic_safetensors_save), but this writer predates/escaped that sweep.

```
        tmp = self.file.with_suffix(".jsonl.tmp")
```

### 46. enigma_engine/core/dataset.py:201 - [major/CONFIRMED/reuse] (found by: simplification)

_chunked_read_text is a line-for-line duplicate of _iter_chunked_read_text (same open/remainder/rfind/clean/progress/sleep loop, ~50 lines) differing only in parts.append(...) vs yield.

**Failure scenario:** Two copies of the chunked-reader must be patched in lockstep: a fix to the remainder handling, progress math, or newline-split logic applied to one copy and not the other silently diverges the list path (_process_file at line 191) from the iterator path (iter_text_chunks). The simpler form is the one-liner: def _chunked_read_text(path, *, on_progress=None): return list(_iter_chunked_read_text(path, on_progress=on_progress)) -- identical output, deletes ~50 duplicated lines.

```
def _chunked_read_text(
```

### 47. collect_pretraining_data.py:633 - [major/CONFIRMED/efficiency] (found by: collectors-1)

process_wiki_dump globs the 4.7M-file wikipedia_dump directory twice on the resume path, the second glob doing a stat() syscall on every file just to print a GB total.

**Failure scenario:** On any resume of a populated dump, line 626 builds a stem set from one glob and line 633 runs a second full glob issuing 4.7M `f.stat().st_size` syscalls purely to compute a cosmetic 'X GB already extracted' number. combine_all_sources was explicitly rewritten to os.scandir to avoid exactly this cost at wiki_dump scale; this path still pays minutes of directory I/O per resume.

```
    total_bytes_written = sum(f.stat().st_size for f in WIKI_DUMP_DIR.glob("*.txt"))
```

### 48. resume_training.ps1:26 - [minor/CONFIRMED/correctness] (found by: train-scripts)

The checkpoint guard hard-exits when latest.pth is missing, but pretrain_enigma.py deliberately falls back to prev.pth in exactly that case — so after the one documented crash window (between the two renames in atomic_torch_save's rotation) or a deleted-corrupt latest, the desktop-shortcut resume refuses to launch a run that the trainer itself would resume fine.

**Failure scenario:** Power loss or power_guardian's Stop-Process -Force lands between os.replace(path, rotate_to) and os.replace(tmp_path, path) in atomic_torch_save (safe_save.py:54-56) — the state the pretrain fallback was built for: prev.pth exists, latest.pth does not. Next morning the user double-clicks 'Resume Enigma Training'; the script prints 'Checkpoint not found ... Cannot resume (was the model moved?)' and exits 1, even though 'python pretrain_enigma.py --resume models/enigma_pretrain_large/latest.pth' would succeed via its prev.pth fallback (pretrain_enigma.py:199-201). The self-service resume workflow is dead until manual intervention.

```
if (-not (Test-Path $ckpt)) {
```

### 49. make_dpo_data.py:86 - [minor/CONFIRMED/correctness] (found by: train-scripts)

The adversarial-denial loops index the answer pools with j % len(pool) where j only ever takes values 0 and 1 (enumerate over rng.sample(qs, 2)), so answer variants 2 and 3 of both _DENY_MODEL_A and _DENY_COMPANY_A can never appear in dpo_pairs.jsonl — half the authored chosen-answer diversity is unreachable.

**Failure scenario:** Run make_dpo_data.py: for every one of the 15 _ORGS_MODELS, j iterates {0, 1}, so _DENY_MODEL_A[j % 4] selects only indices 0 and 1; identically _DENY_COMPANY_A[j % 4] for all 10 companies. The generated preference file trains the denial behavior on exactly 2 of the 4 authored phrasings per pool ('No, and there's no trick to it...' and 'Nope. I'm Enigma, built from zero...' never occur), undercutting the file's own stated design ('the diversity lesson applies to preferences too') and biasing DPO toward two fixed surface forms. The modulo is dead code signaling the intended full rotation.

```
_DENY_MODEL_A[j % len(_DENY_MODEL_A)],
```

### 50. eval_behavior.py:129 - [minor/CONFIRMED/correctness] (found by: serve)

The tool-probe detail string embeds the model-emitted tool-call name without the _ascii() guard, and eval_behavior never reconfigures stdout to UTF-8, so a non-ASCII character in a generated tool name crashes the eval mid-run on the cp1252 console.

**Failure scenario:** On a tool/restraint probe the model hallucinates a tool call whose name contains any non-cp1252 character (the name is free-form generated text passed through _openai_tool_calls verbatim). `print(f"[{cat...}] ... -> {detail}")` at line 136 raises UnicodeEncodeError, aborting the whole scorecard with a traceback instead of grading the probe as a failure. Content probes were ASCII-guarded in the 2026-07-06 fix pass (`_ascii(content[:60])`, line 133) but this sibling path was missed; the file imports no `sys` and has no `sys.stdout.reconfigure` fallback.

```
            detail = f"tool={called}"
```

### 51. serve_enigma.py:850 - [minor/CONFIRMED/correctness] (found by: serve)

POST /v1/memory with empty or whitespace-only text returns a 500 with a stack trace: MemoryStore.add raises ValueError("empty memory") and the endpoint does not map it to a 400.

**Failure scenario:** Client POSTs {"text": ""} (or "   ") to /v1/memory. MemoryStore.add normalizes with `" ".join(str(text).split())` and raises ValueError (memory_store.py:114-116); FastAPI converts the uncaught exception into a generic 500 Internal Server Error — a client-input error reported as a server crash, contradicting the file's own error-mapping pattern (the role check at lines 717-722 exists precisely to turn ValueError-class client mistakes into 400s, and _execute_builtin returns "error: nothing to remember" for the same input on the tool path).

```
    return {"ok": True, "memory": MEMORY.add(req.text, kind=req.kind)}
```

### 52. eval_behavior.py:146 - [minor/CONFIRMED/correctness] (found by: serve)

An unknown category in behavior_probes.jsonl (typo or new category without a threshold) is graded against a default threshold of 0.0, so it always prints PASS and never gates — silently corrupting the scorecard the SFT runs are compared on.

**Failure scenario:** A probe line carries "category": "advesarial" (typo) or a newly added category not yet in THRESHOLDS. Its results land in their own bucket; `THRESHOLDS.get(cat, 0.0)` yields 0.0 and `rate >= 0.0` is always true, so the bucket prints "PASS" even at 0/4, while the real "adversarial" bucket quietly shrinks. Exit code stays 0 and the regression ships. The file already distinguishes None-threshold informational categories (line 147), so a missing key should fail loudly rather than default-pass.

```
        thr = THRESHOLDS.get(cat, 0.0)
```

### 53. identity_paraphrases.py:146 - [minor/CONFIRMED/correctness] (found by: collectors-2)

Company-denial templates that embed 'a {c}' produce double-article garbage ('I'm not a a startup model', 'a some Silicon Valley lab model') because _ORGS_COMPANIES contains entries that already start with an article, leaking malformed template artifacts into the identity training data.

**Failure scenario:** Running gen_identity_paraphrases() (confirmed by execution) emits 4 corrupted records, e.g. Q: "You're really a a startup model, right?" and A: "Wrong guess. I'm not a a big tech company model -- ..." / "I'm not a some Silicon Valley lab model". These are oversampled x8 into mix.jsonl by make_sft_data, teaching the 182M model ungrammatical identity denials; answer template index 2 is deterministically used for every company (j % 4 over 3 sampled questions), so the artifact always ships.

```
    "Wrong guess. I'm not a {c} model -- I was trained from scratch by SirRulean, locally, from random weights.",
```

### 54. enigma_engine/training/training_evaluation.py:246 - [minor/CONFIRMED/correctness] (found by: training-aux)

evaluate_tool_usage divides successes by len(test_cases), counting un-gradeable/skipped cases as failures and deflating the reported tool-call success rate.

**Failure scenario:** A test set contains a case with a non-empty 'expected_command' but no 'expected_tool' (line 213-219 only logs a warning and cannot score it) or an empty 'prompt' (line 210-211 does `continue`). These cases never increment successes yet are counted in `total = len(test_cases)` (line 246), and `failures = total - successes` (line 253) reports them as failures. A model that correctly calls every gradeable tool is scored below 1.0 purely because of malformed/legacy entries in the list.

```
        total = len(test_cases)
```

### 55. collect_pretraining_data.py:3727 - [minor/CONFIRMED/correctness] (found by: collectors-1)

The `--fandom` help text says '0 = unlimited/all', but the dispatch guard `args.fandom > 0 or args.fandom_all` skips Fandom entirely when `--fandom 0` is passed without `--fandom-all`.

**Failure scenario:** A user reads the help ('Max articles from Fandom wikis (0 = unlimited/all)') and runs `--fandom 0` expecting to collect every Fandom article. Because `args.fandom > 0` is False and `--fandom-all` was not given, the Fandom block never runs and zero articles are collected, silently contradicting the documented behavior.

```
        if args.fandom > 0 or args.fandom_all:
```

### 56. Start-Enigma.ps1:11 - [minor/CONFIRMED/correctness] (found by: sweep-scripts)

The already-running check treats ANY process listening on port 8000 as 'Enigma already serving' and exits 0 without verifying the listener is serve_enigma, so a foreign app on 8000 silently prevents Enigma from ever starting.

**Failure scenario:** Any other local service grabs port 8000 (a very common dev-server default: Django runserver, http.server, another model server). Desktop launch runs Start-Enigma.ps1, Get-NetTCPConnection finds the listener, the script prints 'Enigma already serving on port 8000 (pid ...)' and exits 0. Odysseus then sends /v1/chat/completions to whatever is actually on 8000 and gets 404s/garbage, while Enigma never comes up and nothing in serve_enigma.log explains why. The sibling guard in resume_training.ps1 checks the actual process command line ('*pretrain_enigma*'); this script checks only the port.

```
$up = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
```

### 57. enigma_engine/core/model_presets.py:232 - [minor/CONFIRMED/correctness] (found by: sweep-package)

The GQA validation `self.n_heads % self.n_kv_heads != 0` divides by n_kv_heads before checking it is non-zero, so an explicit n_kv_heads=0 raises an uninformative ZeroDivisionError instead of the clear ValueError the check exists to produce.

**Failure scenario:** Constructing `ForgeConfig(n_kv_heads=0)` or, more realistically, `ForgeConfig.from_dict({...,"n_kv_heads":0})` when loading a hand-edited/corrupt checkpoint config: __post_init__ line 232 evaluates `n_heads % 0` and throws `ZeroDivisionError: integer division or modulo by zero` with no context, rather than the intended message 'n_kv_heads must divide evenly into n_heads'. The read-only `validate()` path has the identical hole at line 290 (`n_kv` resolves to 0, `n_heads % n_kv`).

```
        if self.n_heads % self.n_kv_heads != 0:
```

### 58. enigma_engine/core/progressive_growing.py:591 - [minor/PLAUSIBLE/correctness] (found by: adapt-merge)

GradualUnfreezer freezes with a substring test ('layers.' in name) but unfreezes with a prefix test (name.startswith(f'layers.{i}.')), so on any name-prefixed model (PEFT 'base_model.model.layers.0...', DataParallel 'module.layers.0...') every layer is frozen at init and step() can never unfreeze one.

**Failure scenario:** setup_gradual_unfreeze(peft_model, n_layers) after growth: _freeze_all_layers matches 'layers.' as a substring and sets requires_grad=False on all transformer params; _unfreeze_head leaves them frozen (their names contain 'layers.'); later step() calls _unfreeze_layer(i) whose startswith(f'layers.{i}.') matches zero parameters because names begin with 'base_model.model.' — yet it still logs 'Unfroze layer N', so training silently proceeds forever with only embeddings/norm/output trainable and no error. LISAScheduler has the same startswith assumption (lines 652/665) — on a wrapped model it freezes nothing, silently voiding its memory-saving contract.

```
            if name.startswith(prefix):
```

### 59. identity_anchors.py:364 - [minor/PLAUSIBLE/correctness] (found by: collectors-2)

The capabilities anchor flatly trains 'Can you see my screen?' -> 'No -- I only see what you type or hand me', directly contradicting the see_screen tool-call training data ('What's on my screen right now?' -> call see_screen), and the identity side is oversampled x8 vs tools x5 in the mix.

**Failure scenario:** At serve time with a client offering see_screen (an avatar/mod tool), the user asks 'Can you see my screen?' or 'What's on my screen?': the x8-repeated flat denial competes head-on with the x5 tool examples on near-identical question surfaces, so the model either refuses a capability it has (denial wins) or claims screen access when no tool is offered (tool phrasing wins). Unlike the neighboring 'Can you browse the internet?' anchor, this answer has no 'unless you wire that up' hedge conditioning it on tool availability.

```
"No — I only see what you type or hand me. If you want me to work with "
```

### 60. enigma_engine/core/tokenizer.py:152 - [minor/PLAUSIBLE/correctness] (found by: tokenizer-data)

encode_text() evaluates `if ids` before the isinstance(list) guard, so a tokenizer whose callable interface returns a multi-element tensor crashes on Tensor truthiness instead of being converted via .tolist().

**Failure scenario:** For a HuggingFace-style callable tokenizer, ids = result.get("input_ids") can be a torch tensor. Line 152 `if ids and isinstance(ids, list) and ...` evaluates bool(ids) first; for a tensor with >1 element this raises RuntimeError: 'Boolean value of Tensor with more than one element is ambiguous', before the tensor-handling branch at line 156 (`if hasattr(ids, 'tolist')`) can run. The list guard is placed after the truthiness test so it cannot protect against it.

```
    if ids and isinstance(ids, list) and isinstance(ids[0], list):
```

### 61. make_sft_data.py:795 - [minor/PLAUSIBLE/correctness] (found by: sweep-scripts)

fit_mix_to_block's ASCII fast path budgets only message `content` characters plus a fixed 64-token margin, but render_training additionally emits the tool_calls JSON payloads (via json.dumps -> _enc_content) and per-message template ids, which the fast path never counts -- so a near-limit tool conversation can be stamped 'fits' while actually rendering past block+1, recreating the silent train-time skip this function exists to prevent.

**Failure scenario:** A tool_multiturn record whose message contents sum to just under limit-64 chars (large distractor tool-spec system block + tool results) also carries two uncounted tool-call payloads like {"name": "avatar_say", "arguments": {"text": "The guest wifi password is hunter2-guest."}} (~90 chars each) plus ~8 messages x header/IM ids. Under the char-mode tokenizer the fast-path comment explicitly claims to support (token_count == char_count there), the uncounted payloads+template alone exceed the 64-token margin, the line passes through untrimmed, render_training at load time produces len(ids) > block+1, and finetune_enigma.load_examples drops it as skipped_long -- while make_sft_data's summary reported it as fitted, silently thinning exactly the multiturn tool data the mix is weighted to oversample.

```
        if all(c.isascii() for c in contents) and sum(len(c) for c in contents) + 64 <= limit:
```

### 62. enigma_engine/core/kv_cache.py:1341 - [minor/CONFIRMED/simplification] (found by: simplification)

StreamingLLMCache.update hand-rolls its window-shift copy as a quantize/non-quantize if/else that duplicates the _cache_k/_cache_v copy statements verbatim in both branches, instead of iterating the _rollable_buffers() hook that exists precisely to name the buffers a shift must move together.

**Failure scenario:** 28 lines (1341-1368) of six near-identical slice-copy statements where the else branch is a strict subset of the if branch; any new per-position buffer added to the cache family (the way H2O adds _attn_scores and TurboQuant adds four INT4 buffers via _rollable_buffers) is silently skipped by this hand-rolled shift, leaving stale entries decoded against shifted data -- the exact desync class the _rollable_buffers abstraction was added to prevent. Simpler form: for name in self._rollable_buffers(): buf = getattr(self, name); buf[:, dst_start:dst_start+actual_keep] = buf[:, src_start:src_start+actual_keep].clone() -- identical behavior for both quantized and float caches.

```
                if self.quantize:
```

### 63. pretrain_enigma.py:515 - [minor/CONFIRMED/efficiency] (found by: efficiency)

The live pretrain step loop calls loss.item() after every micro-batch backward (grad_accum=16 -> 16 full host-device syncs per optimizer step), serializing CPU-side get_batch work (random memmap reads over the 226 GB tokens.bin, which cannot all be page-cached in 61 GB RAM) against GPU compute.

**Failure scenario:** Each micro-batch: CPU launches fwd/bwd then blocks on .item() until the GPU drains; only then does get_batch run its 24 random memmap slice reads (potential NVMe hits on a file 4x larger than RAM) + np.stack + H2D copy while the GPU sits idle -- an estimated few percent of the ~150 ms micro-batch time, i.e. hours over a 287k-step run. Cheaper: accumulate loss.detach() into a GPU tensor inside the accum loop and call .item() once per optimizer step (or only at the every-10-step log line and the save-time finite check), restoring CPU/GPU overlap; pinning the staging arrays would additionally make the non_blocking=True copies at lines 310-311 actually asynchronous.

```
loss_acc += loss.item()
```

### 64. sample_enigma.py:47 - [minor/CONFIRMED/conventions] (found by: serve)

sample_enigma loads checkpoints with strict=False and discards the missing/unexpected key lists, so a checkpoint that does not match the constructed architecture silently runs with randomly initialized weights for the missing tensors.

**Failure scenario:** User points --ckpt at a checkpoint whose state dict diverges from ForgeConfig-built module names (a differently configured run, a partially converted file, or a future arch tweak): load_state_dict(strict=False) drops the mismatched keys without a word, the probe prints plausible-looking metadata (step, param count) and then generates garbage that gets misattributed to model quality. serve_enigma.py:91 loads the same checkpoints with strict=True, and even pretrain_enigma.py:385 captures `missing, unexpected` when it uses strict=False — this is the only loader that both relaxes strictness and ignores the report.

```
    model.load_state_dict(ck["model_state_dict"], strict=False)
```

### 65. serve_enigma.py:293 - [minor/CONFIRMED/altitude] (found by: altitude)

The strip-trailing-EOS / ensure-BOS dance after tokenizer.encode() is a per-caller convention duplicated in serve_enigma._generate_text and sample_enigma.py instead of one encode-for-generation helper at the tokenizer/chat_format boundary.

**Failure scenario:** Every new generation entry point (next eval script, a REPL probe, training_evaluation extension) must re-remember both steps; forgetting the EOS strip reproduces the already-shipped 2026-06-11 serve bug (model sees a finished document and replies with EOS/new-document), and KNOWN_ISSUES item 8 has to document the trap in prose precisely because no helper encodes it. The two existing copies are the concrete duplication cost: the same 4-line idiom maintained twice (serve 292-306, sample 63-67), each with its own comment re-explaining the gotcha.

```
    if ids and ids[-1] == EOS_ID:
```

### 66. collect_distill_data.py:445 - [minor/CONFIRMED/reuse] (found by: reuse)

_rewrite_combined_text() re-implements the canonical 'User: ...\n\nAssistant: ...' block writer that collect_finetuning_data._write_combined_text already provides (same empty-skip rule, same blank-line separator, same block f-string) instead of building the deduped pair list and calling the shared writer.

**Failure scenario:** The canonical plain-transcript training format is now defined in two files. collect_search_data.py:51-54 imports _write_combined_text from collect_finetuning_data specifically 'so dual-emit format stays identical' -- collect_distill_data breaks that single-source rule, so any format fix in the canonical writer (the empty-block rationale documented at collect_finetuning_data.py:83-86, or a future separator change) leaves the distill .txt silently emitting a divergent shape into data/finetune, which combine/pretokenize then mixes into training.

```
            out.write(f"User: {prompt}\n\nAssistant: {completion}\n")
```

### 67. enigma_engine/core/tokenizer.py:314 - [minor/CONFIRMED/simplification] (found by: simplification)

SimpleTokenizer.__init__ hand-duplicates the eight convenience ID assignments (pad/bos/eos/unk/think_start/think_end/search_start/search_end) that _sync_special_ids() already derives from the special_tokens map, including a duplicated copy of the Stage B-1 explanatory comment.

**Failure scenario:** The ID mapping now lives in three hand-synced places: the special_tokens dict (lines 302-311), the __init__ literals (314-324), and _sync_special_ids' defaults (482-491). Adding or renumbering a special token requires editing all three; missing the __init__ block leaves fresh (default-vocab) tokenizers with IDs that disagree with the map -- exactly the aliasing class the _load_vocab rebuild was added to prevent. Simpler form: after defining self.special_tokens in __init__, call self._sync_special_ids() and delete the literal block.

```
        self.pad_token_id = 0
```

### 68. dpo_enigma.py:115 - [minor/CONFIRMED/reuse] (found by: reuse)

dpo_enigma.py duplicates finetune_enigma.py's checkpoint load-and-validate ritual verbatim (dict-shape check, ForgeConfig.from_dict, strict=False load, the freqs_cis/causal_mask-filtered real_missing computation, the same arch-mismatch SystemExit), a third hand-rolled copy of the loader alongside serve_enigma.py's.

**Failure scenario:** Concrete maintenance cost: the tolerated-buffer list is load-bearing (a new non-persistent buffer added to the model -- another rotary table, a new mask cache -- must be added to BOTH filter copies or one of the SFT/DPO passes hard-exits with a spurious 'arch mismatch' on every valid checkpoint), and the checkpoint-shape contract ('model_state_dict' + 'config') is asserted in three separately-worded places. serve/finetune/dpo are one live lineage, not the dormant FORGE fork, so 'bespoke by design' does not cover this triplication.

```
    real_missing = [k for k in missing if "freqs_cis" not in k and "causal_mask" not in k]
```

### 69. pretrain_enigma.py:169 - [minor/CONFIRMED/simplification] (found by: simplification)

The tokenizer object bound to tok is never read after the vocab-size check -- the except-branch 'tok = None' is a dead assignment and the comment's stated purpose ('for readable samples') has no implementing code anywhere in the training loop.

**Failure scenario:** The variable and its None fallback exist only to be discarded: grep for \btok\b in pretrain_enigma.py hits lines 161-169 and nothing else (the loop prints loss/lr/tok-per-s only, never decodes a sample). A reader trusts the comment 'We still try to load the exact one ... for readable samples' (line 154-156) and expects sample text during training that never appears. Simpler form: keep the vocab-mismatch WARN but drop the binding, e.g. compare getattr(get_tokenizer("bpe"), "vocab_size", None) inline, and fix the stale comment.

```
        tok = None
```

### 70. serve_enigma.py:384 - [minor/CONFIRMED/reuse] (found by: reuse)

_CALC_TOOL and _REMEMBER_TOOL hardcode a second copy of the calculate/remember name+description+parameters that make_sft_data.py TOOLS defines -- the served spec must stay byte-identical to the trained one, but nothing shares or asserts the constants.

**Failure scenario:** The comment at serve_enigma.py:374-375 states the contract ('in the same spec shape make_sft_data trains on'), yet it is enforced only by eyeball. Rewording either copy's description (e.g. tweaking make_sft_data's calculate blurb for a new SFT run without touching serve) makes the model see a system-prompt spec it was never trained on; per this repo's own measurements that degrades routing silently (the 2026-07-05 schema-shape finding in chat_format._flat_params is this exact failure class) and no test catches it.

```
        "description": "Evaluate an arithmetic expression and return the exact result.",
```

### 71. make_sft_data.py:891 - [minor/CONFIRMED/conventions] (found by: collectors-2)

The eval-probe filter on tool examples is silent (no count printed, unlike identity's n_leak and general's n_gen_leak), and it silently deletes three RESTRAINT entries that duplicate held-out eval probes verbatim ('Good evening.', 'How's it going?', 'Nice to meet you.'), contradicting both the module's 'no silent caps' promise and the in-file rule 'never the exact probe strings'.

**Failure scenario:** Author reads RESTRAINT lines 357/359/392 and believes those greeting surfaces are trained; they never reach tool_calls.jsonl because they exactly match data/eval/behavior_probes.jsonl questions (verified against the probes file), and the tool_calls print reports no held-out count so the removal is invisible. Cost: three intended restraint surfaces silently untrained, plus a latent eval-gaming hazard -- if a probe's punctuation is ever edited, the identical RESTRAINT string starts training the (near-)exact probe.

```
    tools = [r for r in gen_tool_examples() if _norm_q(r) not in eval_qs]
```

### 72. serve_enigma.py:502 - [minor/CONFIRMED/reuse] (found by: reuse)

The byte-critical tool-system preamble 'You are Enigma. You can use tools when they are needed; answer directly when they are not.\n' is copy-pasted in three files (serve_enigma.py, make_sft_data.py, training_evaluation.py) with no shared constant, despite chat_format.py existing to centralize exactly such train==serve strings (it already does this for TOOL_SYNTAX).

**Failure scenario:** serve_enigma.py:497-500 documents that 'Training's tool examples ALWAYS lead with this exact preamble ... a system message that OPENS with "Available tools:" is a shape the model never saw' -- i.e. the string is a learned contract. A wording tweak in make_sft_data for the next SFT run that is not mirrored in serve (or vice versa) puts serving off-distribution on every tool request; the eval harness would only show it as a diffuse tool-score drop, not point at the cause.

```
                "You are Enigma. You can use tools when they are needed; "
```

### 73. enigma_engine/core/dataset.py:535 - [minor/CONFIRMED/simplification] (found by: simplification)

_process_directory re-tests 'if suffix in _TEXT_SUFFIXES' on files that were already filtered by exactly that suffix set two lines earlier, so the condition is always true and the suffix variable exists only for the dead check.

**Failure scenario:** Pure dead branching: the sorted(...) comprehension on line 532 admits only files whose f.suffix.lower() is in _TEXT_SUFFIXES, so the inner check can never be false and the else path (skip file) is unreachable. Cost: the reader hunts for what non-text file could reach here, and a future edit to one of the two suffix sets but not the other silently changes behavior. Simpler form: for f in files: chunk = _process_file(f, text_key=text_key); if chunk: parts.append(chunk).

```
        if suffix in _TEXT_SUFFIXES:
```

### 74. collect_distill_data.py:151 - [minor/CONFIRMED/conventions] (found by: conventions)

A RuntimeError message contains a U+2192 arrow ("user→assistant") that is surfaced through logger.warning at line 516, the exact bug class the collector fix pass (9 arrow logger lines) was supposed to eliminate.

**Failure scenario:** During a distill collection run against a base/mis-templated teacher, the magpie parse fails, the RuntimeError propagates to `logger.warning("prompt %d/%d failed: %s", idx, len(prompts), exc)` (line 516); U+2192 is NOT encodable in cp1252, so the console StreamHandler raises UnicodeEncodeError and the warning is replaced by a '--- Logging error ---' traceback -- the operator loses the diagnostic the message was designed to deliver ("fail loud").

```
            f"magpie parse failed: user→assistant marker not found "
```

### 75. pretrain_enigma.py:170 - [minor/CONFIRMED/conventions] (found by: conventions)

The LIVE pretrain script prints em-dashes (U+2014) in at least four console output lines (170, 287, 483, 546), violating the ASCII-console rule that explicitly names this character.

**Failure scenario:** Running `python pretrain_enigma.py` with a missing tokenizer, an active val-gen window, `--sanity`, or a non-finite-loss checkpoint skip emits non-ASCII to the cp1252 console/redirected train_large.log; CLAUDE.md states this class of output breaks on this box, and the project's own CODE_REVIEW.md counted em-dash lines as ASCII-rule violations when it fixed the collector batch -- these survivors sit in the most-run script in the repo.

```
        print(f"  (tokenizer unavailable — training on raw IDs: {exc})", flush=True)
```

### 76. enigma_engine/core/adaptive_trainer.py:217 - [minor/CONFIRMED/conventions] (found by: conventions)

TrainingPlan.summary() builds its human-readable display string with a U+2190 left arrow, the same cp1252-crash bug class that was fixed in TrainingQueue.summary().

**Failure scenario:** Any caller that prints the plan summary (its documented purpose: "Human-readable summary of plan state") hits UnicodeEncodeError on the cp1252 console because U+2190 is not in cp1252; training_queue.py:486 shows the repo already ruled summaries must be ASCII ("ASCII only: summary() prints to the Windows cp1252 console") -- this sibling summary() was missed.

```
            f"Training Plan: {self.student_name} ← {self.trainer_name}",
```

### 77. enigma_engine/training/training_monitor.py:361 - [minor/CONFIRMED/reuse] (found by: training-aux)

The NaN-aware moving-average computation in get_chart_data is a verbatim duplicate of moving_average(), so the two copies must be kept in sync by hand.

**Failure scenario:** get_chart_data() (lines 361-376) reimplements the identical sliding-window sum with the same NaN/inf guard logic as moving_average() (lines 317-334). A future correctness fix or window-semantics change applied to one copy silently leaves the other computing a different chart series; there is no shared helper, so a reviewer must notice both sites.

```
        for i in range(len(losses_snap)):
```

### 78. finetune_enigma.py:270 - [minor/CONFIRMED/conventions] (found by: conventions)

The live SFT script prints em-dashes in console output at lines 270 and 383, violating the ASCII-console rule.

**Failure scenario:** Every first instruct pass over a base checkpoint prints the chat-row re-init banner (line 270), and every `--sanity` run prints line 383, both with U+2014 to the cp1252 console -- the same violation class CODE_REVIEW.md counted and fixed elsewhere ("6 more em-dash lines violate the ASCII rule"), now in a live-pipeline script.

```
f"init: chat-token embedding rows {rows} re-initialized (mean + noise) — "
```

### 79. enigma_engine/core/model.py:220 - [minor/CONFIRMED/conventions] (found by: conventions)

logger.info calls in Enigma.__init__ embed U+2192 arrows (lines 220-221 vision projection, line 228 audio projection), which cp1252 cannot encode at all.

**Failure scenario:** Constructing any Enigma model with `vision_hidden_size` or `audio_hidden_size` set (multimodal config) while a console StreamHandler is attached raises UnicodeEncodeError inside logging on the cp1252 console -- the info line is swallowed into a '--- Logging error ---' traceback spam at every model construction.

```
                f"{self.config.vision_hidden_size} → {self.config.dim} "
```

### 80. enigma_engine/core/weight_mapping.py:338 - [minor/CONFIRMED/conventions] (found by: conventions)

logger.info (line 338) and logger.debug (line 201) messages contain U+2192 arrows, unencodable in cp1252.

**Failure scenario:** Importing foreign weights through the shape-matching path logs `Shape-matched: {name} → tok_embeddings.weight` at INFO level; on the cp1252 console the encode fails and logging emits an error traceback instead of the mapping receipt, hiding exactly the audit trail weight mapping is supposed to produce.

```
                logger.info(f"Shape-matched: {name} → tok_embeddings.weight")
```

## Appendix: 18 unverified candidates (verifier lost to session limit)

Treat these as plausible-but-unconfirmed; each needs a verification pass.

### U. collect_pretraining_data.py:3171 - [unverified/correctness]

The gated-access error message directs the user to accept the license for `bigcode/the-stack-v2`, but the code loads `bigcode/the-stack` (v1), so following the guidance does not fix the auth failure.

**Failure scenario:** User runs `--code 10` without HF access. load_dataset("bigcode/the-stack", ...) raises a gated/403 error; the handler prints 'accept the license at https://huggingface.co/datasets/bigcode/the-stack-v2'. The user accepts the v2 license, reruns, and is STILL denied because v1 (bigcode/the-stack) is the gated dataset actually being read. The docstring itself states v1 is loaded, confirming the URL is a copy-paste error.

### U. collect_pretraining_data.py:1616 - [unverified/reuse]

fetch_fineweb_edu duplicates ~160 lines of _fetch_hf_streaming almost verbatim instead of delegating to it like every other HF source.

**Failure scenario:** openwebtext/c4/dclm/finemath all route through the generic _fetch_hf_streaming (which supports filter_ai=False), but FineWeb-Edu has its own near-identical copy of the stream/batch/resume/flush logic. The two copies have already drifted (the generic one guards eta with `if speed_mb > 0`, the fineweb copy at line 1720 does not), so every future fix to the streaming/resume path must be made in two places and can be forgotten in one.

### U. enigma_engine/core/progressive_growing.py:482 - [unverified/correctness]

_init_identity_layer() creates attention biases when use_bias=True but omits all feed-forward biases (w1/w2/w3 or up/down .bias) and norm biases, so a use_bias model's grown state dict is missing keys and the 'depth growth is output-preserving' guarantee is broken.

**Failure scenario:** Grow a use_bias=True model in depth (identity layers inserted). The emitted state dict has layers.N.attention.wq.bias etc. but no layers.N.feed_forward.w2.bias / w1.bias / w3.bias (model_components.py builds all FFN Linears with bias=config.use_bias). load_state_dict(strict=True) fails with missing keys; with strict=False the model keeps nn.Linear's default uniform-random w2.bias, which is added into the residual stream even though w2.weight is zero -> the grown model does NOT reproduce the source model's output, contradicting the module docstring's output-preservation claim. Same omission for ls_attn/ls_ffn when use_layer_scale=True and for LayerNorm biases when use_rms_norm=False, both of which _expand_layer DOES handle for mapped layers.

### U. serve_enigma.py:691 - [unverified/correctness]

In instruct mode, assistant content generated on intermediate builtin-tool hops is streamed to SSE clients but silently dropped from non-stream responses — the same request returns different content depending on stream=true/false, and the stream path glues hop texts together with no separator.

**Failure scenario:** Model answers a math ask with "Let me work that out. <|tool_call|>{calculate...}<|/tool_call|>", serve executes calculate and loops. stream=true: the client already received "Let me work that out." as deltas during hop 0 (lines 586-604 stream content_ids before _loop_on_builtins is known), then the hop-1 answer is appended directly after it with no newline. stream=false: `out` is overwritten each hop and line 691 joins only the LAST hop's content, so "Let me work that out." never reaches the client. Identical prompts thus produce divergent transcripts across the two modes.

### U. serve_enigma.py:857 - [unverified/correctness]

GET /v1/memory with k=0 returns ALL memories instead of none: the negative-slice fallback `MEMORY.all()[-k:]` degenerates to `[0:]` when k is 0 (and inverts for negative k).

**Failure scenario:** A client (or the Odysseus UI) calls GET /v1/memory?k=0 intending an empty page or a count-only probe: `[-0:]` slices the whole list, dumping every stored record; `count` (len(MEMORY)) is unaffected so the mismatch is invisible. With q set the same k=0 flows into BM25 search which returns up to 0 records — the two branches disagree about what k=0 means. Negative k (e.g. -5) returns everything but the 5 oldest instead of erroring.

### U. enigma_engine/core/model.py:1096 - [unverified/conventions]

quantize(mode='int8_static') and quantize(mode='int4' without bitsandbytes) silently substitute dynamic INT8 and return success, violating the 'fail honestly (feature absent) rather than guess' rule.

**Failure scenario:** A caller requesting static INT8 (calibrated) or true INT4 gets a dynamically-INT8-quantized model back with a successful return; downstream memory/accuracy expectations (INT4 is half the weight footprint of INT8) are wrong with no exception -- the same module's from_huggingface/from_gguf/from_onnx were deliberately converted to honest NotImplementedError per this rule, but the quantization fallbacks were left substituting.

### U. enigma_engine/core/plugin_loader.py:227 - [unverified/conventions]

A cluster of logger warning/error lines across core modules still emits em-dashes (U+2014) in console-bound log output: plugin_loader.py 227/239/246/271, lora_utils.py 120/947/1008, hardware_detection.py 82, dataset.py 297/347, model_registry.py 209.

**Failure scenario:** These are runtime operator-facing warnings (untrusted plugin skipped, OOM abort, oversized file skipped, corrupt checkpoint) emitted with non-ASCII on the cp1252 console -- exactly the survivors the fixed collector batch (CODE_REVIEW.md: '6 more em-dash lines violate the ASCII rule') shows the project treats as violations; the cost is inconsistent enforcement of a stated repo-wide rule in messages meant to be read when something is going wrong.

### U. serve_enigma.py:594 - [unverified/simplification]

The SSE chunk envelope dict ({id, object, created, model, choices}) is copy-pasted inline eight times across the two streaming paths (six chat.completion.chunk blocks, two text_completion blocks), each wrapped in the same 'data: ' + json.dumps(...) + '\n\n' boilerplate.

**Failure scenario:** Any change to the envelope (e.g. adding usage or system_fingerprint to chunks, changing the id scheme) requires eight coordinated edits across serve_enigma.py lines ~589-661, 740-765, and 797-823; a missed one ships an inconsistent stream. A 5-line helper like _sse(obj_type, choices) -> str built once from cid/created/MODEL_ID collapses ~90 lines of duplicated literal into one place with identical output.

### U. enigma_engine/core/kv_cache.py:1385 - [unverified/simplification]

StreamingLLMCache.get is a pure pass-through override that only calls super().get(up_to_position), adding no behavior over KVCache.get.

**Failure scenario:** Six dead lines whose docstring ('returns sinks + recent window') implies streaming-specific selection logic that does not exist -- the sink/window layout is maintained entirely by update(), and get() is byte-for-byte the parent's behavior. A reader debugging retrieval order audits this override expecting eviction-aware indexing and finds none. Simpler form: delete the method; Python resolves to KVCache.get identically.

### U. make_sft_data.py:449 - [unverified/altitude]

The tool system-prompt (preamble + 'Available tools:' block) is hand-duplicated in make_sft_data._system instead of being rendered by chat_format.render_tools_system, so the byte-level train==serve contract is maintained by comment discipline across three files.

**Failure scenario:** Someone tweaks the wording, the single-vs-double newline, or the spec-JSON key order in make_sft_data._system for the next SFT run and retrains; serve_enigma._with_context (lines 495-504) and chat_format.render_tools_system still emit the old byte shape, so the served system prompt no longer matches what the model was trained on -- the exact drift class the model is measured to be sensitive to (chat_format._flat_params docstring: given an unseen schema shape 'it mimics the schema shape in its arguments', measured 2026-07-05). Nothing fails loudly; tool-call quality silently degrades.

### U. dpo_enigma.py:115 - [unverified/altitude]

Enigma checkpoint loading/validation is re-implemented in four entry points with three different strictness conventions, including a duplicated freqs_cis/causal_mask missing-key filter and one loader (sample_enigma) with no architecture check at all.

**Failure scenario:** The buffer-name filter is a truth that must be edited in two places: if the model gains or renames a non-persistent buffer, finetune_enigma.py:260 and dpo_enigma.py:115 must both change or one script hard-fails (or silently passes) on valid checkpoints. Meanwhile sample_enigma.py:47 loads with bare strict=False and no missing/unexpected check, so pointing it at a checkpoint from a different architecture (e.g. a merged or foreign .pth with a config dict) loads partially-random weights and prints garbage samples with no diagnostic -- the failure mode the other three loaders each re-built their own guard against. The 'is an Enigma checkpoint (model_state_dict + config)' probe is also triplicated (serve 87-88, finetune 217-218, dpo 106-107).

### U. enigma_engine/core/model.py:1336 - [unverified/correctness]

export_to_safetensors always crashes: safetensors.save_file refuses the weight-tied state dict (output.weight shares storage with tok_embeddings.weight, and tying is unconditional at model.py:261).

**Failure scenario:** Any call to model.export_to_safetensors(path) on any Enigma instance raises RuntimeError before writing anything. Empirically reproduced with safetensors 0.7.0: "Some tensors share memory... [{'tok_embeddings.weight', 'output.weight'}]". The documented from_safetensors round-trip is therefore impossible to produce with this exporter; the repo's atomic_safetensors_save helper has the same gap and no caller strips the tied key first.

### U. enigma_engine/core/model_presets.py:881 - [unverified/correctness]

config_for_param_target's rope_theta fix is incomplete: the custom (no-close-preset) path still rebuilds ForgeConfig from a handful of fields, silently resetting rope_theta from the reference preset's 500000 to the 10000 default (CODE_REVIEW lists this as Closed, but only the within-20% preset path was fixed via deepcopy at line 805).

**Failure scenario:** config_for_param_target(450_000_000) picks 'large'/'xl' (rope_theta=500000) as the template but returns a custom config with rope_theta=10000 — verified live for 450M/3.5B/60B targets, all returned rope_theta=10000.0. A model built from it gets a 50x-shorter RoPE base than every long-context preset it was scaled from; the regression test (tests/test_audit_regressions.py:183) only covers the preset-match path.

### U. enigma_engine/core/model_components.py:1104 - [unverified/correctness]

The Mixture-of-Depths load-balancing auxiliary loss is computed and stored on the block but never consumed: get_mod_aux_loss() has zero callers, so MoD trains without the balancing term the code documents, and the stored tensor keeps the last batch's autograd graph alive on the module between steps.

**Failure scenario:** Enable use_mixture_of_depths: the router trains with no load-balancing pressure (the documented aux loss is silently dropped — grep shows get_mod_aux_loss defined at line 1174 and referenced nowhere else; the trainer only aggregates get_moe_aux_loss at training/training.py:3308-3309), risking router collapse. Additionally `self._mod_aux_loss = (...)` retains a graph-attached tensor on the module after backward, holding activation memory until the next forward and breaking deepcopy(model) for DPO/KTO reference-model construction, same failure mode as the fixed _shared_kv pinning.

### U. enigma_engine/core/model_utils.py:118 - [unverified/efficiency]

apply_repetition_penalty takes the set-based branch (window=128 < 1000 always) which does a GPU->CPU .tolist() sync plus a Python loop issuing ~3-5 tiny CUDA kernels per unique token, on every decoded token of the serve path (generate_stream defaults repetition_penalty=1.1 and serve_enigma never overrides it).

**Failure scenario:** Serving any chat/completion request: each generated token pays one forced device sync (tokens.view(-1).tolist() on a CUDA tensor) plus up to ~128 iterations x (index + torch.where + scatter-assign) = several hundred kernel launches at ~5us each, i.e. roughly 1-3 ms of pure launch/sync overhead per token -- comparable to or larger than the 182M model's actual decode step, so decode throughput can drop by tens of percent. Cheaper: vectorize with torch.unique(tokens) then one gather + torch.where + one scatter (3 kernels total, no host sync) -- essentially the bincount branch that already exists below but is unreachable because seq_len is capped at the 128-token window.

### U. enigma_engine/core/model.py:897 - [unverified/efficiency]

generate_stream's prefill forward computes the output-head projection and materializes logits for every prompt position, but only logits[:, -1, :] is ever sampled.

**Failure scenario:** Every serve request with a near-full prompt (~960 ids after truncation): the lm_head GEMM runs over all positions (~2 x 960 x 1024 x 4736 = 9.3 GFLOP in non-TF32 fp32) and allocates a [1, 960, 4736] fp32 logits tensor (~18 MB) that is immediately discarded except for one row -- pure waste added to first-token latency on every request (generate() at line 807 pays the same). Cheaper: slice the hidden state to the last position before the output projection when generating (e.g. a last_logits_only path in forward, or self.output(h_normed[:, -1:]) for the use_cache prefill), reducing the head cost by ~960x.

### U. make_sft_data.py:450 - [unverified/reuse]

_system() re-implements chat_format.render_tools_system byte-for-byte (same spec JSON via _tool_spec, same 'Available tools:' block, same TOOL_SYNTAX tail) instead of calling the canonical helper that exists precisely to keep the train==serve tool prompt from diverging.

**Failure scenario:** The 'Available tools' block that the SFT corpus trains on is now defined in two places. Any future edit to render_tools_system (e.g. adding a field to the spec JSON, changing the joiner) changes what serve_enigma.py sends at inference while make_sft_data.py keeps emitting the old shape into training data -- exactly the byte-shape drift chat_format.py's docstring says it exists to prevent, and it fails silently as degraded tool-call accuracy, not an error.

### U. pretokenize_data.py:149 - [unverified/reuse]

The scan/filter/paragraph-dedup loop (os.scandir, .txt filter, read_text errors='replace', MIN_PARAGRAPH_LENGTH gates, split('\n\n'), sha256[:8] seen_hashes set with capacity warning) is a verbatim copy of collect_pretraining_data.combine_all_sources' loop, kept in sync only by hand.

**Failure scenario:** These two copies define WHAT ENTERS THE TRAINING CORPUS, and they have already drifted once with real damage: the closed CODE_REVIEW finding 'pretokenize_data.SOURCE_DIRS omits dclm/finemath/the_stack -- corpora the collector writes never enter tokens.bin, silently' was exactly this dual-maintenance failing. The dedup-cap constant, the min-length rule, or a new source dir changed in one file and not the other silently produces a tokens.bin that disagrees with combined.txt -- no error, just a skewed corpus discovered (if ever) after a multi-week pretrain.

