I'll give you a **deduped list of "top + latest"** AI repos that contain **raw model code** (architectures/training loops) across major areas. Because "latest" changes, I web-browsed just now and biased toward **actively maintained** repos plus well-known "top" ones. ([github.com](https://github.com/open-mmlab/mmagic?utm_source=openai))

```text name=TOP_AND_LATEST_AI_RAW_MODEL_CODE_REPOS_NO_DUPES.txt
# ============================================================
# REVIEW STATUS (March 2026)
# ✅ = Reviewed against Enigma Engine source code
# ⬜ = Not yet reviewed (relevant to Enigma)
#
# STALENESS NOTE (2026-07-18): the LoRA / RLHF / SelfPlay / Trainer /
# rl_training.py / reward_functions.py capability references below are
# HISTORICAL — that whole dormant stack was deleted in the 2026-07-18
# compression pass (git history holds it). Idea-mappings to those files
# now mean "a future bespoke script", not existing code. BLIP eyes note
# is also stale: eyes are NATIVE (her own encoder) since 2026-07-17.
# ============================================================
#
# ENIGMA CAPABILITIES (determines what repos are relevant):
#   - Text LLM: transformer, training, LoRA, DPO, RLHF, SelfPlay
#   - Vision encoder: custom ViT (trainable from scratch), hybrid CNN+ViT
#   - Vision projection + audio projection in main model (forward_multimodal)
#   - train_vision() in Trainer — joint image-text training
#   - ORGANS (local services behind serve_enigma.py flags — the old mods/
#     process zoo is gone; capabilities live in-process now):
#       image-gen (--image-gen) — Stable Diffusion sd-turbo via diffusers;
#                                 imagine tool + /v1/images/generations
#       eyes (--eyes)           — her own distilled ViT captioning; image
#                                 messages captioned into context +
#                                 /v1/images/describe
#       voice (--voice)         — Kokoro-82M TTS; speak tool + /v1/audio/speech
#       ears (--ears)           — faster-whisper ASR; /v1/audio/transcriptions
#       (audiogen/videogen/threed have NO current equivalent — future organs;
#        codegen is just the model itself)
#   - RL: RewardModel, RewardTrainer, RLHFTrainer, SelfPlayTrainer
#
# ALL repos below should be reviewed for patterns that improve
# Enigma's existing capabilities or unlock new ones.
# ============================================================

# -------------------------------------------------------
# LLMs / text (training + from-scratch pretraining)
# -------------------------------------------------------
- Lightning-AI/litgpt               ✅ architecture match, SequentialLR warmup, chunked CE, LoRA
- karpathy/nanoGPT                   ✅ training loop, manual get_lr warmup+cosine, fused AdamW, vocab pad 64
- karpathy/llama2.c                  ✅ same arch as litgpt/nanoGPT (RoPE, RMSNorm, SwiGLU, GQA), no new patterns
- karpathy/llm.c                     ✅ cosine/linear/WSD schedulers, correct warmup decay_ratio, outlier detection
- EleutherAI/gpt-neox                ✅ AnnealingLR correct warmup, DPO/KTO/REINFORCE, grad noise scale, µP
- NVIDIA/Megatron-LM                 ✅ tensor/pipeline/sequence parallelism, fused CUDA kernels — not actionable at Enigma's scale (1-2 GPU local)
- microsoft/DeepSpeedExamples        ✅ ZeRO sharding, activation checkpointing — library integration not code patterns, PyTorch native AMP/checkpoint covers it
- facebookresearch/fairseq           ✅ label smoothing (useful, ~3 lines), bucket batching (overlaps sequence packing roadmap), registry arch overkill for Enigma
- facebookresearch/fairseq2          ✅ Rust+Python rewrite of fairseq, same concepts, no additional patterns
- allenai/OLMo                       ✅ DataCollator padding/attention_mask/label_mask patterns
- huggingface/transformers           ✅ reference implementation — Enigma uses its own architecture, not HF model classes; tokenizer/config loading used via model_contexts for external models
- huggingface/trl                    ✅ DPO loss variants (sigmoid/hinge/ipo/bco), ref model, LoRA+DPO, GRPO/KTO
- OpenRLHF/OpenRLHF                  ✅ GRPO (group ranking, no critic needed), KTO (unpaired prefs), iterative DPO, rejection sampling — 3 actionable patterns
- RWKV/RWKV-LM                      ✅ fundamentally different arch (linear-attention RNN, not transformer), constant-memory inference — not applicable without new model class

# -------------------------------------------------------
# Diffusion / generative image
# Relevant: the image-gen organ uses diffusers (sd-turbo), could add
# fine-tuning, ControlNet, inpainting, SDXL/Flux, scheduler
# improvements, and better prompt handling.
# -------------------------------------------------------
- openai/guided-diffusion            ✅ REVIEWED: Classifier-guided diffusion with DDPM/DDIM schedulers. Key patterns: (1) Classifier guidance formula: score = unconditional_score + scale * grad(log p(y|x)). Enigma imagegen has no guidance control — it calls diffusers pipeline with a prompt string only. (2) DDIM deterministic sampling (fewer steps, same quality) — diffusers already implements this, Enigma just needs to expose scheduler choice to user. (3) Model architecture uses AdaGN (adaptive group norm conditioned on timestep) — standard in diffusion, not applicable to Enigma's LLM. Verdict: low actionable value — guidance and schedulers are better accessed via the diffusers library directly (next entry).
- huggingface/diffusers              ✅ REVIEWED: Enigma already imports this for SD inference. Key patterns Enigma's image-gen organ is missing: (1) **ControlNet** — condition generation on edge maps, depth maps, poses. `StableDiffusionControlNetPipeline` is ~5 lines to swap in, enables structure-guided generation. (2) **LoRA for SD** — `load_lora_weights()` lets users load community LoRA styles without retraining. ~3 lines. (3) **img2img** — `StableDiffusionImg2ImgPipeline` takes input image + prompt, outputs modified image. Enigma only has txt2img. (4) **Inpainting** — `StableDiffusionInpaintPipeline` paints into masked regions. Very useful with images the eyes organ ingests. (5) **Scheduler options** — Enigma uses whatever default the pipeline loads. Exposing DPMSolverMultistep (fast, 20 steps) vs EulerDiscrete (quality, 50 steps) gives user speed/quality tradeoff. (6) **SDXL pipeline** — higher quality base model, same API. (7) **Prompt weighting** — `compel` or `(word:1.5)` syntax for emphasizing prompt parts. Not built in but commonly added. Verdict: 5 high-value additions (ControlNet, img2img, inpainting, LoRA loading, scheduler choice), all low effort since diffusers does the heavy lifting.
- open-mmlab/mmagic                  ✅ REVIEWED: Unified framework for image restoration/generation. Covers super-resolution (ESRGAN, Real-ESRGAN, SwinIR), inpainting (DeepFillv2, MAT), style transfer (AdaIN), denoising. Key patterns: (1) **Super-resolution** — Real-ESRGAN upscales 4x with pretrained weights. Could improve imagegen output quality by upscaling after generation. Single pretrained model, ~10 lines integration. (2) **Inpainting architecture** — partial convolutions that only operate on valid (non-masked) pixels. More principled than diffusion inpainting for small edits. (3) Configuration registry pattern (MMCV) is heavily over-engineered for Enigma. Verdict: super-resolution post-processing is the one actionable pattern — upscale generated images to higher resolution. Everything else overlaps with diffusers or is architecturally different.
- mlfoundations/open-diffusion       ✅ REVIEWED: Open-source diffusion training. Key patterns: (1) **SD fine-tuning on custom data** — uses LAION-style (image, caption) pairs with CLIP text encoder frozen, only UNet trained. Relevant if user wants to fine-tune imagegen on their own images. (2) **DreamBooth** — fine-tune SD to learn specific subjects from 5-10 images. Very popular for personalization. (3) **Text encoder fine-tuning** — optional unfreeze of CLIP text encoder for domain adaptation. Verdict: DreamBooth-style fine-tuning (teach SD a new subject from few images) is the highest-value pattern here. Could expose in the image-gen organ as "train on my images" — but significant compute requirement (needs GPU, ~20min per subject).

# -------------------------------------------------------
# GANs / generative modeling
# Relevant: could add GAN-based image generation as an
# alternative backend in the image-gen organ, or use discriminator
# patterns for training quality scoring.
# -------------------------------------------------------
- NVlabs/stylegan2-ada-pytorch       ✅ REVIEWED: Adaptive discriminator augmentation (ADA) enables high-quality GAN training on limited data (as few as ~1,000 images). Key patterns: (1) **ADA** — dynamically adjusts augmentation probability during training based on discriminator overfitting signal (r_t metric). Prevents discriminator from memorizing small datasets. Concept applicable to Enigma's vision encoder training: monitor train/val gap and auto-increase augmentation. (2) **Progressive growing** — trains low-res first, adds layers for higher res. Not directly applicable to ViT but the curriculum concept (easy→hard) matches Enigma's adaptive training. (3) **Discriminator as quality scorer** — GAN discriminator inherently learns "is this real?" which is a quality metric. Could train a lightweight discriminator on good vs bad text generations as an alternative to neural reward model. Verdict: ADA overfitting detection concept is transferable. Discriminator-as-reward-scorer is a creative alternative to RewardModel but lower priority than fixing PPO. StyleGAN itself is too GPU-heavy for local generation vs diffusion.
- NVlabs/stylegan3                   ✅ REVIEWED: Alias-free generation — addresses texture sticking artifacts in StyleGAN2. Core innovation is continuous signal processing (anti-aliased up/downsampling, filtered nonlinearities). Key patterns: (1) **Filtered activations** — applies low-pass filter after every nonlinearity to prevent aliasing. Theoretically applicable to any neural network but overhead is high and benefit is image-generation-specific. (2) **Fourier features** instead of learned positional embeddings for spatial consistency. Enigma uses learnable positional embeddings in ViT — Fourier features are an alternative but not clearly better for ViT. Verdict: no actionable patterns for Enigma. The alias-free innovations are specific to continuous generative architectures, not applicable to transformers or ViT.

# -------------------------------------------------------
# Vision (classification/detection/segmentation)
# Relevant: Enigma has a custom ViT (vision_encoder.py) trained
# from scratch. These repos have pretrained weights, better
# architectures, data augmentation, and training recipes that
# would improve Enigma's vision capabilities.
# -------------------------------------------------------
- pytorch/vision                     ✅ REVIEWED: torchvision is the standard image transform library. Key patterns Enigma is missing: (1) **Training augmentation pipeline** — `transforms.Compose([RandomResizedCrop(224), RandomHorizontalFlip(), ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), ToTensor(), Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])`. Enigma's `preprocess_image()` only does resize+normalize to [-1,1] — no randomness, no augmentation. This is the #1 vision training improvement. (2) **ImageNet normalization** — standard mean/std `[0.485,0.456,0.406]/[0.229,0.224,0.225]` used by ALL pretrained models. Enigma normalizes to [-1,1] which is incompatible with pretrained weights. Must match if loading pretrained ViT. (3) **RandAugment / AutoAugment** — learned augmentation policies. `transforms.RandAugment(num_ops=2, magnitude=9)` is the modern standard, applies random combinations of color, geometric, and filter transforms. (4) **CutMix / MixUp** — `torchvision.transforms.v2.CutMix` and `MixUp` blend training images and labels for regularization. Proven 1-3% accuracy improvement on classification. (5) **Functional transforms** — `torchvision.transforms.functional` lets you apply identical random transforms to image+mask pairs (needed for segmentation training). (6) **AugMix** — stochastic depth augmentation chains for robustness. Verdict: 3 high-value items: (a) add training augmentation pipeline to `preprocess_image()`, (b) switch to ImageNet normalization if loading pretrained weights, (c) RandAugment for automated augmentation policy.
- rwightman/timm                     ✅ REVIEWED: 700+ pretrained ViT weights (DINO, CLIP, SigLIP, AIM-v2, DINOv2). Key patterns Enigma lacks: (1) drop_path / stochastic depth — linearly increasing drop rates per layer, major regularization, Enigma has none. (2) LayerScale (init_values=1e-5) — learnable scaling on residual outputs, stabilizes deep training. (3) QK normalization — prevents attention logit explosion on long sequences. (4) PatchDropout — drop random patches during ViT training for regularization+speed. (5) Register tokens (reg_tokens=4) — extra learnable tokens that absorb attention noise artifacts. (6) Newer ViTs use SwiGLU+RMSNorm (same as Enigma LLM, but Enigma ViT already uses this — good). (7) Stochastic depth decay rule: rate increases linearly from 0→max per block. Pretrained weights: `timm.create_model('vit_small_patch16_224', pretrained=True)` is the single highest-value change for Enigma vision.
- facebookresearch/detectron2        ✅ REVIEWED: Object detection and segmentation framework. Architecture: Faster R-CNN, Mask R-CNN, RetinaNet with FPN (Feature Pyramid Network) backbone. Key patterns: (1) **FPN backbone** — multi-scale feature extraction from ResNet/ViT, outputs feature maps at 4+ scales. Essential for detecting objects of different sizes. Enigma's ViT outputs single-scale features (all patches at same resolution). FPN is the standard fix — but adding it to Enigma's ViT requires architectural changes (~50 lines). (2) **RoI pooling** — crops and resizes feature regions to fixed size for per-object classification. Only relevant if adding bounding box detection. (3) **Anchor-based detection** — predefined boxes at multiple scales/ratios. Standard but complex setup. Verdict: FPN multi-scale features from ViT is conceptually valuable (better at different object sizes) but Enigma's vision encoder is used for captioning (global image understanding), not detection. Object detection in the eyes organ would be better served by a pretrained YOLO model (next entry) than modifying the ViT. Lower priority.
- open-mmlab/mmclassification        ✅ REVIEWED: Image classification training recipes. Key patterns: (1) **Training recipes** — complete configs for training ViT from scratch: lr=1e-3, weight_decay=0.3, warmup=5 epochs, total=300 epochs, label_smoothing=0.1, drop_path_rate=0.1, mixup_alpha=0.8, cutmix_alpha=1.0. Enigma's vision training has none of these hyperparameters tuned. (2) **Knowledge distillation** — train small model to mimic large model's outputs. Relevant if wanting to distill a large pretrained ViT into Enigma's small ViT. (3) **EMA (Exponential Moving Average)** — maintain a running average of model weights during training, use for inference. Smooths training noise, ~2 lines to add. Proven improvement for ViT training. Verdict: EMA weight averaging is the most actionable pattern — easy to implement, proven benefit. Training recipe hyperparameters are a good reference for vision encoder training config.
- open-mmlab/mmdetection             ✅ REVIEWED: Detection architectures (YOLO, Faster R-CNN, DETR). Key patterns: (1) **DETR (Detection Transformer)** — end-to-end detection using transformer decoder with learned object queries. Novel architecture but heavy. (2) **Multi-scale feature maps** — same FPN pattern as detectron2. (3) **Deformable attention** — attention that learns which spatial positions to attend to, instead of attending to all positions. Could improve Enigma's attention efficiency on vision tasks. Verdict: deformable attention is interesting for vision (attend to relevant regions, not all patches) but implementation complexity is high (~200 lines + custom CUDA kernel for speed). Not actionable at current stage.
- open-mmlab/mmsegmentation          ✅ REVIEWED: Segmentation architectures. Key patterns: (1) **SegFormer** — lightweight ViT variant with hierarchical features (multi-scale patch embedding). Efficient for dense prediction. (2) **MLP decoder** — simple MLP head on ViT features for segmentation (no complex decoder needed). Shows ViT features are rich enough for pixel-level tasks with minimal overhead. Verdict: no directly actionable patterns for Enigma. Segmentation is a separate task, and the architectural patterns (hierarchical ViT, MLP decoder) don't improve Enigma's current captioning use case.
- open-mmlab/mmaction2               ✅ REVIEWED: Video action recognition. Key patterns: (1) **TimeSformer** — extends ViT to video by adding temporal attention. Two variants: divided attention (spatial-then-temporal per layer) or joint attention (attend across space+time). Enigma's `encode_video_frames()` processes frames independently — no temporal modeling. (2) **Video data pipeline** — decodes video → sample N frames → spatial augmentation per frame → temporal augmentation across frames. Enigma extracts frames but doesn't apply augmentation. (3) **Temporal sampling strategies** — uniform (evenly spaced), dense (consecutive chunk), or multi-scale (different rates for different clips). Enigma uses uniform but with basic PIL frame extraction. (4) **SlowFast** — dual-pathway: slow path (low frame rate, high resolution) + fast path (high frame rate, low resolution). Captures both appearance and motion. Verdict: temporal attention between video frames (TimeSformer pattern) is the key missing piece for Enigma's video understanding. Currently each frame is processed independently — adding temporal attention would let the model understand motion and sequence. Medium effort (~100 lines to add temporal attention heads).
- facebookresearch/pytorchvideo      ✅ REVIEWED: Video model architectures and transforms. Key patterns: (1) **Video transforms** — `UniformTemporalSubsample(num_samples=8)`, `ShortSideScale(size=256)`, `CenterCropVideo(crop_size=224)`, `NormalizeVideo(mean, std)`. Proper video preprocessing pipeline vs Enigma's basic PIL frame extraction. (2) **X3D** — efficient video model that's 4-5x faster than SlowFast. Uses channel-wise separable spatial/temporal convolutions. Good for real-time video understanding. (3) **MViT (Multiscale Vision Transformer)** — pooling attention that reduces sequence length at deeper layers. Handles more frames without quadratic attention cost. (4) **Uniform clip sampling** — standardized way to extract training clips from long videos (random temporal offset, fixed duration). Enigma doesn't do this. Verdict: video preprocessing pipeline (proper temporal sampling + spatial transforms) is the highest-value pattern. MViT's pooling attention for efficient long-sequence vision processing is architecturally interesting but complex. Video transforms are straightforward to add.
- ultralytics/ultralytics            ✅ REVIEWED: YOLOv8 — fast object detection. Key patterns: (1) **Single-shot detection** — detects objects in one forward pass (~5ms on GPU). Enigma's eyes organ captions images (BLIP) but cannot detect objects. (2) **Pretrained models** — `YOLO('yolov8n.pt')` is 6MB, runs in real-time. Could add object detection to the eyes organ alongside captioning. (3) **Python API simplicity** — `model('image.jpg')` returns boxes, labels, confidences. Very clean integration. (4) **Tracking** — `model.track(frame)` adds object tracking across video frames. (5) **Export** — ONNX, TensorRT, CoreML, TFLite. Cross-platform inference. Verdict: YOLO integration for the eyes organ is high-value, low-effort. The eyes organ already ingests images — adding `YOLO('image.jpg')` for object detection alongside captioning gives much richer scene understanding. ~20 lines to integrate. Pretrained YOLOv8-nano is only 6MB.

# -------------------------------------------------------
# Multimodal / vision-language
# Already reviewed — these directly improve Enigma's
# multimodal projections and training.
# -------------------------------------------------------
- salesforce/LAVIS                   ✅ Q-Former learnable bridge (learned queries + cross-attn) >> linear projection, frozen-component 2-stage train — substantial but better than linear proj
- mlfoundations/open_flamingo        ✅ gated cross-attention injected between LLM layers (gate starts at 0), Perceiver Resampler — cleaner vision injection than concatenation
- haotian-liu/LLaVA                  ✅ CLIP vision tower, MLP2x projector, 2-stage train, LoRA excl. projector
- mlfoundations/open_clip            ✅ pretrained CLIP ViT weights >> training VisionEncoder from scratch — from_pretrained_clip() factory would be high-value low-effort

# -------------------------------------------------------
# Speech / audio
# Relevant: Enigma has audio_projection in the model; the ears
# organ uses faster-whisper (ASR), the voice organ uses Kokoro-82M
# (neural TTS, swapped in 2026-07-23). Missing: audio encoder
# training runs, audio feature extraction for multimodal training.
# NOTE: the reviews below were written when the voice organ was
# still pyttsx3/SAPI, so their "neural TTS would be an upgrade"
# verdicts are already banked -- read them for the ARCHITECTURE
# notes (Conformer, mel params, audio tokenization), not for the
# current state of the organ.
# -------------------------------------------------------
- openai/whisper                     ✅ REVIEWED: AudioEncoder architecture is clean and reusable: Conv1d(n_mels→n_state, k=3, p=1) → GELU → Conv1d(n_state→n_state, k=3, stride=2, p=1) → GELU → sinusoidal_pos_embed → N transformer blocks (LayerNorm + MultiHeadAttention + FFN with GELU) → LayerNorm. Dimensions: tiny(n_mels=80, n_state=384, n_head=6, n_layer=4), base(n_state=512, n_head=8, n_layer=6), small(n_state=768, n_head=12, n_layer=12). Key insight: sinusoidal positional embeddings (not learned) for audio sequences. Stride-2 conv downsamples mel frames 2x. This is the exact template for Enigma's missing audio encoder — Conv1d frontend + transformer blocks, outputs features that feed into audio_projection.
- espnet/espnet                      ✅ REVIEWED: End-to-end speech processing (ASR, TTS, speech translation). Key patterns: (1) **Conformer encoder** — combines convolution (local patterns) with transformer attention (global patterns). Architecture: FFN → MultiHeadAttention → Conv → FFN per block (macaron-style). Outperforms pure transformer for audio by ~10-15% WER. Enigma's Whisper-inspired audio encoder template uses pure transformer — Conformer is the modern upgrade. ~30 lines to add Conv module inside each audio encoder block. (2) **Mel spectrogram extraction** — espnet uses 80-dim mel filterbanks, 25ms window, 10ms hop, log-mel scaling. Same as Whisper. Enigma doesn't have mel extraction yet — this confirms the standard params. (3) **CTC loss** — Connectionist Temporal Classification for sequence-to-sequence alignment without explicit alignment labels. Used alongside attention loss in hybrid CTC/attention ASR. Relevant if Enigma adds speech recognition training. (4) **TTS: Tacotron2/FastSpeech2** — text→mel→waveform pipeline. FastSpeech2 is non-autoregressive (fast). Enigma's voice organ uses pyttsx3/SAPI (system TTS) — local neural TTS would be a major upgrade but requires pretrained vocoder (HiFi-GAN). Verdict: Conformer encoder architecture is the main actionable finding — a Conv module inside each transformer block improves audio processing. Mel spectrogram params confirmed. Neural TTS is aspirational (needs significant infrastructure).
- speechbrain/speechbrain            ✅ REVIEWED: Speech toolkit with easy-to-use APIs. Key patterns: (1) **Audio feature extraction** — `speechbrain.processing.features.MFCC` and `Fbank` (mel filterbank). Multiple feature types: MFCC (compact, traditional), Fbank (richer, modern), raw waveform. Enigma has audio_projection but no audio feature extractor — SpeechBrain's Fbank implementation is clean reference for extracting mel features before feeding into audio encoder. (2) **Speaker embedding** — ECAPA-TDNN model produces speaker identity vectors from audio. Could enable speaker-aware conversation (identify who's speaking in multi-speaker audio). (3) **Audio augmentation** — speed perturbation (0.9-1.1x), SpecAugment (mask frequency/time bands), noise injection, room impulse response simulation. Same concept as image augmentation but for audio. Enigma has zero audio augmentation. (4) **Pretrained models** — `speechbrain/asr-wav2vec2-commonvoice-en` etc. Fine-tunable on custom data. (5) **VAD (Voice Activity Detection)** — detect speech segments in audio. Useful for the ears organ to skip silence. Verdict: audio feature extraction (Fbank/mel) and SpecAugment are the actionable patterns. Speaker embedding is a nice future capability. VAD could improve ears-organ efficiency.
- wenet-e2e/wenet                    ✅ REVIEWED: Production speech recognition with streaming support. Key patterns: (1) **Streaming/chunked attention** — processes audio in fixed-size chunks with look-ahead, enabling real-time ASR without waiting for full utterance. Enigma's ears organ processes complete audio files — streaming would enable live transcription. (2) **U2 (Unified Streaming/Non-streaming)** — single model architecture that works in both streaming and offline modes. Uses dynamic chunk training where chunk size varies during training (including full-context). At inference, choose chunk size for latency/accuracy tradeoff. (3) **CTC prefix beam search** — improved decoding that uses CTC scores to guide attention decoder. Better accuracy than pure attention decoding. (4) **Shared encoder, two decoders** — CTC + attention decoders share the same encoder features. CTC for streaming, attention for accuracy. Verdict: streaming ASR (chunked attention) is the key pattern for a future live-mic mode in the ears organ. U2 architecture is elegant but complex to implement. Chunked processing for real-time transcription is the practical takeaway.
- NVIDIA/NeMo                        ✅ REVIEWED: Large-scale speech models with modern architectures. Key patterns: (1) **FastConformer** — optimized Conformer with 8x downsampling (vs 4x standard) for faster processing. Uses depthwise separable convolutions. Speed improvement without quality loss. (2) **Multi-task learning** — single audio encoder trained on ASR + speaker ID + language ID simultaneously. Shared representations, better generalization. Enigma could train audio encoder on captioning + speech recognition jointly. (3) **TTS: XTTS / VITS** — modern TTS architectures that clone voice from ~10s of reference audio. VITS is end-to-end (text→waveform, no vocoder needed). Most relevant for a voice-organ upgrade from pyttsx3. (4) **Audio tokenization (EnCodec-style)** — converts audio to discrete tokens that an LLM can process. This bridges the gap between audio and text — the LLM treats audio tokens same as text tokens. Enigma's audio_projection is continuous features → linear projection. Tokenized audio would allow audio in the same token stream as text. (5) **Prompt tuning for ASR** — adapt trained ASR model to new domains with small prompt prefix instead of full fine-tuning. Verdict: audio tokenization (discrete audio tokens for LLM) is the most impactful long-term pattern — it unifies audio and text modalities in the same token space. FastConformer and VITS TTS are good architecture references. Multi-task audio training is practical.

# -------------------------------------------------------
# Graph neural networks
# Lower priority — reviewed for knowledge graph 
# capabilities in RAG, or graph-based reasoning.
# -------------------------------------------------------
- pyg-team/pytorch_geometric         ✅ REVIEWED: GNN architectures for graph-structured data. Key patterns: (1) **Message passing framework** — nodes aggregate information from neighbors via learnable functions. Core concept: `node_feature = UPDATE(node, AGGREGATE(neighbors))`. Could enhance RAG by building a knowledge graph of documents/facts where related docs are connected, and GNN propagates relevance scores. More principled than vector similarity alone. (2) **Graph attention (GAT)** — attention-weighted neighbor aggregation. Similar to transformer attention but over graph edges not sequence positions. (3) **Node/edge embedding** — standard patterns for encoding structured data as vectors. Could represent memory facts as a graph (fact→related_fact edges) for better memory.search() retrieval. Verdict: knowledge graph for RAG/memory is conceptually valuable but requires significant infrastructure (graph construction, entity extraction, relationship identification). Not actionable until RAG system is more mature. Lower priority.
- dmlc/dgl                           ✅ REVIEWED: Deep graph library, alternative to PyG. Similar API patterns, less community support than PyG. Key pattern: **heterogeneous graphs** — different node/edge types in same graph (e.g., "user"→"asked"→"question", "question"→"related_to"→"topic"). Natural fit for conversation memory graphs. Same verdict as PyG: conceptually interesting but low priority until RAG is more mature.

# -------------------------------------------------------
# Reinforcement learning
# Relevant: Enigma has RewardModel, RLHFTrainer, SelfPlayTrainer
# but they're partially implemented. These repos have production
# RL training loops.
# -------------------------------------------------------
- Farama-Foundation/Gymnasium        ✅ REVIEWED: Standard RL environment interface. Key patterns: (1) **Environment abstraction** — `env.reset() → observation`, `env.step(action) → (obs, reward, done, info)`. Clean interface that decouples policy from environment. Relevant if building agent/tool-use RL where model learns when to call commands. Enigma's SelfPlayTrainer generates + scores but doesn't use this environment pattern. (2) **Observation/action spaces** — typed specifications (`Box`, `Discrete`, `MultiBinary`) that define valid inputs/outputs. Useful for constraining what the model can do in RL training. (3) **Wrappers** — composable environment transformations (TimeLimit, RecordVideo, NormalizeObservation). Pattern of wrapping core object with optional behaviors. Verdict: the `step(action) → reward` abstraction could clean up SelfPlayTrainer's loop, but it's a design pattern, not a bug fix. Lower priority than fixing PPO fundamentals (SB3 review).
- DL-RM/stable-baselines3            ✅ REVIEWED: PPO implementation reveals Enigma's RLHFTrainer is far more broken than 'clip_range unused'. Missing: (1) NO value function — PPO needs V(s) head to compute advantages, Enigma has none. (2) NO advantage estimation (GAE) — advantages = rewards - values, Enigma uses raw rewards directly. (3) NO entropy bonus — ent_coef * entropy_loss prevents mode collapse, Enigma doesn't compute entropy. (4) NO proper clipped surrogate loss — ratio = exp(new_logp - old_logp), loss = -min(adv*ratio, adv*clamp(ratio, 1-clip, 1+clip)). (5) NO KL divergence early stopping (target_kl). (6) NO advantage normalization ((adv - mean) / std). (7) NO multi-epoch updates over rollout buffer. SB3 defaults: lr=3e-4, clip=0.2, gae_lambda=0.95, n_epochs=10, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5. Enigma needs a complete PPO rewrite, not a patch.
- ray-project/ray                    ✅ REVIEWED: Distributed RL framework (RLlib). Key patterns: (1) **Replay buffer implementations** — prioritized experience replay (PER) where high-TD-error transitions are sampled more often. Enigma's RL training has no replay buffer — each rollout is used once then discarded. PER would improve data efficiency for RLHF. (2) **Multi-GPU training** — model parallelism + data parallelism for RL. Not applicable to Enigma's local setup. (3) **Algorithm implementations** — PPO, SAC, DQN with distributed rollouts. PPO implementation is less readable than SB3 (distributed complexity). (4) **Offline RL** — train policies from pre-collected data (no live interaction). Could enable training RLHF from saved conversation ratings instead of live generation. Verdict: replay buffer (prioritized experience replay) is the one practical pattern — store generated responses + rewards, sample high-error ones more often. Offline RL from saved ratings is conceptually interesting but needs infrastructure.
- google-deepmind/acme               ✅ REVIEWED: Modular RL research framework. Key patterns: (1) **Actor-learner separation** — clean split between data collection (actor) and parameter updates (learner). Enigma's RLHFTrainer mixes generation and training in same loop. Separation would make the code cleaner but isn't performance-critical for local use. (2) **Reverb replay system** — efficient replay buffer with priorities. Same finding as ray. (3) **Distributional RL** — model the distribution of returns, not just expected value. Theoretically better but much more complex than standard value estimation. (4) **MCTS (Monte Carlo Tree Search)** — search-based planning (AlphaZero-style). Could enable multi-step command planning but massive implementation effort. Verdict: actor-learner separation is a clean design pattern that could improve RLHFTrainer readability. Not a bug fix, more of a code quality improvement. Lower priority than PPO rewrite.
- openai/spinningup                  ✅ REVIEWED: Educational RL implementations — clean, readable reference code. Key patterns: (1) **PPO implementation** — ~200 lines, matches SB3 findings exactly. Cleaner to read: GAE computation is a clear backward pass over rewards+values, surrogate loss is explicit ratio*advantage with clipping, entropy bonus added to loss. Better reference for Enigma's PPO rewrite than SB3 (which has distributed/vectorized complexity). (2) **VPG (Vanilla Policy Gradient)** — simplest policy gradient. Good starting point before PPO. Shows how to compute log-probs, rewards-to-go, and baseline subtraction. (3) **SAC (Soft Actor-Critic)** — off-policy RL with entropy maximization. Uses two Q-networks + one policy. More sample-efficient than PPO but harder to implement for LLM training. (4) **Reward-to-go vs full returns** — computes discounted cumulative rewards from each timestep forward (not from beginning). More accurate advantage estimation. Verdict: spinningup's PPO is the best reference for Enigma's PPO rewrite — cleaner than SB3, same components, ~200 lines. Confirms all 7 missing components identified in SB3 review. Use this as the primary implementation template.

# -------------------------------------------------------
# Robotics + embodied AI
# Relevant for agent capabilities: tool use, planning,
# multi-step reasoning, real-world interaction patterns.
# -------------------------------------------------------
- ARISE-Initiative/robomimic         ✅ REVIEWED: Imitation learning from demonstrations. Key patterns: (1) **Behavioral cloning (BC)** — train policy to mimic expert actions via supervised learning on (state, action) pairs. Directly applicable to Enigma: train model to mimic user's command patterns by collecting (context, user_action) pairs from chat history. The model learns "in this situation, the user usually does X." (2) **BC-RNN** — sequential behavioral cloning using LSTM/GRU to model temporal dependencies in action sequences. Could predict multi-step command chains. (3) **Observation history** — uses windowed observation history (N recent frames) not just current frame. Similar pattern to Enigma's chat history truncation, validates the "recent window" approach. (4) **Action chunking** — predict a sequence of actions at once instead of one at a time. Could predict "user will ask about X, then edit file Y, then run Z" from context. Verdict: behavioral cloning from chat history is a realistic new training mode — learn user patterns from saved sessions. The data already exists (session JSON files with role/content pairs). Medium effort, unique capability.
- real-stanford/diffusion_policy     ✅ REVIEWED: Diffusion models for action prediction. Key patterns: (1) **Action as diffusion** — instead of predicting next action directly, generate a trajectory of actions via iterative denoising. More multimodal (handles ambiguous situations where multiple actions are valid). (2) **Temporal ensembling** — average overlapping predicted action chunks for smoother execution. Reduces jitter. Verdict: creative architecture but not applicable to text-based AI assistant. Diffusion-based action prediction makes sense for continuous control (robotics), not discrete command generation. No actionable patterns.
- huggingface/lerobot                ✅ REVIEWED: Robot learning toolkit. Key patterns: (1) **Action tokenization** — converts continuous actions into discrete tokens for transformer processing. Same concept as audio tokenization (NeMo): discretize continuous signals so LLM can process them. (2) **Dataset format** — standardized (observation, action, reward, done) tuples stored as Parquet files. Efficient storage + random access. Enigma stores chat history as JSON arrays — Parquet would be more efficient for large session archives but adds a dependency. (3) **Trajectory prediction** — predict future trajectory from current state. For Enigma: predict likely conversation direction from current context (proactive responses). Verdict: action tokenization validates the pattern of discretizing non-text modalities for LLM processing. No directly actionable patterns for current Enigma capabilities.
- isaac-sim/IsaacLab                 ✅ REVIEWED: GPU-accelerated RL environments using NVIDIA Isaac Sim. Key patterns: (1) **Vectorized environments** — run thousands of environment instances in parallel on GPU. Not applicable to Enigma's single-user local setup. (2) **Sim-to-real transfer** — domain randomization during training makes policies robust to real-world variation. Concept parallels data augmentation for generalization. Verdict: no actionable patterns. Infrastructure is GPU-accelerated physics simulation, not applicable to text/vision AI.
- NVlabs/OmniIsaacGymEnvs           ✅ REVIEWED: Similar to IsaacLab (GPU-accelerated RL environments). Same verdict: no actionable patterns for Enigma. Physics simulation infrastructure.

# -------------------------------------------------------
# Cameras / calibration + SLAM/VIO
# Relevant: the eyes organ ingests images; camera input for live
# video understanding / spatial awareness could come later.
# -------------------------------------------------------
- ethz-asl/kalibr                    ✅ REVIEWED: Camera-IMU calibration tool. Computes intrinsic (focal length, distortion) and extrinsic (relative position) parameters. Only relevant if Enigma adds physical camera input with precise 3D geometry requirements. Verdict: not actionable. Enigma's eyes organ captions images, not physical camera feeds. If webcam input is added later, OpenCV's calibration is simpler.
- AprilRobotics/apriltag             ✅ REVIEWED: Fiducial marker detection (square markers with coded patterns). Used for AR, robotics localization. Verdict: not actionable for Enigma. No use case for physical marker detection in a text/vision AI assistant.
- UZ-SLAMLab/ORB_SLAM3              ✅ REVIEWED: Visual SLAM (Simultaneous Localization and Mapping). Builds 3D map of environment from camera video while tracking camera position. Key patterns: (1) **Feature extraction + matching** — ORB features for fast visual matching between frames. (2) **Loop closure** — detects when camera revisits a previous location, corrects accumulated drift. (3) **Multi-map** — maintains multiple maps that can be merged. Verdict: conceptually interesting for spatial awareness but Enigma operates on screen content, not physical camera feeds. No actionable patterns unless adding robotic/AR capabilities.
- HKUST-Aerial-Robotics/VINS-Fusion ✅ REVIEWED: Visual-inertial odometry. Fuses camera + IMU for precise motion tracking. Verdict: not actionable. Requires hardware sensor input, not applicable to desktop AI.
- rpng/open_vins                     ✅ REVIEWED: Similar to VINS-Fusion, visual-inertial navigation. Verdict: same — not actionable for desktop AI.

# -------------------------------------------------------
# 3D reconstruction / SfM
# Relevant: there is NO 3D organ today (the old threed mod is
# gone). These repos are the prior art for real 3D-from-images
# if a future 3D organ gets built.
# -------------------------------------------------------
- colmap/colmap                      ✅ REVIEWED: Structure-from-Motion — reconstructs 3D point cloud + camera poses from unordered images. Key patterns: (1) **Feature matching pipeline** — SIFT features → exhaustive/sequential matching → geometric verification → bundle adjustment. Full pipeline from input images to 3D reconstruction. (2) **Dense reconstruction** — after sparse SfM, densifies to detailed surface mesh. (3) **Integration pattern** — COLMAP is a standalone binary, typically called via subprocess with config files. A future 3D organ could shell out to it. Verdict: most capable 3D-from-images solution but C++ binary dependency is heavy. If user wants "upload photos → get 3D model," this is the pipeline. Practical integration: call the COLMAP binary via subprocess from a future 3D organ. Medium effort, high capability.
- openMVG/openMVG                    ✅ REVIEWED: Multi-view geometry library. Cleaner API than COLMAP, library-based (not just binary). Key patterns: (1) **Incremental SfM** — add images one at a time, growing the reconstruction. COLMAP also does this. (2) **Global SfM** — estimate all camera poses simultaneously (faster but less robust). Trade-off: speed vs accuracy. (3) **Python bindings** — easier integration than COLMAP subprocess. Verdict: same capability as COLMAP but with Python bindings for cleaner integration. Less battle-tested than COLMAP. Either works for a future 3D organ.
- openMVS/openMVS                    ✅ REVIEWED: Multi-view stereo — takes COLMAP/openMVG sparse reconstruction and densifies to mesh. Key patterns: (1) **Dense point cloud** → surface mesh → texture mapping. Full pipeline from sparse points to textured 3D model. (2) **Mesh refinement** — iteratively improves mesh quality. Verdict: complementary to COLMAP/openMVG. The full pipeline would be: images → COLMAP (sparse) → openMVS (dense mesh + texture) → OBJ export. Real 3D reconstruction for a future 3D organ, but a heavy dependency chain.

# -------------------------------------------------------
# Autonomous driving
# Lower priority — relevant patterns for: multi-sensor
# fusion, real-time inference, planning under uncertainty.
# -------------------------------------------------------
- carla-simulator/carla              ✅ REVIEWED: Driving simulator with Python API. Key patterns: (1) **Multi-sensor fusion** — combines camera, LiDAR, radar, GPS data into unified representation. Architecturally: each sensor has encoder → features concatenated or cross-attended → decoder. Same concept as Enigma's multimodal (vision + audio + text) but with more modalities. Validates Enigma's approach of separate projection per modality → concatenate. (2) **Real-time processing** — strict latency requirements (~100ms loops). Irrelevant to Enigma's interactive chat. Verdict: multi-sensor fusion validates Enigma's multimodal concatenation approach. No new actionable patterns.
- autowarefoundation/autoware        ✅ REVIEWED: Full autonomous driving stack. Key patterns: (1) **Perception → Planning → Control pipeline** — clean separation of concerns. Each stage has defined inputs/outputs. Enigma's router (perception of user intent → planning response → generating output) follows a similar pattern loosely. (2) **Uncertainty estimation** — outputs confidence scores with predictions, uses them for decision-making. Enigma could benefit from confidence scoring on responses (currently doesn't report how confident it is). Verdict: uncertainty/confidence estimation on model outputs is a useful concept but implementation (MC Dropout, ensemble disagreement) adds significant inference overhead. Lower priority.
- ApolloAuto/apollo                  ✅ REVIEWED: Baidu's autonomous driving platform. Key patterns: (1) **Prediction module** — predicts future trajectories of other agents using LSTM + attention. Multi-modal predictions (several possible futures, each with probability). (2) **Behavior planning** — decision tree + cost function for action selection under uncertainty. Verdict: trajectory prediction is conceptually interesting for conversation flow prediction (where is this conversation going?) but architecturally different. No actionable patterns.
- commaai/openpilot                  ✅ REVIEWED: Lightweight self-driving on consumer hardware. Key patterns: (1) **Efficient real-time vision** — supercombo model processes camera + desire → path + lane lines at ~20 FPS on mobile chip. Uses efficient convnet backbone. (2) **Quantization for deployment** — INT8 quantization on ONNX models for mobile inference. Enigma already has GGUF quantization support for LLM. (3) **Temporal convolution** — processes recent frame history with 1D temporal convolution instead of recurrence. Lightweight alternative to LSTM/attention for temporal modeling. Relevant for video frame processing. Verdict: temporal convolution for video frames is a lightweight alternative to TimeSformer's temporal attention. Much simpler (~10 lines) — apply 1D conv across frame features. Lower compute than attention. Could be a practical first step before full temporal attention for video understanding.

# -------------------------------------------------------
# Classical ML / tabular
# Lower priority — data preprocessing, ensemble methods
# could improve training data quality and model evaluation.
# -------------------------------------------------------
- scikit-learn/scikit-learn           ✅ REVIEWED: Comprehensive ML library. Key patterns for Enigma: (1) **TF-IDF vectorizer** — `TfidfVectorizer` converts documents to term-frequency-inverse-document-frequency vectors. Better RAG retrieval than simple word matching. Enigma's RAG uses chunk splitting + embedding similarity — TF-IDF is a lightweight alternative for keyword-based retrieval that works without neural embeddings. Could complement existing vector search. (2) **Data preprocessing** — `StandardScaler`, `MinMaxScaler`, `LabelEncoder`. Useful for normalizing training data features but Enigma trains on text (tokenized), not tabular data. (3) **KMeans / DBSCAN clustering** — group similar training examples. Could improve data deduplication in `validate_training_data()`: cluster near-duplicates instead of exact string matching. (4) **Train/test split** — `train_test_split` with stratification. Enigma's `val_split` does random splitting — stratified splitting ensures balanced representation across categories. (5) **Pipeline** — chain preprocessing + model steps. Clean pattern but Enigma already has its own pipeline. Verdict: TF-IDF for lightweight RAG retrieval (no GPU needed, instant indexing) and clustering for better deduplication are the two actionable patterns. Both are stdlib-level (scikit-learn is common enough to not count as "new dependency").
- dmlc/xgboost                       ✅ REVIEWED: Gradient boosting for tabular data. Key patterns: (1) **Fast reward scoring** — XGBoost can train a reward model on (text_features, human_score) pairs in seconds. Enigma's RewardModel is a full transformer with frozen base weights — overkill when you just need "is this response good?" from simple features (length, keyword presence, format compliance). XGBoost reward model: extract ~20 features from response → predict score. Trains in <1s on 1000 examples. (2) **Feature importance** — XGBoost gives per-feature importance scores, showing what makes a response good/bad. Interpretable unlike neural reward model. (3) **Early stopping** — built-in early stopping on validation metric. Enigma implements its own. Verdict: XGBoost as a lightweight reward model alternative is interesting for simple quality scoring (format compliance, length appropriateness) where you don't need semantic understanding. Neural reward model is still needed for semantic quality. Niche use case.
- microsoft/LightGBM                 ✅ REVIEWED: Fast gradient boosting (2-10x faster than XGBoost). Same verdict as XGBoost — lightweight reward scoring alternative. LightGBM is faster but XGBoost has better documentation. Either works. No additional patterns beyond XGBoost.
- catboost/catboost                  ✅ REVIEWED: Categorical gradient boosting — handles categorical features natively without one-hot encoding. Same use case as XGBoost/LightGBM. Slightly better for categorical data but Enigma's data is text-based. Verdict: no additional patterns beyond XGBoost.

# -------------------------------------------------------
# Extra
# -------------------------------------------------------
- edenaion/EZ-CorridorKey            ✅ REVIEWED: Niche repo — corridor detection for indoor navigation. Uses simple CNN for binary classification (corridor vs not-corridor). No actionable patterns for Enigma. Architectural patterns are basic (Conv2d → ReLU → MaxPool → FC).
```

## Summary (March 2026)

### Review Status
**Pass 1 complete (16 repos):** LLMs, multimodal/vision-language — core transformer training patterns.
**Pass 2 complete (50 repos):** ALL remaining categories reviewed. Every repo now has ✅ status with actionable findings documented.
**Pass 3 complete (source-verified):** All repo findings re-verified against actual Enigma source code (deep read of model.py, training.py, rl_training.py, tokenizer.py, vision_encoder.py, model_context.py, rag.py, engine_generation.py, lora_utils.py, defaults.py). Found new gaps: DPO doubles VRAM via deepcopy (now P1#6), chat history has no token budgeting (now P0#7), Phase 3 adaptive scoring broken (now P0#6), SelfPlay incomplete, RAG already has BM25/TF-IDF (Pass 2 item #19 marked redundant). All findings consolidated in SUGGESTIONS.md with realistic effort/impact grouping. P1 renumbered sequentially (1-35, no gaps).

### New Review Topics (Pass 2)
In addition to model architecture and training, Pass 2 specifically looked at:
- **Tokenizer patterns** — pre-tokenization regex, byte-level handling, vocab strategies
- **History/context management** — how models handle conversation history, session persistence, context windows, summarization
- **File setup** — how models load/save state, config management, model identity persistence
- **Data pipelines** — augmentation, preprocessing, efficient loading
- **Audio/video processing** — encoders, feature extraction, temporal modeling

### Why all categories matter

| Domain | Enigma Component | Why It's Relevant |
|--------|-----------------|-------------------|
| Diffusion / image gen | image-gen organ (`core/imagegen.py`, uses diffusers) | Fine-tuning SD on user images, ControlNet, inpainting, SDXL, better schedulers |
| GANs | image-gen organ | Alternative local generation, discriminator for quality scoring |
| Vision (cls/det/seg) | `vision_encoder.py`, eyes organ (`core/eyes.py`) | Pretrained ViT weights, data augmentation, object detection in ingested images |
| Speech / audio | `audio_projection` + `core/audio_encoder.py`, voice organ (`core/tts.py`), ears organ (`core/asr.py`) | Audio encoder for training, better TTS, audio feature extraction |
| RL | `rl_training.py` (RewardModel, RLHF, SelfPlay) | PPO clipping fix, advantage estimation, proper RL training loops |
| Robotics / embodied | Agent capabilities | Tool-use RL, multi-step planning, behavioral cloning from user history |
| 3D reconstruction | (no 3D organ yet — future) | Real 3D from images, if/when a 3D organ is built |
| Video understanding | `encode_video_frames()` (vision_encoder.py); no video-gen organ yet | Video training data, action recognition, better frame extraction |
| Classical ML | Training data pipeline, RAG | Data preprocessing, lightweight reward models, better retrieval |
| Cameras/SLAM | eyes organ (image ingest only) | Not actionable — hardware-dependent patterns |
| Driving/planning | Agent architecture | Temporal convolution for video, uncertainty estimation |
| Graph neural networks | RAG system | Knowledge graph for memory retrieval — future |

### Pass 1 findings (already implemented)
1. **Warmup scheduler bug** (P0) — ✅ FIXED: SequentialLR(warmup → cosine) now in Trainer and DPO
2. **DPO approach validated** — trl and gpt-neox both implement DPO with same reference model pattern
3. **Multimodal projector validated** — LLaVA's MLP projector matches Enigma's approach
4. **Low-hanging fruit:** vocab padding (P1), ~~fused AdamW~~ ✅ DONE, full checkpoint resume (P1)
5. **Nice-to-have:** outlier detection (llm.c), gradient noise scale (gpt-neox), WSD scheduler (llm.c)
6. ✅ **Label smoothing** in CE loss (fairseq)
7. ✅ **min_p sampling** — full pipeline (llama.cpp/vLLM/Kobold pattern)
8. ✅ **RLHF reward normalization** — running mean/std (OpenRLHF pattern)
9. ✅ **DPO max_length** — reads model config instead of hardcoded 512

### Pass 2 findings (newly actionable items)

**High-value findings from Pass 2:**

#### Model / Architecture
1. **EMA weight averaging** (mmclassification) — maintain running average of model weights during training, use for inference. ~5 lines, proven benefit for training stability. Applicable to both LLM and ViT training.
2. **Temporal attention for video** (mmaction2/TimeSformer) — add temporal attention between video frames. Currently each frame processed independently. ~100 lines.
3. **Temporal convolution for video** (openpilot) — lighter alternative to temporal attention. 1D conv across frame features. ~10 lines. Good first step.
4. **Conformer blocks for audio encoder** (espnet) — add Conv module inside each audio transformer block. Improves audio processing 10-15%. ~30 lines per block.
5. **Audio tokenization** (NeMo/EnCodec-style) — convert audio to discrete tokens for LLM processing. Unifies audio+text in same token stream. High effort but highest-impact audio improvement.

#### Training / Data
6. **Image augmentation pipeline** (pytorch/vision) — `RandomResizedCrop + RandomHorizontalFlip + ColorJitter + Normalize`. Enigma's vision training has zero augmentation. #1 vision training improvement.
7. **RandAugment** (pytorch/vision) — automated augmentation policy. `transforms.RandAugment(num_ops=2, magnitude=9)`. Modern standard.
8. **SpecAugment for audio** (speechbrain) — mask frequency/time bands in mel spectrograms during training. Audio equivalent of image augmentation.
9. **ImageNet normalization** (pytorch/vision) — switch to `mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]` for pretrained weight compatibility. Currently [-1,1].
10. **Behavioral cloning from chat history** (robomimic) — train model on (context, user_action) pairs from saved sessions. Session data already exists. New training mode.
11. **Replay buffer for RLHF** (ray/acme) — store generated responses + rewards, sample high-error ones more often. Better data efficiency for RLHF training.

#### Imagegen Mod
12. **ControlNet** (diffusers) — condition generation on edge maps, depth maps, poses. ~5 lines to swap pipeline.
13. **img2img** (diffusers) — take input image + prompt → modified image. Currently only txt2img.
14. **Inpainting** (diffusers) — paint into masked regions. Useful with images the eyes organ ingests.
15. **LoRA loading for SD** (diffusers) — load community LoRA styles. ~3 lines.
16. **Scheduler choice** (diffusers) — DPMSolver (fast, 20 steps) vs EulerDiscrete (quality, 50 steps).
17. **Super-resolution post-processing** (mmagic/Real-ESRGAN) — upscale generated images 4x. Pretrained model, ~10 lines.

#### Vision Mod
18. **YOLO object detection** (ultralytics) — `YOLO('yolov8n.pt')` for object detection in screenshots alongside OCR. 6MB model, ~20 lines. High value, low effort.

#### Tokenizer / RAG
19. ~~**TF-IDF for RAG**~~ — **REDUNDANT**: RAG already uses BM25 with TF-IDF-style scoring. Real upgrade is semantic embedding search.

#### History / Context (NEW TOPIC)
20. **Conversation summarization** — no repo implements exactly what Enigma needs, but the gap is clear: Enigma truncates old history by dropping messages. Better approach: summarize old messages into a compact context paragraph. Validated by robomimic's observation history windowing + LLaVA's training approach (compress visual context).
21. **Video preprocessing pipeline** (pytorchvideo) — proper temporal sampling + spatial transforms for video frames. Currently basic PIL extraction.

#### Not Actionable (reviewed but no patterns apply)
- Camera/SLAM repos (5) — hardware-dependent, not applicable to desktop AI
- NVlabs/stylegan3 — alias-free innovations specific to continuous generation
- IsaacLab, OmniIsaacGymEnvs — GPU physics simulation infrastructure
- real-stanford/diffusion_policy — continuous robotics control, not text
- EZ-CorridorKey — basic CNN, no useful patterns

---

### Pass 3 findings (source-verified gaps)

Gaps discovered by reading actual Enigma source code, not just repo comparisons:

| # | Finding | Impact | Where in Enigma |
|---|---------|--------|-----------------|
| 1 | **DPO uses deepcopy for ref model** — doubles VRAM. RLHF already has LoRA disable pattern. | High — saves ~2x VRAM | training.py DPO section |
| 2 | **Chat history has no token budget** — ModelContext keeps all messages, can exceed model's max_seq_len | High — causes context overflow | model_context.py |
| 3 | **Phase 3 scoring returns empty** — adaptive curriculum never advances by merit in COMMANDS stage | High — undermines adaptive training | Adaptive pipeline scoring |
| 4 | **SelfPlay trainer incomplete** — loop exists but training update is minimal | Medium — blocks SelfPlay usage | rl_training.py |
| 5 | **PPO missing 7 components** — not just clipping, but value function, GAE, entropy, buffer | Critical — PPO doesn't work | rl_training.py |
| 6 | **RAG already has BM25/TF-IDF** — Pass 2 item #19 was redundant | Correction — remove from backlog | rag.py |
| 7 | **No attention_mask in training batches** — model attends to pad tokens | High — wasted compute + noise | training.py _create_batches() |

---

**Cross-reference:** The consolidated backlog of all actionable findings from these reviews is in `BACKLOG.md` (the `SUGGESTIONS.md` "Open Backlog" section this line used to point at was consolidated there). This file is the per-repo detail reference.

---

## Curriculum / course references (not raw model code — learning material, mine when training Enigma further)

⬜ = bookmarked, not yet mined for actionable patterns.

- **rohitg00/ai-engineering-from-scratch** ⬜ — 503-lesson, 20-phase AI-engineering *curriculum* (Python/TS/Rust/Julia; PyTorch/JAX). 33.1k★, 5.4k forks, MIT. Not a model/library — a course that teaches the whole field from math → production. It is essentially the textbook for what Enigma builds by hand. Phases that map onto Enigma's training stack, to mine when we train her more:
  - Phase 6 (transformers), Phase 10 (LLM training), Phase 11 (LLM engineering) → `pretrain_enigma.py` / `finetune_enigma.py` / `model.py`
  - Phase 13 (tools/protocols/MCP), Phases 14–15 (agents) → the serve tool layer (built-ins + client tools) + Odysseus
  - **Highest value where Enigma is thinnest/unproven:** eval rigor, RLHF/reward modeling (`rl_training.py`, `reward_functions.py` — Pass-3 flagged PPO as missing 7 components), multimodal, alignment.
  - Added 2026-06-15.

## Comparable systems / prior art (shipped products in Enigma's problem space)

### Chat-memory systems (researched 2026-07-06 for the `remember` tool decision)

Question asked: adopt an existing memory layer for Enigma serve, or build on `memory_store.py`?
Verdict: **build the thin `remember` built-in tool on our MemoryStore** — every serious framework
delegates the intelligent parts (extract/summarize/conflict-resolve) to a frontier-class LLM, which
a 182M from-scratch model cannot be and which a routed Ollama model would make un-hers (Muppet
pattern). The shipped prior art for OUR design is ChatGPT itself: its saved memories are a `bio`
TOOL the model calls, injected into context next turn (0xeb/TheBigPromptLibrary documents it).
Even the SLM-memory research floor (LightMem, arXiv 2604.07798) is 1B+ online models.

- **letta-ai/letta** (MemGPT) ⬜ — agent runtime where the model self-edits labeled memory blocks
  via tool calls (core/recall/archival tiers; "sleep-time" background agents consolidate). Apache-2.0.
  NOT adoptable (it's a whole runtime, wants frontier models; sleep-time agent is explicitly "use a
  larger model"). MINE THE IDEAS: memory as model-editable blocks; a later offline consolidation
  pass when a bigger Enigma exists. Added 2026-07-06.
- **mem0ai/mem0** ⬜ — memory library, ~48k★. Self-hosted still REQUIRES an extractor LLM (default
  OpenAI gpt-5-mini) + embedding model + vector store; the LLM does extraction + conflict
  resolution. Its LOCOMO benchmark war with Zep (both claim SOTA, both scores collapse under the
  other's re-evaluation, 84%->58% and back) is the case study in not trusting "said to work well."
  MINE THE IDEA: update-on-contradiction memory ops (ADD/UPDATE/DELETE decided per new fact).
  Added 2026-07-06.
- **getzep/zep** ⬜ — temporal knowledge graph memory (community edition self-hostable). Strongest
  at "what was true in Q1?" temporal reasoning; same extractor-LLM dependency, Go service + graph
  DB to babysit. Overkill for personal-facts scale. Added 2026-07-06.
- **CaviraOSS/OpenMemory** ⬜ — the only genuinely local-first one: Python/Node SDK, SQLite,
  embeddings optional (Ollama or a no-model "synthetic fallback"), runs offline. But it's a
  storage/retrieval engine — the decide-what-to-save step is still the assistant's job (MCP tool
  calls), which is precisely the part we have to train Enigma to do regardless. Adopting it would
  replace our working BM25 store with a dependency, not solve the hard part. Re-visit if we ever
  want its decay/sector model. Added 2026-07-06.

- **tinyhumansai/openhuman** ⬜ — private, local-first desktop AI companion. Rust 61.6% / TS 35.5%, Tauri+CEF shell, SQLite. 32.3k★, 3.1k forks, GPL-3.0 (copyleft — borrow *ideas*, not code), early beta. The closest shipped parallel to the **Odysseus + Enigma + avatar** trinity: desktop mascot w/ voice + "background thinking" (≈ avatar), local "Memory Tree" w/ hierarchical summarization + Obsidian wiki (≈ Odysseus memory), model routing per workload (Enigma deliberately has no router — one from-scratch model), native tools web/fs/git/voice (≈ the serve built-in tools + organs), optional Ollama. **What they have that we don't:** 118+ one-click OAuth integrations via **Composio** + auto-fetch (every 20 min) into the Memory Tree → "warm-start" memory; **TokenJuice** context compression (~80% cost cut). **What we have that they don't:** we forge our *own* model (they only route to existing LLMs); a real rigged GLB avatar engine (theirs is a mascot gimmick). Mine for: Composio integration layer + auto-ingesting hierarchical Memory Tree (directly upgrades `memory_store.py`). Added 2026-06-15.
