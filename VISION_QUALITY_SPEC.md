# Vision Quality Spec — best-quality seeing and making (2026-07-17)

> User rulings, verbatim intent: best quality out of the vision work; her
> training imagery will NOT be everyday web photos (domain = user's choice,
> TBD); generated images must be in the style the user likes ("I do not
> like bad art at all"); and she should handle FINE-GRAINED discrimination
> — the named target: two near-identical subjects (twins) with one visible
> difference, and she can tell you what it is.
> Status: design only. Training is LAST (user ruling); nothing here runs
> until the user reopens it.

## 1. Why the current stack cannot pass the twins test (honest limits)

1. **Resolution floor.** Everything is squeezed to 224x224 before her
   encoder sees it. A distinguishing mole/earring/scar can be a handful of
   pixels — destroyed before the encoder ever gets a chance.
2. **The caption bottleneck.** Serve's pipeline captions the image ONCE
   into one line, then she answers from that text. "Two women standing
   together" throws the difference away even if the encoder features kept
   it. The information dies at the caption, not in her.
3. **Question-blindness.** The caption is generated before/without the
   question, so "what's different between them?" cannot steer what her
   eyes extract.
4. **182M + stage-1 alignment** articulates coarse scene content, not
   subtle comparisons — with the current data, which contains no
   comparative supervision at all.

## 2. The four levers that fix it (all buildable in this repo)

| Lever | What it does | Prerequisite |
|---|---|---|
| **A. Resolution / tiling** (LLaVA-NeXT-style AnyRes: global 224 view + NxN 224 crops) | Small details survive into patch features | Phase 4 context extension — 4 tiles + global = ~980 image tokens; block 1024 can't also hold a conversation. **Fine-grained vision makes 1024→2048 a hard prerequisite, not a nice-to-have.** |
| **B. Native patch injection** (image begin/end tokens ids 4724+, patches in her context WITH the question) | She "looks while thinking" — the question conditions attention over the pixels; two images can sit side by side in context and be compared | Multimodal SFT data carrying the token shape (next data cycle); Phase 4 for two-image prompts |
| **C. Contrast-pair training data** ("spot the difference" corpus: two near-identical images + text naming the ONE difference) | Discrimination is learned from data that demands it — no amount of architecture substitutes | Buildable synthetically: take a source image, edit ONE detail (inpainting/img2img), auto-caption the edit as ground truth. Domain can be the user's chosen imagery. |
| **D. Question-conditioned captioning (interim VQA shape)** | Even before B, feed the question into the caption pass so her eyes extract what was asked | Caption-with-question training pairs; cheapest of the four, weakest ceiling |

Realistic order: **context headroom → B (+D data folded in) → A → C grows
alongside whatever domain corpus the user picks.** C is also the answer to
"how do I correlate things in an image for her": you don't hand-author
correlations — the paired-difference data teaches her attention to find them.

**The context prerequisite is satisfied by v2, not only by Phase 4.** If the
v2 pretrain proceeds, the `v2_deep_*` presets train at a native 8192 context
and the v2 tokenizer carries 2.41x more text per token — together these clear
lever A's ~980-image-token problem without a v1 length-extension pass. Read
every "Phase 4" prerequisite in this document as "context headroom, from v2 or
Phase 4, whichever lands".

## 3. Generation in the user's style ("her imagination")

- The lever is CURATION, not architecture: a style corpus of only images
  the user actually likes (even 50-300 images is enough for a style LoRA).
- Interim: LoRA on the local SD backbone (sd-turbo/sdxl-turbo) trained on
  the style corpus -> she generates only in that style. Pip-only, hours.
- Long-term (Phase 4.5 step 8): her own small generator trained on the
  style corpus + domain imagery — owned weights, same arc as eyes/ears.
- Quality gate idea: the user thumbs-up/down her generations; keepers join
  the style corpus (the teach-loop pattern, applied to art).

## 3b. External grounding (2026-07-18, adversarially fact-checked)

Three findings from the tiny-VLM literature bear directly on this plan:

1. **The DINOv2 teacher choice deserves a re-decision.** Her encoder was
   distilled from DINOv2-S, which is SELF-SUPERVISED and NOT language-aligned.
   Every successful sub-1B VLM (SmolVLM, Moondream, the LLaVA lineage) uses a
   SigLIP/CLIP-class contrastive encoder precisely because projector alignment
   onto an LM is far easier from a language-adjacent representation; the
   literature shows DINOv2 only as an AUXILIARY spatial encoder in dual-encoder
   setups, never alone. Expect the current align stage to need more LM-side
   training than a SigLIP-teacher version would. Options: re-distill from
   SigLIP, distill from BOTH (dual teacher), or keep DINOv2 and pay for it in
   stage-2 training. Decide before spending the align GPU-days.
   (arxiv 2504.05299 SmolVLM; Moondream = SigLIP.)
2. **Frozen-LM projector-only alignment is stage 1, not the finish line.** The
   Idefics-lineage result ("What matters when building VLMs", arxiv 2405.02246)
   is that fully-frozen autoregressive training diverges, and LoRA-unfreezing
   the LM gained +12.9 points. So plan a stage 2 that trains the LM — full or
   a small bespoke LoRA. NOTE: the old `lora_utils.py` was deleted in the
   2026-07-18 compression pass; if stage-2 LoRA is the pick, write a minimal
   one against current `model.py` rather than reviving the old stack.
3. **Compress visual tokens harder than instinct says.** SmolVLM's central
   small-scale finding: SMALLER VLMs benefit from MORE aggressive compression
   — pixel-shuffle r=4 (16x reduction) vs r=2 for big models, landing ~64-128
   visual tokens/image. This materially softens lever A's context problem: at
   ~64-128 tokens/tile, tiling fits far sooner than the ~980-token estimate in
   §2 (which assumed uncompressed 196-patch tiles). Add pixel-shuffle to the
   encoder->projection path and re-do that arithmetic before treating Phase 4
   as an absolute blocker. Sub-image boundaries want LEARNED positional tokens
   (SmolVLM found these beat raw-text separators for compact models).

## 4. Open decisions (user's, before any of this trains)

1. Image DOMAIN for her eyes (game frames? anime? screenshots? mixed?).
2. Style corpus source for generation (folder of liked art? links? her
   sd-turbo keepers?).
3. Resolution target for fine detail (336? 448? tiled 224s?) — decides
   encoder re-distill settings.
4. Whether Phase 4 (context 1024→2048) is approved as the vision
   prerequisite it actually is — **re-check after pixel-shuffle (§3b.3); it
   may drop from hard-prereq to merely helpful. Moot if the v2 pretrain
   proceeds: the `v2_deep_*` presets train at a native 8192 context.**
5. **Encoder teacher: keep DINOv2, switch to SigLIP, or dual (§3b.1).**
   This one gates any re-distill and should be settled first.

## 5. What already exists and carries over unchanged

Distill pipeline (distill_vision_encoder.py — re-runnable on any imagery),
align pipeline (align_vision.py --data <any jsonl>), batched train_vision,
serve wiring (--eyes, projection graft, native captioner), encoder
persistence, the [image: ...] marker path. Swapping the data does not
change one line of this machinery.
