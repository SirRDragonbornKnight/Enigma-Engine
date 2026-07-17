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

Realistic order: **Phase 4 length extension → B (+D data folded in) → A → C
grows alongside whatever domain corpus the user picks.** C is also the
answer to "how do I correlate things in an image for her": you don't
hand-author correlations — the paired-difference data teaches her attention
to find them.

## 3. Generation in the user's style ("her imagination")

- The lever is CURATION, not architecture: a style corpus of only images
  the user actually likes (even 50-300 images is enough for a style LoRA).
- Interim: LoRA on the local SD backbone (sd-turbo/sdxl-turbo) trained on
  the style corpus -> she generates only in that style. Pip-only, hours.
- Long-term (Phase 4.5 step 8): her own small generator trained on the
  style corpus + domain imagery — owned weights, same arc as eyes/ears.
- Quality gate idea: the user thumbs-up/down her generations; keepers join
  the style corpus (the teach-loop pattern, applied to art).

## 4. Open decisions (user's, before any of this trains)

1. Image DOMAIN for her eyes (game frames? anime? screenshots? mixed?).
2. Style corpus source for generation (folder of liked art? links? her
   sd-turbo keepers?).
3. Resolution target for fine detail (336? 448? tiled 224s?) — decides
   encoder re-distill settings.
4. Whether Phase 4 (context 1024→2048) is approved as the vision
   prerequisite it actually is.

## 5. What already exists and carries over unchanged

Distill pipeline (distill_vision_encoder.py — re-runnable on any imagery),
align pipeline (align_vision.py --data <any jsonl>), batched train_vision,
serve wiring (--eyes, projection graft, native captioner), encoder
persistence, the [image: ...] marker path. Swapping the data does not
change one line of this machinery.
