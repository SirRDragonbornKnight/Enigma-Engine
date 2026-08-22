"""Image understanding primitive -- her OWN eyes.

Phase 4.5 step 5 ("wire and retire", 2026-07-17): the BLIP scaffold this
module used to wrap is RETIRED. Captions now come from her own weights end
to end -- the distilled-and-aligned VisionEncoder + the vision_projection
trained by align_vision.py + the served frozen text model. Generation runs
in exactly the align-trained shape: [projected patches][BOS -> caption ->
EOS], greedy, no KV cache (a ~250-position forward on a 182M model is
millisecond-scale; recomputing per token keeps this free of cache surgery).

flatten_image_content() is the pure ingestion half: it rewrites ONE message's
OpenAI multimodal content list into plain text, honestly marking anything the
server cannot see, so serve stays a thin caller and the logic stays testable.
The caption enters chat as the "[image: ...]" marker shape the adopted model
was SFT-trained on -- token-level image injection into templated chat is the
NEXT training cycle's work (she has no trained image-delimiter tokens yet).
"""

from __future__ import annotations

import base64
import hashlib
import io
import threading
from collections import OrderedDict
from pathlib import Path

# How many captions one Eyes remembers. Every chat turn resends the WHOLE
# history, and the flatten path captions every image part of every message on
# every request -- so a picture three turns back re-paid for itself, up to
# max_caption_tokens full forwards, holding the shared generation lock, before
# the model could answer at all (audit 2026-08-22). LRU: a re-shown image that
# has aged out simply captions again.
CAPTION_CACHE_SIZE = 32


class EyesError(Exception):
    """The captioning backend is unavailable or a caption failed."""


class _InFlight:
    """One caption being computed right now, shared by every caller that asks
    for the SAME bytes while it runs.

    The cache alone only helps the second turn: the chat page's own history
    replay sends one picture in several parallel requests, and eight threads
    with one payload measured eight full captioning passes (audit 2026-08-22)
    -- all of them queued on the generation lock, ahead of the answer the user
    is waiting for. The first caller in publishes the slot; the rest wait on
    it and share whatever it produced, success or failure."""

    __slots__ = ("done", "caption", "error")

    def __init__(self):
        self.done = threading.Event()
        self.caption: str | None = None
        self.error: BaseException | None = None


class Eyes:
    """One captioner, one image at a time (the shared lock serializes GPU use
    with text generation -- captioning borrows the served model)."""

    def __init__(
        self,
        model=None,
        tokenizer=None,
        encoder=None,
        gen_lock: threading.Lock | None = None,
        max_caption_tokens: int = 48,
        captioner_factory=None,
        repetition_penalty: float = 1.3,
        cache_size: int = CAPTION_CACHE_SIZE,
    ):
        self._lock = gen_lock if gen_lock is not None else threading.Lock()
        # Its OWN lock, never the generation one: a cache hit must answer
        # without queueing behind whatever is generating.
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._cache_size = max(0, int(cache_size))
        # Keyed the same way as the cache: the captions being computed RIGHT
        # NOW, so concurrent callers of one payload share a single pass.
        self._inflight: dict[str, _InFlight] = {}
        if captioner_factory is not None:  # tests inject a fake: PIL.Image -> str
            self._caption = captioner_factory()
            return
        if model is None or tokenizer is None or encoder is None:
            raise EyesError(
                "native eyes need the served model + tokenizer + her aligned "
                "vision encoder (train one with distill_vision_encoder.py then "
                "align_vision.py)"
            )
        if getattr(model, "vision_projection", None) is None:
            raise EyesError(
                "the served model has no vision_projection -- serve must open "
                "the vision port and graft the aligned projection weights"
            )
        import torch

        device = next(model.parameters()).device
        encoder = encoder.to(device).eval()
        image_size = getattr(getattr(encoder, "config", None), "image_size", 224)
        eos = getattr(tokenizer, "eos_token_id", 2)
        bos = getattr(tokenizer, "bos_token_id", 1)
        # Live vocab width for the argmax slice below (None for config-less
        # test stubs, which then decode over the full head unchanged).
        vocab = getattr(getattr(model, "config", None), "vocab_size", None)

        def _caption(img) -> str:
            from enigma_engine.core.vision_encoder import preprocess_image

            # Her native from-scratch normalization ([-1, 1]) -- the SAME
            # preprocessing distillation and alignment trained under.
            x = preprocess_image(img, image_size=image_size, imagenet_normalize=False).to(device)
            autocast = torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")
            )
            with torch.no_grad(), autocast:
                feats = encoder(x)
                ids = [bos]
                for _ in range(max_caption_tokens):
                    logits = model.forward_multimodal(
                        input_ids=torch.tensor([ids], dtype=torch.long, device=device),
                        vision_features=feats,
                    )
                    # Slice off the vocab-alignment padding columns before
                    # argmax -- ids >= vocab_size have no decode (the model's
                    # generate() paths mask them; this native loop must too).
                    step = logits[0, -1]
                    if vocab is not None and step.shape[-1] > vocab:
                        step = step[:vocab]
                    # Greedy at 183M degenerates into loops ("watch watch
                    # watch" -- measured live 2026-07-20); a repetition
                    # penalty over HER generated ids breaks them while
                    # keeping the loop deterministic (still argmax).
                    if repetition_penalty != 1.0 and len(ids) > 1:
                        from enigma_engine.core.model_utils import apply_repetition_penalty

                        step = apply_repetition_penalty(
                            step, torch.tensor(ids[1:], device=step.device), repetition_penalty
                        )
                    nxt = int(step.argmax())
                    if nxt == eos:
                        break
                    ids.append(nxt)
            return tokenizer.decode(ids[1:], skip_special_tokens=True).strip()

        self._caption = _caption

    def describe(self, image) -> str:
        """Caption an image given as bytes, a path, or a PIL.Image.

        A BYTES payload -- what the chat path always hands over, since every
        image_url part is base64-decoded before it gets here -- is memoized by
        sha256 (see CAPTION_CACHE_SIZE), so an image already in the history is
        captioned once per process rather than once per turn. Callers that
        arrive while that first pass is still running wait for it and share
        its result instead of starting passes of their own (_InFlight). A path
        or a PIL.Image has no stable key and always captions. Only successful
        captions are stored: a failure is worth retrying, not remembering.

        The cache lock is held for lookups and publication ONLY, never across
        the caption itself -- captioning takes the shared generation lock, and
        a cache that queued behind generation would be slower than no cache.
        """
        key = None
        if isinstance(image, (bytes, bytearray)):
            key = hashlib.sha256(bytes(image)).hexdigest()
            while True:
                with self._cache_lock:
                    if key in self._cache:
                        self._cache.move_to_end(key)
                        return self._cache[key]
                    shared = self._inflight.get(key)
                    if shared is None:
                        self._inflight[key] = _InFlight()
                        break  # this caller computes it; the rest wait below
                shared.done.wait()  # outside the lock: see the docstring
                if shared.error is not None:
                    raise shared.error
                if shared.caption is not None:
                    return shared.caption
                # Neither published -- whoever owned the slot never finished.
                # Start over rather than wait on a promise nobody will keep.
        try:
            caption = self._describe_uncached(image)
        except BaseException as exc:
            self._publish(key, None, exc)
            raise
        self._publish(key, caption, None)
        return caption

    def _describe_uncached(self, image) -> str:
        """Decode the payload and caption it -- no cache, no in-flight slot."""
        try:
            from PIL import Image
        except ImportError as exc:
            raise EyesError("Pillow not installed -- pip install 'enigma-engine[eyes]'") from exc
        try:
            if isinstance(image, (bytes, bytearray)):
                img = Image.open(io.BytesIO(bytes(image)))
            elif isinstance(image, (str, Path)):
                img = Image.open(image)
            else:
                img = image
            img = img.convert("RGB")
        except EyesError:
            raise
        except Exception as exc:
            raise EyesError(f"could not read image: {exc}") from exc
        with self._lock:
            try:
                caption = (self._caption(img) or "").strip()
            except Exception as exc:
                raise EyesError(f"captioning failed: {exc}") from exc
        if not caption:
            raise EyesError("captioner returned nothing")
        return caption

    def _publish(self, key, caption: str | None, error: BaseException | None) -> None:
        """Hand a finished caption -- or its failure -- to whoever waited.

        The cache insert and the in-flight release happen under ONE cache-lock
        acquisition, so a caller arriving at that instant finds either the
        answer or the slot to wait on, never neither. A failure releases the
        slot too: waiters see the same error rather than hanging on it, and
        the next caller retries (a failure is not cached)."""
        if key is None:
            return
        with self._cache_lock:
            shared = self._inflight.pop(key, None)
            if caption is not None and self._cache_size:
                self._cache[key] = caption
                self._cache.move_to_end(key)
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)  # oldest USE, not oldest write
        if shared is not None:
            shared.caption, shared.error = caption, error
            shared.done.set()


def flatten_image_content(content: list, describe=None) -> str:
    """Rewrite one message's OpenAI multimodal content list into plain text.

    describe: bytes -> caption, or None when the eyes organ is off. Every part
    the model cannot see becomes an HONEST inline marker instead of silently
    vanishing -- she should never answer as if she saw an image she didn't.
    Only data: URLs are accepted; the server never fetches remote images.
    """
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
            continue
        if not isinstance(part, dict):
            parts.append(f"[unsupported content part: {type(part).__name__}]")
            continue
        ptype = part.get("type")
        if ptype == "text":
            parts.append(str(part.get("text") or ""))
        elif ptype == "image_url":
            url = str((part.get("image_url") or {}).get("url") or "")
            if describe is None:
                parts.append("[image ignored -- eyes disabled (start serve with --eyes)]")
            elif url.startswith("data:"):
                try:
                    payload = base64.b64decode(url.split(",", 1)[1])
                    parts.append(f"[image: {describe(payload)}]")
                except Exception as exc:
                    parts.append(f"[image error: {exc}]")
            else:
                parts.append("[image ignored -- only data: URLs are supported]")
        else:
            parts.append(f"[unsupported content part: {ptype!r}]")
    return "\n".join(p for p in parts if p).strip()
