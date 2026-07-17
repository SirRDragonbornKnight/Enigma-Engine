"""
External-Teacher Distillation Data Collector for Enigma Engine
==============================================================

Talks HTTP to any OpenAI-compatible chat endpoint (Ollama, llama.cpp server,
vLLM, our own ``run.py --serve``, OpenAI proper) and writes the responses
as a fine-tune corpus the existing FORGE Distill mode can train on.

This is the "external teacher" half of N-19. There is **no** in-process
teacher: the teacher AI runs as its own process, anywhere. We just consume
its replies as text. That keeps Enigma fully local + black-box (the
endpoint defaults to localhost) and avoids the cost of holding two models
in the same address space.

Output format
-------------
Same as ``collect_finetuning_data.py``:

  - ``data/finetune/distill_<tag>.jsonl``  — ``{"prompt", "completion"}`` rows
  - ``data/finetune/distill_<tag>.txt``    — ``User: …\\n\\nAssistant: …`` blocks

Usage
-----

  python collect_distill_data.py \\
      --endpoint http://localhost:11434/v1 \\
      --model qwen3:8b \\
      --prompts data/distill_prompts.txt \\
      --tag qwen3_8b \\
      --max-tokens 512 --temperature 0.7

  # Resume an interrupted run (skips prompts already in the JSONL):
  python collect_distill_data.py --resume --tag qwen3_8b ...

Privacy / safety
----------------
  - Endpoint defaults to ``http://localhost:11434/v1`` (Ollama default port).
  - If the endpoint hostname is **not** localhost / 127.0.0.1, a WARNING is
    logged on every run so a misconfiguration can't silently leak prompts.
  - Stdlib only (``urllib.request``). No new dependencies.
  - No telemetry. No retry-with-modified-prompt. Failed prompts are skipped
    with a logged WARNING and counted in the summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/finetune")
DEFAULT_ENDPOINT = "http://localhost:11434/v1"
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", ""}


# ── Magpie templates ──────────────────────────────────────────────────
#
# Magpie (arxiv:2406.08464) extracts instruction-tuning data from a chat
# model by sending it an *empty user turn* in the model's own chat
# template. A well-aligned chat model will then emit BOTH a synthetic
# user instruction AND its own assistant answer in one completion call.
# We parse the result on the user→assistant role marker.
#
# These template strings are the literal raw-text prefixes you'd send
# to ``/v1/completions`` (not ``/v1/chat/completions``). Wrong template
# for the target model = every parse fails. The failure rate is
# observable in the run summary; misconfiguration cannot silently leak
# malformed rows because empty-instruction / empty-answer / missing-
# marker rows all raise and are counted as failed.

_MAGPIE_TEMPLATES: dict[str, dict[str, str]] = {
    "chatml": {
        "prefix": "<|im_start|>user\n",
        "user_end": "<|im_end|>\n<|im_start|>assistant\n",
        "assistant_end": "<|im_end|>",
    },
    "llama3": {
        "prefix": ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"),
        "user_end": ("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
        "assistant_end": "<|eot_id|>",
    },
}


def _resolve_magpie_template(
    name: str,
    *,
    custom_prefix: str | None = None,
    custom_user_end: str | None = None,
    custom_assistant_end: str | None = None,
) -> dict[str, str]:
    """Return the template dict for ``name``.

    ``"custom"`` requires all three of ``custom_prefix`` / ``custom_user_end`` /
    ``custom_assistant_end`` to be non-empty strings. Anything else raises
    ``ValueError`` so a misconfiguration fails at parse time, not silently
    after a long run.
    """
    if name == "custom":
        for label, value in (
            ("--magpie-prefix", custom_prefix),
            ("--magpie-user-end", custom_user_end),
            ("--magpie-assistant-end", custom_assistant_end),
        ):
            if not value:
                raise ValueError(f"--template custom requires {label} (got empty value)")
        return {
            "prefix": custom_prefix or "",
            "user_end": custom_user_end or "",
            "assistant_end": custom_assistant_end or "",
        }
    if name not in _MAGPIE_TEMPLATES:
        raise ValueError(f"unknown magpie template '{name}'. Choices: {sorted(_MAGPIE_TEMPLATES) + ['custom']}")
    return _MAGPIE_TEMPLATES[name]


def _parse_magpie_completion(raw: str, template: dict[str, str]) -> tuple[str, str]:
    """Split a raw Magpie completion into (instruction, answer).

    The model's completion starts at the position right after our
    sent prefix (``<|im_start|>user\\n``) and is expected to contain:

        <user instruction>  user_end  <assistant answer>  [assistant_end]

    Raises ``RuntimeError`` with a descriptive message when:
      - ``user_end`` marker is missing entirely (wrong template, or model
        is a base model without chat training)
      - the user instruction (pre-marker text) is empty after strip
        (model emitted only the assistant marker)
      - the assistant answer (post-marker text, with ``assistant_end``
        stripped) is empty after strip
    """
    user_end = template["user_end"]
    assistant_end = template["assistant_end"]
    if user_end not in raw:
        raise RuntimeError(
            f"magpie parse failed: user->assistant marker not found "
            f"(model may not be chat-tuned for this template, or temperature "
            f"caused early stop). First 200 chars: {raw[:200]!r}"
        )
    instr_part, _, ans_part = raw.partition(user_end)
    instruction = instr_part.strip()
    if not instruction:
        raise RuntimeError(
            "magpie parse failed: empty user instruction (model emitted assistant marker without content)"
        )
    if assistant_end and assistant_end in ans_part:
        ans_part = ans_part.split(assistant_end, 1)[0]
    answer = ans_part.strip()
    if not answer:
        raise RuntimeError(
            "magpie parse failed: empty assistant answer "
            "(model emitted instruction but no reply -- try higher max_tokens)"
        )
    return instruction, answer


# ── Prompt loading ────────────────────────────────────────────────────


def load_prompts(path: Path) -> list[str]:
    """Load prompts from a file.

    Supports two formats, auto-detected:
      - ``.jsonl`` — one JSON object per line, with a ``"prompt"`` field
                     (other fields ignored).
      - anything else — one prompt per line, blank lines + ``#`` comments
                        skipped.
    """
    if not path.exists():
        raise FileNotFoundError(f"prompts file not found: {path}")
    text = path.read_text(encoding="utf-8")
    prompts: list[str] = []
    if path.suffix.lower() == ".jsonl":
        for ln, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("skip malformed JSONL line %d: %s", ln, exc)
                continue
            prompt = (row.get("prompt") or "").strip()
            if prompt:
                prompts.append(prompt)
    else:
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            prompts.append(stripped)
    return prompts


def _prompt_key(prompt: str) -> str:
    """Stable short hash for resume-deduplication."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _load_done_keys(jsonl_path: Path) -> set[str]:
    """Read existing JSONL output and return the set of prompt-keys already
    written, so a resumed run can skip them.

    Malformed lines are skipped silently — they'll be overwritten on the
    next clean write of the same file.
    """
    if not jsonl_path.exists():
        return set()
    done: set[str] = set()
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        prompt = row.get("prompt")
        if isinstance(prompt, str):
            done.add(_prompt_key(prompt))
    return done


# ── HTTP client ───────────────────────────────────────────────────────


class TeacherClient:
    """Minimal stdlib client for OpenAI-compatible /v1/chat/completions.

    No streaming. No retry. One request per ``ask()`` call. Failures raise
    ``RuntimeError`` with the upstream status / message attached so the
    caller can log + skip without losing the prompt context.
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        system_prompt: str | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.system_prompt = system_prompt
        self._url = f"{self.endpoint}/chat/completions"

    def ask(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
    ) -> str:
        """Send one prompt, return the assistant's text reply."""
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "stream": False,
        }
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self._url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"teacher returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"teacher unreachable at {self._url}: {exc.reason}") from exc
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"teacher returned non-JSON body: {payload[:200]}") from exc
        try:
            return obj["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"teacher response missing choices[0].message.content: {payload[:200]}") from exc


class MagpieClient:
    """Minimal stdlib client for OpenAI-compatible /v1/completions.

    Magpie depends on raw-prefix completion, NOT chat completions: we
    send the model its own chat-template prefix with an empty user turn
    and let it emit both a synthetic instruction and its own reply.
    Most OpenAI-compatible servers expose /v1/completions (Ollama,
    llama.cpp server, vLLM, our own --serve, OpenAI legacy). If the
    target server only supports chat completions, magpie cannot run
    against it — same loud-on-real-issue rule as the chat client:
    failures raise from ``ask_pair`` and the driver counts them.
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        template: dict[str, str],
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.template = template
        self.api_key = api_key
        self.timeout = timeout
        self._url = f"{self.endpoint}/completions"

    def ask_pair(
        self,
        *,
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> tuple[str, str]:
        """Generate one (instruction, answer) pair via empty-prefix completion.

        Returns the parsed pair or raises ``RuntimeError`` on any HTTP /
        decode / parse failure. The driver swallows the error, logs at
        WARNING, counts it, and continues to the next attempt.
        """
        body = {
            "model": self.model,
            "prompt": self.template["prefix"],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "stream": False,
        }
        # Stop on the assistant_end marker so we don't bleed into a
        # second turn the model sometimes hallucinates. Servers ignore
        # unsupported keys; this is a soft hint, not a correctness gate
        # (the parser strips the marker again as a belt-and-braces).
        if self.template.get("assistant_end"):
            body["stop"] = [self.template["assistant_end"]]
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self._url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"magpie endpoint returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"magpie endpoint unreachable at {self._url}: {exc.reason}") from exc
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"magpie endpoint returned non-JSON body: {payload[:200]}") from exc
        try:
            raw = obj["choices"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"magpie response missing choices[0].text: {payload[:200]}") from exc
        if not isinstance(raw, str):
            raise RuntimeError(f"magpie response choices[0].text not a string: {type(raw).__name__}")
        return _parse_magpie_completion(raw, self.template)


# ── Driver ────────────────────────────────────────────────────────────


def _warn_if_remote(endpoint: str) -> None:
    """Log a WARNING when the endpoint isn't localhost — privacy guard."""
    host = (urlparse(endpoint).hostname or "").lower()
    if host not in LOCAL_HOSTS:
        logger.warning(
            "endpoint host '%s' is NOT localhost -- prompts will leave this machine. "
            "Set --endpoint http://localhost:... if that's not what you want.",
            host,
        )


def _append_jsonl(path: Path, row: dict) -> None:
    """Append one row to JSONL. Creates parent dir + file if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _rewrite_combined_text(jsonl_path: Path, txt_path: Path) -> int:
    """Rebuild the canonical .txt file from the .jsonl file.

    Done as a final pass (not append-per-row) so the .txt output is always
    a clean dedup'd snapshot of whatever's in the .jsonl, even after
    resumed runs.
    """
    if not jsonl_path.exists():
        return 0
    written = 0
    seen: set[str] = set()
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as out:
        first = True
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = (row.get("prompt") or "").strip()
            completion = (row.get("completion") or "").strip()
            if not prompt or not completion:
                continue
            # First row per prompt wins -- same rule the resume skip uses,
            # so .jsonl files that accumulated duplicates stay clean here.
            key = _prompt_key(prompt)
            if key in seen:
                continue
            seen.add(key)
            if not first:
                out.write("\n")
            out.write(f"User: {prompt}\n\nAssistant: {completion}\n")
            first = False
            written += 1
    return written


def collect(
    *,
    endpoint: str,
    model: str,
    prompts: list[str],
    tag: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 1.0,
    api_key: str | None = None,
    system_prompt: str | None = None,
    timeout: float = 120.0,
    resume: bool = False,
    output_dir: Path = OUTPUT_DIR,
    sleep_between: float = 0.0,
    client: TeacherClient | None = None,
) -> dict:
    """Run the full collection pass.

    Returns a summary dict:
      ``{"requested": N, "skipped_done": N, "ok": N, "failed": N,
         "jsonl": str, "txt": str}``
    """
    _warn_if_remote(endpoint)

    jsonl_path = output_dir / f"distill_{tag}.jsonl"
    txt_path = output_dir / f"distill_{tag}.txt"

    if not resume and jsonl_path.exists():
        logger.warning(
            "%s exists and --resume not set -- overwriting (rename if you want to keep it)",
            jsonl_path,
        )
        jsonl_path.unlink()

    done = _load_done_keys(jsonl_path) if resume else set()
    if done:
        logger.info("resume: %d prompt(s) already in %s, will skip", len(done), jsonl_path)

    if client is None:
        client = TeacherClient(
            endpoint=endpoint,
            model=model,
            api_key=api_key,
            timeout=timeout,
            system_prompt=system_prompt,
        )

    ok = 0
    failed = 0
    skipped = 0
    started = time.time()
    for idx, prompt in enumerate(prompts, 1):
        key = _prompt_key(prompt)
        if key in done:
            skipped += 1
            continue
        try:
            completion = client.ask(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        except RuntimeError as exc:
            logger.warning("prompt %d/%d failed: %s", idx, len(prompts), exc)
            failed += 1
            continue
        completion = (completion or "").strip()
        if not completion:
            logger.warning("prompt %d/%d returned empty completion -- skipped", idx, len(prompts))
            failed += 1
            continue
        _append_jsonl(jsonl_path, {"prompt": prompt, "completion": completion})
        done.add(key)
        ok += 1
        if idx % 10 == 0 or idx == len(prompts):
            elapsed = time.time() - started
            rate = ok / elapsed if elapsed > 0 else 0.0
            logger.info(
                "[%d/%d] ok=%d failed=%d skipped=%d (%.2f rows/s)",
                idx,
                len(prompts),
                ok,
                failed,
                skipped,
                rate,
            )
        if sleep_between > 0:
            time.sleep(sleep_between)

    written = _rewrite_combined_text(jsonl_path, txt_path)
    logger.info(
        "done: requested=%d ok=%d failed=%d skipped=%d -> %s (%d text blocks)",
        len(prompts),
        ok,
        failed,
        skipped,
        txt_path,
        written,
    )
    return {
        "requested": len(prompts),
        "skipped_done": skipped,
        "ok": ok,
        "failed": failed,
        "jsonl": str(jsonl_path),
        "txt": str(txt_path),
    }


def magpie_collect(
    *,
    endpoint: str,
    model: str,
    n: int,
    tag: str,
    template_name: str = "chatml",
    custom_prefix: str | None = None,
    custom_user_end: str | None = None,
    custom_assistant_end: str | None = None,
    max_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
    api_key: str | None = None,
    timeout: float = 120.0,
    resume: bool = False,
    output_dir: Path = OUTPUT_DIR,
    sleep_between: float = 0.0,
    client: MagpieClient | None = None,
) -> dict:
    """Run a Magpie-style empty-prefix collection pass.

    Generates ``n`` (instruction, answer) pairs via independent
    completion calls. The instruction comes from the model itself, not
    from a prompts file — that's the whole point of Magpie. Output
    format matches :func:`collect`: each pair becomes a JSONL row
    ``{"prompt": instruction, "completion": answer}`` and is folded
    into the canonical ``.txt`` snapshot.

    Resume semantics are content-based: the SHA-256 prefix of the
    *generated instruction* is treated as the dedup key, same as the
    prompts-driven path. Re-running with ``--resume`` skips
    instructions already in the JSONL but always appends NEW pairs up
    to ``n``. Without ``--resume`` the existing JSONL is overwritten.

    Failures (HTTP error, parse error, empty pair) are counted and
    skipped — they never abort the run. A consistently high failure
    rate (>50%) is logged as a WARNING at end-of-run because that
    almost always means the wrong template was chosen for the model.
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"magpie requires n >= 1 (got n={n!r})")

    _warn_if_remote(endpoint)
    template = _resolve_magpie_template(
        template_name,
        custom_prefix=custom_prefix,
        custom_user_end=custom_user_end,
        custom_assistant_end=custom_assistant_end,
    )

    jsonl_path = output_dir / f"distill_{tag}.jsonl"
    txt_path = output_dir / f"distill_{tag}.txt"

    if not resume and jsonl_path.exists():
        logger.warning(
            "%s exists and --resume not set -- overwriting (rename if you want to keep it)",
            jsonl_path,
        )
        jsonl_path.unlink()

    done = _load_done_keys(jsonl_path) if resume else set()
    if done:
        logger.info(
            "resume: %d instruction(s) already in %s, will skip duplicates",
            len(done),
            jsonl_path,
        )

    if client is None:
        client = MagpieClient(
            endpoint=endpoint,
            model=model,
            template=template,
            api_key=api_key,
            timeout=timeout,
        )

    ok = 0
    failed = 0
    duplicate = 0
    started = time.time()
    for idx in range(1, n + 1):
        try:
            instruction, answer = client.ask_pair(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        except RuntimeError as exc:
            logger.warning("magpie %d/%d failed: %s", idx, n, exc)
            failed += 1
            continue
        key = _prompt_key(instruction)
        if key in done:
            duplicate += 1
            continue
        done.add(key)
        _append_jsonl(jsonl_path, {"prompt": instruction, "completion": answer})
        ok += 1
        if idx % 10 == 0 or idx == n:
            elapsed = time.time() - started
            rate = ok / elapsed if elapsed > 0 else 0.0
            logger.info(
                "[%d/%d] ok=%d failed=%d duplicate=%d (%.2f rows/s)",
                idx,
                n,
                ok,
                failed,
                duplicate,
                rate,
            )
        if sleep_between > 0:
            time.sleep(sleep_between)

    if n > 0 and failed / n > 0.5:
        logger.warning(
            "magpie failure rate %.0f%% (%d/%d) -- wrong --template for model? "
            "expected template markers may not match the model's chat training",
            100.0 * failed / n,
            failed,
            n,
        )

    written = _rewrite_combined_text(jsonl_path, txt_path)
    logger.info(
        "magpie done: requested=%d ok=%d failed=%d duplicate=%d -> %s (%d text blocks)",
        n,
        ok,
        failed,
        duplicate,
        txt_path,
        written,
    )
    return {
        "requested": n,
        "ok": ok,
        "failed": failed,
        "duplicate": duplicate,
        "jsonl": str(jsonl_path),
        "txt": str(txt_path),
    }


# ── CLI ───────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Collect distillation corpus from an external teacher AI.",
    )
    p.add_argument(
        "--endpoint", default=DEFAULT_ENDPOINT, help=f"OpenAI-compatible base URL (default: {DEFAULT_ENDPOINT})"
    )
    p.add_argument("--model", required=True, help="Model name on the teacher (e.g. 'qwen3:8b' for Ollama)")
    p.add_argument(
        "--prompts",
        type=Path,
        default=None,
        help='Path to a .jsonl ({"prompt":...}) or .txt (one per line) file. '
        "Required for the prompts-driven mode; ignored when --magpie is set.",
    )
    p.add_argument(
        "--magpie",
        type=int,
        default=0,
        metavar="N",
        help="Magpie mode (arxiv:2406.08464): generate N instruction/answer pairs "
        "via empty-prefix completion. Mutually exclusive with --prompts.",
    )
    p.add_argument(
        "--template",
        default="chatml",
        choices=sorted(_MAGPIE_TEMPLATES) + ["custom"],
        help="Magpie chat-template family (default: chatml -- covers Qwen2/3, "
        "ChatGLM3+, and most ChatML-trained models). Use 'custom' to supply "
        "your own marker strings via --magpie-prefix / --magpie-user-end / "
        "--magpie-assistant-end.",
    )
    p.add_argument(
        "--magpie-prefix",
        default=None,
        help='Custom raw-prefix string for --template custom (e.g. "<|im_start|>user\\n")',
    )
    p.add_argument("--magpie-user-end", default=None, help="Custom user->assistant marker for --template custom")
    p.add_argument(
        "--magpie-assistant-end",
        default=None,
        help="Custom assistant-end marker for --template custom (used as stop string)",
    )
    p.add_argument(
        "--tag", default="external", help="Output filename tag -- writes data/finetune/distill_<tag>.{jsonl,txt}"
    )
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature. When unset, defaults to 0.7 in "
        "prompts mode and 1.0 in Magpie mode (per arxiv:2406.08464 "
        "empty-prefix-synthesis recommendation).",
    )
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system message prepended to every chat request (prompts mode only -- Magpie ignores this)",
    )
    p.add_argument("--api-key", default=None, help="Bearer token (only needed for cloud endpoints)")
    p.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout in seconds (default 120)")
    p.add_argument(
        "--resume", action="store_true", help="Skip prompts/instructions already present in the existing .jsonl output"
    )
    p.add_argument("--limit", type=int, default=0, help="(prompts mode) Stop after this many prompts (0 = no limit)")
    p.add_argument(
        "--sleep-between", type=float, default=0.0, help="Seconds to sleep between requests (rate-limit friendliness)"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    # Mutual exclusion: prompts-driven and Magpie are different data
    # collection paradigms. Asking for both is a misconfiguration.
    if args.magpie and args.prompts:
        logger.error("--magpie and --prompts are mutually exclusive (Magpie generates its own instructions). Drop one.")
        return 2
    if not args.magpie and not args.prompts:
        logger.error("either --prompts <file> or --magpie <N> is required")
        return 2

    if args.magpie:
        # Magpie wants higher diversity than prompts-driven distill — the
        # paper (arxiv:2406.08464) recommends T=1.0 for empty-prefix
        # instruction synthesis. Resolve here so users who pass
        # --temperature explicitly still get their value through.
        if args.temperature is None:
            magpie_temperature = 1.0
            logger.info("Magpie default temperature 1.0 (paper recommendation; pass --temperature to override)")
        else:
            magpie_temperature = args.temperature
        try:
            summary = magpie_collect(
                endpoint=args.endpoint,
                model=args.model,
                n=args.magpie,
                tag=args.tag,
                template_name=args.template,
                custom_prefix=args.magpie_prefix,
                custom_user_end=args.magpie_user_end,
                custom_assistant_end=args.magpie_assistant_end,
                max_tokens=args.max_tokens,
                temperature=magpie_temperature,
                top_p=args.top_p,
                api_key=args.api_key,
                timeout=args.timeout,
                resume=args.resume,
                sleep_between=args.sleep_between,
            )
        except ValueError as exc:
            logger.error(str(exc))
            return 2
        if summary["ok"] == 0:
            logger.error(
                "no successful pairs written -- check --template matches model + endpoint supports /v1/completions"
            )
            return 1
        return 0

    try:
        prompts = load_prompts(args.prompts)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 2
    if not prompts:
        logger.error("no prompts loaded from %s", args.prompts)
        return 2
    if args.limit and args.limit > 0:
        prompts = prompts[: args.limit]
    logger.info("loaded %d prompt(s) from %s", len(prompts), args.prompts)
    prompts_temperature = 0.7 if args.temperature is None else args.temperature
    summary = collect(
        endpoint=args.endpoint,
        model=args.model,
        prompts=prompts,
        tag=args.tag,
        max_tokens=args.max_tokens,
        temperature=prompts_temperature,
        top_p=args.top_p,
        api_key=args.api_key,
        system_prompt=args.system_prompt,
        timeout=args.timeout,
        resume=args.resume,
        sleep_between=args.sleep_between,
    )
    if summary["ok"] == 0:
        logger.error("no successful rows written -- check teacher endpoint + model name")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
