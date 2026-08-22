"""Web lookup primitive -- the search organ, the sixth of the family.

Counterpart of tts/asr/eyes/imagegen: a capability living beside the model,
not in its weights. The model emits ``<search>query</search>`` (native vocab
ids -- the v2 table carves 12/13 for exactly this; legacy vocabs carry None
and the feature honestly does not exist for them), serve intercepts the
closed tag, this organ runs the query, and the results are spliced back as a
tool turn for the next generation hop.

Privacy posture (ruled 2026-07-29: privacy, not zero egress): queries go ONLY
to the machine's own SearXNG instance -- a metasearch service the user runs
and configured themselves (WSL2 docker on this box). What leaves the machine
leaves through THAT service's user-chosen engines; this organ never talks to
the internet directly and defaults to localhost.

The transport is injectable so tests never touch the network, matching the
model_factory pattern the other organs use. A down backend degrades honestly
per query -- SearXNG living in WSL means it can come up AFTER serve does, so
a boot-time failure must not disable search for the server's whole life.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_BASE_URL = "http://127.0.0.1:8888"
DEFAULT_K = 5
# One hit renders as one line; a snippet longer than this is cut so a single
# verbose engine cannot eat the context budget the answer needs.
SNIPPET_CHARS = 240
_TIMEOUT_S = 8.0


class SearchError(Exception):
    """The search backend is unreachable or returned something unusable."""


def _default_fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "enigma-engine-search-organ"})
    base = url.split("/search")[0]
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        # HTTPError IS a URLError, so it must be caught first -- an answered
        # request is not "unreachable". SearXNG's stock answer to a JSON
        # request without 'json' in search.formats is exactly 403, so name
        # that cause where it actually surfaces.
        hint = " (is 'json' in SearXNG's search.formats?)" if exc.code == 403 else ""
        raise SearchError(f"search backend at {base} answered HTTP {exc.code}{hint}") from exc
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise SearchError(f"search backend unreachable at {base}: {exc}") from exc


class Searcher:
    """One SearXNG endpoint, queried per request; no state between queries."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, fetch=None):
        base = (base_url or "").strip().rstrip("/")
        if not base.startswith(("http://", "https://")):
            raise SearchError(f"search base URL must be http(s), got {base_url!r}")
        self.base_url = base
        self._fetch = fetch if fetch is not None else _default_fetch

    def probe(self) -> bool:
        """Whether the backend answers right now. Boot uses this for an
        informative line only -- never to disable the organ, since the WSL
        service may start after serve does."""
        try:
            self.query("ping", k=1)
            return True
        except SearchError:
            return False

    def query(self, text: str, k: int = DEFAULT_K) -> list[dict]:
        """Top-k results as [{title, url, snippet}]. Zero hits is a valid
        answer (empty list), not an error -- "nothing found" and "backend
        down" must stay distinguishable or the model is taught to treat an
        outage as settled absence."""
        text = (text or "").strip()
        if not text:
            raise SearchError("empty search query")
        k = max(1, int(k))
        url = f"{self.base_url}/search?q={urllib.parse.quote_plus(text)}&format=json"
        body = self._fetch(url)
        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            # SearXNG answers 403 HTML when the json format is not enabled in
            # its settings.yml -- name that likely cause instead of "bad JSON".
            raise SearchError(
                f"search backend at {self.base_url} did not return JSON "
                "(is 'json' in SearXNG's search.formats?)"
            ) from exc
        raw = data.get("results")
        if not isinstance(raw, list):
            raise SearchError(f"search backend response carries no results list ({self.base_url})")
        hits: list[dict] = []
        for r in raw:
            if not isinstance(r, dict):
                continue
            title = _one_line(r.get("title"))
            # Coerced the same way the title is: `.strip()` on a truthy non-str
            # url raised AttributeError, which is not a SearchError, so it blew
            # past serve's _apply_search handler and answered 500 instead of
            # degrading to "nothing found". A malformed result is skipped here,
            # never allowed to take the whole query down.
            hit_url = _one_line(r.get("url"))
            if not title or not hit_url:
                continue
            hits.append({
                "title": title,
                "url": hit_url,
                "snippet": _one_line(r.get("content"))[:SNIPPET_CHARS],
            })
            if len(hits) >= k:
                break
        return hits


def _one_line(text) -> str:
    """Collapse whitespace so one hit renders as one line."""
    return " ".join(str(text or "").split())


def render_results(query: str, hits: list[dict]) -> str:
    """The tool-turn text the model reads back -- ONE renderer for serve and
    for the SFT data that teaches the shape, so the trained bytes and the
    served bytes cannot drift (the BUILTIN_TOOLS lesson applied from birth).
    """
    if not hits:
        return f'search results for "{query}": none found'
    lines = [f'search results for "{query}":']
    for i, h in enumerate(hits, 1):
        snippet = f" -- {h['snippet']}" if h.get("snippet") else ""
        lines.append(f"{i}. {h['title']}{snippet} ({h['url']})")
    return "\n".join(lines)
