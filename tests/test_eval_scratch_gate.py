r"""The eval harness's scratch gate and the capabilities fields it trusts.

Round-2 review (2026-08-13): the gate was PORT-ONLY -- `http://192.168.1.50:8123`
and a serve on 8123 carrying the launcher's REAL --memory-dir both passed, and
the run's first act is an unrecoverable `clear()` of that store (backup=False
by delete semantics). The gate now requires all three: a scratch port, a local
host, and a memory dir the target REPORTS as somewhere other than the live
store. Audit 2026-08-14 tightened the residue: an UNANSWERED capabilities
probe ({}) is unknown, not "no store", and dir identity is same-FILE, not
string equality (alias spellings like \\?\C:\... dodge resolve()).
"""
from __future__ import annotations

from pathlib import Path

import eval_behavior
from eval_behavior import _is_scratch_target


def _caps(mem_dir):
    if mem_dir is None:
        return {"memory": False, "memory_dir": None}
    return {"memory": True, "memory_dir": str(mem_dir)}


def test_scratch_needs_port_host_and_a_disposable_store(tmp_path):
    scratch_dir = tmp_path / "memory_eval"
    scratch_dir.mkdir()
    scratch = _caps(scratch_dir)

    assert _is_scratch_target("http://127.0.0.1:8123", scratch) is True
    assert _is_scratch_target("http://localhost:8123", scratch) is True
    # the server AFFIRMS it has no store -> the clear is a no-op -> fine
    assert _is_scratch_target("http://127.0.0.1:8123", _caps(None)) is True

    # wrong port
    assert _is_scratch_target("http://127.0.0.1:8000", scratch) is False
    # non-local host, right port -- the LAN-wipe case
    assert _is_scratch_target("http://192.168.1.50:8123", scratch) is False
    assert _is_scratch_target("http://some-remote-box:8123/", scratch) is False
    # malformed
    assert _is_scratch_target("http://127.0.0.1:notaport", scratch) is False


def test_an_unanswered_capabilities_probe_is_not_scratch():
    """{} is what _capabilities returns for a timeout, an older build's 404,
    or a proxy's HTML body -- every one reachable on a LIVE serve that is
    momentarily busy. _wait_for_server polls /v1/models, so this gate is the
    only thing between {} and the unrecoverable clear (audit 2026-08-14:
    the shipped code read {} as "no store" and would have wiped)."""
    assert _is_scratch_target("http://127.0.0.1:8123", {}) is False


def test_the_live_store_is_never_scratch(tmp_path, monkeypatch):
    """The exact case that motivated the fix: serve restarted on 8123 with
    the launcher's real --memory-dir -- under ANY spelling of that dir."""
    live = tmp_path / "live_memory"
    live.mkdir()
    monkeypatch.setattr(eval_behavior, "_LIVE_MEMORY_DIR", live.resolve())

    assert _is_scratch_target("http://127.0.0.1:8123", _caps(live)) is False
    # alias spellings: same-FILE identity catches a case variant; the \\?\
    # form refuses either way (samefile matches, or stat fails on the alias
    # and an unprovable identity refuses)
    assert _is_scratch_target("http://127.0.0.1:8123",
                              _caps(str(live.resolve()).upper())) is False
    assert _is_scratch_target("http://127.0.0.1:8123",
                              _caps("\\\\?\\" + str(live.resolve()))) is False
    # a server that has a store but cannot SAY where it lives is not
    # verifiably disposable (old server, proxy, anything) -> refuse
    assert _is_scratch_target("http://127.0.0.1:8123",
                              {"memory": True, "memory_dir": None}) is False
    assert _is_scratch_target("http://127.0.0.1:8123", {"memory": True}) is False
    # a non-string dir has no provable identity -> refuse, not a traceback
    assert _is_scratch_target("http://127.0.0.1:8123",
                              {"memory": True, "memory_dir": 123}) is False
    # a REAL scratch dir is still recognized under the patched live root
    scratch_dir = tmp_path / "memory_eval"
    scratch_dir.mkdir()
    assert _is_scratch_target("http://127.0.0.1:8123", _caps(scratch_dir)) is True


def test_a_client_tool_shadowing_a_builtin_is_refused(monkeypatch):
    """Dispatch is by NAME alone: a client tool named `forget` was consumed
    by the server-side loop -- deleting a REAL memory -- while the client's
    own tool never ran and could not tell (review 2026-08-13). Colliding
    names 400 honestly before anything executes -- in BOTH spec shapes the
    renderer accepts (audit 2026-08-14: the flat shape sailed past a gate
    that only read the nested key)."""
    from fastapi.testclient import TestClient

    import serve_enigma as serve

    monkeypatch.setattr(serve, "_BOOTED", True)
    client = TestClient(serve.app)
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function",
                   "function": {"name": "forget", "parameters": {}}}],
    })
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "forget" in detail and "built-in" in detail

    # the FLAT spec chat_format's renderer equally accepts (t.get("function", t))
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "name": "forget", "parameters": {}}],
    })
    assert r.status_code == 400
    assert "forget" in r.json()["detail"]

    # a spec whose "function" is not an object is a client error, not a 500
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": None}],
    })
    assert r.status_code == 400
    assert "malformed" in r.json()["detail"]


def test_capabilities_reports_memory_dir_and_base_checkpoint_honesty(tmp_path, monkeypatch):
    """serve's /v1/capabilities must (a) name the store dir the eval gate
    verifies against, and (b) stop advertising builtins/search on a BASE
    checkpoint where the tool loop is unreachable -- the dict is a
    self-report consumers act on (the chat page builds its controls from
    it), so a dead surface must not read as live."""
    import serve_enigma as serve

    class _Store:
        def __init__(self, d: Path):
            self.dir = d

    monkeypatch.setattr(serve, "MEMORY", _Store(tmp_path))
    monkeypatch.setattr(serve, "INSTRUCT", True)
    monkeypatch.setattr(serve, "SEARCHER", object())
    caps = serve.capabilities()
    assert caps["memory_dir"] == str(tmp_path.resolve())
    assert "calculate" in caps["builtins"]
    assert caps["search"] is True

    monkeypatch.setattr(serve, "INSTRUCT", False)
    caps = serve.capabilities()
    assert caps["builtins"] == [], "a base checkpoint cannot run the builtin loop"
    assert caps["search"] is False

    monkeypatch.setattr(serve, "MEMORY", None)
    caps = serve.capabilities()
    assert caps["memory_dir"] is None


def test_the_persona_fields_are_additive_to_the_gate_that_reads_this_dict(
        tmp_path, monkeypatch):
    """capabilities now also says WHO is serving, and this harness is its one
    machine consumer: the same dict decides whether the store this run is
    about to clear is disposable. Additive means additive -- every key the
    gate and the chat page already read is still there, still carrying what
    it carried, and the verdict is unchanged.

    It is unchanged for a PACK too: a server is judged by the store it
    reports, never by the name it answers to. A gate that started reading the
    name would refuse a legitimate scratch server, or bless a live one."""
    import serve_enigma as serve
    from enigma_engine.core.persona import Persona

    class _Store:
        def __init__(self, d: Path):
            self.dir = d

    scratch = tmp_path / "memory_eval"
    scratch.mkdir()
    monkeypatch.setattr(serve, "MEMORY", _Store(scratch))

    caps = serve.capabilities()
    assert caps["persona"] == "Enigma"
    assert caps["persona_is_default"] is True
    assert {"memory", "memory_dir", "voice", "ears", "eyes", "image_gen",
            "search", "instruct", "max_context", "builtins"} <= set(caps)
    assert caps["memory_dir"] == str(scratch.resolve())
    assert _is_scratch_target("http://127.0.0.1:8123", caps) is True

    monkeypatch.setattr(serve, "PERSONA", Persona(name="Atlas", data_dirname=".atlas"))
    caps = serve.capabilities()
    assert caps["persona"] == "Atlas"
    assert caps["persona_is_default"] is False
    assert caps["memory_dir"] == str(scratch.resolve())
    assert _is_scratch_target("http://127.0.0.1:8123", caps) is True

    # ...and the live store still refuses, whoever answers for it
    monkeypatch.setattr(serve, "MEMORY", _Store(eval_behavior._LIVE_MEMORY_DIR))
    assert _is_scratch_target("http://127.0.0.1:8123", serve.capabilities()) is False
