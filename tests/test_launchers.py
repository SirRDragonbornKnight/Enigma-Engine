"""The launcher chain, parameterized: one Start/Stop/Tray/Quiet that can bring
up ANY AI in this checkout instead of a forked copy per persona.

Two things are pinned here and they pull in opposite directions. Enigma's
launch must be byte-compatible -- the argument list, the log names and the
memory store she has always had, proven by running the real .ps1 with -DryRun
rather than asserted in prose. A pack's launch must be genuinely hers -- her
port, her memory under her own home, her pack path quoted so a directory with
a space in it survives the trip.

The PowerShell tests shell out to powershell.exe, so they are skipped where it
is absent; the derivation tests are the drift detector between
Enigma-Persona.ps1's Get-PersonaSlug and core/persona.py's _slug, which are
the same rule written twice in two languages."""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from enigma_engine.core.persona import ENIGMA, Persona
from make_persona_launchers import START_BAT, TALK_BAT, write_launchers

REPO_ROOT = Path(__file__).resolve().parent.parent
LIB = REPO_ROOT / "Enigma-Persona.ps1"
CHAIN = ("Start-Enigma.ps1", "Stop-Enigma.ps1", "Enigma-Tray.ps1", "Enigma-Quiet.ps1")

_POWERSHELL = shutil.which("powershell")
needs_powershell = pytest.mark.skipif(_POWERSHELL is None, reason="no powershell on this box")

# What Start-Enigma.ps1 passed to serve before it took a -Persona, argument for
# argument. The adopted checkpoint, the 2048 context the v2 lineage needs, all
# four organs, and data\memory -- her real long-term memory, in the repo, which
# this wave does NOT move.
HER_SERVE_ARGS = [
    "serve_enigma.py",
    "--port", "8000",
    "--model", r"models\enigma_v2_sft2\model.pth",
    "--max-context", "2048",
    "--memory-dir", r"data\memory",
    "--eyes", "--ears", "--voice", "--image-gen",
]


def _run_ps(script: str, *args: str) -> list[str]:
    """A launcher's stdout lines, run for real."""
    proc = subprocess.run(
        [_POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
         str(REPO_ROOT / script), *args],
        capture_output=True, text=True, timeout=120, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, f"{script} exited {proc.returncode}: {proc.stderr}"
    return proc.stdout.splitlines()


def _tokens(line: str) -> list[str]:
    """A printed command line split the way the C runtime splits it -- quotes
    group, and they are not part of the value python receives."""
    return [t[1:-1] if t.startswith('"') and t.endswith('"') else t
            for t in shlex.split(line, posix=False)]


def _resolve(pack: Path | None) -> dict:
    """Resolve-EnigmaPersona's answer for a pack (or Enigma), as a dict."""
    pack_dir = "" if pack is None else str(pack)
    proc = subprocess.run(
        [_POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command",
         f". '{LIB}'; Resolve-EnigmaPersona -EngineDir '{REPO_ROOT}' "
         f"-PackDir '{pack_dir}' | ConvertTo-Json -Compress"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def _write_pack(pack: Path, **fields) -> Path:
    pack.mkdir(parents=True, exist_ok=True)
    blob = {"name": "Atlas Prime", "data_dirname": ".atlas_prime", "port": 8300}
    blob.update(fields)
    (pack / "pack.json").write_text(json.dumps(blob), encoding="utf-8")
    return pack


# ---------------------------------------------------------------------------
# byte-compatibility: what her own launch resolves to, run for real
# ---------------------------------------------------------------------------


@needs_powershell
def test_her_serve_command_line_is_the_one_it_has_always_been():
    """The whole wave rests on this: parameterizing the launchers changed
    nothing about the launch that takes no parameters. Asserted against the
    REAL script through -DryRun, because "byte-compatible" written in a
    comment is a claim and this is a measurement."""
    lines = _run_ps("Start-Enigma.ps1", "-DryRun")
    serve = next(ln for ln in lines if ln.startswith("DRYRUN serve: "))
    tokens = _tokens(serve[len("DRYRUN serve: "):])
    assert tokens[0] == str(REPO_ROOT / "venv" / "Scripts" / "python.exe")
    assert tokens[1:] == HER_SERVE_ARGS
    logs = next(ln for ln in lines if ln.startswith("DRYRUN logs: "))
    assert logs == (f"DRYRUN logs: {REPO_ROOT / 'serve_enigma.log'} | "
                    f"{REPO_ROOT / 'serve_enigma.err.log'}")


@needs_powershell
def test_her_stop_targets_the_same_port_and_window_it_always_did():
    lines = _run_ps("Stop-Enigma.ps1", "-DryRun")
    assert lines[0] == "DRYRUN stop: persona=Enigma port=8000"
    # ...and the one refinement: the window a PACK opened is not hers to close.
    assert lines[1] == ("DRYRUN window: match=*enigma_window.py* extra= "
                        "exclude=*--persona*")


# ---------------------------------------------------------------------------
# ...and what a pack's launch resolves to instead
# ---------------------------------------------------------------------------


@needs_powershell
def test_a_pack_launch_binds_her_port_and_her_own_memory(tmp_path):
    """Every value a second AI needs to not collide with the first: her port,
    her memory store under her OWN home, her pack on the serve command line,
    and a log pair of her own. The pack directory carries a SPACE on purpose
    -- that is the argument that split in half while -ArgumentList took an
    array."""
    pack = _write_pack(tmp_path / "pack with space")
    lines = _run_ps("Start-Enigma.ps1", "-Persona", str(pack), "-DryRun")
    tokens = _tokens(next(ln for ln in lines if ln.startswith("DRYRUN serve: "))
                     [len("DRYRUN serve: "):])
    assert tokens[1:] == [
        "serve_enigma.py",
        "--port", "8300",
        "--model", r"models\enigma_v2_sft2\model.pth",
        "--max-context", "2048",
        "--memory-dir", str(Path.home() / ".atlas_prime" / "memory"),
        "--eyes", "--ears", "--voice", "--image-gen",
        "--persona", str(pack),
    ]
    logs = next(ln for ln in lines if ln.startswith("DRYRUN logs: "))
    assert logs == (f"DRYRUN logs: {REPO_ROOT / 'serve_atlas_prime.log'} | "
                    f"{REPO_ROOT / 'serve_atlas_prime.err.log'}")


@needs_powershell
def test_the_port_flag_beats_the_packs_own_field(tmp_path):
    """The pack says which port she binds; -Port is the override for the run
    where it is already taken."""
    pack = _write_pack(tmp_path / "atlas")
    lines = _run_ps("Start-Enigma.ps1", "-Persona", str(pack), "-Port", "8455", "-DryRun")
    tokens = _tokens(next(ln for ln in lines if ln.startswith("DRYRUN serve: "))
                     [len("DRYRUN serve: "):])
    assert tokens[tokens.index("--port") + 1] == "8455"
    assert _run_ps("Stop-Enigma.ps1", "-Persona", str(pack), "-Port", "8455", "-DryRun")[0] == \
        "DRYRUN stop: persona=Atlas Prime port=8455"


@needs_powershell
def test_a_pack_stop_leaves_the_other_ais_window_alone(tmp_path):
    pack = _write_pack(tmp_path / "atlas")
    lines = _run_ps("Stop-Enigma.ps1", "-Persona", str(pack), "-DryRun")
    assert lines[0] == "DRYRUN stop: persona=Atlas Prime port=8300"
    assert lines[1] == (f"DRYRUN window: match=*enigma_window.py* extra=*{pack}* "
                        "exclude=")


# ---------------------------------------------------------------------------
# the derivation, against the Python it mirrors
# ---------------------------------------------------------------------------


@needs_powershell
def test_enigmas_memory_store_does_not_move():
    """STATED PLAINLY because it was only ever implied: a pack's memory rides
    her home, but HERS stays the repo-relative data\\memory it has always
    been. The wave-1 mute/talk migration does not extend to her memories."""
    her = _resolve(None)
    assert her["MemoryDir"] == r"data\memory"
    assert str(Path.home()) not in her["MemoryDir"]
    assert her["Name"] == "Enigma" and her["Slug"] == "enigma"
    assert her["Mutex"] == r"Local\EnigmaTray"
    assert Path(her["Log"]).name == "serve_enigma.log"


@needs_powershell
@pytest.mark.parametrize("name, dirname", [
    ("Atlas", None),           # the plain case: home derived from the slug
    ("Atlas Prime", None),     # a SPACE folds to one underscore...
    ("Atlas-Prime-", None),    # ...and a trailing run of them is trimmed off
    ("Atlas", ".chosen_home"), # an explicit home wins over the derivation
])
def test_the_launcher_derives_the_home_python_derives(tmp_path, name, dirname):
    """Get-PersonaSlug in Enigma-Persona.ps1 is core/persona.py's `_slug`
    rewritten in PowerShell, and the launcher hands the result to serve as
    --memory-dir. If the two ever disagree, a pack writes her memories into a
    directory the loader does not think is hers, and nothing else in the suite
    can see it -- so the two implementations are compared here directly."""
    blob = {"name": name, "port": 8300}  # no data_dirname = both sides derive it
    if dirname is not None:
        blob["data_dirname"] = dirname
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "pack.json").write_text(json.dumps(blob), encoding="utf-8")

    persona = Persona.load(pack)
    resolved = _resolve(pack)
    assert resolved["Name"] == persona.name
    assert resolved["Slug"] == persona.slug
    assert resolved["MemoryDir"] == str(persona.home / "memory")
    assert resolved["Port"] == persona.port


@needs_powershell
def test_a_pack_without_a_manifest_refuses_at_the_launcher(tmp_path):
    """Fail closed: a half-built pack directory must not resolve to Enigma's
    values and start HER server under someone else's name."""
    proc = subprocess.run(
        [_POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
         str(REPO_ROOT / "Start-Enigma.ps1"), "-Persona", str(tmp_path), "-DryRun"],
        capture_output=True, text=True, timeout=120, cwd=str(REPO_ROOT),
    )
    assert proc.returncode != 0
    assert "pack.json" in proc.stderr
    assert "DRYRUN" not in proc.stdout


# ---------------------------------------------------------------------------
# one copy of the ownership question
# ---------------------------------------------------------------------------


def test_the_chain_defines_get_servingpersona_exactly_once():
    """It shipped as byte-identical copies in Start and Stop, which is the
    shape that drifts: the script that decides a port is hers and the script
    that decides it is hers ENOUGH TO KILL must never disagree. One definition
    in the dot-sourced file, and nothing else in the chain carries one."""
    lib = LIB.read_text(encoding="utf-8")
    assert lib.count("function Get-ServingPersona") == 1
    for name in CHAIN:
        text = (REPO_ROOT / name).read_text(encoding="utf-8")
        assert text.count("function Get-ServingPersona") == 0, f"{name} carries a second copy"
        assert '. (Join-Path $PSScriptRoot "Enigma-Persona.ps1")' in text, (
            f"{name} does not source the shared launcher helpers"
        )


def test_the_fallback_that_keeps_a_wedged_server_killable_survives():
    """The one behavior 3a shipped that a convergence could silently drop: an
    empty answer means UNKNOWN, never "not ours", so a serve wedged
    mid-generation is still recognized by its process and stays killable."""
    lib = LIB.read_text(encoding="utf-8")
    assert 'return ""' in lib
    for name in ("Start-Enigma.ps1", "Stop-Enigma.ps1"):
        text = (REPO_ROOT / name).read_text(encoding="utf-8")
        assert '$cmd -like "*serve_enigma.py*"' in text
        assert '$procName -in @("enigma", "enigma-ai")' in text


# ---------------------------------------------------------------------------
# the generated shims
# ---------------------------------------------------------------------------


def test_the_generated_shims_are_thin(tmp_path):
    """A shim knows the pack and the port and NOTHING else: the checkpoint,
    the organs, the memory store and the ownership check all stay in the one
    .ps1 both AIs call. A copied launcher is how this chain would rot -- the
    stale copy is always the AI that stops booting."""
    pack = _write_pack(tmp_path / "atlas pack")
    persona, written = write_launchers(pack)
    assert persona.name == "Atlas Prime"
    assert [p.name for p in written] == [START_BAT, TALK_BAT]

    start = (pack / START_BAT).read_text(encoding="ascii")
    assert start.splitlines()[-1] == (
        f'powershell -NoProfile -ExecutionPolicy Bypass -File '
        f'"{REPO_ROOT / "Start-Enigma.ps1"}" -Persona "{pack.resolve()}" -Port 8300'
    )
    talk = (pack / TALK_BAT).read_text(encoding="ascii")
    assert f'call "%~dp0{START_BAT}"' in talk
    assert "if errorlevel 1 exit /b 1" in talk
    assert talk.splitlines()[-1] == (
        f'start "" py -3.12 "{REPO_ROOT / "enigma_window.py"}" '
        f'--persona "{pack.resolve()}" --url http://127.0.0.1:8300/'
    )
    for name, text in ((START_BAT, start), (TALK_BAT, talk)):
        for leaked in ("--model", "--memory-dir", "--eyes", "enigma_v2_sft2", "venv"):
            assert leaked not in text, f"{name} duplicates launcher logic ({leaked})"
        raw = (pack / name).read_bytes()
        assert raw.count(b"\r\n") == raw.count(b"\n"), f"{name} needs CRLF for cmd.exe"


def test_the_generator_refuses_to_shim_enigma(tmp_path):
    """Her launchers exist -- Talk to Enigma.bat and the Desktop shortcuts
    that point at it. A generated pair beside them is a second way to start
    the same AI, and the two would drift."""
    pack = _write_pack(tmp_path / "hers", name=ENIGMA["name"],
                       data_dirname=ENIGMA["data_dirname"],
                       name_meaning=ENIGMA["name_meaning"])
    with pytest.raises(SystemExit, match="IS Enigma"):
        write_launchers(pack)
    assert not (pack / START_BAT).exists()
    assert not (pack / TALK_BAT).exists()


def test_the_generator_refuses_a_pack_that_declares_no_port(tmp_path):
    """Explicit beats clever (ruled): a pack with no port would inherit 8000
    and collide with whoever is serving there, discovered at boot inside a
    hidden window."""
    pack = tmp_path / "portless"
    pack.mkdir()
    (pack / "pack.json").write_text(
        json.dumps({"name": "Atlas", "data_dirname": ".atlas"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="declares no port"):
        write_launchers(pack)
    assert not (pack / START_BAT).exists()


def test_persona_packs_are_local_only():
    """RULED: packs are user data, never the repo's -- other people install
    this engine and author their own AIs on their own machines."""
    ignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "personas/" in [line.strip() for line in ignore.splitlines()]
    git = shutil.which("git")
    if git:
        tracked = subprocess.run([git, "ls-files", "personas"], cwd=str(REPO_ROOT),
                                 capture_output=True, text=True, timeout=60)
        assert tracked.stdout.strip() == "", "a pack file is tracked in git"
