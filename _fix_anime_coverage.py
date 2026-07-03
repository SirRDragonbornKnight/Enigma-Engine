"""Coverage fix pass: recover the 5 mis-named wikis (try alias subdomains) and
re-crawl the marquee wikis that came back broken-thin (Bleach=1, Naruto=326, ...)
with a LOWER prose-length gate so infobox-heavy anime pages qualify.
Un-marks the targeted wikis in progress.json so fetch_fandom retries them.
GPU-free: network + disk only.
"""

import sys
from itertools import chain

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, r"C:\Users\SirKn\Enigma Engine")
import collect_pretraining_data as C

# Lower the prose-length gate (was 1500) so character/episode pages survive.
C.MIN_FANDOM_LENGTH = 800
MIN_ARTICLES = 100

# 5 mis-named wikis from the last run -> try alias subdomains in priority order.
RECOVER = [
    (
        "Rising of the Shield Hero",
        ["shieldhero", "tate-no-yuusha-no-nariagari", "risingoftheshieldhero", "the-rising-of-the-shield-hero"],
    ),
    ("Classroom of the Elite", ["youkoso-jitsuryoku", "classroomoftheelite", "you-zitsu", "cote"]),
    (
        "Irregular at Magic High School",
        ["mahouka-koukou-no-rettousei", "the-irregular-at-magic-high-school", "irregularatmagichighschool"],
    ),
    ("Goblin Slayer", ["goblin-slayer", "goblinslayer-anime"]),
    ("Madoka Magica", ["madoka-magica", "puellamagi", "madokamagica", "magireco"]),
]
# Marquee wikis that came back broken-thin -> un-mark + re-crawl at lower threshold.
REDO = [
    ("Bleach", ["bleach"]),
    ("Naruto", ["naruto"]),
    ("Fate", ["fate"]),
    ("Berserk", ["berserk"]),
]


def probe_one(sub):
    try:
        r = C.SESSION.get(
            f"https://{sub}.fandom.com/api.php",
            params={"action": "query", "format": "json", "meta": "siteinfo", "siprop": "statistics|general"},
            timeout=8,
        )
        if r.status_code == 200:
            q = r.json().get("query", {})
            arts = q.get("statistics", {}).get("articles", 0)
            g = q.get("general", {})
            if g.get("lang", "en") == "en" and arts >= MIN_ARTICLES:
                return (sub, g.get("sitename", sub), arts)
    except Exception:
        return None
    return None


def first_valid(label, aliases):
    for sub in aliases:
        res = probe_one(sub)
        if res:
            print(f"  OK   {label:32} -> {res[0]:30} {res[2]:>7,}  [{res[1]}]", flush=True)
            return (res[0], label)
    print(f"  MISS {label:32} (tried: {', '.join(aliases)})", flush=True)
    return None


print("probing recovery + marquee-redo targets (prose gate lowered to 800 chars)...", flush=True)
valid = []
for label, aliases in chain(RECOVER, REDO):
    v = first_valid(label, aliases)
    if v:
        valid.append(v)

progress = C.load_progress()
done = set(progress.get("fandom_done_wikis", []))
target_subs = {sub for sub, _ in valid}
unmarked = sorted(target_subs & done)
if unmarked:
    progress["fandom_done_wikis"] = [w for w in progress["fandom_done_wikis"] if w not in target_subs]
    print(f"\nun-marked for re-crawl: {', '.join(unmarked)}", flush=True)

before = len(list(C.FANDOM_DIR.glob("*.txt")))
print(f"\nfandom .txt before = {before:,}", flush=True)
print(f"crawling {len(valid)} targets (unlimited, threshold=800)...", flush=True)

C.fetch_fandom(valid, 0, progress)
C.save_progress(progress)

after = len(list(C.FANDOM_DIR.glob("*.txt")))
print(f"\nfandom .txt after = {after:,}  (+{after - before:,} new)", flush=True)
print("[done] coverage fix pass complete", flush=True)
