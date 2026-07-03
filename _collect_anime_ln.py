"""Targeted Fandom crawl: missing big anime/manga wikis + light-novel series wikis.

Reuses the proven fetch_fandom pipeline from collect_pretraining_data.py
(heuristic quality + AI-content filters only -- NO GPU / model / network-AI).
Validates each candidate subdomain (English, >= MIN_ARTICLES) before crawling and
LOGS every dropped guess (no silent misses), skips wikis already in progress.json,
crawls a bounded first batch, and persists progress at the end.

GPU-free: network + disk only. Light-novel PROSE is copyrighted and NOT scraped;
this collects the light-novel *fan wikis* (summaries/lore), same as the anime wikis.
"""

import sys
from itertools import zip_longest

try:  # Fandom sitenames contain non-cp1252 chars (e.g. "Haikyū") -> force UTF-8 stdout
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, r"C:\Users\SirKn\Enigma Engine")
import collect_pretraining_data as C

MIN_ARTICLES = 150  # inclusive of smaller LN wikis; quality filtered later
NEW_ARTICLE_BUDGET = 50_000  # bound this first batch

ANIME = [
    ("naruto", "Naruto"),
    ("onepiece", "One Piece"),
    ("bleach", "Bleach"),
    ("attackontitan", "Attack on Titan"),
    ("myheroacademia", "My Hero Academia"),
    ("kimetsu-no-yaiba", "Demon Slayer"),
    ("jujutsu-kaisen", "Jujutsu Kaisen"),
    ("hunterxhunter", "Hunter x Hunter"),
    ("fairytail", "Fairy Tail"),
    ("swordartonline", "Sword Art Online"),
    ("tokyoghoul", "Tokyo Ghoul"),
    ("deathnote", "Death Note"),
    ("fma", "Fullmetal Alchemist"),
    ("evangelion", "Evangelion"),
    ("sailormoon", "Sailor Moon"),
    ("inuyasha", "InuYasha"),
    ("cowboybebop", "Cowboy Bebop"),
    ("steins-gate", "Steins;Gate"),
    ("onepunchman", "One Punch Man"),
    ("berserk", "Berserk"),
    ("spy-x-family", "Spy x Family"),
    ("chainsaw-man", "Chainsaw Man"),
    ("gintama", "Gintama"),
    ("haikyuu", "Haikyuu"),
    ("detectiveconan", "Detective Conan"),
    ("ghibli", "Studio Ghibli"),
    ("vinlandsaga", "Vinland Saga"),
    ("blackclover", "Black Clover"),
    ("jojo", "JoJo's Bizarre Adventure"),
    ("codegeass", "Code Geass"),
    ("puella-magi", "Madoka Magica"),
    ("hellsing", "Hellsing"),
    ("fate", "Fate"),
]
LIGHT_NOVELS = [
    ("overlordmaruyama", "Overlord"),
    ("mushokutensei", "Mushoku Tensei"),
    ("rezero", "Re:Zero"),
    ("konosuba", "KonoSuba"),
    ("shieldhero", "Rising of the Shield Hero"),
    ("danmachi", "DanMachi"),
    ("classroom-of-the-elite", "Classroom of the Elite"),
    ("oregairu", "Oregairu"),
    ("no-game-no-life", "No Game No Life"),
    ("tensura", "Reincarnated as a Slime"),
    ("youjo-senki", "Saga of Tanya"),
    ("mahouka", "Irregular at Magic High School"),
    ("spiceandwolf", "Spice and Wolf"),
    ("log-horizon", "Log Horizon"),
    ("bofuri", "BOFURI"),
    ("goblinslayer", "Goblin Slayer"),
    ("haruhi", "Haruhi Suzumiya"),
    ("bakemonogatari", "Monogatari"),
    ("accelworld", "Accel World"),
    ("durarara", "Durarara"),
    ("toarumajutsunoindex", "A Certain Magical Index"),
    ("frieren", "Frieren"),
]

# Interleave (LN first in each pair) so a bounded budget still covers BOTH.
CANDIDATES = []
for ln, an in zip_longest(LIGHT_NOVELS, ANIME):
    if ln:
        CANDIDATES.append(ln)
    if an:
        CANDIDATES.append(an)


def probe(sub):
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


print(f"probing {len(CANDIDATES)} candidate wikis (anime + light-novel)...", flush=True)
valid, dropped = [], []
for sub, label in CANDIDATES:
    res = probe(sub)
    if res:
        valid.append((res[0], label))
        print(f"  OK   {sub:24} {res[2]:>8,} articles  [{res[1]}]", flush=True)
    else:
        dropped.append(sub)
        print(f"  DROP {sub:24} (not found / non-en / <{MIN_ARTICLES} articles)", flush=True)

print(f"\nvalid={len(valid)}  dropped={len(dropped)}", flush=True)
if dropped:
    print("DROPPED subdomains (fix names for a follow-up pass): " + ", ".join(dropped), flush=True)

progress = C.load_progress()
done = set(progress.get("fandom_done_wikis", []))
todo = [(w, l) for (w, l) in valid if w not in done]
skipped_done = [w for (w, _) in valid if w in done]
if skipped_done:
    print(f"already-done, skipping: {', '.join(skipped_done)}", flush=True)

before = len(list(C.FANDOM_DIR.glob("*.txt"))) if C.FANDOM_DIR.exists() else 0
budget = before + NEW_ARTICLE_BUDGET
print(f"\nfandom .txt before = {before:,}", flush=True)
print(
    f"crawling {len(todo)} new targeted wikis, budget = +{NEW_ARTICLE_BUDGET:,} (stop at {budget:,} total)", flush=True
)

C.fetch_fandom(todo, budget, progress)
C.save_progress(progress)

after = len(list(C.FANDOM_DIR.glob("*.txt")))
print(f"\nfandom .txt after = {after:,}  (+{after - before:,} new articles)", flush=True)
print("[done] targeted anime + light-novel crawl complete", flush=True)
