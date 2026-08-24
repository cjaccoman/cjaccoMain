"""Regenerate all existing player cards in Documents/PlayerCards/2026/.

For each .png found, converts the slug back to a candidate name, looks it up
in aaa_2026.csv first (AAA card), then rk_2026.csv (Rk card). Skips files
that can't be matched in either file.
"""

import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_player_card import generate_card, find_player, normalize_name

DATA_DIR  = Path(__file__).resolve().parent.parent / "data" / "rankings"
CARDS_DIR = Path.home() / "Documents" / "prospectFiles" / "PlayerCards" / "2026"


def slug_to_name(slug: str) -> str:
    """Convert filename slug to title-cased candidate name."""
    name = slug.replace("_", " ").strip()
    return " ".join(w.capitalize() for w in name.split())


SKIP_NAMES = {
    "henry bolte",
    "travis bazzana",
}


def main():
    aaa = pd.read_csv(DATA_DIR / "aaa_2026.csv")
    rk  = pd.read_csv(DATA_DIR / "rk_2026.csv")

    cards = sorted(CARDS_DIR.rglob("*.png"))
    print(f"Found {len(cards)} cards to regenerate.\n")

    ok = skipped = 0
    for path in cards:
        slug = path.stem
        # Strip trailing _cat suffix (category variant cards)
        slug = re.sub(r"_cat$", "", slug)
        candidate = slug_to_name(slug)

        if candidate.lower() in SKIP_NAMES:
            print(f"  SKIP  {path.relative_to(CARDS_DIR)} — on skip list")
            skipped += 1
            continue

        # Try AAA first, then Rk
        p = find_player(aaa, candidate)
        if p is not None:
            level = "aaa"
        else:
            p = find_player(rk, candidate)
            if p is not None:
                level = "rk"
            else:
                print(f"  SKIP  {path.relative_to(CARDS_DIR)} — no match for '{candidate}'")
                skipped += 1
                continue

        generate_card(p["Name"], year=2026, level=level)
        ok += 1

    print(f"\nDone. {ok} regenerated, {skipped} skipped.")


if __name__ == "__main__":
    main()
