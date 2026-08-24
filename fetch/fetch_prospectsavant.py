"""Fetch ProspectSavant hitter leaderboard data for all levels and seasons.

Saves to data/prospectSavant/ps_{Level}_{year}.csv — replaces manual exports.
No authentication required.

URL: https://oriolebird.pythonanywhere.com/leaders/hitters/{level}/{year}/{min_pa}/{min_age}/{max_age}

Usage:
    python fetch_prospectsavant.py          # fetch all levels, all seasons
    python fetch_prospectsavant.py 2026     # fetch all levels for one season
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

PS_DIR   = Path(__file__).resolve().parent.parent / "data" / "prospectSavant"
BASE_URL = "https://oriolebird.pythonanywhere.com/leaders/hitters"

SEASONS    = [2023, 2024, 2025, 2026]
LEVELS     = ["AAA", "AA", "A+", "A", "Rk"]   # Rk = rookie ball (2026+ only)
MIN_PA     = 50
MIN_AGE    = 16
MAX_AGE    = 28
CALL_DELAY = 0.4

# Columns to keep — raw values only (we compute our own percentile ranks in build scripts)
KEEP_COLS = {
    # Identity / metadata
    "Name":         "name",
    "Org":          "MLB_AbbName",
    "MLBAMId":      "MLBAMId",
    "MinorMasterId":"MinorMasterId",
    "Age":          "age",
    "AgeDays":      "age_days",       # exact age in days at time of data pull
    "PA":           "pa",
    "AB":           "ab",
    # Traditional
    "BA":           "ba",
    "OBP":          "obp",
    "SLG":          "slg",
    "ISO":          "iso",
    "HR":           "hr",
    "BB":           "bb",
    "K":            "k",
    "BB%":          "bbrate",
    "K%":           "krate",
    "wRC+":         "wrcplus",
    "BABIP":        "babip",
    "wOBA":         "woba",
    "xwOBA":        "xwoba",
    # Discipline
    "Chase%":       "chaserate",
    "ZContact%":    "zcontact",
    "SwStr%":       "swstr",
    "Zone%":        "zonerate",
    "ZSwing%":      "zswing",
    "Whiff%":       "whiffrate",
    # Power / Batted ball
    "MaxEV":        "maxev",
    "EV90":         "ev90",
    "AvgEV":        "ev",
    "HardHit%":     "hhrate",
    "Barrel%BBE":   "barrelbbe",
    "Barrel%PA":    "barrelpa",
    "xBA":          "xba",
    "xSLG":         "xslg",
    "PullAir%":     "pullair",
    "GB%":          "gbrate",
    "FB%":          "fbrate",
    "LD%":          "ldrate",
    "LA":           "langle",
    # Speed
    "Spd":          "spd",
    "wBsR":         "wbsr",
    "wBsR/PA":      "wbsr_pa",
    # PS composite
    "PSScore":      "pscore",
}


def fetch_level_season(level: str, year: int) -> pd.DataFrame | None:
    url = f"{BASE_URL}/{level}/{year}/{MIN_PA}/{MIN_AGE}/{MAX_AGE}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "prospectsMain/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read()).get("data", [])
        if not data:
            return None
        df = pd.DataFrame(data)

        # Rename to friendly output names, keep only available cols
        rename = {v: k for k, v in KEEP_COLS.items() if v in df.columns}
        df = df.rename(columns=rename)
        out_cols = [k for k in KEEP_COLS if k in df.columns]
        df = df[out_cols].copy()

        df["Season"] = year
        df["Level"]  = level
        df = df.sort_values("PA", ascending=False).reset_index(drop=True)
        return df
    except Exception as e:
        print(f"  Warning: {level} {year} failed — {e}")
        return None


def main() -> None:
    PS_DIR.mkdir(exist_ok=True)

    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        seasons = [int(sys.argv[1])]
    else:
        seasons = SEASONS

    for year in seasons:
        for level in LEVELS:
            print(f"  Fetching {level} {year}...", end=" ", flush=True)
            df = fetch_level_season(level, year)
            if df is None:
                print("0 players")
                continue
            safe_level = level.replace("+", "p")
            out_path = PS_DIR / f"ps_{safe_level}_{year}.csv"
            df.to_csv(out_path, index=False)
            print(f"{len(df)} players -> {out_path.name}")
            time.sleep(CALL_DELAY)

    print("\nDone.")


if __name__ == "__main__":
    main()

