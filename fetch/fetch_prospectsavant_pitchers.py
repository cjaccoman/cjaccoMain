"""Fetch ProspectSavant pitcher leaderboard data for all levels and seasons.

Mirrors fetch_prospectsavant.py for the pitching side.
Saves to data/prospectSavant/prospect_savant_pitchers.csv.
No authentication required.

URL: https://oriolebird.pythonanywhere.com/leaders/pitchers/{level}/{year}/{min_ip}/{min_age}/{max_age}

Coverage: 2023–2026, all 5 levels (2022 and earlier return 500 errors).

Usage:
    python fetch/fetch_prospectsavant_pitchers.py          # fetch all seasons
    python fetch/fetch_prospectsavant_pitchers.py 2026     # fetch one season
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

PS_DIR   = Path(__file__).resolve().parent.parent / "data" / "prospectSavant"
BASE_URL = "https://oriolebird.pythonanywhere.com/leaders/pitchers"

SEASONS    = [2023, 2024, 2025, 2026]
LEVELS     = ["AAA", "AA", "A+", "A", "Rk"]
MIN_IP     = 10
MIN_AGE    = 16
MAX_AGE    = 28
CALL_DELAY = 0.4

# Columns to keep — raw values; _p suffix = PS percentile rank (0–1)
KEEP_COLS = {
    # Identity / metadata
    "name":          "Name",
    "MLB_AbbName":   "Org",
    "MLBAMId":       "MLBAMId",
    "MinorMasterId": "MinorMasterId",
    "role":          "Role",
    "Throws":        "Throws",
    "age":           "Age",
    "age_days":      "AgeDays",
    "ip":            "IP",
    "tbf":           "TBF",
    # Traditional results
    "era":           "ERA",
    "fip":           "FIP",
    "xfip":          "xFIP",
    "whip":          "WHIP",
    "krate":         "K%",
    "bbrate":        "BB%",
    "kbb_rate":      "K-BB%",
    "hr9":           "HR9",
    # Stuff / velocity
    "max_velo":      "MaxVelo",
    "velocity":      "AvgVelo",
    "effective_speed": "EffVelo",
    "spin_rate":     "SpinRate",
    "release_extension": "Extension",
    "arm_angle":     "ArmAngle",
    # Pitch movement
    "api_break_x_arm":         "HBreak_Arm",
    "api_break_z_induced":     "IVBreak",
    "api_break_z_with_gravity":"TVBreak",
    # Whiff / chase / zone
    "swstr":         "SwStr%",
    "whiffrate":     "Whiff%",
    "chaserate":     "Chase%",
    "zcontact":      "ZContact%",
    "zonerate":      "Zone%",
    "zswing":        "ZSwing%",
    "strikerate":    "Strike%",
    # Batted ball
    "gbrate":        "GB%",
    "fbrate":        "FB%",
    "ldrate":        "LD%",
    "barrelbbe":     "Barrel%BBE",
    "barrelpa":      "Barrel%PA",
    "hhrate":        "HardHit%",
    "langle":        "LA",
    "ev":            "AvgEV",
    # Expected stats
    "xwoba":         "xwOBA",
    "xba":           "xBA",
    "xslg":          "xSLG",
    "woba":          "wOBA",
    # PS percentile ranks (0–1)
    "krate_p":       "K%_p",
    "bbrate_p":      "BB%_p",
    "kbb_p":         "KBB_p",
    "whiffrate_p":   "Whiff%_p",
    "chaserate_p":   "Chase%_p",
    "zcontact_p":    "ZContact%_p",
    "zonerate_p":    "Zone%_p",
    "velo_p":        "Velo_p",
    "gbrate_p":      "GB%_p",
    "fip_p":         "FIP_p",
    "xfip_p":        "xFIP_p",
    "strikerate_p":  "Strike%_p",
    "ev_p":          "AvgEV_p",
    # PS composite
    "pscore":        "PSScore",
}


def fetch_level_season(level: str, year: int) -> pd.DataFrame | None:
    url = f"{BASE_URL}/{level}/{year}/{MIN_IP}/{MIN_AGE}/{MAX_AGE}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "prospectsMain/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read()).get("data", [])
        if not data:
            return None
        df = pd.DataFrame(data)

        rename = {k: v for k, v in KEEP_COLS.items() if k in df.columns}
        df = df.rename(columns=rename)
        out_cols = [v for v in KEEP_COLS.values() if v in df.columns]
        df = df[out_cols].copy()

        df["Season"] = year
        df["Level"]  = level
        df = df.sort_values("IP", ascending=False).reset_index(drop=True)
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

    all_frames = []
    for year in seasons:
        for level in LEVELS:
            print(f"  Fetching {level} {year}...", end=" ", flush=True)
            df = fetch_level_season(level, year)
            if df is None:
                print("0 pitchers")
                continue
            all_frames.append(df)
            print(f"{len(df)} pitchers")
            time.sleep(CALL_DELAY)

    if all_frames:
        merged = pd.concat(all_frames, ignore_index=True)
        out_path = PS_DIR / "prospect_savant_pitchers.csv"
        merged.to_csv(out_path, index=False)
        print(f"\nWrote {len(merged):,} rows -> {out_path.name}")
        print(f"Seasons: {sorted(merged['Season'].unique())}")
        print(f"Levels:  {sorted(merged['Level'].unique())}")
    else:
        print("\nNo data fetched.")

    print("Done.")


if __name__ == "__main__":
    main()
