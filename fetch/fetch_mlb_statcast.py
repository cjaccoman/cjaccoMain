"""Fetch season-by-season MLB Statcast data from Baseball Savant leaderboards.

Pulls three endpoints per season and merges into one wide file:
  1. Statcast EV leaderboard  — MaxEV, AvgEV, EV50, Barrel%, SweetSpot%, EV95%
  2. Expected stats            — xBA, xSLG, xwOBA (and actual deltas), PA, BIP
  3. Sprint speed              — sprint_speed, bolts, hp_to_1b

Plus one career-aggregate pull (same data regardless of year filter):
  4. Bat tracking              — avg_bat_speed, swing_length, hard_swing_rate, etc.

Coverage:
  Statcast / Expected / Sprint speed : 2015–current
  Bat tracking                        : career aggregate only (not per-season)

Outputs:
  data/api/mlb_statcast.csv      — one row per player × season (2015–current)
  data/api/mlb_bat_tracking.csv  — one row per player, career aggregate

Usage:
  python fetch/fetch_mlb_statcast.py           # incremental — current season only
  python fetch/fetch_mlb_statcast.py --full    # full refetch 2015–current
"""

import argparse
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR     = Path(__file__).resolve().parent.parent / "data"
API_DIR      = DATA_DIR / "api"
API_DIR.mkdir(parents=True, exist_ok=True)

SEASON_OUT   = API_DIR / "mlb_statcast.csv"
BATTRACK_OUT = API_DIR / "mlb_bat_tracking.csv"

FIRST_SEASON   = 2015   # Statcast tracking began in MLB
CURRENT_SEASON = 2026
SEASONS        = list(range(FIRST_SEASON, CURRENT_SEASON + 1))

SLEEP_SEC = 1.2
HEADERS   = {"User-Agent": "prospectsMain/1.0 (baseball research)"}

# Savant leaderboard URLs
URL_EV = (
    "https://baseballsavant.mlb.com/leaderboard/statcast"
    "?type=batter&year={year}&position=&team=&min=0&csv=true"
)
URL_XSTATS = (
    "https://baseballsavant.mlb.com/leaderboard/expected_statistics"
    "?type=batter&year={year}&position=&team=&min=0&csv=true"
)
URL_SPEED = (
    "https://baseballsavant.mlb.com/leaderboard/sprint_speed"
    "?min_opp=0&position=&team=&year={year}&csv=true"
)
URL_BATTRACK = (
    "https://baseballsavant.mlb.com/leaderboard/bat-tracking"
    "?attackZone=&batSide=&contactType=&count=&dateStart=&dateEnd="
    "&gameType=&isHardHit=&minSwings=0&minGroupSwings=1&pitchType="
    "&position=&seasonStart=&seasonEnd=&team=&type=batter&year=2026&csv=true"
)


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _fetch_csv(url: str, retries: int = 2) -> pd.DataFrame | None:
    for attempt in range(retries + 1):
        try:
            time.sleep(SLEEP_SEC)
            r = requests.get(url, timeout=30, headers=HEADERS)
            r.raise_for_status()
            df = pd.read_csv(StringIO(r.text))
            if df.empty or len(df.columns) < 2:
                return None
            return df
        except Exception as e:
            if attempt == retries:
                print(f"    WARN: failed {url[:80]} : {e}")
                return None
            time.sleep(2 ** attempt)
    return None


# ---------------------------------------------------------------------------
# Per-season parsers
# ---------------------------------------------------------------------------

def parse_ev(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Parse statcast EV leaderboard."""
    df = df.copy()
    df["Season"]    = year
    df["MLBAM_ID"]  = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")

    # Name: 'last_name, first_name' -> 'First Last'
    name_col = "last_name, first_name"
    if name_col in df.columns:
        df["Name"] = df[name_col].apply(
            lambda s: " ".join(reversed([p.strip() for p in str(s).split(",")])) if pd.notna(s) else ""
        )

    rename = {
        "attempts":               "BattedBalls",
        "max_hit_speed":          "MaxEV",
        "avg_hit_speed":          "AvgEV",
        "ev50":                   "EV50",
        "fbld":                   "EV_FBLD",     # avg EV on FB+LD
        "gb":                     "EV_GB",        # avg EV on GB
        "avg_hit_angle":          "AvgLaunchAngle",
        "anglesweetspotpercent":  "SweetSpot%",
        "max_distance":           "MaxDist",
        "avg_distance":           "AvgDist",
        "avg_hr_distance":        "AvgHRDist",
        "ev95plus":               "EV95plus",
        "ev95percent":            "EV95%",
        "barrels":                "Barrels",
        "brl_percent":            "Brl%",
        "brl_pa":                 "Brl/PA",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    keep = ["Season", "MLBAM_ID", "Name"] + [v for v in rename.values() if v in df.columns]
    return df[[c for c in keep if c in df.columns]]


def parse_xstats(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Parse expected statistics leaderboard."""
    df = df.copy()
    df["Season"]   = year
    df["MLBAM_ID"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")

    name_col = "last_name, first_name"
    if name_col in df.columns:
        df["Name"] = df[name_col].apply(
            lambda s: " ".join(reversed([p.strip() for p in str(s).split(",")])) if pd.notna(s) else ""
        )

    rename = {
        "pa":                       "PA",
        "bip":                      "BIP",
        "ba":                       "BA",
        "est_ba":                   "xBA",
        "est_ba_minus_ba_diff":     "xBA_diff",
        "slg":                      "SLG",
        "est_slg":                  "xSLG",
        "est_slg_minus_slg_diff":   "xSLG_diff",
        "woba":                     "wOBA",
        "est_woba":                 "xwOBA",
        "est_woba_minus_woba_diff": "xwOBA_diff",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    keep = ["Season", "MLBAM_ID"] + [v for v in rename.values() if v in df.columns]
    return df[[c for c in keep if c in df.columns]]


def parse_speed(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Parse sprint speed leaderboard."""
    df = df.copy()
    df["Season"]   = year
    df["MLBAM_ID"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")

    rename = {
        "sprint_speed":    "SprintSpeed",
        "competitive_runs":"CompetitiveRuns",
        "bolts":           "Bolts",
        "hp_to_1b":        "HP_to_1B",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    keep = ["Season", "MLBAM_ID"] + [v for v in rename.values() if v in df.columns]
    return df[[c for c in keep if c in df.columns]]


def parse_battrack(df: pd.DataFrame) -> pd.DataFrame:
    """Parse bat tracking leaderboard (career aggregate)."""
    df = df.copy()
    df["MLBAM_ID"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")

    name_col = "name"
    if name_col in df.columns:
        df["Name"] = df[name_col]

    rename = {
        "swings_competitive":         "SwingsCompetitive",
        "avg_bat_speed":              "AvgBatSpeed",
        "hard_swing_rate":            "HardSwing%",
        "squared_up_per_swing":       "SquaredUp/Swing",
        "blast_per_swing":            "Blast/Swing",
        "swing_length":               "SwingLength",
        "whiff_per_swing":            "Whiff/Swing",
        "batted_ball_event_per_swing":"BBE/Swing",
        "swords":                     "Swords",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    keep = ["MLBAM_ID", "Name"] + [v for v in rename.values() if v in df.columns]
    return df[[c for c in keep if c in df.columns]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Refetch all seasons 2015–current instead of incremental")
    args = parser.parse_args()

    print("=== fetch_mlb_statcast.py ===\n")

    # Determine scope
    full_mode = args.full or not SEASON_OUT.exists()
    if full_mode:
        print(f"Full mode — fetching seasons {FIRST_SEASON}–{CURRENT_SEASON}")
        existing = pd.DataFrame()
        fetch_seasons = SEASONS
    else:
        print(f"Incremental mode — refreshing {CURRENT_SEASON} only")
        existing = pd.read_csv(SEASON_OUT, dtype={"MLBAM_ID": "Int64"})
        existing = existing[existing["Season"] != CURRENT_SEASON]
        fetch_seasons = [CURRENT_SEASON]

    all_season_rows: list[pd.DataFrame] = []

    print(f"\nFetching {len(fetch_seasons)} season(s) × 3 endpoints each\n")
    for i, year in enumerate(fetch_seasons, 1):
        print(f"[{i:3}/{len(fetch_seasons)}] {year}", end="  ")

        ev_df    = _fetch_csv(URL_EV.format(year=year))
        xst_df   = _fetch_csv(URL_XSTATS.format(year=year))
        spd_df   = _fetch_csv(URL_SPEED.format(year=year))

        counts = []

        if ev_df is not None:
            ev_parsed = parse_ev(ev_df, year)
            counts.append(f"EV:{len(ev_parsed)}")
        else:
            ev_parsed = pd.DataFrame()
            counts.append("EV:0")

        if xst_df is not None:
            xst_parsed = parse_xstats(xst_df, year)
            counts.append(f"xStats:{len(xst_parsed)}")
        else:
            xst_parsed = pd.DataFrame()
            counts.append("xStats:0")

        if spd_df is not None:
            spd_parsed = parse_speed(spd_df, year)
            counts.append(f"Spd:{len(spd_parsed)}")
        else:
            spd_parsed = pd.DataFrame()
            counts.append("Spd:0")

        print("  ".join(counts))

        # Merge the three into one wide frame keyed on Season + MLBAM_ID
        if ev_parsed.empty and xst_parsed.empty and spd_parsed.empty:
            continue

        base = ev_parsed if not ev_parsed.empty else pd.DataFrame(columns=["Season", "MLBAM_ID"])
        if not xst_parsed.empty:
            base = base.merge(xst_parsed, on=["Season", "MLBAM_ID"], how="outer")
        if not spd_parsed.empty:
            base = base.merge(spd_parsed, on=["Season", "MLBAM_ID"], how="outer")

        # Consolidate Name column (may come from EV or xStats)
        name_cols = [c for c in base.columns if c == "Name"]
        if len(name_cols) > 1:
            base["Name"] = base["Name_x"].combine_first(base["Name_y"])
            base = base.drop(columns=["Name_x", "Name_y"], errors="ignore")

        all_season_rows.append(base)

    # Combine with existing
    if all_season_rows:
        new_df = pd.concat(all_season_rows, ignore_index=True)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = existing

    # Sort and save
    combined = combined.sort_values(["Season", "MLBAM_ID"]).reset_index(drop=True)
    combined.to_csv(SEASON_OUT, index=False)
    print(f"\nWrote {len(combined):,} rows -> {SEASON_OUT}")
    print(f"Columns ({len(combined.columns)}): {list(combined.columns)}")

    seasons_present = sorted(combined["Season"].dropna().unique())
    print(f"Seasons: {int(seasons_present[0])}–{int(seasons_present[-1])}, "
          f"{len(seasons_present)} seasons, "
          f"{combined['MLBAM_ID'].nunique():,} unique players")

    # -----------------------------------------------------------------------
    # Bat tracking — career aggregate (pull once)
    # -----------------------------------------------------------------------
    print("\nFetching bat tracking (career aggregate)...")
    bt_df = _fetch_csv(URL_BATTRACK)
    if bt_df is not None:
        bt_parsed = parse_battrack(bt_df)
        bt_parsed.to_csv(BATTRACK_OUT, index=False)
        print(f"Wrote {len(bt_parsed):,} rows -> {BATTRACK_OUT}")
        print(f"Columns: {list(bt_parsed.columns)}")
    else:
        print("  WARN: bat tracking fetch failed — skipping")

    print("\nDone.")


if __name__ == "__main__":
    main()
