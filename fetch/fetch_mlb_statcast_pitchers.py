"""Fetch season-by-season MLB pitcher Statcast data from Baseball Savant.

Mirrors fetch_mlb_statcast.py for the pitching side. Pulls three endpoints
per season and merges into one wide file:

  1. Expected stats (pitcher)  — BA/xBA, SLG/xSLG, wOBA/xwOBA, ERA, xERA
  2. Percentile rankings       — K%, BB%, Whiff%, Chase%, fb_velocity, fb_spin,
                                 curve_spin, xERA (0–100 pct ranks, not raw)
  3. Pitch movement (FF)       — fastball avg_speed, IVBreak, HBreak, Tail, Rise

Coverage:
  xStats + Percentile rankings : 2015–current
  Pitch movement               : 2015–current

Output:
  data/api/mlb_statcast_pitchers.csv  — one row per pitcher × season

Usage:
  python fetch/fetch_mlb_statcast_pitchers.py           # incremental
  python fetch/fetch_mlb_statcast_pitchers.py --full    # full 2015–current
"""

import argparse
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

DATA_DIR  = Path(__file__).resolve().parent.parent / "data"
API_DIR   = DATA_DIR / "api"
API_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = API_DIR / "mlb_statcast_pitchers.csv"

FIRST_SEASON   = 2015
CURRENT_SEASON = 2026
SEASONS        = list(range(FIRST_SEASON, CURRENT_SEASON + 1))

SLEEP_SEC = 1.2
HEADERS   = {"User-Agent": "prospectsMain/1.0 (baseball research)"}

URL_XSTATS = (
    "https://baseballsavant.mlb.com/leaderboard/expected_statistics"
    "?type=pitcher&year={year}&position=&team=&min=0&csv=true"
)
URL_PCTRANKS = (
    "https://baseballsavant.mlb.com/leaderboard/percentile-rankings"
    "?type=pitcher&year={year}&min=0&csv=true"
)
# Fastball movement — one row per pitcher per pitch type; we filter to FF
URL_MOVEMENT = (
    "https://baseballsavant.mlb.com/leaderboard/pitch-movement"
    "?year={year}&team=&min=25&pitch_type=FF&hand=&csv=true"
)


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


def _flip_name(s) -> str:
    if pd.isna(s):
        return ""
    parts = [p.strip() for p in str(s).split(",")]
    return " ".join(reversed(parts))


def parse_xstats(df: pd.DataFrame, year: int) -> pd.DataFrame:
    df = df.copy()
    df["Season"]   = year
    df["MLBAM_ID"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    if "last_name, first_name" in df.columns:
        df["Name"] = df["last_name, first_name"].apply(_flip_name)

    rename = {
        "pa":                        "PA",
        "bip":                       "BIP",
        "ba":                        "BA_against",
        "est_ba":                    "xBA_against",
        "est_ba_minus_ba_diff":      "xBA_diff",
        "slg":                       "SLG_against",
        "est_slg":                   "xSLG_against",
        "est_slg_minus_slg_diff":    "xSLG_diff",
        "woba":                      "wOBA_against",
        "est_woba":                  "xwOBA_against",
        "est_woba_minus_woba_diff":  "xwOBA_diff",
        "era":                       "ERA",
        "xera":                      "xERA",
        "era_minus_xera_diff":       "ERA_minus_xERA",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    keep = ["Season", "MLBAM_ID", "Name"] + [v for v in rename.values() if v in df.columns]
    return df[[c for c in keep if c in df.columns]]


def parse_pctranks(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Percentile rankings — values are 0–100 pct ranks, not raw stats."""
    df = df.copy()
    df["Season"]   = year
    df["MLBAM_ID"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")

    rename = {
        "xwoba":            "pct_xwOBA",
        "xera":             "pct_xERA",
        "exit_velocity":    "pct_AvgEV_against",
        "max_ev":           "pct_MaxEV_against",
        "hard_hit_percent": "pct_HardHit%",
        "brl_percent":      "pct_Brl%",
        "k_percent":        "pct_K%",
        "bb_percent":       "pct_BB%",
        "whiff_percent":    "pct_Whiff%",
        "chase_percent":    "pct_Chase%",
        "fb_velocity":      "pct_FBVelo",
        "fb_spin":          "pct_FBSpin",
        "curve_spin":       "pct_CurveSpin",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    keep = ["Season", "MLBAM_ID"] + [v for v in rename.values() if v in df.columns]
    return df[[c for c in keep if c in df.columns]]


def parse_movement(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Fastball movement — already filtered to FF at fetch time."""
    df = df.copy()
    df["Season"]   = year
    id_col = "pitcher_id" if "pitcher_id" in df.columns else "player_id"
    df["MLBAM_ID"] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")

    rename = {
        "avg_speed":               "FF_AvgVelo",
        "pitches_thrown":          "FF_Pitches",
        "pitcher_break_z_induced": "FF_IVBreak",
        "pitcher_break_x":         "FF_HBreak",
        "rise":                    "FF_Rise",
        "tail":                    "FF_Tail",
        "diff_z":                  "FF_IVBreak_vs_Lg",
        "diff_x":                  "FF_HBreak_vs_Lg",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    keep = ["Season", "MLBAM_ID"] + [v for v in rename.values() if v in df.columns]
    return df[[c for c in keep if c in df.columns]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Refetch all seasons 2015–current")
    args = parser.parse_args()

    print("=== fetch_mlb_statcast_pitchers.py ===\n")

    full_mode = args.full or not OUT_PATH.exists()
    if full_mode:
        print(f"Full mode — fetching {FIRST_SEASON}–{CURRENT_SEASON}")
        existing = pd.DataFrame()
        fetch_seasons = SEASONS
    else:
        print(f"Incremental mode — refreshing {CURRENT_SEASON} only")
        existing = pd.read_csv(OUT_PATH, dtype={"MLBAM_ID": "Int64"})
        existing = existing[existing["Season"] != CURRENT_SEASON]
        fetch_seasons = [CURRENT_SEASON]

    all_rows: list[pd.DataFrame] = []
    total = len(fetch_seasons)
    print(f"Fetching {total} season(s) × 3 endpoints each\n")

    for i, year in enumerate(fetch_seasons, 1):
        print(f"[{i:3}/{total}] {year}", end="  ")

        xst_df = _fetch_csv(URL_XSTATS.format(year=year))
        pct_df = _fetch_csv(URL_PCTRANKS.format(year=year))
        mov_df = _fetch_csv(URL_MOVEMENT.format(year=year))

        parts = []
        tags  = []

        if xst_df is not None:
            p = parse_xstats(xst_df, year); parts.append(p); tags.append(f"xSt:{len(p)}")
        else:
            tags.append("xSt:0")

        if pct_df is not None:
            p = parse_pctranks(pct_df, year); parts.append(p); tags.append(f"Pct:{len(p)}")
        else:
            tags.append("Pct:0")

        if mov_df is not None:
            p = parse_movement(mov_df, year); parts.append(p); tags.append(f"FF:{len(p)}")
        else:
            tags.append("FF:0")

        print("  ".join(tags))

        if not parts:
            continue

        base = parts[0]
        for other in parts[1:]:
            base = base.merge(other, on=["Season", "MLBAM_ID"], how="outer")

        name_cols = [c for c in base.columns if c.startswith("Name")]
        if len(name_cols) > 1:
            base["Name"] = base[name_cols[0]].combine_first(base[name_cols[1]])
            for nc in name_cols[1:]:
                base = base.drop(columns=[nc], errors="ignore")

        all_rows.append(base)

    if all_rows:
        new_df   = pd.concat(all_rows, ignore_index=True)
        combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    else:
        combined = existing

    combined = combined.sort_values(["Season", "MLBAM_ID"]).reset_index(drop=True)
    combined.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(combined):,} rows -> {OUT_PATH.name}")
    print(f"Columns ({len(combined.columns)}): {list(combined.columns)}")
    seasons = sorted(combined["Season"].dropna().unique())
    print(f"Seasons: {int(seasons[0])}–{int(seasons[-1])}, {combined['MLBAM_ID'].nunique():,} unique pitchers")
    print("\nDone.")


if __name__ == "__main__":
    main()
