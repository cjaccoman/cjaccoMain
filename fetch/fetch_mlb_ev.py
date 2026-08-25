"""Fetch MLB Statcast exit-velocity stats from Baseball Savant leaderboard.

Downloads per-player per-season data (2015–present) and saves career
averages (PA-weighted MaxEV and EV90) for use as a fallback in player_comps.csv
when MiLB EV data is unavailable.

Output: data/api/mlb_statcast_ev.csv
  One row per player. Columns: MLBAM_ID, Name, MaxEV_mlb, EV90_mlb, MLB_EV_PA

Usage:
  python fetch/fetch_mlb_ev.py
"""

import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = DATA_DIR / "api" / "mlb_statcast_ev.csv"

FIRST_SEASON = 2015  # Statcast tracking began in MLB in 2015
LAST_SEASON  = 2026
SLEEP_SEC    = 1.5

# Baseball Savant statcast batting leaderboard — CSV export
URL = (
    "https://baseballsavant.mlb.com/leaderboard/statcast"
    "?type=batter&year={year}&position=&team=&min=0&csv=true"
)

# Baseball Savant exit-velocity leaderboard columns (as of 2024):
#   player_id       — MLBAM ID
#   last_name, first_name — name (comma-separated in one column)
#   max_hit_speed   — MaxEV
#   attempts        — batted balls (used as PA proxy for weighting)
# Note: ev90 is not available from this endpoint; EV90_mlb is left null.


def fetch_season(year: int) -> pd.DataFrame | None:
    url = URL.format(year=year)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
        if df.empty:
            return None

        if "player_id" not in df.columns or "max_hit_speed" not in df.columns:
            print(f"  {year}: unexpected columns: {list(df.columns[:12])}")
            return None

        out = pd.DataFrame()
        out["MLBAM_ID"] = pd.to_numeric(df["player_id"], errors="coerce")
        # Name column is "last_name, first_name" — flip to "First Last"
        if "last_name, first_name" in df.columns:
            parts = df["last_name, first_name"].str.split(", ", n=1, expand=True)
            out["Name"] = parts[1].str.strip() + " " + parts[0].str.strip()
        out["MaxEV"]  = pd.to_numeric(df["max_hit_speed"], errors="coerce").where(df["max_hit_speed"] > 0)
        out["PA"]     = pd.to_numeric(df.get("attempts", pd.Series(dtype=float)), errors="coerce").fillna(0)
        out["Season"] = year
        return out[out["MaxEV"].notna()]

    except Exception as exc:
        print(f"  {year}: ERROR — {exc}")
        return None


def main() -> None:
    current_year = pd.Timestamp.now().year
    seasons = range(FIRST_SEASON, min(current_year, LAST_SEASON) + 1)

    frames = []
    for year in seasons:
        print(f"Fetching {year}… ", end="", flush=True)
        df = fetch_season(year)
        if df is not None and len(df):
            frames.append(df)
            print(f"{len(df)} players")
        else:
            print("no data")
        time.sleep(SLEEP_SEC)

    if not frames:
        print("No data fetched.")
        return

    all_seasons = pd.concat(frames, ignore_index=True)
    print(f"\nTotal season-rows: {len(all_seasons):,}")

    # Career PA-weighted averages per player
    def career_ev(g):
        pa = g["PA"].fillna(0)
        total_pa = pa.sum()
        if total_pa == 0 or g["MaxEV"].isna().all():
            return pd.Series({"MaxEV_mlb": float("nan"), "MLB_EV_PA": 0})
        max_ev = (g["MaxEV"] * pa).sum() / total_pa
        return pd.Series({"MaxEV_mlb": round(max_ev, 1), "MLB_EV_PA": int(total_pa)})

    career = (
        all_seasons.groupby("MLBAM_ID")
        .apply(career_ev)
        .reset_index()
    )

    # Attach best name (last season seen)
    name_map = (
        all_seasons.sort_values("Season")
        .dropna(subset=["Name"])
        .drop_duplicates("MLBAM_ID", keep="last")
        [["MLBAM_ID", "Name"]]
    ) if "Name" in all_seasons.columns else pd.DataFrame(columns=["MLBAM_ID", "Name"])

    career = career.merge(name_map, on="MLBAM_ID", how="left")
    career = career[["MLBAM_ID", "Name", "MaxEV_mlb", "MLB_EV_PA"]]
    career = career[career["MaxEV_mlb"].notna()]

    career.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(career):,} players → {OUT_PATH}")
    print(f"  MaxEV_mlb range: {career['MaxEV_mlb'].min():.1f}–{career['MaxEV_mlb'].max():.1f}")


if __name__ == "__main__":
    main()
