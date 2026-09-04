"""Fetch additional MiLB split types from the MLB Stats API.

Confirmed working bulk endpoints (from full probe of all 61 MLB Stats API stat types):

  expectedStatistics     — per-season xBA, xSLG, xwOBA, xwOBAcon (2015+, all levels)
  statsSingleSeasonAdv   — per-season batted-ball type breakdown (fly/line/ground/pop
                           hits+outs, raw swing counts, gidpOpp); goes back to 1988;
                           season param is IGNORED — always returns all-time (~44k/level)
  careerAdvanced         — career-aggregate version of statsSingleSeasonAdv (optional)

Dead (confirmed 0 rows / errors for all 5 MiLB levels):
  advanced, atGameStart, byDateRange*, byDayOfWeek, byMonth, careerPlayoffs,
  careerRegularSeason, firstYearStats, gameTypeStats, homeAndAway, lastYearStats,
  metricAverages*, opponentsFaced, pitchArsenal, projected*, rankings, sabermetrics,
  sprayChart, standard, tracking, vsOpponents, vsTeam*, winLoss, yearByYearAdvanced
  (* = 400/500 error, not just 0 rows)

Outputs:
  data/api/milb_expected_stats.csv    — Season, Level, MLBAMID, Name, xBA, xSLG,
                                        xwOBA, xwOBAcon
  data/api/milb_season_advanced.csv   — Season, Level, MLBAMID, Name + per-season
                                        batted-ball type breakdown (optional)
  data/api/milb_career_advanced.csv   — Level, MLBAMID, Name + career-aggregate
                                        batted-ball breakdown (optional)

Usage:
  python fetch/fetch_milb_splits.py                   # expectedStatistics, incremental
  python fetch/fetch_milb_splits.py --full            # expectedStatistics, all seasons
  python fetch/fetch_milb_splits.py --season-advanced # also build milb_season_advanced
  python fetch/fetch_milb_splits.py --career-advanced # also build milb_career_advanced
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR    = Path(__file__).resolve().parent.parent / "data"
API_DIR     = DATA_DIR / "api"
API_DIR.mkdir(parents=True, exist_ok=True)

OUT_EXPECTED       = API_DIR / "milb_expected_stats.csv"
OUT_SEASON_ADV     = API_DIR / "milb_season_advanced.csv"
OUT_CAREER         = API_DIR / "milb_career_advanced.csv"

BASE_URL     = "https://statsapi.mlb.com/api/v1"

SPORT_IDS = {"AAA": 11, "AA": 12, "A+": 13, "A": 14, "R": 16}

FIRST_SEASON    = 2015
CURRENT_SEASON  = 2026
SKIP_SEASONS    = {2020}

PAGE_SIZE   = 1000   # server caps at 1000 regardless of limit param
CALL_DELAY  = 0.35


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _get(url: str, retries: int = 2, timeout: int = 60) -> dict | None:
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "prospectsMain/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as e:
            if attempt == retries:
                print(f"    WARN: failed {url[:80]} : {e}", flush=True)
                return None
            time.sleep(2 + attempt * 2)
    return None


def _api(path: str, **params) -> dict | None:
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{BASE_URL}/{path}?{qs}" if qs else f"{BASE_URL}/{path}"
    time.sleep(CALL_DELAY)
    return _get(url)


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------

def fetch_paginated(stat_type: str, sport_id: int, season: int | None = None) -> list[dict]:
    """Fetch all rows for a stat type + level using offset pagination."""
    rows = []
    offset = 0
    while True:
        params = dict(
            stats=stat_type,
            playerPool="all",
            group="hitting",
            gameType="R",
            sportId=sport_id,
            limit=PAGE_SIZE,
            offset=offset,
        )
        if season is not None:
            params["season"] = season
        data = _api("stats", **params)
        if not data:
            break
        page_rows = []
        for block in data.get("stats", []):
            page_rows.extend(block.get("splits", []))
        rows.extend(page_rows)
        if len(page_rows) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return rows


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _parse_dot(v) -> float | None:
    if v is None or str(v).strip() in ("---", ".---", "-.--", ""):
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _int(v) -> int | None:
    try:
        return int(v) if v is not None else None
    except (ValueError, TypeError):
        return None


def parse_expected_row(sp: dict, level: str, season: int) -> dict:
    stat, player = sp.get("stat", {}), sp.get("player", {})
    return {
        "Season":   season,
        "Level":    level,
        "MLBAMID":  player.get("id"),
        "Name":     player.get("fullName", ""),
        "xBA":      _parse_dot(stat.get("avg")),
        "xSLG":     _parse_dot(stat.get("slg")),
        "xwOBA":    _parse_dot(stat.get("woba")),
        "xwOBAcon": _parse_dot(stat.get("wobaCon")),
    }


def parse_season_adv_row(sp: dict, level: str) -> dict:
    stat, player = sp.get("stat", {}), sp.get("player", {})
    season = sp.get("season")
    row = {
        "Season":        int(season) if season else None,
        "Level":         level,
        "MLBAMID":       player.get("id"),
        "Name":          player.get("fullName", ""),
        "PA":            _int(stat.get("plateAppearances")),
        "HBP":           _int(stat.get("hitByPitch")),
        "GIDP":          _int(stat.get("gidp")),
        "GIDPopp":       _int(stat.get("gidpOpp")),
        "Pitches":       _int(stat.get("numberOfPitches")),
        "TotalSwings":   _int(stat.get("totalSwings")),
        "SwingMiss":     _int(stat.get("swingAndMisses")),
        "BIP":           _int(stat.get("ballsInPlay")),
        "FlyOuts":       _int(stat.get("flyOuts")),
        "PopOuts":       _int(stat.get("popOuts")),
        "LineOuts":      _int(stat.get("lineOuts")),
        "GroundOuts":    _int(stat.get("groundOuts")),
        "FlyHits":       _int(stat.get("flyHits")),
        "PopHits":       _int(stat.get("popHits")),
        "LineHits":      _int(stat.get("lineHits")),
        "GroundHits":    _int(stat.get("groundHits")),
        "ExtraBaseHits": _int(stat.get("extraBaseHits")),
        "PitchesPerPA":  _parse_dot(stat.get("pitchesPerPlateAppearance")),
        "BB_K":          _parse_dot(stat.get("walksPerStrikeout")),
        "BABIP":         _parse_dot(stat.get("babip")),
    }
    # Derived batted-ball rates
    total_bip = (
        (row["FlyOuts"] or 0) + (row["PopOuts"] or 0) +
        (row["LineOuts"] or 0) + (row["GroundOuts"] or 0) +
        (row["FlyHits"] or 0) + (row["PopHits"] or 0) +
        (row["LineHits"] or 0) + (row["GroundHits"] or 0)
    )
    def _pct(n): return round(n / total_bip, 4) if total_bip > 0 else None
    row["LD_pct"]  = _pct((row["LineHits"] or 0) + (row["LineOuts"] or 0))
    row["FB_pct"]  = _pct((row["FlyHits"] or 0) + (row["FlyOuts"] or 0))
    row["GB_pct"]  = _pct((row["GroundHits"] or 0) + (row["GroundOuts"] or 0))
    row["POP_pct"] = _pct((row["PopHits"] or 0) + (row["PopOuts"] or 0))
    ts = row["TotalSwings"] or 0
    row["SwStr_pct"] = round((row["SwingMiss"] or 0) / ts, 4) if ts > 0 else None
    return row


def parse_career_adv_row(sp: dict, level: str) -> dict:
    row = parse_season_adv_row(sp, level)
    # For career rows, season is always None — prefix keys with Career_
    out = {"Level": level, "MLBAMID": row["MLBAMID"], "Name": row["Name"]}
    skip = {"Season", "Level", "MLBAMID", "Name"}
    for k, v in row.items():
        if k not in skip:
            out[f"Career_{k}"] = v
    return out


# ---------------------------------------------------------------------------
# expectedStatistics
# ---------------------------------------------------------------------------

def fetch_expected_stats(seasons: list[int]) -> pd.DataFrame:
    all_rows, total, i = [], len(seasons) * len(SPORT_IDS), 0
    for season in seasons:
        for level, sport_id in SPORT_IDS.items():
            i += 1
            print(f"  [{i:3}/{total}] expectedStatistics {level} {season}", end="  ", flush=True)
            splits = fetch_paginated("expectedStatistics", sport_id, season)
            rows = [parse_expected_row(sp, level, season) for sp in splits]
            rows = [r for r in rows if r["MLBAMID"] is not None]
            all_rows.extend(rows)
            print(f"{len(rows)} rows", flush=True)
    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# statsSingleSeasonAdvanced
# NOTE: season param is IGNORED by this endpoint — always returns all-time
# rows sorted by player name. We pull once per level (no season loop needed).
# ---------------------------------------------------------------------------

def fetch_season_advanced() -> pd.DataFrame:
    all_rows = []
    for level, sport_id in SPORT_IDS.items():
        print(f"  statsSingleSeasonAdvanced {level}", end="  ", flush=True)
        splits = fetch_paginated("statsSingleSeasonAdvanced", sport_id, season=None)
        rows = [parse_season_adv_row(sp, level) for sp in splits]
        rows = [r for r in rows if r["MLBAMID"] is not None and r["Season"] is not None]
        all_rows.extend(rows)
        print(f"{len(rows)} rows", flush=True)
    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# careerAdvanced
# ---------------------------------------------------------------------------

def fetch_career_advanced() -> pd.DataFrame:
    all_rows = []
    for level, sport_id in SPORT_IDS.items():
        print(f"  careerAdvanced {level}", end="  ", flush=True)
        splits = fetch_paginated("careerAdvanced", sport_id, season=None)
        rows = [parse_career_adv_row(sp, level) for sp in splits]
        rows = [r for r in rows if r["MLBAMID"] is not None]
        all_rows.extend(rows)
        print(f"{len(rows)} rows", flush=True)
    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Refetch all seasons 2015–current (default: current season only)")
    parser.add_argument("--season-advanced", action="store_true",
                        help="Build milb_season_advanced.csv (per-season batted-ball breakdown; slow — ~45k rows/level)")
    parser.add_argument("--career-advanced", action="store_true",
                        help="Build milb_career_advanced.csv (career-aggregate batted-ball breakdown; slow)")
    args = parser.parse_args()

    print("=== fetch_milb_splits.py ===\n", flush=True)

    # -----------------------------------------------------------------------
    # expectedStatistics
    # -----------------------------------------------------------------------
    all_seasons = [y for y in range(FIRST_SEASON, CURRENT_SEASON + 1) if y not in SKIP_SEASONS]
    full_mode   = args.full or not OUT_EXPECTED.exists()

    if full_mode:
        fetch_seasons  = all_seasons
        existing_exp   = pd.DataFrame()
        print(f"Full mode: expectedStatistics {FIRST_SEASON}–{CURRENT_SEASON} "
              f"({len(fetch_seasons)} seasons × {len(SPORT_IDS)} levels)\n", flush=True)
    else:
        fetch_seasons  = [CURRENT_SEASON]
        existing_exp   = pd.read_csv(OUT_EXPECTED, dtype={"MLBAMID": "Int64"})
        existing_exp   = existing_exp[existing_exp["Season"] != CURRENT_SEASON]
        print(f"Incremental mode: refreshing {CURRENT_SEASON} only\n", flush=True)

    new_exp = fetch_expected_stats(fetch_seasons)

    if not new_exp.empty:
        combined = pd.concat([existing_exp, new_exp], ignore_index=True)
        combined["MLBAMID"] = pd.to_numeric(combined["MLBAMID"], errors="coerce").astype("Int64")
        combined = combined.sort_values(["Season", "Level", "Name"]).reset_index(drop=True)
        combined.to_csv(OUT_EXPECTED, index=False)
        print(f"\nWrote {len(combined):,} rows -> {OUT_EXPECTED}", flush=True)
        print(f"Seasons: {sorted(combined['Season'].unique())}")
        print(f"Levels:  {combined['Level'].unique().tolist()}")
    else:
        print("No new expected stats rows.", flush=True)

    # -----------------------------------------------------------------------
    # statsSingleSeasonAdvanced (optional)
    # -----------------------------------------------------------------------
    if args.season_advanced:
        print(f"\n=== statsSingleSeasonAdvanced (all-time per-season breakdown) ===\n", flush=True)
        print("NOTE: season param ignored by API — full all-time pull per level", flush=True)
        sa_df = fetch_season_advanced()
        if not sa_df.empty:
            sa_df["MLBAMID"] = pd.to_numeric(sa_df["MLBAMID"], errors="coerce").astype("Int64")
            sa_df = sa_df.sort_values(["Season", "Level", "Name"]).reset_index(drop=True)
            sa_df.to_csv(OUT_SEASON_ADV, index=False)
            print(f"\nWrote {len(sa_df):,} rows -> {OUT_SEASON_ADV}", flush=True)
            for level in SPORT_IDS:
                n = (sa_df["Level"] == level).sum()
                yr_min = sa_df.loc[sa_df["Level"] == level, "Season"].min()
                yr_max = sa_df.loc[sa_df["Level"] == level, "Season"].max()
                print(f"  {level}: {n:,} rows ({yr_min}–{yr_max})")
        else:
            print("No season-advanced rows fetched.", flush=True)

    # -----------------------------------------------------------------------
    # careerAdvanced (optional)
    # -----------------------------------------------------------------------
    if args.career_advanced:
        print(f"\n=== careerAdvanced (career-aggregate) ===\n", flush=True)
        ca_df = fetch_career_advanced()
        if not ca_df.empty:
            ca_df["MLBAMID"] = pd.to_numeric(ca_df["MLBAMID"], errors="coerce").astype("Int64")
            ca_df = ca_df.sort_values(["Level", "Name"]).reset_index(drop=True)
            ca_df.to_csv(OUT_CAREER, index=False)
            print(f"\nWrote {len(ca_df):,} rows -> {OUT_CAREER}", flush=True)
            for level in SPORT_IDS:
                n = (ca_df["Level"] == level).sum()
                print(f"  {level}: {n:,} players")
        else:
            print("No careerAdvanced rows fetched.", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
