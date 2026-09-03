"""Fetch MLB batting stats from the MLB Stats API (2006–current).

Replaces the FanGraphs manual export workflow for hist_mlb_data.csv.

Output:
  data/historical/hist_mlb_data.csv  — schema-compatible with the existing file:
    Season, Name, Team, G, PA, 1B, 2B, 3B, HR, R, RBI, BB, IBB, SO, GDP,
    SB, CS, TB, NameASCII, PlayerId, MLBAMID, PPG, PPPA, PPPA_Z

Usage:
  python fetch/fetch_mlb_data.py           # incremental — refreshes current season only
  python fetch/fetch_mlb_data.py --full    # full refetch 2006–current
"""

import argparse
import json
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR   = Path(__file__).resolve().parent.parent / "data"
API_DIR    = DATA_DIR / "api"
HIST_DIR   = DATA_DIR / "historical"
API_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL        = "https://statsapi.mlb.com/api/v1"
CHADWICK_CACHE  = API_DIR / "chadwick.csv"
OUT_PATH        = HIST_DIR / "hist_mlb_data.csv"

MLB_SPORT_ID    = 1
SEASONS         = list(range(2006, 2027))
CURRENT_SEASON  = SEASONS[-1]
MIN_PA          = 1    # include everyone; downstream filters by PA >= 50
CALL_DELAY      = 0.3  # seconds between API calls

# Fantasy scoring formula weights
SCORING = dict(
    w1B=1, w2B=2, w3B=3, wHR=4,
    wR=1, wRBI=2, wBB=1, wIBB=0.5,   # IBB = +1.5 total = BB share +1 + 0.5 extra
    wSO=-2, wGDP=-1.5, wSB=3, wCS=-1.5, wTB=1,
)

STATS_LIMIT = 5000


# ---------------------------------------------------------------------------
# HTTP helpers  (identical to fetch_milb_data.py)
# ---------------------------------------------------------------------------

def _get(url: str, retries: int = 2) -> dict | None:
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "prospectsMain/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except Exception as e:
            if attempt == retries:
                print(f"    WARN: failed {url[:80]}... : {e}")
                return None
            time.sleep(1 + attempt)
    return None


def _api(path: str, **params) -> dict | None:
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{BASE_URL}/{path}?{qs}" if qs else f"{BASE_URL}/{path}"
    time.sleep(CALL_DELAY)
    return _get(url)


# ---------------------------------------------------------------------------
# Chadwick crosswalk  (shared cache with fetch_milb_data.py)
# ---------------------------------------------------------------------------

def load_chadwick() -> pd.DataFrame:
    if not CHADWICK_CACHE.exists():
        print("  WARNING: chadwick.csv not found — run fetch_milb_data.py first.")
        print("  PlayerId will fall back to MLBAM ID for all players.")
        return pd.DataFrame(columns=["key_mlbam", "key_fangraphs"])

    ck = pd.read_csv(
        CHADWICK_CACHE,
        usecols=["key_mlbam", "key_fangraphs"],
        dtype={"key_mlbam": "Int64", "key_fangraphs": "Int64"},
    ).dropna(subset=["key_mlbam"])
    print(f"  Chadwick (cached): {len(ck):,} players, "
          f"{ck['key_fangraphs'].notna().sum():,} with FanGraphs IDs")
    return ck


def apply_crosswalk(df: pd.DataFrame, chadwick: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MLBAMID"] = pd.to_numeric(df["MLBAMID"], errors="coerce").astype("Int64")
    merged = df.merge(chadwick, left_on="MLBAMID", right_on="key_mlbam", how="left")
    fg    = merged["key_fangraphs"]
    mlbam = merged["MLBAMID"]
    merged["PlayerId"] = fg.where(fg.notna(), mlbam).apply(
        lambda x: str(int(x)) if pd.notna(x) else None
    )
    return merged.drop(columns=["key_mlbam", "key_fangraphs"], errors="ignore")


# ---------------------------------------------------------------------------
# Team lookup for MLB (sportId=1)
# ---------------------------------------------------------------------------

def build_team_lookup(season: int) -> dict[int, str]:
    """Returns {team_id: abbreviation}."""
    data = _api("teams", sportId=MLB_SPORT_ID, season=season)
    if not data:
        return {}
    return {t["id"]: t.get("abbreviation", "UNK") for t in data.get("teams", [])}


# ---------------------------------------------------------------------------
# Fetch + parse
# ---------------------------------------------------------------------------

def _int(v) -> int | None:
    try:
        return int(v) if v is not None else None
    except (ValueError, TypeError):
        return None


def fetch_season(season: int) -> list[dict]:
    """Fetch season hitting stats for all MLB players in one season."""
    data = _api(
        "stats",
        stats="season",
        playerPool="all",
        group="hitting",
        gameType="R",
        season=season,
        sportId=MLB_SPORT_ID,
        limit=STATS_LIMIT,
    )
    if not data:
        return []

    splits = []
    for block in data.get("stats", []):
        block_splits = block.get("splits", [])
        total = block.get("totalSplits", len(block_splits))
        if len(block_splits) < total:
            print(f"    WARN: got {len(block_splits)}/{total} splits — "
                  f"consider increasing STATS_LIMIT")
        splits.extend(block_splits)
    return splits


def parse_splits(splits: list, team_lkp: dict, season: int) -> list[dict]:
    rows = []
    for sp in splits:
        stat   = sp.get("stat", {})
        player = sp.get("player", {})
        team   = sp.get("team", {})

        hits    = _int(stat.get("hits")) or 0
        doubles = _int(stat.get("doubles")) or 0
        triples = _int(stat.get("triples")) or 0
        hr      = _int(stat.get("homeRuns")) or 0
        singles = hits - doubles - triples - hr

        bb  = _int(stat.get("baseOnBalls")) or 0
        ibb = _int(stat.get("intentionalWalks")) or 0
        so  = _int(stat.get("strikeOuts")) or 0
        gdp = _int(stat.get("groundIntoDoublePlay")) or 0
        sb  = _int(stat.get("stolenBases")) or 0
        cs  = _int(stat.get("caughtStealing")) or 0
        r   = _int(stat.get("runs")) or 0
        rbi = _int(stat.get("rbi")) or 0
        tb  = _int(stat.get("totalBases")) or 0
        pa  = _int(stat.get("plateAppearances")) or 0
        g   = _int(stat.get("gamesPlayed")) or 0

        rows.append({
            "MLBAMID": player.get("id"),
            "Season":  season,
            "Name":    player.get("fullName", ""),
            "Team":    team_lkp.get(team.get("id"), team.get("abbreviation", "UNK")),
            "G":   g,
            "PA":  pa,
            "1B":  singles,
            "2B":  doubles,
            "3B":  triples,
            "HR":  hr,
            "R":   r,
            "RBI": rbi,
            "BB":  bb,
            "IBB": ibb,
            "SO":  so,
            "GDP": gdp,
            "SB":  sb,
            "CS":  cs,
            "TB":  tb,
        })
    return rows


# ---------------------------------------------------------------------------
# PPPA computation
# ---------------------------------------------------------------------------

def _norm_name(name: str) -> str:
    return unicodedata.normalize("NFD", name).encode("ascii", "ignore").decode()


def compute_pppa(df: pd.DataFrame) -> pd.DataFrame:
    """Add TB (verify/recompute), TP, PPG, PPPA, PPPA_Z, NameASCII."""
    df = df.copy()

    # TB from API is authoritative; recompute as sanity check fallback
    tb_computed = df["1B"] + 2 * df["2B"] + 3 * df["3B"] + 4 * df["HR"]
    df["TB"] = df["TB"].where(df["TB"] > 0, tb_computed)

    # Fantasy points: TP = 1B + 2*2B + 3*3B + 4*HR + R + 2*RBI + BB + 1.5*IBB
    #                      + TB - 2*SO - 1.5*GDP + 3*SB - 1.5*CS
    df["TP"] = (
        df["1B"] + 2 * df["2B"] + 3 * df["3B"] + 4 * df["HR"]
        + df["R"] + 2 * df["RBI"]
        + df["BB"] + 1.5 * df["IBB"]
        + df["TB"]
        - 2 * df["SO"] - 1.5 * df["GDP"]
        + 3 * df["SB"] - 1.5 * df["CS"]
    )
    df["PPG"]  = (df["TP"] / df["G"]).round(4).where(df["G"] > 0)
    df["PPPA"] = (df["TP"] / df["PA"]).round(4).where(df["PA"] > 0)

    # PPPA_Z: within-season z-score (PA >= 50 to set baseline, applied to all)
    season_stats = (
        df[df["PA"] >= 50]
        .groupby("Season")["PPPA"]
        .agg(mu="mean", sigma="std")
        .reset_index()
    )
    df = df.merge(season_stats, on="Season", how="left")
    df["PPPA_Z"] = ((df["PPPA"] - df["mu"]) / df["sigma"]).round(4)
    df = df.drop(columns=["mu", "sigma", "TP"])

    df["NameASCII"] = df["Name"].apply(_norm_name)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Refetch all seasons 2006-current instead of incremental")
    args = parser.parse_args()

    print("=== fetch_mlb_data.py ===\n")

    print("Step 1: Chadwick register")
    chadwick = load_chadwick()

    # Determine fetch scope
    full_mode = args.full or not OUT_PATH.exists()
    if full_mode:
        print(f"\nFull mode — fetching seasons {SEASONS[0]}–{CURRENT_SEASON}")
        # Preserve pre-2006 rows from existing file (legacy data, not re-fetched)
        if OUT_PATH.exists():
            legacy = pd.read_csv(OUT_PATH, dtype={"PlayerId": str, "MLBAMID": str})
            legacy = legacy[legacy["Season"] < SEASONS[0]]
            print(f"  Preserving {len(legacy):,} pre-{SEASONS[0]} rows from existing file")
        else:
            legacy = pd.DataFrame()
        existing = legacy
        fetch_seasons = SEASONS
    else:
        print(f"\nIncremental mode — refreshing {CURRENT_SEASON} only")
        existing = pd.read_csv(OUT_PATH, dtype={"PlayerId": str, "MLBAMID": str})
        existing = existing[existing["Season"] != CURRENT_SEASON]
        fetch_seasons = [CURRENT_SEASON]

    all_rows: list[dict] = []
    total = len(fetch_seasons)

    print(f"\nStep 2: Fetching {total} season(s)\n")
    for i, season in enumerate(fetch_seasons, 1):
        print(f"[{i:3}/{total}] {season}", end="  ")
        team_lkp = build_team_lookup(season)
        splits   = fetch_season(season)
        if splits:
            rows = parse_splits(splits, team_lkp, season)
            rows = [r for r in rows if (r.get("PA") or 0) >= MIN_PA]
            all_rows.extend(rows)
            print(f"{len(rows):4} players")
        else:
            print("no data")

    print(f"\nFetched {len(all_rows):,} new rows")

    if not all_rows:
        print("Nothing new to write.")
        return

    # Build new rows DataFrame
    new_df = pd.DataFrame(all_rows)
    new_df = apply_crosswalk(new_df, chadwick)

    # Combine with existing history
    combined = pd.concat([existing, new_df], ignore_index=True)

    # Compute PPPA across full dataset for consistent PPPA_Z baselines
    print("\nComputing PPPA and PPPA_Z...")
    combined = compute_pppa(combined)

    # Final column order matching existing hist_mlb_data.csv schema
    cols = [
        "Season", "Name", "Team", "G", "PA",
        "1B", "2B", "3B", "HR", "R", "RBI", "BB", "IBB", "SO", "GDP",
        "SB", "CS", "TB", "NameASCII", "PlayerId", "MLBAMID",
        "PPG", "PPPA", "PPPA_Z",
    ]
    # Keep any extra columns from existing file that aren't in cols
    extra = [c for c in combined.columns if c not in cols]
    combined = combined[[c for c in cols if c in combined.columns] + extra]

    combined.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(combined):,} rows -> {OUT_PATH}")

    # Summary
    seasons_present = sorted(combined["Season"].unique())
    print(f"\nSeasons covered: {seasons_present[0]}–{seasons_present[-1]} "
          f"({len(seasons_present)} seasons)")
    curr_rows = combined[combined["Season"] == CURRENT_SEASON]
    print(f"{CURRENT_SEASON}: {len(curr_rows):,} players, "
          f"{(curr_rows['PA'] >= 50).sum():,} with PA >= 50, "
          f"{(curr_rows['PA'] >= 50).sum():,} eligible for graduation filter")
    print("\nDone.")


if __name__ == "__main__":
    main()
