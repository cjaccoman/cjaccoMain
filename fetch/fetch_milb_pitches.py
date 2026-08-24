"""Fetch MiLB pitch-level data from game feeds to compute Chase%, Z-Contact%, PullAir%.

Data source: MLB Stats API game feeds (/api/v1.1/game/{gamePk}/feed/live)
Coverage: all affiliated MiLB levels (AAA through R), 2006-present

Metric definitions:
  Chase%     = swings at out-of-zone pitches (zones 11-14) / total out-of-zone pitches
  Z-Contact% = contact (non-miss) on in-zone swings (zones 1-9) / total in-zone swings
  PullAir%   = pulled fly balls + line drives / total batted balls in play

Outputs (data/api/):
  milb_pitches_agg.csv    -- final metrics, one row per player-season-level
  milb_pitches_games.csv  -- intermediate: one row per gamePk-batter (raw counts, cache)
  milb_games_done.csv     -- completed gamePks (skip on subsequent runs)

Run modes:
  First run (no milb_pitches_agg.csv): full history 2006-present (~9 hrs all levels)
  Incremental (output exists): only new current-season games (~seconds to minutes)
"""

import argparse
import json
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

BASE_URL    = "https://statsapi.mlb.com/api/v1"
BASE_URL_11 = "https://statsapi.mlb.com/api/v1.1"

SPORT_IDS       = {"AAA": 11, "AA": 12, "A+": 13, "A": 14, "R": 16}
LEVEL_FOR_SPORT = {v: k for k, v in SPORT_IDS.items()}

SEASONS        = list(range(2006, 2027))
CURRENT_SEASON = SEASONS[-1]
CALL_DELAY     = 0.3   # seconds between API calls
SAVE_INTERVAL  = 100   # flush to disk every N games (crash safety)

AGG_OUT    = API_DIR / "milb_pitches_agg.csv"
GAMES_OUT  = API_DIR / "milb_pitches_games.csv"
CACHE_OUT  = API_DIR / "milb_games_done.csv"
CHAD_CACHE = API_DIR / "chadwick.csv"

# Pitch zone codes
IN_ZONE  = frozenset(range(1, 10))    # zones 1-9: in strike zone
OUT_ZONE = frozenset(range(11, 15))   # zones 11-14: out of zone / chase zone

# Pitch outcome codes
SWING_CODES   = frozenset({"S", "F", "X", "T", "O", "L"})  # all batter swings
CONTACT_CODES = frozenset({"F", "X", "L"})                  # swing + ball contacted
WHIFF_CODES   = frozenset({"S", "T", "O"})                  # swinging strikes (miss)

# Batted ball types that count as "air balls" for PullAir%
AIR_TRAJ = frozenset({"fly_ball", "line_drive"})

# Pull direction threshold (TV-view x coordinate).
# coordX < PULL_MID = left field (RHB pull side)
# coordX > PULL_MID = right field (LHB pull side)
PULL_MID = 126.0


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(url: str, retries: int = 3) -> dict | None:
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "prospectsMain/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except Exception as e:
            if attempt == retries:
                print(f"      WARN: failed {url[:80]}... : {e}")
                return None
            time.sleep(2 ** attempt)
    return None


def _api(path: str, **params) -> dict | None:
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{BASE_URL}/{path}?{qs}" if qs else f"{BASE_URL}/{path}"
    time.sleep(CALL_DELAY)
    return _get(url)


def _game_feed(game_pk: int) -> dict | None:
    url = f"{BASE_URL_11}/game/{game_pk}/feed/live"
    time.sleep(CALL_DELAY)
    return _get(url)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def load_cache() -> set[int]:
    if CACHE_OUT.exists():
        df = pd.read_csv(CACHE_OUT, dtype={"gamePk": int})
        return set(df["gamePk"].tolist())
    return set()


def save_cache(done_pks: set[int]) -> None:
    pd.DataFrame({"gamePk": sorted(done_pks)}).to_csv(CACHE_OUT, index=False)


def flush_game_rows(rows: list[dict]) -> None:
    """Append raw game-batter rows to milb_pitches_games.csv."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    write_header = not GAMES_OUT.exists() or GAMES_OUT.stat().st_size == 0
    df.to_csv(GAMES_OUT, mode="a", header=write_header, index=False)


# ---------------------------------------------------------------------------
# Chadwick crosswalk (read-only; run fetch_milb_data.py to populate)
# ---------------------------------------------------------------------------

def load_chadwick() -> pd.DataFrame:
    if CHAD_CACHE.exists():
        return pd.read_csv(
            CHAD_CACHE,
            usecols=["key_mlbam", "key_fangraphs"],
            dtype={"key_mlbam": "Int64", "key_fangraphs": "Int64"},
        ).dropna(subset=["key_mlbam"])
    print("  WARN: chadwick.csv not found — run fetch_milb_data.py first. PlayerId will use MLBAM IDs.")
    return pd.DataFrame(columns=["key_mlbam", "key_fangraphs"])


def apply_crosswalk(df: pd.DataFrame, chadwick: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MLBAM_ID"] = pd.to_numeric(df["MLBAM_ID"], errors="coerce").astype("Int64")
    merged = df.merge(chadwick, left_on="MLBAM_ID", right_on="key_mlbam", how="left")
    fg    = merged["key_fangraphs"]
    mlbam = merged["MLBAM_ID"]
    merged["PlayerId"] = fg.where(fg.notna(), mlbam).apply(
        lambda x: str(int(x)) if pd.notna(x) else None
    )
    return merged.drop(columns=["key_mlbam", "key_fangraphs"], errors="ignore")


# ---------------------------------------------------------------------------
# Schedule fetcher
# ---------------------------------------------------------------------------

def fetch_schedule(sport_id: int, season: int) -> list[int]:
    """Return gamePks for all completed regular-season games at this level/season."""
    data = _api("schedule", sportId=sport_id, season=season, gameType="R")
    if not data:
        return []
    pks = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            state = (game.get("status") or {}).get("abstractGameState", "")
            if state == "Final":
                pk = game.get("gamePk")
                if pk:
                    pks.append(pk)
    return pks


# ---------------------------------------------------------------------------
# Game feed parser
# ---------------------------------------------------------------------------

def _zone(v) -> int:
    try:
        return int(v) if v is not None else 0
    except (ValueError, TypeError):
        return 0


def parse_game_feed(data: dict, game_pk: int, season: int, level: str) -> list[dict]:
    """Parse all pitch events in a game feed into per-batter count rows."""
    live  = data.get("liveData") or {}
    plays = (live.get("plays") or {}).get("allPlays", [])

    counts: dict[int, dict] = {}
    names:  dict[int, str]  = {}
    sides:  dict[int, str]  = {}

    for play in plays:
        matchup   = play.get("matchup") or {}
        batter    = matchup.get("batter") or {}
        batter_id = batter.get("id")
        if not batter_id:
            continue

        names[batter_id] = batter.get("fullName", "")
        sides[batter_id] = (matchup.get("batSide") or {}).get("code", "")

        if batter_id not in counts:
            counts[batter_id] = dict(
                OutsidePitches=0, OutsideSwings=0,
                InZonePitches=0, InZoneSwings=0, InZoneContacts=0,
                BattedBalls=0, AirBalls=0, PullAirBalls=0,
                TotalPitches=0, SwStr=0,
            )
        agg      = counts[batter_id]
        bat_side = sides[batter_id]

        for event in play.get("playEvents", []):
            if not event.get("isPitch", False):
                continue

            zone      = _zone((event.get("pitchData") or {}).get("zone"))
            call_code = ((event.get("details") or {}).get("call") or {}).get("code", "")
            is_swing  = call_code in SWING_CODES

            agg["TotalPitches"] += 1
            if call_code in WHIFF_CODES:
                agg["SwStr"] += 1

            if zone in IN_ZONE:
                agg["InZonePitches"] += 1
                if is_swing:
                    agg["InZoneSwings"] += 1
                    if call_code in CONTACT_CODES:
                        agg["InZoneContacts"] += 1
            elif zone in OUT_ZONE:
                agg["OutsidePitches"] += 1
                if is_swing:
                    agg["OutsideSwings"] += 1

            # Batted ball (in-play only)
            if call_code == "X":
                hit_data   = event.get("hitData") or {}
                trajectory = hit_data.get("trajectory", "")
                agg["BattedBalls"] += 1
                if trajectory in AIR_TRAJ:
                    agg["AirBalls"] += 1
                    coords = hit_data.get("coordinates") or {}
                    cx = coords.get("coordX")
                    if cx is not None:
                        is_pull = (bat_side == "R" and cx < PULL_MID) or \
                                  (bat_side == "L" and cx > PULL_MID)
                        if is_pull:
                            agg["PullAirBalls"] += 1

    rows = []
    for batter_id, agg in counts.items():
        rows.append({
            "gamePk": game_pk,
            "Season": season,
            "Level":  level,
            "MLBAM_ID": batter_id,
            "Name":   names[batter_id],
            "BatSide": sides.get(batter_id, ""),
            **agg,
        })
    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

COUNT_COLS = [
    "OutsidePitches", "OutsideSwings",
    "InZonePitches",  "InZoneSwings", "InZoneContacts",
    "BattedBalls",    "AirBalls",     "PullAirBalls",
    "TotalPitches",   "SwStr",
]


def build_agg(games_df: pd.DataFrame, chadwick: pd.DataFrame) -> pd.DataFrame:
    """Group per-game-batter rows into per-player-season-level metric rows."""
    if games_df.empty:
        return pd.DataFrame()

    # Back-fill columns added after the initial cache build
    if "TotalPitches" not in games_df.columns:
        games_df = games_df.copy()
        games_df["TotalPitches"] = games_df.get("OutsidePitches", 0) + games_df.get("InZonePitches", 0)
    if "SwStr" not in games_df.columns:
        games_df = games_df.copy()
        games_df["SwStr"] = 0  # unknown from old cache; will fill as new games are fetched

    # Deduplicate: one row per (gamePk, MLBAM_ID) prevents double-counting on re-runs
    games_df = games_df.drop_duplicates(subset=["gamePk", "MLBAM_ID"])

    grp  = ["MLBAM_ID", "Season", "Level"]
    agg  = games_df.groupby(grp, as_index=False)[COUNT_COLS].sum()
    gcnt = games_df.groupby(grp, as_index=False)["gamePk"].nunique().rename(
        columns={"gamePk": "Games"}
    )
    # Last name seen (arbitrary; name join via milb_hitting.csv is authoritative)
    nm   = games_df.groupby(grp)["Name"].last().reset_index()

    agg = agg.merge(gcnt, on=grp).merge(nm, on=grp)

    # Compute metrics (null when denominator is zero)
    agg["Chase%"] = (
        agg["OutsideSwings"] / agg["OutsidePitches"]
    ).where(agg["OutsidePitches"] > 0).round(3)

    agg["Z-Contact%"] = (
        agg["InZoneContacts"] / agg["InZoneSwings"]
    ).where(agg["InZoneSwings"] > 0).round(3)

    agg["PullAir%"] = (
        agg["PullAirBalls"] / agg["BattedBalls"]
    ).where(agg["BattedBalls"] > 0).round(3)

    total_swings = agg["InZoneSwings"] + agg["OutsideSwings"]
    agg["Whiff%"] = (
        agg["SwStr"] / total_swings
    ).where(total_swings > 0).round(3)

    agg = apply_crosswalk(agg, chadwick)

    col_order = [
        "PlayerId", "MLBAM_ID", "Season", "Level", "Name",
        "Chase%", "Z-Contact%", "PullAir%", "Whiff%",
        "OutsidePitches", "OutsideSwings",
        "InZonePitches",  "InZoneSwings", "InZoneContacts",
        "BattedBalls",    "AirBalls",     "PullAirBalls",
        "TotalPitches",   "SwStr",
        "Games",
    ]
    return agg[[c for c in col_order if c in agg.columns]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch MiLB pitch metrics from game feeds")
    parser.add_argument(
        "--season", type=int, default=None,
        help="Fetch a specific season only (e.g. --season 2026). "
             "Default: current season if output exists, full history if not.",
    )
    args = parser.parse_args()

    print("=== fetch_milb_pitches.py ===\n")

    chadwick  = load_chadwick()
    done_pks  = load_cache()
    print(f"Cache: {len(done_pks):,} games already processed")

    if args.season is not None:
        fetch_seasons = [args.season] if args.season != 2020 else []
        print(f"Single-season mode — fetching {args.season} only\n")
    elif AGG_OUT.exists():
        fetch_seasons = [CURRENT_SEASON]
        print(f"Incremental mode — fetching new {CURRENT_SEASON} games only\n")
    else:
        fetch_seasons = [s for s in SEASONS if s != 2020]  # 2020 MiLB cancelled
        print(f"Full history mode — {SEASONS[0]}-{SEASONS[-1]} (2020 skipped)\n")

    pending_rows: list[dict] = []
    total_new_games = 0

    for season in fetch_seasons:
        for level, sport_id in SPORT_IDS.items():
            schedule = fetch_schedule(sport_id, season)
            pending  = [pk for pk in schedule if pk not in done_pks]

            print(f"  {season} {level:3}: {len(schedule):4} completed games, "
                  f"{len(pending):4} new")

            if not pending:
                continue

            for i, game_pk in enumerate(pending, 1):
                feed = _game_feed(game_pk)
                if feed:
                    rows = parse_game_feed(feed, game_pk, season, level)
                    pending_rows.extend(rows)
                done_pks.add(game_pk)
                total_new_games += 1

                if i % SAVE_INTERVAL == 0:
                    flush_game_rows(pending_rows)
                    pending_rows = []
                    save_cache(done_pks)
                    print(f"      [{i}/{len(pending)}] flushed to disk")

            # Flush remaining rows after each level
            flush_game_rows(pending_rows)
            pending_rows = []
            save_cache(done_pks)

    print(f"\nProcessed {total_new_games:,} new games")

    # Rebuild aggregation from full games file
    print("\nRebuilding milb_pitches_agg.csv...")
    if GAMES_OUT.exists():
        games_df = pd.read_csv(
            GAMES_OUT,
            dtype={"MLBAM_ID": "Int64", "gamePk": int},
            on_bad_lines="skip",
        )
        agg = build_agg(games_df, chadwick)
        agg.to_csv(AGG_OUT, index=False)
        print(f"  Wrote {len(agg):,} player-season-level rows -> {AGG_OUT.name}")

        # Summary
        seasons_covered = sorted(games_df["Season"].unique())
        print(f"  Seasons: {seasons_covered[0]}-{seasons_covered[-1]}")
        print(f"  Levels: {sorted(games_df['Level'].unique())}")
        print(f"  Unique players: {games_df['MLBAM_ID'].nunique():,}")
    else:
        print("  No game data found — nothing to aggregate.")

    print("\nDone.")


if __name__ == "__main__":
    main()

