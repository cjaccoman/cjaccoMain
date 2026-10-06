"""Fetch MiLB pitch-level data from game feeds to compute Chase%, Z-Contact%, PullAir%,
velocity-bucketed metrics (P95+), and exit velocity (MaxEV).

Data source: MLB Stats API game feeds (/api/v1.1/game/{gamePk}/feed/live)
Coverage: all affiliated MiLB levels (AAA through R), 2006-present

Metric definitions:
  Chase%      = swings at out-of-zone pitches (zones 11-14) / total out-of-zone pitches
  Z-Contact%  = contact (non-miss) on in-zone swings (zones 1-9) / total in-zone swings
  PullAir%    = pulled fly balls + line drives / total batted balls in play
  P95_Whiff%  = whiffs on 95+ mph pitches / swings on 95+ mph pitches
  P95_Chase%  = swings on 95+ mph out-of-zone pitches / total 95+ mph out-of-zone pitches
  P95_pct     = 95+ mph pitches seen / total pitches seen
  MaxEV       = max hitData.launchSpeed across all in-play events in the season
                (requires Hawk-Eye; populated at AAA 2023+ and A-ball FSL parks 2021+)

Velocity coverage: pitchData.startSpeed requires Trackman/Hawk-Eye at the park.
  AAA 2023+: near-complete (~100%). Earlier seasons and lower levels: partial or absent.
  Rows with P95_Pitches=0 have null P95_Whiff% and P95_Chase%.

Exit velocity coverage: hitData.launchSpeed requires Hawk-Eye.
  AAA 2023+: ~100%. A-ball FSL parks (Clearwater/Palm Beach/St. Lucie/Bradenton/Lakeland)
  2021+: ~100%. All other levels/parks: null.

Outputs (data/api/):
  milb_pitches_agg.csv    -- final metrics, one row per player-season-level
  milb_pitches_games.csv  -- intermediate: one row per gamePk-batter (raw counts, cache)
  milb_games_done.csv     -- completed gamePks (skip on subsequent runs)

Run modes:
  First run (no milb_pitches_agg.csv): full history 2006-present (~9 hrs all levels)
  Incremental (output exists): only new current-season games (~seconds to minutes)
  --rebuild-velocity YEAR: re-process a season to add P95 velocity columns
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

def staging_path(level: str, year: int) -> Path:
    """Per-level staging CSV used during parallel --rebuild-velocity runs."""
    return API_DIR / f"milb_pitches_games_{level}_{year}.csv"

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
                # Velocity bucket (95+ mph)
                P95_Pitches=0, P95_Swings=0, P95_Whiff=0,
                P95_OutsidePitches=0, P95_OutsideSwings=0,
                # Exit velocity (Hawk-Eye parks only: AAA 2023+, A-ball FSL 2021+)
                EV_Max=None, EV_Count=0,
            )
        agg      = counts[batter_id]
        bat_side = sides[batter_id]

        for event in play.get("playEvents", []):
            if not event.get("isPitch", False):
                continue

            pitch_data = event.get("pitchData") or {}
            zone       = _zone(pitch_data.get("zone"))
            call_code  = ((event.get("details") or {}).get("call") or {}).get("code", "")
            is_swing   = call_code in SWING_CODES

            # Velocity bucket
            start_speed = pitch_data.get("startSpeed")
            is_p95 = start_speed is not None and start_speed >= 95.0

            agg["TotalPitches"] += 1
            if call_code in WHIFF_CODES:
                agg["SwStr"] += 1

            if is_p95:
                agg["P95_Pitches"] += 1
                if is_swing:
                    agg["P95_Swings"] += 1
                    if call_code in WHIFF_CODES:
                        agg["P95_Whiff"] += 1
                if zone in OUT_ZONE:
                    agg["P95_OutsidePitches"] += 1
                    if is_swing:
                        agg["P95_OutsideSwings"] += 1

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
                # Exit velocity (requires Hawk-Eye; null where not tracked)
                ls = hit_data.get("launchSpeed")
                if ls is not None:
                    try:
                        ls_f = float(ls)
                        if ls_f > 0:
                            agg["EV_Count"] += 1
                            if agg["EV_Max"] is None or ls_f > agg["EV_Max"]:
                                agg["EV_Max"] = ls_f
                    except (TypeError, ValueError):
                        pass

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
    "P95_Pitches",    "P95_Swings",   "P95_Whiff",
    "P95_OutsidePitches", "P95_OutsideSwings",
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
    # Back-fill P95 columns (0 = no velocity data; P95_Pitches=0 → null computed metrics)
    for col in ["P95_Pitches", "P95_Swings", "P95_Whiff", "P95_OutsidePitches", "P95_OutsideSwings"]:
        if col not in games_df.columns:
            games_df = games_df.copy()
            games_df[col] = 0
    # Back-fill EV columns (added after initial cache build; EV_Max=NaN means no Hawk-Eye data)
    if "EV_Max" not in games_df.columns:
        games_df = games_df.copy()
        games_df["EV_Max"] = None
    if "EV_Count" not in games_df.columns:
        games_df = games_df.copy()
        games_df["EV_Count"] = 0

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

    # EV aggregation: MaxEV = max of per-game maxes; EV_Count = sum
    # These can't go into COUNT_COLS since EV_Max is not summable.
    ev_max = games_df.groupby(grp, as_index=False)["EV_Max"].max()
    ev_cnt = games_df.groupby(grp, as_index=False)["EV_Count"].sum()
    agg = agg.merge(ev_max, on=grp, how="left").merge(ev_cnt, on=grp, how="left")
    # Null out MaxEV where no EV data was captured (EV_Count=0 means no Hawk-Eye present)
    no_ev = agg["EV_Count"].fillna(0) == 0
    agg.loc[no_ev, "EV_Max"] = None

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

    # P95+ velocity metrics (null when no velocity data available)
    agg["P95_Whiff%"] = (
        agg["P95_Whiff"] / agg["P95_Swings"]
    ).where(agg["P95_Swings"] > 0).round(3)

    agg["P95_Chase%"] = (
        agg["P95_OutsideSwings"] / agg["P95_OutsidePitches"]
    ).where(agg["P95_OutsidePitches"] > 0).round(3)

    agg["P95_pct"] = (
        agg["P95_Pitches"] / agg["TotalPitches"]
    ).where(agg["TotalPitches"] > 0).round(3)
    # Zero P95_Pitches means no velocity data — null out the rate
    agg.loc[agg["P95_Pitches"] == 0, ["P95_Whiff%", "P95_Chase%", "P95_pct"]] = None

    agg = apply_crosswalk(agg, chadwick)

    col_order = [
        "PlayerId", "MLBAM_ID", "Season", "Level", "Name",
        "Chase%", "Z-Contact%", "PullAir%", "Whiff%",
        "P95_Whiff%", "P95_Chase%", "P95_pct",
        "MaxEV", "EV_Count",
        "OutsidePitches", "OutsideSwings",
        "InZonePitches",  "InZoneSwings", "InZoneContacts",
        "BattedBalls",    "AirBalls",     "PullAirBalls",
        "TotalPitches",   "SwStr",
        "P95_Pitches",    "P95_Swings",   "P95_Whiff",
        "P95_OutsidePitches", "P95_OutsideSwings",
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
    parser.add_argument(
        "--rebuild-velocity", type=int, default=None, metavar="YEAR",
        help="Re-process a season to add P95 velocity columns (e.g. --rebuild-velocity 2024). "
             "Without --level: removes all levels for that season and re-fetches into main CSV. "
             "With --level: writes only that level to a staging file (parallel-safe).",
    )
    parser.add_argument(
        "--level", default=None, choices=list(SPORT_IDS.keys()),
        help="Process only this level. Use with --rebuild-velocity for parallel runs. "
             "Writes to milb_pitches_games_{LEVEL}_{YEAR}.csv (staging). "
             "Run --merge-year YEAR after all levels finish.",
    )
    parser.add_argument(
        "--merge-year", type=int, default=None, metavar="YEAR",
        help="Merge per-level staging files for YEAR into the main games CSV and rebuild agg. "
             "Run this after all --rebuild-velocity YEAR --level X jobs finish.",
    )
    args = parser.parse_args()

    print("=== fetch_milb_pitches.py ===\n")

    chadwick  = load_chadwick()

    # ------------------------------------------------------------------
    # --merge-year: combine staging files into main CSV, rebuild agg
    # ------------------------------------------------------------------
    if args.merge_year is not None:
        my = args.merge_year
        print(f"Merge mode — combining {my} staging files into main games CSV\n")
        parts: list[pd.DataFrame] = []
        for level in SPORT_IDS:
            sp = staging_path(level, my)
            if sp.exists():
                df = pd.read_csv(sp, dtype={"MLBAM_ID": "Int64", "gamePk": int}, on_bad_lines="skip")
                print(f"  {level}: {len(df):,} rows from staging file")
                parts.append(df)
                sp.unlink()
            else:
                print(f"  {level}: no staging file found (skipped)")
        if not parts:
            print("No staging files found. Nothing to merge.")
            return

        # Load main games CSV (excluding this year's rows to avoid duplication)
        if GAMES_OUT.exists():
            main_df = pd.read_csv(GAMES_OUT, dtype={"MLBAM_ID": "Int64", "gamePk": int}, on_bad_lines="skip")
            main_df = main_df[main_df["Season"] != my]
            print(f"  Main CSV (after dropping {my}): {len(main_df):,} rows")
            parts.insert(0, main_df)

        combined = pd.concat(parts, ignore_index=True)
        # Ensure P95 columns exist
        for p95c in ["P95_Pitches", "P95_Swings", "P95_Whiff", "P95_OutsidePitches", "P95_OutsideSwings"]:
            if p95c not in combined.columns:
                combined[p95c] = 0
        combined.fillna({c: 0 for c in ["P95_Pitches", "P95_Swings", "P95_Whiff", "P95_OutsidePitches", "P95_OutsideSwings"]}, inplace=True)
        # Ensure EV columns exist (added after initial cache build)
        if "EV_Max" not in combined.columns:
            combined["EV_Max"] = None
        if "EV_Count" not in combined.columns:
            combined["EV_Count"] = 0
        combined.to_csv(GAMES_OUT, index=False)
        print(f"\n  Wrote {len(combined):,} rows -> {GAMES_OUT.name}")

        # Update main done tracker with all merged gamePks
        done_pks = load_cache()
        done_pks.update(combined["gamePk"].astype(int).tolist())
        save_cache(done_pks)
        print(f"  Done tracker updated: {len(done_pks):,} total gamePks")

        print("\nRebuilding milb_pitches_agg.csv...")
        agg = build_agg(combined, chadwick)
        agg.to_csv(AGG_OUT, index=False)
        print(f"  Wrote {len(agg):,} player-season-level rows -> {AGG_OUT.name}")
        seasons_covered = sorted(combined["Season"].unique())
        print(f"  Seasons: {seasons_covered[0]}-{seasons_covered[-1]}")
        print(f"  Levels: {sorted(combined['Level'].unique())}")
        print("\nDone.")
        return

    done_pks  = load_cache()
    print(f"Cache: {len(done_pks):,} games already processed")

    if args.rebuild_velocity is not None:
        rv_year = args.rebuild_velocity

        if args.level is not None:
            # ------------------------------------------------------------------
            # Per-level staging mode (parallel-safe): write to staging CSV only,
            # do NOT touch the main games CSV or main done tracker.
            # ------------------------------------------------------------------
            level     = args.level
            sport_id  = SPORT_IDS[level]
            out_path  = staging_path(level, rv_year)
            print(f"Rebuild-velocity (staged) — {rv_year} {level} -> {out_path.name}\n")

            schedule = fetch_schedule(sport_id, rv_year)
            print(f"  {rv_year} {level}: {len(schedule):4} games to fetch")

            pending_rows: list[dict] = []
            total_new_games = 0
            for i, game_pk in enumerate(schedule, 1):
                feed = _game_feed(game_pk)
                if feed:
                    rows = parse_game_feed(feed, game_pk, rv_year, level)
                    pending_rows.extend(rows)
                total_new_games += 1

                if i % SAVE_INTERVAL == 0:
                    chunk = pd.DataFrame(pending_rows)
                    if not out_path.exists():
                        chunk.to_csv(out_path, index=False)
                    else:
                        chunk.to_csv(out_path, mode="a", header=False, index=False)
                    pending_rows = []
                    print(f"      [{i}/{len(schedule)}] flushed to {out_path.name}")

            if pending_rows:
                chunk = pd.DataFrame(pending_rows)
                if not out_path.exists():
                    chunk.to_csv(out_path, index=False)
                else:
                    chunk.to_csv(out_path, mode="a", header=False, index=False)

            print(f"\nProcessed {total_new_games:,} games -> {out_path.name}")
            print("Run --merge-year after all level jobs finish to combine and rebuild agg.")
            print("\nDone.")
            return

        # ------------------------------------------------------------------
        # Original all-levels mode (sequential, modifies main CSV)
        # ------------------------------------------------------------------
        print(f"Rebuild-velocity mode — re-processing {rv_year} to add P95/EV columns\n")
        if GAMES_OUT.exists():
            games_df_existing = pd.read_csv(GAMES_OUT, dtype={"MLBAM_ID": "Int64", "gamePk": int}, on_bad_lines="skip")
            old_count = len(games_df_existing)
            games_df_existing = games_df_existing[games_df_existing["Season"] != rv_year]
            for p95c in ["P95_Pitches", "P95_Swings", "P95_Whiff", "P95_OutsidePitches", "P95_OutsideSwings"]:
                if p95c not in games_df_existing.columns:
                    games_df_existing[p95c] = 0
            if "EV_Max" not in games_df_existing.columns:
                games_df_existing["EV_Max"] = None
            if "EV_Count" not in games_df_existing.columns:
                games_df_existing["EV_Count"] = 0
            games_df_existing.to_csv(GAMES_OUT, index=False)
            removed = old_count - len(games_df_existing)
            print(f"  Dropped {removed:,} rows for {rv_year} from milb_pitches_games.csv")
        season_pks_to_remove: set[int] = set()
        for level, sport_id in SPORT_IDS.items():
            sched = fetch_schedule(sport_id, rv_year)
            season_pks_to_remove.update(sched)
        removed_from_cache = done_pks & season_pks_to_remove
        done_pks -= season_pks_to_remove
        save_cache(done_pks)
        print(f"  Removed {len(removed_from_cache):,} game PKs from done cache")
        fetch_seasons = [rv_year]

    elif args.season is not None:
        fetch_seasons = [args.season] if args.season != 2020 else []
        print(f"Single-season mode — fetching {args.season} only\n")
    elif AGG_OUT.exists():
        fetch_seasons = [CURRENT_SEASON]
        print(f"Incremental mode — fetching new {CURRENT_SEASON} games only\n")
    else:
        fetch_seasons = [s for s in SEASONS if s != 2020]  # 2020 MiLB cancelled
        print(f"Full history mode — {SEASONS[0]}-{SEASONS[-1]} (2020 skipped)\n")

    pending_rows_main: list[dict] = []
    total_new_games = 0
    levels_to_fetch = {args.level: SPORT_IDS[args.level]} if args.level else SPORT_IDS

    for season in fetch_seasons:
        for level, sport_id in levels_to_fetch.items():
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
                    pending_rows_main.extend(rows)
                done_pks.add(game_pk)
                total_new_games += 1

                if i % SAVE_INTERVAL == 0:
                    flush_game_rows(pending_rows_main)
                    pending_rows_main = []
                    save_cache(done_pks)
                    print(f"      [{i}/{len(pending)}] flushed to disk")

            flush_game_rows(pending_rows_main)
            pending_rows_main = []
            save_cache(done_pks)

    print(f"\nProcessed {total_new_games:,} new games")

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
        seasons_covered = sorted(games_df["Season"].unique())
        print(f"  Seasons: {seasons_covered[0]}-{seasons_covered[-1]}")
        print(f"  Levels: {sorted(games_df['Level'].unique())}")
        print(f"  Unique players: {games_df['MLBAM_ID'].nunique():,}")
    else:
        print("  No game data found — nothing to aggregate.")

    print("\nDone.")


if __name__ == "__main__":
    main()

