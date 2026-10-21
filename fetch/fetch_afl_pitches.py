"""Fetch Arizona Fall League pitch-level Statcast data from game feeds.

All six AFL stadiums in Phoenix have full Hawk-Eye coverage (confirmed 2023+),
giving 100% pitch velocity/movement/spin and ~14% exit velocity (BIP only).

Metrics computed per batter (season aggregate):

  Swing/contact quality:
    Chase%         swings at out-of-zone pitches / total out-of-zone pitches
    Z-Contact%     contact on in-zone swings / total in-zone swings
    Whiff%         swinging strikes / total swings
    PullAir%       pulled fly balls + line drives / all batted balls
    P95_Whiff%     whiffs on 95+ mph pitches / swings on 95+ mph pitches
    P95_Chase%     chases on 95+ mph pitches / 95+ out-of-zone pitches
    P95_pct        fraction of pitches seen that were 95+ mph

  Contact quality / power:
    MaxEV          season max launchSpeed (mph) on balls in play
    EV_MaxLA       launch angle on the MaxEV contact
    AvgEV          mean launchSpeed on balls in play
    HardHit%       BIP with launchSpeed >= 95 mph / total BIP

  Production — overall:
    PA             total plate appearances
    PPPA           fantasy points per PA (personal scoring formula)
                   TP = 2×1B + 4×2B + 6×3B + 8×HR + R + 2×RBI
                        + BB + 1.5×IBB − 2×SO − 1.5×GIDP + 3×SB − 1.5×CS
    P95_PA         PA where pitcher threw >= 1 pitch at 95+ mph
    P95_PPPA       PPPA on those P95 plate appearances (excl. R/SB/CS)
                   null when P95_PA < 10

  Production — by pitch category (decisive-pitch method):
    FA_PPPA / FA_PA    PPPA on PAs where the decisive pitch was a fastball
                       (FF, SI, FC — 4-seam, sinker, cutter)
    BR_PPPA / BR_PA    PPPA on PAs where the decisive pitch was a breaking ball
                       (SL, CU, SV, ST, KC, CS — slider/curve/sweeper/slurve)
    OS_PPPA / OS_PA    PPPA on PAs where the decisive pitch was offspeed
                       (CH, FS, FO, SC — changeup/splitter/forkball/screwball)

    Decisive pitch = the last pitch of the PA (the pitch that ended the AB —
    a contact pitch, a strikeout pitch, or ball 4 of a walk). This mirrors how
    MLB Network / Baseball Savant report "BA vs. changeup" etc. — production is
    credited to the pitch the at-bat turned on, not every pitch seen in the AB.
    Requires >= 5 PA in that bucket to report; null otherwise.

Per-pitch-type seen columns (for each of FF SI FC SL CU CH):
  {type}_pct       fraction of total pitches
  {type}_velo      average startSpeed
  {type}_spin      average spinRate
  {type}_ivb       average induced vertical break (breakVerticalInduced)
  {type}_hbreak    average horizontal break (breakHorizontal, pitcher-side sign)

Outputs:
  data/api/afl_pitches_agg.csv    one row per player × season
  data/api/afl_pitches_games.csv  per-game cache (crash safety / incremental)

Usage:
  python fetch/fetch_afl_pitches.py                 # current season, incremental
  python fetch/fetch_afl_pitches.py --season 2025   # specific past season
  python fetch/fetch_afl_pitches.py --full          # 2023-present (Hawk-Eye era)

Join key: MLBAM_ID matches afl_hitting.csv and afl_pitching.csv.
Pre-2023 AFL games have partial/absent velocity and no EV — not fetched by default.
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR    = Path(__file__).resolve().parent.parent / "data"
API_DIR     = DATA_DIR / "api"
API_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL    = "https://statsapi.mlb.com/api/v1"
BASE_URL_11 = "https://statsapi.mlb.com/api/v1.1"

AFL_SPORT_ID  = 17
AFL_LEAGUE_ID = 119
HAWK_EYE_FIRST_YEAR = 2023   # full EV/spin/movement coverage from this year
CURRENT_SEASON      = 2026
NO_SEASON           = {2020}  # AFL cancelled (COVID)

CALL_DELAY    = 0.3    # seconds between API calls
SAVE_INTERVAL = 50     # flush games CSV every N games

AGG_OUT   = API_DIR / "afl_pitches_agg.csv"
GAMES_OUT = API_DIR / "afl_pitches_games.csv"
CACHE_OUT = API_DIR / "afl_pitches_done.csv"

# Zone definitions (match fetch_milb_pitches.py conventions)
IN_ZONE  = frozenset(range(1, 10))   # zones 1-9: strike zone
OUT_ZONE = frozenset(range(11, 15))  # zones 11-14: chase zone

# Pitch outcome codes
SWING_CODES   = frozenset({"S", "F", "X", "T", "O", "L"})
CONTACT_CODES = frozenset({"F", "X", "L"})
WHIFF_CODES   = frozenset({"S", "T", "O"})

# Batted ball air trajectories for PullAir%
AIR_TRAJ = frozenset({"fly_ball", "line_drive"})
PULL_MID = 126.0   # TV-view x coordinate; left of mid = pulled for RHH

# Pitch types tracked individually in per-type breakdown columns
PITCH_TYPES = ["FF", "SI", "FC", "SL", "CU", "CH"]

HARD_HIT_THRESH = 95.0   # mph, for HardHit%
MIN_P95_PA      = 10     # minimum P95 PAs to report P95_PPPA
MIN_CAT_PA      =  5     # minimum PA per pitch category to report FA/BR/OS_PPPA

# ---------------------------------------------------------------------------
# Pitch category taxonomy (mirrors MLB/Savant groupings used in TV graphics)
# Each PA is classified by its decisive pitch (last pitch of the PA).
# ---------------------------------------------------------------------------

FA_TYPES = frozenset({"FF", "SI", "FC"})           # fastball family
BR_TYPES = frozenset({"SL", "CU", "SV", "ST",      # breaking ball family
                      "KC", "CS", "SC"})
OS_TYPES = frozenset({"CH", "FS", "FO", "EP"})     # offspeed family

PITCH_CATEGORIES = {"FA": FA_TYPES, "BR": BR_TYPES, "OS": OS_TYPES}


def _pitch_category(pitch_code: str) -> Optional[str]:
    """Return 'FA', 'BR', 'OS', or None for unknown/knuckleball."""
    for cat, codes in PITCH_CATEGORIES.items():
        if pitch_code in codes:
            return cat
    return None


# ---------------------------------------------------------------------------
# Fantasy scoring — play-level point values
# Personal formula: 1B+2×2B+3×3B+4×HR + R + 2×RBI + BB + 1.5×IBB + TB
#                   − 2×SO − 1.5×GIDP + 3×SB − 1.5×CS
# TB = 1×1B + 2×2B + 3×3B + 4×HR, so:
#   single = 1+1 = 2, double = 2+2 = 4, triple = 3+3 = 6, HR = 4+4 = 8.
# R, SB, CS tracked via runner events and added in build_agg().
# ---------------------------------------------------------------------------

BATTING_TP: dict = {
    "Single":                     2.0,
    "Double":                     4.0,
    "Triple":                     6.0,
    "Home Run":                   8.0,
    "Walk":                       1.0,
    "Intent Walk":                1.5,
    "Hit By Pitch":               0.0,   # PA, no points
    "Strikeout":                 -2.0,
    "Strikeout - Double Play":   -2.0,
    "Grounded Into Double Play": -1.5,
    # All other events (flyouts, groundouts, FC, DP, sac, errors, etc.): 0.0
}
RBI_WEIGHT = 2.0
R_TP   =  1.0
SB_TP  =  3.0
CS_TP  = -1.5


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(url: str, retries: int = 3) -> Optional[dict]:
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


def _api(path: str, **params) -> Optional[dict]:
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{BASE_URL}/{path}?{qs}" if qs else f"{BASE_URL}/{path}"
    time.sleep(CALL_DELAY)
    return _get(url)


def _game_feed(game_pk: int) -> Optional[dict]:
    url = f"{BASE_URL_11}/game/{game_pk}/feed/live"
    time.sleep(CALL_DELAY)
    return _get(url)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def load_cache() -> set:
    if CACHE_OUT.exists():
        df = pd.read_csv(CACHE_OUT, dtype={"gamePk": int})
        return set(df["gamePk"].tolist())
    return set()


def save_cache(done: set) -> None:
    pd.DataFrame({"gamePk": sorted(done)}).to_csv(CACHE_OUT, index=False)


def flush_rows(rows: list) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    write_header = not GAMES_OUT.exists() or GAMES_OUT.stat().st_size == 0
    df.to_csv(GAMES_OUT, mode="a", header=write_header, index=False)


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

def fetch_schedule(season: int) -> list:
    """Return gamePks for all Final AFL regular-season games in this season."""
    data = _api("schedule", sportId=AFL_SPORT_ID, season=season,
                gameType="R", leagueId=AFL_LEAGUE_ID)
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


def _fresh_counts() -> dict:
    counts = dict(
        # Pitch-level aggregates
        OutsidePitches=0, OutsideSwings=0,
        InZonePitches=0,  InZoneSwings=0, InZoneContacts=0,
        BattedBalls=0,    AirBalls=0,     PullAirBalls=0,
        TotalPitches=0,   SwStr=0,
        P95_Pitches=0,    P95_Swings=0,   P95_Whiff=0,
        P95_OutsidePitches=0, P95_OutsideSwings=0,
        # EV (on balls in play)
        EV_Max=None, EV_Sum=0.0, EV_Count=0, HardHit=0,
        EV_MaxLA=None,
        # PA / production accumulators (runner-based R/SB/CS added separately)
        PA=0,     Batting_TP=0.0,           # all PAs
        P95_PA=0, P95_Batting_TP=0.0,       # PAs with >= 1 95+ mph pitch
        # By-pitch-category accumulators (decisive-pitch method)
        FA_PA=0,  FA_Batting_TP=0.0,        # fastball decisive PAs
        BR_PA=0,  BR_Batting_TP=0.0,        # breaking ball decisive PAs
        OS_PA=0,  OS_Batting_TP=0.0,        # offspeed decisive PAs
    )
    # Per-pitch-type counters
    for pt in PITCH_TYPES:
        counts[f"PT_{pt}_n"]      = 0
        counts[f"PT_{pt}_velo"]   = 0.0
        counts[f"PT_{pt}_spin"]   = 0
        counts[f"PT_{pt}_spin_n"] = 0
        counts[f"PT_{pt}_ivb"]    = 0.0
        counts[f"PT_{pt}_ivb_n"]  = 0
        counts[f"PT_{pt}_hbk"]    = 0.0
        counts[f"PT_{pt}_hbk_n"]  = 0
    return counts


def parse_game_feed(data: dict, game_pk: int, season: int) -> list:
    """Parse all play/pitch events into per-batter count rows (one row per batter per game)."""
    live  = data.get("liveData") or {}
    plays = (live.get("plays") or {}).get("allPlays", [])

    counts: dict       = {}   # batter MLBAM_ID → _fresh_counts()
    names:  dict       = {}
    sides:  dict       = {}
    runner_stats: dict = {}   # any MLBAM_ID → {R, SB, CS}

    for play in plays:
        matchup   = play.get("matchup") or {}
        batter    = matchup.get("batter") or {}
        batter_id = batter.get("id")
        if not batter_id:
            continue

        names[batter_id] = batter.get("fullName", "")
        sides[batter_id] = (matchup.get("batSide") or {}).get("code", "")

        if batter_id not in counts:
            counts[batter_id] = _fresh_counts()

        agg      = counts[batter_id]
        bat_side = sides[batter_id]

        # ── pitch loop ───────────────────────────────────────────────────────
        play_had_pitch  = False
        pa_had_p95      = False
        last_pitch_type = None     # updated on every pitch; used after loop

        for event in play.get("playEvents", []):
            if not event.get("isPitch", False):
                continue

            play_had_pitch = True

            details    = event.get("details") or {}
            pitch_data = event.get("pitchData") or {}
            hit_data   = event.get("hitData") or {}

            zone      = _zone(pitch_data.get("zone"))
            call_code = (details.get("call") or {}).get("code", "")
            is_swing  = call_code in SWING_CODES

            start_speed     = pitch_data.get("startSpeed")
            is_p95          = start_speed is not None and start_speed >= 95.0
            if is_p95:
                pa_had_p95 = True

            pitch_type_code = (details.get("type") or {}).get("code", "")
            if pitch_type_code:
                last_pitch_type = pitch_type_code   # keep updating; final value = decisive pitch

            breaks    = pitch_data.get("breaks") or {}
            spin_rate = breaks.get("spinRate")
            ivb       = breaks.get("breakVerticalInduced")
            hbreak    = breaks.get("breakHorizontal")

            # totals
            agg["TotalPitches"] += 1
            if call_code in WHIFF_CODES:
                agg["SwStr"] += 1

            # zone metrics
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

            # P95 bucket
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

            # per-pitch-type breakdown
            if pitch_type_code in PITCH_TYPES:
                pt = pitch_type_code
                agg[f"PT_{pt}_n"] += 1
                if start_speed is not None:
                    agg[f"PT_{pt}_velo"] += start_speed
                if spin_rate is not None:
                    agg[f"PT_{pt}_spin"]   += spin_rate
                    agg[f"PT_{pt}_spin_n"] += 1
                if ivb is not None:
                    agg[f"PT_{pt}_ivb"]   += ivb
                    agg[f"PT_{pt}_ivb_n"] += 1
                if hbreak is not None:
                    agg[f"PT_{pt}_hbk"]   += hbreak
                    agg[f"PT_{pt}_hbk_n"] += 1

            # batted ball / EV
            if call_code == "X":
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

                ls = hit_data.get("launchSpeed")
                if ls is not None:
                    try:
                        ls_f = float(ls)
                        if ls_f > 0:
                            agg["EV_Count"] += 1
                            agg["EV_Sum"]   += ls_f
                            if ls_f >= HARD_HIT_THRESH:
                                agg["HardHit"] += 1
                            if agg["EV_Max"] is None or ls_f > agg["EV_Max"]:
                                agg["EV_Max"] = ls_f
                                la = hit_data.get("launchAngle")
                                agg["EV_MaxLA"] = float(la) if la is not None else None
                    except (TypeError, ValueError):
                        pass

        # ── play outcome (PA-level) ──────────────────────────────────────────
        # Only process if actual pitches were thrown — filters out stolen base
        # attempts, pickoffs, wild pitches that appear as separate play entries.
        if play_had_pitch:
            result     = play.get("result") or {}
            event_name = result.get("event", "")
            rbi        = int(result.get("rbi") or 0)

            bat_tp = BATTING_TP.get(event_name, 0.0) + RBI_WEIGHT * rbi

            # overall PA bucket
            agg["PA"]          += 1
            agg["Batting_TP"]  += bat_tp

            # P95 bucket (PA saw >= 1 95+ mph pitch)
            if pa_had_p95:
                agg["P95_PA"]         += 1
                agg["P95_Batting_TP"] += bat_tp

            # Pitch-category bucket (decisive-pitch method):
            # last_pitch_type = final pitch of this PA (the pitch the AB turned on).
            if last_pitch_type:
                cat = _pitch_category(last_pitch_type)
                if cat:
                    agg[f"{cat}_PA"]         += 1
                    agg[f"{cat}_Batting_TP"] += bat_tp

        # ── runner events (R, SB, CS per player) ────────────────────────────
        for ri in play.get("runners", []):
            runner    = ((ri.get("details") or {}).get("runner") or {})
            runner_id = runner.get("id")
            if not runner_id:
                continue
            if runner_id not in runner_stats:
                runner_stats[runner_id] = {"R": 0, "SB": 0, "CS": 0}

            rev = (ri.get("details") or {}).get("event", "")
            if rev == "Stolen Base":
                runner_stats[runner_id]["SB"] += 1
            elif "Caught Stealing" in rev:
                runner_stats[runner_id]["CS"] += 1

            di = ri.get("details") or {}
            mv = ri.get("movement") or {}
            if di.get("scoringRun") or mv.get("end") == "score":
                runner_stats[runner_id]["R"] += 1

    # ── assemble output rows ─────────────────────────────────────────────────
    rows = []
    for batter_id, agg in counts.items():
        rs = runner_stats.get(batter_id, {"R": 0, "SB": 0, "CS": 0})
        total_tp = (
            agg["Batting_TP"]
            + R_TP  * rs["R"]
            + SB_TP * rs["SB"]
            + CS_TP * rs["CS"]
        )
        rows.append({
            "gamePk":          game_pk,
            "Season":          season,
            "MLBAM_ID":        batter_id,
            "Name":            names[batter_id],
            "BatSide":         sides.get(batter_id, ""),
            "PA":              agg["PA"],
            "Batting_TP":      round(agg["Batting_TP"], 1),
            "R":               rs["R"],
            "SB":              rs["SB"],
            "CS":              rs["CS"],
            "TP":              round(total_tp, 1),
            "P95_PA":          agg["P95_PA"],
            "P95_Batting_TP":  round(agg["P95_Batting_TP"], 1),
            "FA_PA":           agg["FA_PA"],
            "FA_Batting_TP":   round(agg["FA_Batting_TP"], 1),
            "BR_PA":           agg["BR_PA"],
            "BR_Batting_TP":   round(agg["BR_Batting_TP"], 1),
            "OS_PA":           agg["OS_PA"],
            "OS_Batting_TP":   round(agg["OS_Batting_TP"], 1),
            **{k: v for k, v in agg.items()
               if k not in ("Batting_TP", "P95_Batting_TP",
                            "PA", "P95_PA",
                            "FA_PA", "FA_Batting_TP",
                            "BR_PA", "BR_Batting_TP",
                            "OS_PA", "OS_Batting_TP")},
        })
    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

# Integer count columns (simply summed across games)
INT_COLS = [
    "OutsidePitches", "OutsideSwings",
    "InZonePitches",  "InZoneSwings",  "InZoneContacts",
    "BattedBalls",    "AirBalls",      "PullAirBalls",
    "TotalPitches",   "SwStr",
    "P95_Pitches",    "P95_Swings",    "P95_Whiff",
    "P95_OutsidePitches", "P95_OutsideSwings",
    "EV_Count",       "HardHit",
    "PA", "P95_PA", "R", "SB", "CS",
    "FA_PA", "BR_PA", "OS_PA",
]
for _pt in PITCH_TYPES:
    INT_COLS += [f"PT_{_pt}_n", f"PT_{_pt}_spin_n", f"PT_{_pt}_ivb_n", f"PT_{_pt}_hbk_n"]

# Float sum columns
FLOAT_SUM_COLS = [
    "EV_Sum", "TP",
    "Batting_TP", "P95_Batting_TP",
    "FA_Batting_TP", "BR_Batting_TP", "OS_Batting_TP",
]
for _pt in PITCH_TYPES:
    FLOAT_SUM_COLS += [f"PT_{_pt}_velo", f"PT_{_pt}_spin", f"PT_{_pt}_ivb", f"PT_{_pt}_hbk"]


def build_agg(games_df: pd.DataFrame) -> pd.DataFrame:
    if games_df.empty:
        return pd.DataFrame()

    # Back-fill missing columns from older cache rows
    for col in INT_COLS:
        if col not in games_df.columns:
            games_df = games_df.copy()
            games_df[col] = 0
    for col in FLOAT_SUM_COLS:
        if col not in games_df.columns:
            games_df = games_df.copy()
            games_df[col] = 0.0
    for col in ("EV_Max", "EV_MaxLA"):
        if col not in games_df.columns:
            games_df = games_df.copy()
            games_df[col] = None

    games_df = games_df.drop_duplicates(subset=["gamePk", "MLBAM_ID"])

    grp = ["MLBAM_ID", "Season"]

    agg = games_df.groupby(grp, as_index=False)[INT_COLS + FLOAT_SUM_COLS].sum()

    gcnt = (games_df.groupby(grp, as_index=False)["gamePk"]
            .nunique().rename(columns={"gamePk": "Games"}))
    nm   = games_df.groupby(grp)["Name"].last().reset_index()
    bs   = games_df.groupby(grp)["BatSide"].last().reset_index()

    agg = agg.merge(gcnt, on=grp).merge(nm, on=grp).merge(bs, on=grp)

    # MaxEV = max of per-game maxes; take the EV_MaxLA from that same game row.
    # Sort descending so groupby.first() picks the max-EV row per player-season.
    games_with_ev = games_df.dropna(subset=["EV_Max"])
    if not games_with_ev.empty:
        ev_max = (games_with_ev
                  .sort_values("EV_Max", ascending=False)
                  .groupby(grp, as_index=False)[["EV_Max", "EV_MaxLA"]]
                  .first())
        agg = agg.merge(ev_max, on=grp, how="left")
    else:
        agg["EV_Max"]   = None
        agg["EV_MaxLA"] = None
    no_ev = agg["EV_Count"].fillna(0) == 0
    agg.loc[no_ev, ["EV_Max", "EV_MaxLA"]] = None

    # ── swing / contact rates ────────────────────────────────────────────────
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
    agg["Whiff%"] = (agg["SwStr"] / total_swings).where(total_swings > 0).round(3)

    agg["P95_Whiff%"] = (
        agg["P95_Whiff"] / agg["P95_Swings"]
    ).where(agg["P95_Swings"] > 0).round(3)

    agg["P95_Chase%"] = (
        agg["P95_OutsideSwings"] / agg["P95_OutsidePitches"]
    ).where(agg["P95_OutsidePitches"] > 0).round(3)

    agg["P95_pct"] = (
        agg["P95_Pitches"] / agg["TotalPitches"]
    ).where(agg["TotalPitches"] > 0).round(3)
    agg.loc[agg["P95_Pitches"] == 0, ["P95_Whiff%", "P95_Chase%", "P95_pct"]] = None

    # ── EV / contact quality rates ───────────────────────────────────────────
    agg["AvgEV"] = (
        agg["EV_Sum"] / agg["EV_Count"]
    ).where(agg["EV_Count"] > 0).round(1)

    agg["HardHit%"] = (
        agg["HardHit"] / agg["EV_Count"]
    ).where(agg["EV_Count"] > 0).round(3)

    # ── production rates ─────────────────────────────────────────────────────
    # PPPA: full formula. TP already includes R/SB/CS (computed per game row).
    agg["PPPA"] = (
        agg["TP"] / agg["PA"]
    ).where(agg["PA"] > 0).round(3)

    # P95_PPPA: batting outcomes + RBI on P95 PAs only (no R/SB/CS).
    agg["P95_PPPA"] = (
        agg["P95_Batting_TP"] / agg["P95_PA"]
    ).where(agg["P95_PA"] >= MIN_P95_PA).round(3)

    # Pitch-category PPPA (decisive-pitch method, no R/SB/CS — same reasoning).
    for cat in ("FA", "BR", "OS"):
        agg[f"{cat}_PPPA"] = (
            agg[f"{cat}_Batting_TP"] / agg[f"{cat}_PA"]
        ).where(agg[f"{cat}_PA"] >= MIN_CAT_PA).round(3)

    # ── per-pitch-type metrics ───────────────────────────────────────────────
    for pt in PITCH_TYPES:
        n_col   = f"PT_{pt}_n"
        pct_col = f"{pt}_pct"
        vel_col = f"{pt}_velo"
        sp_col  = f"{pt}_spin"
        ivb_col = f"{pt}_ivb"
        hbk_col = f"{pt}_hbreak"

        agg[pct_col] = (agg[n_col] / agg["TotalPitches"]).where(agg["TotalPitches"] > 0).round(3)
        agg[vel_col] = (agg[f"PT_{pt}_velo"] / agg[n_col]).where(agg[n_col] > 0).round(1)
        agg[sp_col]  = (agg[f"PT_{pt}_spin"] / agg[f"PT_{pt}_spin_n"]).where(agg[f"PT_{pt}_spin_n"] > 0).round(0)
        agg[ivb_col] = (agg[f"PT_{pt}_ivb"]  / agg[f"PT_{pt}_ivb_n" ]).where(agg[f"PT_{pt}_ivb_n"]  > 0).round(1)
        agg[hbk_col] = (agg[f"PT_{pt}_hbk"]  / agg[f"PT_{pt}_hbk_n" ]).where(agg[f"PT_{pt}_hbk_n"]  > 0).round(1)

    # Drop raw accumulator columns
    drop_cols = (
        FLOAT_SUM_COLS
        + [c for c in INT_COLS if c.startswith("PT_")]
        + ["EV_Sum", "HardHit", "SwStr",
           "P95_Whiff", "P95_Swings", "P95_Pitches",
           "P95_OutsidePitches", "P95_OutsideSwings",
           "Batting_TP", "P95_Batting_TP",
           "FA_Batting_TP", "BR_Batting_TP", "OS_Batting_TP"]
    )
    agg = agg.drop(columns=[c for c in drop_cols if c in agg.columns])

    # ── column order ─────────────────────────────────────────────────────────
    leading = [
        "MLBAM_ID", "Name", "BatSide", "Season", "Games",
        "PA", "TotalPitches", "BattedBalls",
        # Production — overall
        "TP", "PPPA",
        # Production — velocity context
        "P95_PA", "P95_PPPA",
        # Production — pitch category (decisive-pitch)
        "FA_PA", "FA_PPPA",
        "BR_PA", "BR_PPPA",
        "OS_PA", "OS_PPPA",
        # Runner-based counting stats
        "R", "SB", "CS",
        # Discipline
        "Chase%", "Z-Contact%", "Whiff%",
        # Power / contact quality
        "MaxEV", "EV_MaxLA", "AvgEV", "HardHit%", "EV_Count",
        "PullAir%",
        # Velocity exposure
        "P95_pct", "P95_Whiff%", "P95_Chase%",
    ]
    for pt in PITCH_TYPES:
        leading += [f"{pt}_pct", f"{pt}_velo", f"{pt}_spin", f"{pt}_ivb", f"{pt}_hbreak"]

    remaining = [c for c in agg.columns if c not in leading]
    col_order = leading + remaining
    return agg[[c for c in col_order if c in agg.columns]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch AFL pitch-level Statcast from game feeds")
    parser.add_argument("--season", type=int, default=None,
                        help="Fetch a specific season (e.g. --season 2025)")
    parser.add_argument("--full", action="store_true",
                        help=f"Fetch all Hawk-Eye-era seasons ({HAWK_EYE_FIRST_YEAR}–current)")
    args = parser.parse_args()

    print("=== fetch_afl_pitches.py ===\n")

    if args.full:
        seasons = [yr for yr in range(HAWK_EYE_FIRST_YEAR, CURRENT_SEASON + 1)
                   if yr not in NO_SEASON]
    else:
        seasons = [args.season or CURRENT_SEASON]

    done_pks = load_cache()
    print(f"Cache: {len(done_pks):,} games already processed")

    if GAMES_OUT.exists():
        games_df = pd.read_csv(GAMES_OUT, dtype={"MLBAM_ID": "Int64", "gamePk": int},
                               on_bad_lines="skip")
    else:
        games_df = pd.DataFrame()

    new_rows: list = []
    total_new = 0

    for season in seasons:
        print(f"\n── {season} ──────────────────────────────────────────")
        pks     = fetch_schedule(season)
        pks_new = [pk for pk in pks if pk not in done_pks]
        print(f"  {len(pks)} total games, {len(pks) - len(pks_new)} cached, "
              f"{len(pks_new)} to fetch")

        for i, pk in enumerate(pks_new, 1):
            feed = _game_feed(pk)
            if feed is None:
                print(f"    {i}/{len(pks_new)}  gamePk={pk}  SKIP (feed error)")
                continue

            rows = parse_game_feed(feed, pk, season)
            new_rows.extend(rows)
            done_pks.add(pk)
            total_new += 1

            print(f"    {i}/{len(pks_new)}  gamePk={pk}  {len(rows)} batters")

            if total_new % SAVE_INTERVAL == 0:
                flush_rows(new_rows)
                save_cache(done_pks)
                new_rows = []
                print(f"    [flushed after {total_new} games]")

    if new_rows:
        flush_rows(new_rows)
        save_cache(done_pks)

    print(f"\nFetched {total_new} new games total")

    if GAMES_OUT.exists():
        all_games = pd.read_csv(GAMES_OUT, dtype={"MLBAM_ID": "Int64", "gamePk": int},
                                on_bad_lines="skip")
    elif new_rows:
        all_games = pd.DataFrame(new_rows)
    else:
        print("No data to aggregate.")
        return

    agg = build_agg(all_games)
    if agg.empty:
        print("No rows to write.")
        return

    agg.to_csv(AGG_OUT, index=False)
    print(f"\nWrote {len(agg)} player-season rows → {AGG_OUT.name}")
    print(f"Seasons covered: {sorted(agg['Season'].unique())}")
    print(f"Players: {agg['MLBAM_ID'].nunique()}")

    # ── summary: match back to current prospect pool ─────────────────────────
    scores_path = DATA_DIR / "rankings_fantrax" / "prospect_scores.csv"
    if not scores_path.exists():
        scores_path = DATA_DIR / "rankings" / "prospect_scores.csv"

    season_agg = agg[agg["Season"] == (args.season or CURRENT_SEASON)]
    if scores_path.exists() and not season_agg.empty:
        scores = pd.read_csv(scores_path)
        scores["MLBAM_ID"] = pd.to_numeric(
            scores.get("MLBAM_ID", pd.Series(dtype=str)), errors="coerce"
        ).astype("Int64")
        matched = season_agg.merge(
            scores[["MLBAM_ID", "Combined_Rank", "Name"]].rename(
                columns={"Name": "Prospect_Name"}
            ),
            on="MLBAM_ID", how="inner"
        ).sort_values("Combined_Rank")

        print(f"\n{len(matched)} current prospects with AFL pitch data "
              f"(season {args.season or CURRENT_SEASON}):")
        for _, r in matched.head(20).iterrows():
            pppa_str = f"  PPPA {r['PPPA']:.3f}" if pd.notna(r.get("PPPA")) else ""
            fa_str   = f"  FA {r['FA_PPPA']:.3f}({int(r['FA_PA'])})" if pd.notna(r.get("FA_PPPA")) else ""
            br_str   = f"  BR {r['BR_PPPA']:.3f}({int(r['BR_PA'])})" if pd.notna(r.get("BR_PPPA")) else ""
            os_str   = f"  OS {r['OS_PPPA']:.3f}({int(r['OS_PA'])})" if pd.notna(r.get("OS_PPPA")) else ""
            ev_str   = f"  MaxEV {r['MaxEV']:.1f}" if pd.notna(r.get("MaxEV")) else ""
            print(f"  #{int(r['Combined_Rank']):>3}  {r['Name']:<25}  "
                  f"{int(r['PA'])}PA{pppa_str}{fa_str}{br_str}{os_str}{ev_str}")


if __name__ == "__main__":
    main()
