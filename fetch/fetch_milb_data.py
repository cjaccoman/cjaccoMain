"""Fetch MiLB batting stats from the MLB Stats API (2006–2026).

Outputs written to data/api/:
  milb_hitting.csv   — counting stats; matches minorLeagueData schema (+ MLBAM_ID)
  milb_advanced.csv  — rate + batted-ball stats; matches historical_ml_advanced/batted schema
  chadwick.csv       — MLBAM ↔ FanGraphs ID crosswalk (cached from Chadwick Bureau register)

NOT available from the MLB API for MiLB levels (still require FanGraphs):
  wRC+, wOBA, wRAA, wRC, Spd, wSB, Pull%, Cent%, Oppo%

PlayerId assignment:
  - FanGraphs ID where a Chadwick mapping exists  -> backward-compatible with existing data
  - MLBAM ID (as string) where no FanGraphs mapping -> typically very new/fringe prospects
"""

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
API_DIR = DATA_DIR / "api"
API_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://statsapi.mlb.com/api/v1"
CHADWICK_API = "https://api.github.com/repos/chadwickbureau/register/contents/data"
CHADWICK_RAW = "https://raw.githubusercontent.com/chadwickbureau/register/master/data"
CHADWICK_CACHE = API_DIR / "chadwick.csv"

# sportId -> Level abbreviation used in our data schema
SPORT_IDS = {"AAA": 11, "AA": 12, "A+": 13, "A": 14, "R": 16}
LEVEL_FOR_SPORT = {v: k for k, v in SPORT_IDS.items()}

SEASONS = list(range(2006, 2027))
CURRENT_SEASON = SEASONS[-1]   # only season re-fetched on incremental runs
MIN_PA = 10
MAX_AGE = 26
CALL_DELAY = 0.3  # seconds between API calls — respectful throttle

HITTING_OUT   = API_DIR / "milb_hitting.csv"
ADVANCED_OUT  = API_DIR / "milb_advanced.csv"
BIRTHDATE_OUT = API_DIR / "player_birthdays.csv"
POSITION_OUT  = API_DIR / "player_positions.csv"

BIRTH_BATCH = 50   # max personIds per /api/v1/people request


# ---------------------------------------------------------------------------
# HTTP helpers
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
# Chadwick Bureau register
# ---------------------------------------------------------------------------

def load_chadwick() -> pd.DataFrame:
    if CHADWICK_CACHE.exists():
        ck = pd.read_csv(
            CHADWICK_CACHE,
            usecols=["key_mlbam", "key_fangraphs"],
            dtype={"key_mlbam": "Int64", "key_fangraphs": "Int64"},
        ).dropna(subset=["key_mlbam"])
        print(f"  Chadwick (cached): {len(ck):,} players, "
              f"{ck['key_fangraphs'].notna().sum():,} with FanGraphs IDs")
        return ck

    print("  Discovering Chadwick split files via GitHub API...")
    try:
        req = urllib.request.Request(CHADWICK_API,
                                     headers={"User-Agent": "prospectsMain/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            entries = json.loads(r.read())
        people_files = sorted(
            e["name"] for e in entries if e["name"].startswith("people") and e["name"].endswith(".csv")
        )
        print(f"  Found {len(people_files)} split files: {', '.join(people_files)}")
    except Exception as e:
        print(f"  ERROR listing Chadwick files: {e}; PlayerId will use MLBAM IDs only")
        return pd.DataFrame(columns=["key_mlbam", "key_fangraphs"])

    parts = []
    for fname in people_files:
        url = f"{CHADWICK_RAW}/{fname}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "prospectsMain/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                import io
                chunk = pd.read_csv(
                    io.BytesIO(r.read()),
                    usecols=["key_mlbam", "key_fangraphs"],
                    dtype={"key_mlbam": "Int64", "key_fangraphs": "Int64"},
                )
                parts.append(chunk)
                print(f"    {fname}: {len(chunk):,} rows")
        except Exception as e:
            print(f"    WARN: {fname} failed: {e}")

    if not parts:
        print("  ERROR: no Chadwick files downloaded; PlayerId will use MLBAM IDs only")
        return pd.DataFrame(columns=["key_mlbam", "key_fangraphs"])

    ck = pd.concat(parts, ignore_index=True).dropna(subset=["key_mlbam"])
    ck.to_csv(CHADWICK_CACHE, index=False)
    with_fg = ck["key_fangraphs"].notna().sum()
    print(f"  Chadwick: {len(ck):,} players, {with_fg:,} have FanGraphs IDs -- cached to {CHADWICK_CACHE.name}")
    return ck


# ---------------------------------------------------------------------------
# MLB org abbreviation lookup (one call, uses 2023 as reference)
# ---------------------------------------------------------------------------

def build_mlb_org_lookup() -> dict[int, str]:
    """Returns {parentOrgId: abbreviation} for all 30 MLB franchises."""
    data = _api("teams", sportId=1, season=2023)
    if not data:
        return {}
    return {t["id"]: t.get("abbreviation", "UNK") for t in data.get("teams", [])}


# ---------------------------------------------------------------------------
# Per-season-level lookups
# ---------------------------------------------------------------------------

def build_team_lookup(sport_id: int, season: int, mlb_org: dict) -> dict:
    """Returns {team_id: {"org_abbrev": str, "league_id": int}}."""
    data = _api("teams", sportId=sport_id, season=season)
    if not data:
        return {}
    out = {}
    for t in data.get("teams", []):
        team_id = t.get("id")
        parent_id = (
            t.get("parentOrgId")
            or (t.get("parentOrg") or {}).get("id")
        )
        league_id = (t.get("league") or {}).get("id")
        out[team_id] = {
            "org_abbrev": mlb_org.get(parent_id, t.get("abbreviation", "UNK")),
            "league_id": league_id,
        }
    return out


def build_league_lookup(sport_id: int, season: int) -> dict[int, str]:
    """Returns {league_id: full_league_name}."""
    data = _api("leagues", sportId=sport_id, season=season)
    if not data:
        return {}
    return {lg["id"]: lg.get("name", "DiscLeague") for lg in data.get("leagues", [])}


# ---------------------------------------------------------------------------
# Raw split fetching
# ---------------------------------------------------------------------------

STATS_LIMIT = 5000  # well above any single season/level player count


def fetch_splits(stat_type: str, sport_id: int, season: int) -> list:
    data = _api(
        "stats",
        stats=stat_type,
        playerPool="all",
        group="hitting",
        gameType="R",
        season=season,
        sportId=sport_id,
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
                  f"increase STATS_LIMIT above {STATS_LIMIT}")
        splits.extend(block_splits)
    return splits


# ---------------------------------------------------------------------------
# Type-safe helpers
# ---------------------------------------------------------------------------

def _int(v) -> int | None:
    try:
        return int(v) if v is not None else None
    except (ValueError, TypeError):
        return None


def _float(v) -> float | None:
    try:
        return float(v) if v is not None else None
    except (ValueError, TypeError):
        return None


def _rate(stat: dict, field: str) -> float | None:
    """Parse a rate field; handles both string and float API responses."""
    return _float(stat.get(field))


# ---------------------------------------------------------------------------
# Hitting split parser  (stats=season)
# ---------------------------------------------------------------------------

def parse_hitting_splits(
    splits: list, team_lkp: dict, league_lkp: dict
) -> list[dict]:
    rows = []
    for sp in splits:
        stat = sp.get("stat", {})
        player = sp.get("player", {})
        team_data = sp.get("team", {})
        sport_data = sp.get("sport", {})

        team_id = team_data.get("id")
        sport_id = sport_data.get("id")
        team_info = team_lkp.get(team_id, {})
        league_id = team_info.get("league_id") or (sp.get("league") or {}).get("id")

        hits = _int(stat.get("hits")) or 0
        doubles = _int(stat.get("doubles")) or 0
        triples = _int(stat.get("triples")) or 0
        hr = _int(stat.get("homeRuns")) or 0

        rows.append({
            "MLBAM_ID": player.get("id"),
            "Season": _int(sp.get("season")),
            "Name": player.get("fullName", ""),
            "Team": team_info.get("org_abbrev", "UNK"),
            "Level": LEVEL_FOR_SPORT.get(sport_id, "?"),
            "League": league_lkp.get(league_id, "DiscLeague"),
            "Age": _int(stat.get("age")),
            "G": _int(stat.get("gamesPlayed")),
            "PA": _int(stat.get("plateAppearances")),
            "1B": hits - doubles - triples - hr,
            "2B": doubles,
            "3B": triples,
            "HR": hr,
            "R": _int(stat.get("runs")),
            "RBI": _int(stat.get("rbi")),
            "BB": _int(stat.get("baseOnBalls")),
            "IBB": _int(stat.get("intentionalWalks")),
            "SO": _int(stat.get("strikeOuts")),
            "GIDP": _int(stat.get("groundIntoDoublePlay")),
            "SB": _int(stat.get("stolenBases")),
            "CS": _int(stat.get("caughtStealing")),
        })
    return rows


# ---------------------------------------------------------------------------
# Advanced split parser  (stats=seasonAdvanced)
# ---------------------------------------------------------------------------

def parse_advanced_splits(
    splits: list, team_lkp: dict, league_lkp: dict
) -> list[dict]:
    rows = []
    for sp in splits:
        stat = sp.get("stat", {})
        player = sp.get("player", {})
        team_data = sp.get("team", {})
        sport_data = sp.get("sport", {})

        team_id = team_data.get("id")
        sport_id = sport_data.get("id")
        team_info = team_lkp.get(team_id, {})
        league_id = team_info.get("league_id") or (sp.get("league") or {}).get("id")

        # Batted ball counts (hits + outs per type)
        g_outs = _int(stat.get("groundOuts")) or 0
        f_outs = _int(stat.get("flyOuts")) or 0
        l_outs = _int(stat.get("lineOuts")) or 0
        p_outs = _int(stat.get("popOuts")) or 0
        g_hits = _int(stat.get("groundHits")) or 0
        f_hits = _int(stat.get("flyHits")) or 0
        l_hits = _int(stat.get("lineHits")) or 0
        p_hits = _int(stat.get("popHits")) or 0

        ground = g_outs + g_hits
        fly = f_outs + f_hits
        line = l_outs + l_hits
        pop = p_outs + p_hits
        bip = ground + fly + line + pop
        fly_plus_pop = fly + pop

        gb_pct = round(ground / bip, 3) if bip > 0 else None
        ld_pct = round(line / bip, 3) if bip > 0 else None
        fb_pct = round(fly_plus_pop / bip, 3) if bip > 0 else None
        gb_fb = round(ground / fly_plus_pop, 2) if fly_plus_pop > 0 else None
        iffb_pct = round(pop / fly_plus_pop, 3) if fly_plus_pop > 0 else None

        # Swing-miss rates
        pitches = _int(stat.get("numberOfPitches")) or 0
        misses  = _int(stat.get("swingAndMisses")) or 0
        swings  = _int(stat.get("totalSwings")) or 0
        swstr = round(misses / pitches, 3) if pitches > 0 else None  # misses / pitches
        whiff = round(misses / swings, 3) if swings > 0 else None    # misses / swings

        rows.append({
            "MLBAM_ID": player.get("id"),
            "Season": _int(sp.get("season")),
            "Name": player.get("fullName", ""),
            "Team": team_info.get("org_abbrev", "UNK"),
            "Level": LEVEL_FOR_SPORT.get(sport_id, "?"),
            "League": league_lkp.get(league_id, "DiscLeague"),
            "Age": _int(stat.get("age")),
            "PA": _int(stat.get("plateAppearances")),
            "AVG": _rate(stat, "avg"),
            "OBP": _rate(stat, "obp"),
            "SLG": _rate(stat, "slg"),
            "OPS": _rate(stat, "ops"),
            "ISO": _rate(stat, "iso"),
            "BABIP": _rate(stat, "babip"),
            "BB%": _rate(stat, "walksPerPlateAppearance"),
            "K%": _rate(stat, "strikeoutsPerPlateAppearance"),
            "GB%": gb_pct,
            "LD%": ld_pct,
            "FB%": fb_pct,
            "GB/FB": gb_fb,
            "IFFB%": iffb_pct,
            "SwStr%": swstr,
            "Whiff%": whiff,
            "Pitches": pitches if pitches > 0 else None,
            "_fly_pop": fly_plus_pop,  # used in main() to compute HR/FB; dropped before save
        })
    return rows


# ---------------------------------------------------------------------------
# Chadwick crosswalk
# ---------------------------------------------------------------------------

def apply_crosswalk(df: pd.DataFrame, chadwick: pd.DataFrame) -> pd.DataFrame:
    """Add PlayerId column: FanGraphs ID where mapped, else MLBAM ID as string."""
    df = df.copy()
    df["MLBAM_ID"] = pd.to_numeric(df["MLBAM_ID"], errors="coerce").astype("Int64")
    merged = df.merge(chadwick, left_on="MLBAM_ID", right_on="key_mlbam", how="left")

    # FanGraphs ID wins; fall back to MLBAM ID string for unmapped players
    fg = merged["key_fangraphs"]
    mlbam = merged["MLBAM_ID"]
    merged["PlayerId"] = fg.where(fg.notna(), mlbam).apply(
        lambda x: str(int(x)) if pd.notna(x) else None
    )
    return merged.drop(columns=["key_mlbam", "key_fangraphs"], errors="ignore")


# ---------------------------------------------------------------------------
# Birth date fetch
# ---------------------------------------------------------------------------

def fetch_birth_dates(mlbam_ids: list[int]) -> dict[int, str]:
    """Fetch birth dates for a list of MLBAM IDs via /api/v1/people.

    Returns {mlbam_id: 'YYYY-MM-DD'} for players where birthDate is available.
    Already-cached IDs (from BIRTHDATE_OUT) are skipped to avoid redundant calls.
    """
    existing: dict[int, str] = {}
    if BIRTHDATE_OUT.exists():
        cached = pd.read_csv(BIRTHDATE_OUT, dtype={"MLBAM_ID": int, "BirthDate": str})
        existing = dict(zip(cached["MLBAM_ID"], cached["BirthDate"]))

    to_fetch = [i for i in mlbam_ids if i not in existing]
    if not to_fetch:
        print(f"  Birth dates: all {len(existing)} already cached, nothing to fetch")
        return existing

    print(f"  Birth dates: fetching {len(to_fetch)} new IDs "
          f"({len(existing)} already cached) ...")

    results = dict(existing)
    batches = [to_fetch[i:i + BIRTH_BATCH] for i in range(0, len(to_fetch), BIRTH_BATCH)]
    for idx, batch in enumerate(batches, 1):
        ids_str = ",".join(str(i) for i in batch)
        url = (f"{BASE_URL}/people?personIds={ids_str}"
               f"&fields=people,id,birthDate")
        data = _get(url)
        if data:
            for person in data.get("people", []):
                pid = person.get("id")
                bd  = person.get("birthDate")
                if pid and bd:
                    results[int(pid)] = bd
        if idx < len(batches):
            time.sleep(CALL_DELAY)

    # Persist full cache
    rows = [{"MLBAM_ID": k, "BirthDate": v} for k, v in sorted(results.items())]
    pd.DataFrame(rows).to_csv(BIRTHDATE_OUT, index=False)
    fetched = len(results) - len(existing)
    print(f"  Birth dates: fetched {fetched} new, {len(results)} total cached -> {BIRTHDATE_OUT.name}")
    return results


# ---------------------------------------------------------------------------
# Position fetch
# ---------------------------------------------------------------------------

def fetch_positions(mlbam_ids: list[int]) -> dict[int, str]:
    """Fetch primary position abbreviation for a list of MLBAM IDs via /api/v1/people.

    Returns {mlbam_id: 'SS'} (e.g. C, 1B, 2B, 3B, SS, LF, CF, RF, OF, DH, P).
    Already-cached IDs are skipped. Persists to player_positions.csv.
    """
    existing: dict[int, str] = {}
    if POSITION_OUT.exists():
        cached = pd.read_csv(POSITION_OUT, dtype={"MLBAM_ID": int, "Position": str})
        existing = dict(zip(cached["MLBAM_ID"], cached["Position"]))

    to_fetch = [i for i in mlbam_ids if i not in existing]
    if not to_fetch:
        print(f"  Positions: all {len(existing)} already cached, nothing to fetch")
        return existing

    print(f"  Positions: fetching {len(to_fetch)} new IDs "
          f"({len(existing)} already cached) ...")

    results = dict(existing)
    batches = [to_fetch[i:i + BIRTH_BATCH] for i in range(0, len(to_fetch), BIRTH_BATCH)]
    for idx, batch in enumerate(batches, 1):
        ids_str = ",".join(str(i) for i in batch)
        url = (f"{BASE_URL}/people?personIds={ids_str}"
               f"&fields=people,id,primaryPosition,abbreviation")
        data = _get(url)
        if data:
            for person in data.get("people", []):
                pid = person.get("id")
                pos = (person.get("primaryPosition") or {}).get("abbreviation")
                if pid and pos:
                    results[int(pid)] = pos
        if idx < len(batches):
            time.sleep(CALL_DELAY)

    rows = [{"MLBAM_ID": k, "Position": v} for k, v in sorted(results.items())]
    pd.DataFrame(rows).to_csv(POSITION_OUT, index=False)
    fetched = len(results) - len(existing)
    print(f"  Positions: fetched {fetched} new, {len(results)} total cached -> {POSITION_OUT.name}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== fetch_milb_data.py ===\n")

    print("Step 1: Chadwick register")
    chadwick = load_chadwick()

    print("\nStep 2: MLB org abbreviations")
    mlb_org = build_mlb_org_lookup()
    print(f"  {len(mlb_org)} MLB orgs")

    # --- Determine fetch scope: incremental (2026 only) vs full history ---
    incremental = HITTING_OUT.exists() and ADVANCED_OUT.exists()
    if incremental:
        print(f"\nExisting data found — refreshing {CURRENT_SEASON} only "
              f"(PA >= {MIN_PA}, Age <= {MAX_AGE})")
        existing_hit = pd.read_csv(HITTING_OUT, dtype={"PlayerId": str, "MLBAM_ID": str})
        existing_adv = pd.read_csv(ADVANCED_OUT, dtype={"PlayerId": str, "MLBAM_ID": str})
        # Strip current season and apply age filter retroactively to remaining rows
        existing_hit = existing_hit[
            (existing_hit["Season"] != CURRENT_SEASON) &
            (existing_hit["Age"].fillna(99) <= MAX_AGE)
        ]
        existing_adv = existing_adv[
            (existing_adv["Season"] != CURRENT_SEASON) &
            (existing_adv["Age"].fillna(99) <= MAX_AGE)
        ]
        fetch_seasons = [CURRENT_SEASON]
    else:
        print(f"\nNo existing data — fetching full history "
              f"{SEASONS[0]}-{SEASONS[-1]} (PA >= {MIN_PA}, Age <= {MAX_AGE})")
        existing_hit = pd.DataFrame()
        existing_adv = pd.DataFrame()
        fetch_seasons = SEASONS

    all_hitting: list[dict] = []
    all_advanced: list[dict] = []

    total = len(fetch_seasons) * len(SPORT_IDS)
    done = 0

    print(f"\nStep 3: Fetching {total} season×level combinations\n")

    for season in fetch_seasons:
        for level, sport_id in SPORT_IDS.items():
            done += 1
            tag = f"[{done:3}/{total}] {season} {level:3}"

            team_lkp = build_team_lookup(sport_id, season, mlb_org)
            league_lkp = build_league_lookup(sport_id, season)

            # --- Counting stats ---
            splits = fetch_splits("season", sport_id, season)
            if splits:
                rows = parse_hitting_splits(splits, team_lkp, league_lkp)
                before = len(rows)
                rows = [r for r in rows
                        if (r.get("PA") or 0) >= MIN_PA
                        and (r.get("Age") or 99) <= MAX_AGE]
                all_hitting.extend(rows)
                print(f"{tag}  hitting: {len(rows):4} rows "
                      f"({before - len(rows)} dropped <{MIN_PA} PA or >{MAX_AGE} Age)")
            else:
                print(f"{tag}  hitting: no data")

            # --- Advanced / batted ball stats ---
            adv_splits = fetch_splits("seasonAdvanced", sport_id, season)
            if adv_splits:
                rows = parse_advanced_splits(adv_splits, team_lkp, league_lkp)
                rows = [r for r in rows
                        if (r.get("PA") or 0) >= MIN_PA
                        and (r.get("Age") or 99) <= MAX_AGE]
                all_advanced.extend(rows)

    print(f"\nFetched: {len(all_hitting):,} new hitting rows, {len(all_advanced):,} new advanced rows")

    # --- Build and save hitting file ---
    print("\nBuilding milb_hitting.csv...")
    new_hit_df = pd.DataFrame(all_hitting)
    new_hit_df = apply_crosswalk(new_hit_df, chadwick)
    new_hit_df.loc[new_hit_df["Team"] == "MEX", "Level"] = "R"
    new_hit_df = new_hit_df[[
        "PlayerId", "MLBAM_ID", "Season", "Name", "Team", "Level", "League",
        "Age", "G", "PA", "1B", "2B", "3B", "HR",
        "R", "RBI", "BB", "IBB", "SO", "GIDP", "SB", "CS",
    ]]
    hit_df = pd.concat([existing_hit, new_hit_df], ignore_index=True)
    hit_df.to_csv(HITTING_OUT, index=False)
    print(f"  Wrote {len(hit_df):,} rows -> {HITTING_OUT} "
          f"({len(existing_hit):,} existing + {len(new_hit_df):,} new {CURRENT_SEASON})")

    # --- Build and save advanced file ---
    print("Building milb_advanced.csv...")
    new_adv_df = pd.DataFrame(all_advanced)
    if not new_adv_df.empty:
        new_adv_df = apply_crosswalk(new_adv_df, chadwick)
        new_adv_df.loc[new_adv_df["Team"] == "MEX", "Level"] = "R"

        # HR/FB: join HR from new hitting rows on (MLBAM_ID, Season, Team, Level)
        hr_lookup = (
            pd.DataFrame(all_hitting)[["MLBAM_ID", "Season", "Team", "Level", "HR"]]
            .astype({"MLBAM_ID": "Int64"})
        )
        new_adv_df["MLBAM_ID"] = new_adv_df["MLBAM_ID"].astype("Int64")
        new_adv_df = new_adv_df.merge(hr_lookup, on=["MLBAM_ID", "Season", "Team", "Level"], how="left")
        new_adv_df["HR/FB"] = new_adv_df.apply(
            lambda r: round(r["HR"] / r["_fly_pop"], 3)
            if pd.notna(r["HR"]) and r["_fly_pop"] > 0 else None,
            axis=1,
        )
        new_adv_df = new_adv_df[[
            "PlayerId", "MLBAM_ID", "Season", "Name", "Team", "Level", "League",
            "Age", "PA",
            "AVG", "OBP", "SLG", "OPS", "ISO", "BABIP",
            "BB%", "K%",
            "GB%", "LD%", "FB%", "GB/FB", "IFFB%", "HR/FB", "SwStr%", "Whiff%", "Pitches",
        ]]
        adv_df = pd.concat([existing_adv, new_adv_df], ignore_index=True)
        adv_df.to_csv(ADVANCED_OUT, index=False)
        print(f"  Wrote {len(adv_df):,} rows -> {ADVANCED_OUT} "
              f"({len(existing_adv):,} existing + {len(new_adv_df):,} new {CURRENT_SEASON})")
    else:
        print("  No advanced data collected.")
        adv_df = existing_adv

    # --- Crosswalk summary ---
    total_players = hit_df["MLBAM_ID"].nunique()
    mlbam_only = hit_df[hit_df["PlayerId"] == hit_df["MLBAM_ID"].astype(str)]["MLBAM_ID"].nunique()
    fg_mapped = total_players - mlbam_only
    print(f"\nPlayerId crosswalk summary ({total_players:,} unique players):")
    print(f"  {fg_mapped:,} mapped to FanGraphs IDs")
    print(f"  {mlbam_only:,} using MLBAM ID only (no FanGraphs match)")
    if mlbam_only > 0:
        print(f"  -> These {mlbam_only:,} players likely have minimal MiLB track records")

    # --- Birth dates + positions (new rows only to minimise API calls) ---
    new_mlbam_ids = [int(i) for i in new_hit_df["MLBAM_ID"].dropna().unique()]

    print("\nFetching birth dates...")
    fetch_birth_dates(new_mlbam_ids)

    print("\nFetching positions...")
    fetch_positions(new_mlbam_ids)

    print("\nDone.")


if __name__ == "__main__":
    main()

