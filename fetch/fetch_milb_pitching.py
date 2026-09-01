"""Fetch MiLB pitching stats from the MLB Stats API (2006–2026).

Outputs written to data/api/:
  milb_pitching.csv          — counting stats per player-season-level
  milb_pitching_advanced.csv — rate + batted-ball stats per player-season-level

PlayerId assignment mirrors fetch_milb_data.py (Chadwick crosswalk).
"""

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
API_DIR = DATA_DIR / "api"
API_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://statsapi.mlb.com/api/v1"
CHADWICK_CACHE = API_DIR / "chadwick.csv"

SPORT_IDS = {"AAA": 11, "AA": 12, "A+": 13, "A": 14, "R": 16}
LEVEL_FOR_SPORT = {v: k for k, v in SPORT_IDS.items()}

SEASONS = list(range(2006, 2027))
CURRENT_SEASON = SEASONS[-1]
MIN_IP = 10.0
MAX_AGE = 26
CALL_DELAY = 0.3
STATS_LIMIT = 5000
BIRTH_BATCH = 50

PITCHING_OUT  = API_DIR / "milb_pitching.csv"
ADVANCED_OUT  = API_DIR / "milb_pitching_advanced.csv"
BIRTHDATE_OUT = API_DIR / "player_birthdays.csv"
POSITION_OUT  = API_DIR / "player_positions.csv"


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


def parse_ip(ip_str) -> float | None:
    """Parse '45.2' → 45.667 (decimal part = outs recorded, 0–2)."""
    try:
        s = str(ip_str)
        if "." in s:
            whole, frac = s.split(".", 1)
            return int(whole) + int(frac) / 3
        return float(s)
    except Exception:
        return None


def load_chadwick() -> pd.DataFrame:
    if not CHADWICK_CACHE.exists():
        print("  WARN: chadwick.csv not found — run fetch_milb_data.py first to build it.")
        return pd.DataFrame(columns=["key_mlbam", "key_fangraphs"])
    ck = pd.read_csv(
        CHADWICK_CACHE,
        usecols=["key_mlbam", "key_fangraphs"],
        dtype={"key_mlbam": "Int64", "key_fangraphs": "Int64"},
    ).dropna(subset=["key_mlbam"])
    print(f"  Chadwick (cached): {len(ck):,} players, "
          f"{ck['key_fangraphs'].notna().sum():,} with FanGraphs IDs")
    return ck


def build_mlb_org_lookup() -> dict[int, str]:
    data = _api("teams", sportId=1, season=2023)
    if not data:
        return {}
    return {t["id"]: t.get("abbreviation", "UNK") for t in data.get("teams", [])}


def build_team_lookup(sport_id: int, season: int, mlb_org: dict) -> dict:
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
    data = _api("leagues", sportId=sport_id, season=season)
    if not data:
        return {}
    return {lg["id"]: lg.get("name", "DiscLeague") for lg in data.get("leagues", [])}


def fetch_splits(stat_type: str, sport_id: int, season: int) -> list:
    data = _api(
        "stats",
        stats=stat_type,
        playerPool="all",
        group="pitching",
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


def parse_pitching_splits(splits: list, team_lkp: dict, league_lkp: dict) -> list[dict]:
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

        ip = parse_ip(stat.get("inningsPitched"))

        rows.append({
            "MLBAM_ID": player.get("id"),
            "Season": _int(sp.get("season")),
            "Name": player.get("fullName", ""),
            "Team": team_info.get("org_abbrev", "UNK"),
            "Level": LEVEL_FOR_SPORT.get(sport_id, "?"),
            "League": league_lkp.get(league_id, "DiscLeague"),
            "Age": _int(stat.get("age")),
            "G": _int(stat.get("gamesPlayed")),
            "GS": _int(stat.get("gamesStarted")),
            "IP": ip,
            "ER": _int(stat.get("earnedRuns")),
            "K": _int(stat.get("strikeOuts")),
            "BB": _int(stat.get("baseOnBalls")),
            "IBB": _int(stat.get("intentionalWalks")),
            "HRA": _int(stat.get("homeRuns")),
            "H": _int(stat.get("hits")),
            "HBP": _int(stat.get("hitByPitch")),
            "BF": _int(stat.get("battersFaced")),
            "W": _int(stat.get("wins")),
            "L": _int(stat.get("losses")),
            "SV": _int(stat.get("saves")),
            "SVO": _int(stat.get("saveOpportunities")),
            "HLD": _int(stat.get("holds")),
            "BS": _int(stat.get("blownSaves")),
            "CG": _int(stat.get("completeGames")),
            "SHO": _int(stat.get("shutouts")),
            "WP": _int(stat.get("wildPitches")),
        })
    return rows


def parse_advanced_splits(splits: list, team_lkp: dict, league_lkp: dict) -> list[dict]:
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

        # Batted ball components for GB/FB
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

        # K%, BB%, Whiff% for pitchers (opponent-based)
        k_pct = _float(stat.get("strikeoutsPerPlateAppearance"))
        bb_pct = _float(stat.get("walksPerPlateAppearance"))
        k_bb_pct = _float(stat.get("strikeoutsMinusWalksPercentage"))
        whiff = _float(stat.get("whiffPercentage"))
        babip = _float(stat.get("babip"))
        qs = _int(stat.get("qualityStarts"))

        rows.append({
            "MLBAM_ID": player.get("id"),
            "Season": _int(sp.get("season")),
            "Name": player.get("fullName", ""),
            "Team": team_info.get("org_abbrev", "UNK"),
            "Level": LEVEL_FOR_SPORT.get(sport_id, "?"),
            "League": league_lkp.get(league_id, "DiscLeague"),
            "Age": _int(stat.get("age")),
            "BF": _int(stat.get("battersFaced")),
            "K%": k_pct,
            "BB%": bb_pct,
            "K-BB%": k_bb_pct,
            "Whiff%": whiff,
            "BABIP": babip,
            "QS": qs,
            "GB%": gb_pct,
            "LD%": ld_pct,
            "FB%": fb_pct,
            "GB/FB": gb_fb,
        })
    return rows


def apply_crosswalk(df: pd.DataFrame, chadwick: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MLBAM_ID"] = pd.to_numeric(df["MLBAM_ID"], errors="coerce").astype("Int64")
    merged = df.merge(chadwick, left_on="MLBAM_ID", right_on="key_mlbam", how="left")
    fg = merged["key_fangraphs"]
    mlbam = merged["MLBAM_ID"]
    merged["PlayerId"] = fg.where(fg.notna(), mlbam).apply(
        lambda x: str(int(x)) if pd.notna(x) else None
    )
    return merged.drop(columns=["key_mlbam", "key_fangraphs"], errors="ignore")


def fetch_birth_dates(mlbam_ids: list[int]) -> None:
    existing: dict[int, str] = {}
    if BIRTHDATE_OUT.exists():
        cached = pd.read_csv(BIRTHDATE_OUT, dtype={"MLBAM_ID": int, "BirthDate": str})
        existing = dict(zip(cached["MLBAM_ID"], cached["BirthDate"]))

    to_fetch = [i for i in mlbam_ids if i not in existing]
    if not to_fetch:
        print(f"  Birth dates: all {len(existing)} already cached")
        return

    print(f"  Birth dates: fetching {len(to_fetch)} new IDs ...")
    results = dict(existing)
    batches = [to_fetch[i:i + BIRTH_BATCH] for i in range(0, len(to_fetch), BIRTH_BATCH)]
    for idx, batch in enumerate(batches, 1):
        ids_str = ",".join(str(i) for i in batch)
        url = f"{BASE_URL}/people?personIds={ids_str}&fields=people,id,birthDate"
        data = _get(url)
        if data:
            for person in data.get("people", []):
                pid = person.get("id")
                bd = person.get("birthDate")
                if pid and bd:
                    results[int(pid)] = bd
        if idx < len(batches):
            time.sleep(CALL_DELAY)

    rows = [{"MLBAM_ID": k, "BirthDate": v} for k, v in sorted(results.items())]
    pd.DataFrame(rows).to_csv(BIRTHDATE_OUT, index=False)
    print(f"  Birth dates: {len(results)} total cached")


def fetch_positions(mlbam_ids: list[int]) -> None:
    existing: dict[int, str] = {}
    if POSITION_OUT.exists():
        cached = pd.read_csv(POSITION_OUT, dtype={"MLBAM_ID": int, "Position": str})
        existing = dict(zip(cached["MLBAM_ID"], cached["Position"]))

    to_fetch = [i for i in mlbam_ids if i not in existing]
    if not to_fetch:
        print(f"  Positions: all {len(existing)} already cached")
        return

    print(f"  Positions: fetching {len(to_fetch)} new IDs ...")
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
    print(f"  Positions: {len(results)} total cached")


def main() -> None:
    print("=== fetch_milb_pitching.py ===\n")

    print("Step 1: Chadwick register")
    chadwick = load_chadwick()

    print("\nStep 2: MLB org abbreviations")
    mlb_org = build_mlb_org_lookup()
    print(f"  {len(mlb_org)} MLB orgs")

    incremental = PITCHING_OUT.exists() and ADVANCED_OUT.exists()
    if incremental:
        print(f"\nExisting data found — refreshing {CURRENT_SEASON} only "
              f"(IP >= {MIN_IP}, Age <= {MAX_AGE})")
        existing_pit = pd.read_csv(PITCHING_OUT, dtype={"PlayerId": str, "MLBAM_ID": str})
        existing_adv = pd.read_csv(ADVANCED_OUT, dtype={"PlayerId": str, "MLBAM_ID": str})
        existing_pit = existing_pit[
            (existing_pit["Season"] != CURRENT_SEASON) &
            (existing_pit["Age"].fillna(99) <= MAX_AGE)
        ]
        existing_adv = existing_adv[
            (existing_adv["Season"] != CURRENT_SEASON) &
            (existing_adv["Age"].fillna(99) <= MAX_AGE)
        ]
        fetch_seasons = [CURRENT_SEASON]
    else:
        print(f"\nNo existing data — fetching full history "
              f"{SEASONS[0]}-{SEASONS[-1]} (IP >= {MIN_IP}, Age <= {MAX_AGE})")
        existing_pit = pd.DataFrame()
        existing_adv = pd.DataFrame()
        fetch_seasons = SEASONS

    all_pitching: list[dict] = []
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

            # Counting stats
            splits = fetch_splits("season", sport_id, season)
            if splits:
                rows = parse_pitching_splits(splits, team_lkp, league_lkp)
                before = len(rows)
                rows = [r for r in rows
                        if (r.get("IP") or 0) >= MIN_IP
                        and (r.get("Age") or 99) <= MAX_AGE]
                all_pitching.extend(rows)
                print(f"{tag}  pitching: {len(rows):4} rows "
                      f"({before - len(rows)} dropped <{MIN_IP} IP or >{MAX_AGE} Age)")
            else:
                print(f"{tag}  pitching: no data")

            # Advanced / rate stats
            adv_splits = fetch_splits("seasonAdvanced", sport_id, season)
            if adv_splits:
                rows = parse_advanced_splits(adv_splits, team_lkp, league_lkp)
                rows = [r for r in rows
                        if (r.get("BF") or 0) >= 30
                        and (r.get("Age") or 99) <= MAX_AGE]
                all_advanced.extend(rows)
            else:
                print(f"{tag}  advanced: no data")

    print(f"\nFetched: {len(all_pitching):,} new pitching rows, {len(all_advanced):,} new advanced rows")

    # Save pitching counting stats
    print("\nBuilding milb_pitching.csv...")
    new_pit_df = pd.DataFrame(all_pitching)
    new_pit_df = apply_crosswalk(new_pit_df, chadwick)
    new_pit_df.loc[new_pit_df["Team"] == "MEX", "Level"] = "R"
    new_pit_df = new_pit_df[[
        "PlayerId", "MLBAM_ID", "Season", "Name", "Team", "Level", "League",
        "Age", "G", "GS", "IP", "ER", "K", "BB", "IBB", "HRA",
        "H", "HBP", "BF", "W", "L", "SV", "SVO", "HLD", "BS", "CG", "SHO", "WP",
    ]]
    pit_df = pd.concat([existing_pit, new_pit_df], ignore_index=True)
    pit_df.to_csv(PITCHING_OUT, index=False)
    print(f"  Wrote {len(pit_df):,} rows -> {PITCHING_OUT}")

    # Save advanced stats
    print("Building milb_pitching_advanced.csv...")
    new_adv_df = pd.DataFrame(all_advanced)
    if not new_adv_df.empty:
        new_adv_df = apply_crosswalk(new_adv_df, chadwick)
        new_adv_df.loc[new_adv_df["Team"] == "MEX", "Level"] = "R"
        new_adv_df = new_adv_df[[
            "PlayerId", "MLBAM_ID", "Season", "Name", "Team", "Level", "League",
            "Age", "BF", "K%", "BB%", "K-BB%", "Whiff%", "BABIP",
            "QS", "GB%", "LD%", "FB%", "GB/FB",
        ]]
        adv_df = pd.concat([existing_adv, new_adv_df], ignore_index=True)
        adv_df.to_csv(ADVANCED_OUT, index=False)
        print(f"  Wrote {len(adv_df):,} rows -> {ADVANCED_OUT}")

    # Birth dates + positions for new players
    new_mlbam_ids = [int(i) for i in new_pit_df["MLBAM_ID"].dropna().unique()]
    print("\nFetching birth dates...")
    fetch_birth_dates(new_mlbam_ids)
    print("\nFetching positions...")
    fetch_positions(new_mlbam_ids)

    print("\nDone.")


if __name__ == "__main__":
    main()
