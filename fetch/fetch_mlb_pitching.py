"""Fetch MLB pitching stats from the MLB Stats API (2006–current).

Mirrors fetch_mlb_data.py for the pitching group.  Output is used by
build_pitcher_scores.py to apply the same graduation filter as hitters
(career MLB IP >= 50 → excluded from prospect pool).

Output:
  data/historical/hist_mlb_pitching.parquet
    Season, Name, Team, G, GS, IP, K, BB, ER, HRA, W, L, SV, HLD, BS, CG, SHO, QS,
    PlayerId, MLBAMID

  Note: RW/RL fetched from statSplits?sitCodes=rp (exact W/L in relief appearances).
  Falls back to proration (round(W/L × (G-GS)/G)) if the split endpoint returns
  no data for a player.

Usage:
  python fetch/fetch_mlb_pitching.py           # incremental — current season only
  python fetch/fetch_mlb_pitching.py --full    # full refetch 2006–current
"""

import argparse
import json
import time
import unicodedata
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR       = Path(__file__).resolve().parent.parent / "data"
API_DIR        = DATA_DIR / "api"
HIST_DIR       = DATA_DIR / "historical"
CHADWICK_CACHE = API_DIR / "chadwick.csv"
OUT_PATH       = HIST_DIR / "hist_mlb_pitching.parquet"

BASE_URL       = "https://statsapi.mlb.com/api/v1"
MLB_SPORT_ID   = 1
SEASONS        = list(range(2006, 2027))
CURRENT_SEASON = SEASONS[-1]
CALL_DELAY     = 0.3
STATS_LIMIT    = 5000


def _get(url: str, retries: int = 2) -> dict | None:
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "prospectsMain/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except Exception as e:
            if attempt == retries:
                print(f"    WARN: failed {url[:80]}: {e}")
                return None
            time.sleep(1 + attempt)
    return None


def _api(path: str, **params) -> dict | None:
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{BASE_URL}/{path}?{qs}" if qs else f"{BASE_URL}/{path}"
    time.sleep(CALL_DELAY)
    return _get(url)


def load_chadwick() -> pd.DataFrame:
    if not CHADWICK_CACHE.exists():
        print("  WARNING: chadwick.csv not found — PlayerId will fall back to MLBAM ID.")
        return pd.DataFrame(columns=["key_mlbam", "key_fangraphs"])
    ck = pd.read_csv(
        CHADWICK_CACHE,
        usecols=["key_mlbam", "key_fangraphs"],
        dtype={"key_mlbam": "Int64", "key_fangraphs": "Int64"},
    ).dropna(subset=["key_mlbam"])
    print(f"  Chadwick: {len(ck):,} players, {ck['key_fangraphs'].notna().sum():,} with FG IDs")
    return ck


def apply_crosswalk(df: pd.DataFrame, chadwick: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MLBAMID"] = pd.to_numeric(df["MLBAMID"], errors="coerce").astype("Int64")
    merged = df.merge(chadwick, left_on="MLBAMID", right_on="key_mlbam", how="left")
    fg = merged["key_fangraphs"]
    mlbam = merged["MLBAMID"]
    merged["PlayerId"] = fg.where(fg.notna(), mlbam).apply(
        lambda x: str(int(x)) if pd.notna(x) else None
    )
    return merged.drop(columns=["key_mlbam", "key_fangraphs"], errors="ignore")


def _int(v) -> int:
    try:
        return int(v) if v is not None else 0
    except (ValueError, TypeError):
        return 0


def _ip(v) -> float:
    """Parse MLB API IP string: '45.2' means 45 full innings + 2 outs = 45.667."""
    if v is None:
        return 0.0
    try:
        s = str(v)
        if "." in s:
            full, outs = s.split(".", 1)
            return int(full) + int(outs) / 3
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def fetch_season(season: int) -> list[dict]:
    data = _api(
        "stats",
        stats="season",
        playerPool="all",
        group="pitching",
        gameType="R",
        season=season,
        sportId=MLB_SPORT_ID,
        limit=STATS_LIMIT,
    )
    if not data:
        return []
    splits = []
    for block in data.get("stats", []):
        splits.extend(block.get("splits", []))
    return splits


def fetch_season_advanced(season: int) -> dict[int, int]:
    """Fetch QS from seasonAdvanced endpoint. Returns {mlbam_id: qs}."""
    data = _api(
        "stats",
        stats="seasonAdvanced",
        playerPool="all",
        group="pitching",
        gameType="R",
        season=season,
        sportId=MLB_SPORT_ID,
        limit=STATS_LIMIT,
    )
    if not data:
        return {}
    qs_map: dict[int, int] = {}
    for block in data.get("stats", []):
        for sp in block.get("splits", []):
            pid = sp.get("player", {}).get("id")
            qs  = sp.get("stat", {}).get("qualityStarts")
            if pid is not None and qs is not None:
                qs_map[int(pid)] = int(qs)
    return qs_map


def fetch_season_rp_splits(season: int) -> dict[int, tuple[int, int]]:
    """Fetch exact W/L in relief appearances via statSplits sitCodes=rp.
    Returns {mlbam_id: (rw, rl)}.
    """
    data = _api(
        "stats",
        stats="statSplits",
        playerPool="all",
        group="pitching",
        gameType="R",
        season=season,
        sportId=MLB_SPORT_ID,
        sitCodes="rp",
        limit=STATS_LIMIT,
    )
    if not data:
        return {}
    rp_map: dict[int, tuple[int, int]] = {}
    for block in data.get("stats", []):
        for sp in block.get("splits", []):
            pid = sp.get("player", {}).get("id")
            if pid is None:
                continue
            stat = sp.get("stat", {})
            rw = _int(stat.get("wins"))
            rl = _int(stat.get("losses"))
            rp_map[int(pid)] = (rw, rl)
    return rp_map


def parse_splits(splits: list, season: int,
                 qs_map: dict[int, int] | None = None,
                 rp_map: dict[int, tuple[int, int]] | None = None) -> list[dict]:
    rows = []
    for sp in splits:
        stat   = sp.get("stat", {})
        player = sp.get("player", {})
        team   = sp.get("team", {})
        mlbam  = player.get("id")
        g  = _int(stat.get("gamesPlayed"))
        gs = _int(stat.get("gamesStarted"))
        w  = _int(stat.get("wins"))
        l  = _int(stat.get("losses"))
        # RW/RL: use exact relief-split data when available; prorate as fallback
        if rp_map is not None and mlbam and int(mlbam) in rp_map:
            rw, rl = rp_map[int(mlbam)]
        else:
            relief_share = (g - gs) / g if g > 0 else 0.0
            rw = round(w * relief_share)
            rl = round(l * relief_share)
        rows.append({
            "MLBAMID": mlbam,
            "Season":  season,
            "Name":    player.get("fullName", ""),
            "Team":    team.get("abbreviation", "UNK"),
            "G":       g,
            "GS":      gs,
            "IP":      _ip(stat.get("inningsPitched")),
            "K":       _int(stat.get("strikeOuts")),
            "BB":      _int(stat.get("baseOnBalls")),
            "ER":      _int(stat.get("earnedRuns")),
            "HRA":     _int(stat.get("homeRuns")),
            "W":       w,
            "L":       l,
            "SV":      _int(stat.get("saves")),
            "HLD":     _int(stat.get("holds")),
            "BS":      _int(stat.get("blownSaves")),
            "CG":      _int(stat.get("completeGames")),
            "SHO":     _int(stat.get("shutouts")),
            "QS":      (qs_map or {}).get(int(mlbam), 0) if mlbam else 0,
            "RW":      rw,
            "RL":      rl,
        })
    return rows


def _norm_name(name: str) -> str:
    return unicodedata.normalize("NFD", name).encode("ascii", "ignore").decode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Refetch all seasons 2006-current")
    args = parser.parse_args()

    print("=== fetch_mlb_pitching.py ===\n")
    chadwick = load_chadwick()

    full_mode = args.full or not OUT_PATH.exists()
    if full_mode:
        print(f"Full mode — fetching {SEASONS[0]}–{CURRENT_SEASON}")
        existing = pd.DataFrame()
        fetch_seasons = SEASONS
    else:
        print(f"Incremental mode — refreshing {CURRENT_SEASON} only")
        existing = pd.read_parquet(OUT_PATH)
        existing["PlayerId"] = pd.to_numeric(existing["PlayerId"], errors="coerce")
        existing = existing[existing["Season"] != CURRENT_SEASON]
        fetch_seasons = [CURRENT_SEASON]

    all_rows: list[dict] = []
    total = len(fetch_seasons)
    print(f"\nFetching {total} season(s)\n")
    for i, season in enumerate(fetch_seasons, 1):
        print(f"[{i:3}/{total}] {season}", end="  ")
        splits = fetch_season(season)
        qs_map = fetch_season_advanced(season)
        rp_map = fetch_season_rp_splits(season)
        if splits:
            rows = parse_splits(splits, season, qs_map, rp_map)
            rows = [r for r in rows if r["IP"] > 0]
            all_rows.extend(rows)
            print(f"{len(rows):4} pitchers  (QS: {len(qs_map)}, RP splits: {len(rp_map)})")
        else:
            print("no data")

    if not all_rows:
        print("Nothing new to write.")
        return

    new_df = pd.DataFrame(all_rows)
    new_df["NameASCII"] = new_df["Name"].apply(_norm_name)
    new_df = apply_crosswalk(new_df, chadwick)

    combined = pd.concat([existing, new_df], ignore_index=True)
    if "PlayerId" in combined.columns:
        combined["PlayerId"] = pd.to_numeric(combined["PlayerId"], errors="coerce")
    if "MLBAMID" in combined.columns:
        combined["MLBAMID"] = combined["MLBAMID"].astype(str)

    HIST_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {len(combined):,} rows -> {OUT_PATH}")
    seasons = sorted(combined["Season"].unique())
    print(f"Seasons: {seasons[0]}–{seasons[-1]}")
    print(f"Career IP summary (top 10):")
    career = combined.groupby("PlayerId")["IP"].sum().nlargest(10)
    names = combined.drop_duplicates("PlayerId").set_index("PlayerId")["Name"]
    for pid, ip in career.items():
        print(f"  {names.get(pid,'?'):25s} {ip:.0f} IP")


if __name__ == "__main__":
    main()
