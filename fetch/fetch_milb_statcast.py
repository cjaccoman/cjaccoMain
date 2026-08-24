"""Fetch AAA Statcast pitch-by-pitch data from Baseball Savant MiLB search.

Pulls 2023-present in 3-day chunks to stay under the 25k-row per-request cap.
Incremental: reads existing data and skips already-covered date ranges.

Output: data/api/milb_statcast_aaa.csv
  One row per pitch. Key columns kept (not all 119 raw columns).

Run standalone: python fetch_milb_statcast.py
"""

import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

DATA_DIR  = Path(__file__).resolve().parent.parent / "data"
OUT_PATH  = DATA_DIR / "api" / "milb_statcast_aaa.csv"
DONE_PATH = DATA_DIR / "api" / "milb_statcast_done.txt"  # dates processed but returned 0 rows

FIRST_SEASON = 2023   # first year all AAA parks had Statcast
LAST_SEASON  = 2026
CHUNK_DAYS   = 3      # days per API call (keeps rows < 25k cap)
SLEEP_SEC    = 1.5    # polite delay between requests

# Columns to keep from the 119-column raw response
KEEP_COLS = [
    "batter", "player_name", "game_date", "game_year",
    "stand", "p_throws",
    "on_1b", "on_2b", "on_3b", "outs_when_up", "inning",
    "balls", "strikes",
    "events", "description",
    "home_team", "away_team", "inning_topbot",
    "launch_speed", "launch_angle",
    "estimated_ba_using_speedangle",
    "estimated_woba_using_speedangle",
    "estimated_slg_using_speedangle",
    "woba_value", "woba_denom",
    "babip_value", "iso_value",
    "delta_run_exp",
    "bat_speed", "swing_length",
]

# Season date ranges (roughly April–September for most years)
SEASON_RANGES = {
    2023: (date(2023, 4, 4), date(2023, 9, 24)),
    2024: (date(2024, 4, 5), date(2024, 9, 29)),
    2025: (date(2025, 4, 4), date(2025, 9, 28)),
    2026: (date(2026, 4, 4), date.today()),
}


def date_chunks(start: date, end: date, chunk_days: int):
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=chunk_days - 1), end)
        yield cur, chunk_end
        cur = chunk_end + timedelta(days=1)


def main() -> None:
    try:
        import baseball_stats_python as bsp
    except ImportError:
        raise SystemExit("pip install baseball-stats-python")

    # Load already-covered dates from both existing data and the empty-response cache.
    # Chunks that returned 0 rows won't appear in the CSV but are recorded in DONE_PATH
    # so they don't get re-fetched on every run.
    fetched_dates: set[str] = set()
    if OUT_PATH.exists():
        existing = pd.read_csv(OUT_PATH, usecols=["game_date"])
        fetched_dates.update(existing["game_date"].dropna().unique())
    if DONE_PATH.exists():
        fetched_dates.update(d.strip() for d in DONE_PATH.read_text().splitlines() if d.strip())
    if fetched_dates:
        print(f"Existing: {len(fetched_dates)} unique dates already cached")
    else:
        print("No existing data — full pull")

    current_year = date.today().year
    seasons = [y for y in SEASON_RANGES if y <= min(current_year, LAST_SEASON)]

    chunks_needed = []
    for season, (s_start, s_end) in SEASON_RANGES.items():
        if season > min(current_year, LAST_SEASON):
            continue
        for chunk_start, chunk_end in date_chunks(s_start, s_end, CHUNK_DAYS):
            # Check if any date in this chunk is already fetched
            test_dates = {
                (chunk_start + timedelta(days=d)).isoformat()
                for d in range((chunk_end - chunk_start).days + 1)
            }
            if test_dates.issubset(fetched_dates):
                continue
            chunks_needed.append((chunk_start, chunk_end))

    print(f"Chunks to fetch: {len(chunks_needed)}")
    if not chunks_needed:
        print("Already up to date.")
        return

    new_frames = []
    empty_dates: list[str] = []  # dates that returned 0 rows — recorded to skip next run
    for i, (chunk_start, chunk_end) in enumerate(chunks_needed, 1):
        s_dt = chunk_start.isoformat()
        e_dt = chunk_end.isoformat()
        print(f"[{i}/{len(chunks_needed)}] {s_dt} -> {e_dt} ", end="", flush=True)
        chunk_dates = [
            (chunk_start + timedelta(days=d)).isoformat()
            for d in range((chunk_end - chunk_start).days + 1)
        ]
        try:
            df = bsp.minor_statcast_search(
                season=str(chunk_start.year),
                player_type="batter",
                level="AAA",
                start_dt=s_dt,
                end_dt=e_dt,
            )
            if len(df) == 25000:
                print(f"  WARNING: hit 25k cap — consider smaller chunks")
            if len(df) == 0:
                empty_dates.extend(chunk_dates)
                print("-> 0 rows (cached as done)")
            else:
                keep = [c for c in KEEP_COLS if c in df.columns]
                new_frames.append(df[keep])
                print(f"-> {len(df):,} rows")
        except Exception as exc:
            print(f"  ERROR: {exc}")
        time.sleep(SLEEP_SEC)

    # Persist empty-response dates so they're skipped on future runs
    if empty_dates:
        with open(DONE_PATH, "a") as f:
            f.write("\n".join(empty_dates) + "\n")

    if not new_frames:
        print("No new data fetched.")
        return

    new_data = pd.concat(new_frames, ignore_index=True)

    if OUT_PATH.exists() and fetched_dates:
        old = pd.read_csv(OUT_PATH)
        # Keep only columns that exist in both
        common_cols = [c for c in KEEP_COLS if c in old.columns and c in new_data.columns]
        combined = pd.concat([old[common_cols], new_data[common_cols]], ignore_index=True)
        combined = combined.drop_duplicates()
    else:
        combined = new_data

    combined = combined.sort_values(["game_year", "game_date", "batter"], na_position="last")
    combined.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(combined):,} rows -> {OUT_PATH.name}")
    print(f"Seasons covered: {sorted(combined['game_year'].dropna().unique().astype(int))}")


if __name__ == "__main__":
    main()

