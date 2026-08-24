"""Merge per-worker game files into milb_pitches_games.csv and rebuild the agg.

Run after all fetch_pitches_worker.py processes have completed.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_milb_pitches import build_agg, load_chadwick, AGG_OUT, GAMES_OUT, API_DIR


def load_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, on_bad_lines="skip", low_memory=False)
    df["gamePk"]   = pd.to_numeric(df["gamePk"],   errors="coerce")
    df["Season"]   = pd.to_numeric(df["Season"],   errors="coerce")
    df["MLBAM_ID"] = pd.to_numeric(df["MLBAM_ID"], errors="coerce")
    df = df.dropna(subset=["gamePk", "Season", "Level", "MLBAM_ID"])
    df = df[df["Season"] > 2000]
    df["gamePk"]   = df["gamePk"].astype(int)
    df["Season"]   = df["Season"].astype(int)
    df["MLBAM_ID"] = df["MLBAM_ID"].astype("Int64")
    return df


def main() -> None:
    worker_files = sorted(API_DIR.glob("milb_pitches_games_w*.csv"))
    print(f"Worker files found: {len(worker_files)}")

    dfs = []

    if GAMES_OUT.exists():
        df_existing = load_clean(GAMES_OUT)
        print(f"  Existing games file: {len(df_existing):,} rows")
        dfs.append(df_existing)

    for wf in worker_files:
        df_w = load_clean(wf)
        print(f"  {wf.name}: {len(df_w):,} rows")
        dfs.append(df_w)

    if not dfs:
        print("No data found. Exiting.")
        sys.exit(1)

    merged = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows before dedup: {len(merged):,}")

    # Deduplicate on (gamePk, MLBAM_ID) — same logic as build_agg
    merged = merged.drop_duplicates(subset=["gamePk", "MLBAM_ID"])
    print(f"Total rows after dedup:  {len(merged):,}")

    merged.to_csv(GAMES_OUT, index=False)
    print(f"Wrote merged games file: {GAMES_OUT.name}")

    print("\nRebuilding milb_pitches_agg.csv...")
    chadwick = load_chadwick()
    agg = build_agg(merged, chadwick)
    agg.to_csv(AGG_OUT, index=False)
    print(f"  Wrote {len(agg):,} player-season-level rows -> {AGG_OUT.name}")
    seasons = sorted(merged["Season"].unique())
    print(f"  Seasons: {seasons[0]}-{seasons[-1]}")
    print(f"  Levels: {sorted(merged['Level'].unique())}")
    print(f"  Unique players: {merged['MLBAM_ID'].nunique():,}")
    print(f"\n  Season × Level breakdown:")
    print(agg.groupby(["Season", "Level"]).size().to_string())

    print("\nCleaning up worker files...")
    for wf in worker_files:
        wf.unlink()
        print(f"  Deleted {wf.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()

