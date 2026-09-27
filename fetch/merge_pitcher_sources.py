"""Consolidate pitcher data sources into two wide files.

MiLB combined  (data/api/milb_pitching_combined.parquet):
  milb_pitching.csv (counting stats)
  + milb_pitching_advanced.csv (rate stats)   joined on PlayerId+Season+Level
  + prospect_savant_pitchers.csv (PS metrics) joined on MLBAM_ID+Season+Level

MLB combined   (data/historical/hist_mlb_pitching_combined.parquet):
  hist_mlb_pitching.parquet (counting stats)
  + mlb_statcast_pitchers.csv (Statcast)      joined on MLBAM_ID+Season

Usage:
  python fetch/merge_pitcher_sources.py
"""

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ── Input paths ───────────────────────────────────────────────────────────────
MILB_CNT  = DATA_DIR / "api"        / "milb_pitching.csv"
MILB_ADV  = DATA_DIR / "api"        / "milb_pitching_advanced.csv"
PS_PIT    = DATA_DIR / "prospectSavant" / "prospect_savant_pitchers.csv"
MLB_CNT   = DATA_DIR / "historical" / "hist_mlb_pitching.parquet"
MLB_SC    = DATA_DIR / "api"        / "mlb_statcast_pitchers.csv"

# ── Output paths ──────────────────────────────────────────────────────────────
MILB_OUT  = DATA_DIR / "api"        / "milb_pitching_combined.parquet"
MLB_OUT   = DATA_DIR / "historical" / "hist_mlb_pitching_combined.parquet"

KEY_MILB = ["PlayerId", "Season", "Level"]
KEY_MLB  = ["MLBAM_ID", "Season"]


def build_milb() -> pd.DataFrame:
    print("Building MiLB combined...")

    cnt = pd.read_csv(MILB_CNT, low_memory=False)
    adv = pd.read_csv(MILB_ADV, low_memory=False)

    # Rate-only cols from advanced (drop columns already in counting)
    adv_rate_cols = [c for c in adv.columns
                     if c not in KEY_MILB + ["MLBAM_ID", "Name", "Team", "League", "Age", "BF"]]
    adv_slim = adv[KEY_MILB + adv_rate_cols]

    base = cnt.merge(adv_slim, on=KEY_MILB, how="left")
    print(f"  After cnt+adv merge: {len(base):,} rows")

    # PS pitcher data — join on MLBAM_ID + Season + Level
    ps = pd.read_csv(PS_PIT, low_memory=False)
    ps = ps.rename(columns={"MLBAMId": "MLBAM_ID"})
    ps["MLBAM_ID"] = pd.to_numeric(ps["MLBAM_ID"], errors="coerce")

    # Columns to bring in from PS (exclude identity/overlap cols already in base)
    ps_skip = {"Name", "Org", "MinorMasterId", "Age", "AgeDays", "Season", "Level",
               "IP", "K%", "BB%", "K-BB%", "Whiff%", "GB%", "FB%", "LD%"}
    ps_keep = [c for c in ps.columns if c not in ps_skip]
    ps_slim = ps[["MLBAM_ID", "Season", "Level"] + [c for c in ps_keep
                                                      if c not in ("MLBAM_ID", "Season", "Level")]]
    # One PS row per player-season-level (aggregate if duplicates from split stints)
    ps_slim = ps_slim.groupby(["MLBAM_ID", "Season", "Level"], as_index=False).first()

    base["MLBAM_ID"] = pd.to_numeric(base["MLBAM_ID"], errors="coerce")
    base = base.merge(ps_slim, on=["MLBAM_ID", "Season", "Level"], how="left")
    print(f"  After PS merge: {len(base):,} rows")
    ps_2026 = base[(base.Season == 2026) & base["MaxVelo"].notna()]
    print(f"  PS coverage 2026: {len(ps_2026):,} rows with MaxVelo")

    return base


def build_mlb() -> pd.DataFrame:
    print("Building MLB combined...")

    cnt = pd.read_parquet(MLB_CNT)
    cnt = cnt.rename(columns={"MLBAMID": "MLBAM_ID"})
    cnt["MLBAM_ID"] = pd.to_numeric(cnt["MLBAM_ID"], errors="coerce")

    sc = pd.read_csv(MLB_SC, low_memory=False)
    sc["MLBAM_ID"] = pd.to_numeric(sc["MLBAM_ID"], errors="coerce")

    # Drop duplicate Name from statcast (keep base version)
    sc_cols = [c for c in sc.columns if c not in ("Name",)]
    sc_slim = sc[sc_cols]

    base = cnt.merge(sc_slim, on=KEY_MLB, how="left")
    print(f"  After cnt+statcast merge: {len(base):,} rows")
    sc_2026 = base[(base.Season == 2026) & base["ERA"].notna()]
    print(f"  Statcast coverage 2026: {len(sc_2026):,} rows with ERA")

    return base


def main() -> None:
    print("=== merge_pitcher_sources.py ===\n")

    milb = build_milb()
    milb.to_parquet(MILB_OUT, index=False)
    print(f"  Wrote {len(milb):,} rows -> {MILB_OUT.name}")
    print(f"  Columns ({len(milb.columns)}): {list(milb.columns)}\n")

    mlb = build_mlb()
    mlb.to_parquet(MLB_OUT, index=False)
    print(f"  Wrote {len(mlb):,} rows -> {MLB_OUT.name}")
    print(f"  Columns ({len(mlb.columns)}): {list(mlb.columns)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
