"""Assign traditional player profile archetypes to player_comps.csv.

Three fundamental dimensions — Power, Speed, Discipline — each built by
combining the corresponding TOOLS and ABILITY metrics from the v1.0 model:

  Power      = 0.5 × HRFB  + 0.5 × PullAir
               (TOOLS Raw Power + ABILITY Game Power, same two sources)

  Speed      = SBTalent
               (ABILITY SB Talent; Sprint Speed not available for full history)

  Discipline = BB2K  (= BB% − 2×K%)
               (ABILITY Discipline composite; already merges both sides)

Each dimension is a PA+level-discount weighted career average, then
z-scored within the 17,558-player pool. Profiles are assigned by rule:

  Power_z ≥ 0.75  →  Power flag
  Speed_z ≥ 0.75  →  Speed flag
  Disc_z  ≥ 0.75  →  Discipline flag

  Power + Speed + Discipline  → Five-Tool
  Power + Speed               → Power + Speed
  Power + Discipline          → Disciplined Power
  Power only                  → Power  (sub-label: Free Swinger if K_z > 0.75)
  Speed + Discipline          → Speed / Patient
  Speed only                  → Speed
  Discipline only             → Disciplined
  Kpct_z ≥ 0.75, no flags    → Swing & Miss
  nothing above threshold     → Contact

Usage:
  python analysis/build_comp_profiles.py
  python analysis/build_comp_profiles.py --threshold 0.6   # looser flags
  python analysis/build_comp_profiles.py --show            # print dimension stats
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

DATA  = Path(__file__).resolve().parent.parent / "data"
OUT   = DATA / "computed" / "player_comps.csv"

LEVELS         = ["AAA", "AA", "A+", "A", "R"]
LEVEL_DISCOUNT = {"AAA": 1.00, "AA": 0.59, "A+": 0.34, "A": 0.23, "R": 0.10}
MIN_PA         = 80

DEFAULT_THRESHOLD = 0.75   # z-score to flag a dimension as "notable"
HIGH_K_THRESHOLD  = 0.75   # K_z above this adds "Free Swinger" sub-label


def career_composite(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """PA + level-discount weighted career average for each metric."""
    lk_list  = [l.replace("+", "plus") for l in LEVELS]
    discounts = {l.replace("+", "plus"): LEVEL_DISCOUNT[l] for l in LEVELS}

    rows = []
    for _, row in df.iterrows():
        rec: dict = {"PlayerId": row["PlayerId"], "Name": row["Name"]}
        for m in metrics:
            num, den = 0.0, 0.0
            for lk in lk_list:
                pa  = row.get(f"PA_{lk}")
                val = row.get(f"{m}_{lk}")
                if pd.isna(pa) or pa < MIN_PA or pd.isna(val):
                    continue
                wt   = pa * discounts[lk]
                num  += wt * val
                den  += wt
            rec[m] = num / den if den > 0 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def assign_profile(power_z: float, speed_z: float, disc_z: float,
                   kpct_z: float, threshold: float) -> str:
    p = power_z >= threshold
    s = speed_z >= threshold
    d = disc_z  >= threshold

    if p and s and d:
        return "Five-Tool"
    if p and s:
        return "Power + Speed"
    if p and d:
        return "Disciplined Power"
    if p:
        return "Power / Free Swinger" if kpct_z >= HIGH_K_THRESHOLD else "Power"
    if s and d:
        return "Speed / Patient"
    if s:
        return "Speed"
    if d:
        return "Disciplined"
    if kpct_z >= HIGH_K_THRESHOLD:
        return "Swing & Miss"
    return "Contact"


def run(threshold: float = DEFAULT_THRESHOLD, show: bool = False) -> pd.DataFrame:
    df = pd.read_csv(OUT)
    print(f"Loaded {len(df):,} rows from {OUT}")

    # Raw metrics needed
    raw_metrics = ["HRFB", "PullAir", "SBTalent", "BB2K", "Kpct"]
    comp = career_composite(df, raw_metrics)

    # Z-score each metric within the pool
    for m in raw_metrics:
        mu  = comp[m].mean()
        sig = comp[m].std()
        comp[f"{m}_z"] = (comp[m] - mu) / sig if sig > 1e-9 else 0.0

    # Combine into three dimensions
    comp["Power_z"]     = 0.5 * comp["HRFB_z"] + 0.5 * comp["PullAir_z"]
    comp["Speed_z"]     = comp["SBTalent_z"]
    comp["Disc_z"]      = comp["BB2K_z"]   # BB% − 2×K% already merges both

    # Assign profiles
    def _profile(row):
        if any(pd.isna(row[c]) for c in ["Power_z", "Speed_z", "Disc_z", "Kpct_z"]):
            return np.nan
        return assign_profile(
            row["Power_z"], row["Speed_z"], row["Disc_z"], row["Kpct_z"], threshold
        )

    comp["Profile"] = comp.apply(_profile, axis=1)

    if show:
        print(f"\nDimension stats (pool means / stds):")
        for m in ["HRFB", "PullAir", "SBTalent", "BB2K", "Kpct"]:
            print(f"  {m}: mean={comp[m].mean():.4f}  std={comp[m].std():.4f}")
        print(f"\nProfile counts (threshold z≥{threshold}):")
        print(comp["Profile"].value_counts().to_string())
        print()

    # Merge back into main df
    if "Profile" in df.columns:
        df = df.drop(columns=["Profile"])
    df = df.merge(comp[["PlayerId", "Profile"]], on="PlayerId", how="left")

    cols = [c for c in df.columns if c != "Profile"]
    cols.insert(cols.index("Name") + 1, "Profile")
    df = df[cols]

    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df):,} rows → {OUT}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--show",      action="store_true")
    args = parser.parse_args()
    run(threshold=args.threshold, show=args.show)
