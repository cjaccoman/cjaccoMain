"""Add EraK%, EraHR/FB, EraPPPA, EraSB columns to prospect_features.csv.

Era breakpoints are taken from overhaul02.md — derived via binary segmentation on
year-by-year MLB medians (PA >= 100, N~430/season). MLB-derived breaks are preferred
over MiLB-derived breaks because:
  - They reflect real structural changes (rule changes, ball specs)
  - They avoid MiLB measurement artifacts (e.g. Hawk-Eye intro causing HR/FB spike in 2021)
  - Validated on a larger, cleaner dataset than the MiLB prospect pool

MLB-derived breakpoints (from overhaul02.md):
  K%    : 2015         (18.1% pre / 22.2% post mean)
  HR/FB : 2016, 2022   (9.0% / 12.8% / 11.0% mean by era)
  PPPA  : 2010, 2021   (0.850 / 0.740 / 0.687 mean by era)
  SB/PA : 2023         (0.77% pre / 1.14% post mean)
"""

import pandas as pd
from pathlib import Path

DATA_DIR    = Path(__file__).resolve().parent.parent / "data"
FEATURES_IN = DATA_DIR / "rankings" / "prospect_features.csv"

# ---------------------------------------------------------------------------
# Era definitions — MLB-derived (overhaul02.md)
# ---------------------------------------------------------------------------

ERA_DEFS = {
    "EraK%": {
        "breaks": [2015],
        "names":  ["Contact Era", "High-K Era"],
    },
    "EraHR/FB": {
        "breaks": [2016, 2022],
        "names":  ["Pre-Launch Angle", "Launch Angle Era", "Post-Deadening"],
    },
    "EraPPPA": {
        "breaks": [2010, 2021],
        "names":  ["Early Offensive Era", "Standard Era", "Modern Era"],
    },
    "EraSB": {
        "breaks": [2023],
        "names":  ["Pre-Pitch Clock", "Pitch Clock Era"],
    },
}


def assign_era(season: pd.Series, breaks: list[int], names: list[str]) -> pd.Series:
    out = pd.Series(names[0], index=season.index, dtype="object")
    for brk, name in zip(breaks, names[1:]):
        out[season >= brk] = name
    return out


def main() -> None:
    df = pd.read_csv(FEATURES_IN)

    era_cols = list(ERA_DEFS.keys())

    # Drop any existing era columns and their downstream _adj columns
    adj_cols = [c for c in df.columns if c.endswith("_adj")]
    drop = [c for c in era_cols + adj_cols if c in df.columns]
    if drop:
        df = df.drop(columns=drop)
        print(f"Dropped {len(drop)} stale columns: {drop}\n")

    print("Assigning MLB-derived era labels ...\n")
    added = []
    for col, defn in ERA_DEFS.items():
        df[col] = assign_era(df["Season"], defn["breaks"], defn["names"])
        added.append(col)
        print(f"  {col}  (breaks: {defn['breaks']})")
        for name in defn["names"]:
            mask    = df[col] == name
            seasons = sorted(df.loc[mask, "Season"].unique())
            yr      = f"{seasons[0]}-{seasons[-1]}" if len(seasons) > 1 else str(seasons[0])
            print(f"    {name:<28} ({yr}): {mask.sum():,} rows")
        print()

    # Insert era columns right after Season
    other      = [c for c in df.columns if c not in added]
    season_pos = other.index("Season") + 1
    ordered    = other[:season_pos] + added + other[season_pos:]
    df         = df[ordered]

    df.to_csv(FEATURES_IN, index=False)
    print(f"Wrote {len(df):,} rows -> {FEATURES_IN.name}")
    print(f"Added columns: {added}")


if __name__ == "__main__":
    main()
