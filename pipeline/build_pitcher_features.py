"""Build pitcher_features.csv — one row per player-season-level, 2006-2026.

Sources:
  milb_pitching.csv          — counting stats (IP, K, BB, ER, HR, W, L, SV, HLD, ...)
  milb_pitching_advanced.csv — rate stats (K%, BB%, K-BB%, Whiff%, GB/FB, QS, ...)
  player_birthdays.csv       — for Age_Z_SL

Derived columns:
  ERA       = 9 × ER / IP
  WHIP      = (H + BB) / IP
  K9        = 9 × K / IP
  BB9       = 9 × BB / IP
  HR9       = 9 × HRA / IP
  PPI_skill = 2×K/IP - 0.5×BB/IP - 1×ER/IP - 2×HRA/IP + 0.75  (controllable rate PPI)
  Role      = 'SP' if GS/G >= 0.5 else 'RP'
  Age_Z_SL  = age z-scored within Season × Level peers

Era labels (same MLB-derived breaks as hitter model):
  EraK%  : 2015
  EraHRFB: 2016, 2022
  EraPPPA: 2010, 2021

Era-adjusted z-scores (within Level × Era cell, using players with IP >= 20):
  K%_adj, BB%_adj, KBB_adj, Whiff%_adj, ERA_adj, GB%_adj
  All positive = better (ERA_adj inverted: lower ERA → higher z)
"""

import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR   = Path(__file__).resolve().parent.parent / "data"
PIT_PATH   = DATA_DIR / "api" / "milb_pitching.csv"
ADV_PATH   = DATA_DIR / "api" / "milb_pitching_advanced.csv"
BIRTH_PATH = DATA_DIR / "api" / "player_birthdays.csv"
OUT_PATH   = DATA_DIR / "rankings" / "pitcher_features.csv"

MIN_IP_ADJ  = 20.0   # minimum IP for era-adjustment z-score pools
MIN_IP_FEAT =  5.0   # minimum IP to be included in output at all


def _norm(s) -> str:
    s = unicodedata.normalize("NFD", str(s))
    return "".join(c for c in s if unicodedata.category(c) != "Mn").lower().strip()


def assign_era(season: pd.Series, breaks: list[int], names: list[str]) -> pd.Series:
    out = pd.Series(names[0], index=season.index, dtype="object")
    for brk, name in zip(breaks, names[1:]):
        out[season >= brk] = name
    return out


def z_within_group(df: pd.DataFrame, col: str, group_cols: list[str],
                   min_n: int = 5) -> pd.Series:
    """Z-score col within each group. Groups with < min_n rows fall back to Level-only."""
    result = pd.Series(np.nan, index=df.index)
    for key, grp in df.groupby(group_cols):
        mask = grp[col].notna()
        vals = grp.loc[mask, col]
        if len(vals) < min_n:
            continue
        mu, sd = vals.mean(), vals.std(ddof=1)
        if sd < 1e-9:
            result.loc[grp.index] = 0.0
        else:
            result.loc[grp.index] = (grp[col] - mu) / sd
    # fallback: Level-only z-score for rows still NaN
    for level, lgrp in df.groupby("Level"):
        nan_mask = result.loc[lgrp.index].isna() & lgrp[col].notna()
        if not nan_mask.any():
            continue
        vals = lgrp.loc[lgrp[col].notna(), col]
        if len(vals) < min_n:
            continue
        mu, sd = vals.mean(), vals.std(ddof=1)
        if sd < 1e-9:
            result.loc[lgrp.index[nan_mask]] = 0.0
        else:
            result.loc[lgrp.index[nan_mask]] = (lgrp.loc[nan_mask, col] - mu) / sd
    return result


def main() -> None:
    print("=== build_pitcher_features.py ===\n")

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    print("Loading milb_pitching.csv ...")
    pit = pd.read_csv(PIT_PATH, dtype={"PlayerId": str, "MLBAM_ID": str})
    print(f"  {len(pit):,} rows")

    print("Loading milb_pitching_advanced.csv ...")
    adv = pd.read_csv(ADV_PATH, dtype={"PlayerId": str, "MLBAM_ID": str})
    print(f"  {len(adv):,} rows")

    print("Loading player_birthdays.csv ...")
    bdays = pd.read_csv(BIRTH_PATH, dtype={"MLBAM_ID": str})
    bdays["MLBAM_ID"] = pd.to_numeric(bdays["MLBAM_ID"], errors="coerce")

    # -----------------------------------------------------------------------
    # Filter minimum IP
    # -----------------------------------------------------------------------
    pit = pit[pit["IP"].fillna(0) >= MIN_IP_FEAT].copy()
    print(f"\nAfter IP >= {MIN_IP_FEAT} filter: {len(pit):,} rows")

    # -----------------------------------------------------------------------
    # Merge counting + advanced
    # Advanced uses BF as the join anchor (both files have MLBAM_ID + Season + Level + Team)
    # -----------------------------------------------------------------------
    merge_keys = ["MLBAM_ID", "Season", "Level", "Team"]
    adv_cols = ["K%", "BB%", "K-BB%", "Whiff%", "BABIP", "QS", "GB%", "LD%", "FB%", "GB/FB"]
    adv_sub = adv[merge_keys + adv_cols].copy()

    df = pit.merge(adv_sub, on=merge_keys, how="left")
    print(f"After merge with advanced: {len(df):,} rows "
          f"({df['K%'].notna().sum():,} with K% coverage)")

    # -----------------------------------------------------------------------
    # Age from birth dates
    # -----------------------------------------------------------------------
    df["MLBAM_ID_num"] = pd.to_numeric(df["MLBAM_ID"], errors="coerce")
    bdays_map = dict(zip(bdays["MLBAM_ID"], bdays["BirthDate"]))
    df["BirthDate"] = df["MLBAM_ID_num"].map(bdays_map)
    df["BirthDate"] = pd.to_datetime(df["BirthDate"], errors="coerce")
    if "Age" not in df.columns:
        df["Age"] = np.nan
    # Fill Age from BirthDate where missing
    ref_date = pd.to_datetime(df["Season"].astype(str) + "-07-01")
    age_from_bd = ((ref_date - df["BirthDate"]).dt.days / 365.25).round(1)
    df["Age"] = df["Age"].where(df["Age"].notna(), age_from_bd)
    df = df.drop(columns=["MLBAM_ID_num", "BirthDate"], errors="ignore")

    # -----------------------------------------------------------------------
    # Derived stats
    # -----------------------------------------------------------------------
    df["ERA"]       = (9 * df["ER"] / df["IP"]).where(df["IP"] > 0)
    df["WHIP"]      = ((df["H"] + df["BB"]) / df["IP"]).where(df["IP"] > 0)
    df["K9"]        = (9 * df["K"] / df["IP"]).where(df["IP"] > 0)
    df["BB9"]       = (9 * df["BB"] / df["IP"]).where(df["IP"] > 0)
    df["HR9"]       = (9 * df["HRA"] / df["IP"]).where(df["IP"] > 0)
    df["PPI_skill"] = (
        2 * df["K"] / df["IP"]
        - 0.5 * df["BB"] / df["IP"]
        - 1 * df["ER"] / df["IP"]
        - 2 * df["HRA"] / df["IP"]
        + 0.75
    ).where(df["IP"] > 0)

    # K% from counting if not available from advanced
    if "K%" not in df.columns or df["K%"].isna().all():
        df["K%"] = (df["K"] / df["BF"]).where(df["BF"] > 0)
    if "BB%" not in df.columns or df["BB%"].isna().all():
        df["BB%"] = (df["BB"] / df["BF"]).where(df["BF"] > 0)
    if "K-BB%" not in df.columns or df["K-BB%"].isna().all():
        df["K-BB%"] = df["K%"] - df["BB%"]

    # Role
    df["Role"] = np.where(
        df["GS"].fillna(0) / df["G"].clip(lower=1) >= 0.5,
        "SP", "RP"
    )

    # -----------------------------------------------------------------------
    # Age_Z_SL: age z-scored within Season × Level
    # -----------------------------------------------------------------------
    df["Age_Z_SL"] = np.nan
    for (season, level), grp in df.groupby(["Season", "Level"]):
        ages = grp["Age"].dropna()
        if len(ages) < 5:
            continue
        mu, sd = ages.mean(), ages.std(ddof=1)
        if sd < 1e-9:
            df.loc[grp.index, "Age_Z_SL"] = 0.0
        else:
            df.loc[grp.index, "Age_Z_SL"] = (grp["Age"] - mu) / sd
    print(f"\nAge_Z_SL: {df['Age_Z_SL'].notna().sum():,} rows with age z-score")

    # -----------------------------------------------------------------------
    # Era labels
    # -----------------------------------------------------------------------
    df["EraK%"]   = assign_era(df["Season"], [2015], ["Contact Era", "High-K Era"])
    df["EraHRFB"] = assign_era(df["Season"], [2016, 2022],
                               ["Pre-Launch Angle", "Launch Angle Era", "Post-Deadening"])
    df["EraPPPA"] = assign_era(df["Season"], [2010, 2021],
                               ["Early Offensive Era", "Standard Era", "Modern Era"])

    # -----------------------------------------------------------------------
    # Era-adjusted z-scores (IP >= MIN_IP_ADJ for z-score pool participants)
    # -----------------------------------------------------------------------
    df_adj = df[df["IP"] >= MIN_IP_ADJ].copy()

    # K%_adj: higher K% (for pitchers) = more strikeouts = better → positive z
    df["K%_adj"] = z_within_group(df_adj, "K%", ["Level", "EraK%"]).reindex(df.index)

    # BB%_adj: lower BB% = better command → invert
    df["BB%_adj"] = z_within_group(df_adj, "BB%", ["Level", "EraK%"]).reindex(df.index) * -1

    # KBB_adj: higher K-BB% = better → positive z
    df["KBB_adj"] = z_within_group(df_adj, "K-BB%", ["Level", "EraK%"]).reindex(df.index)

    # Whiff%_adj: higher whiff = more swing-and-miss = better → positive z
    df["Whiff%_adj"] = z_within_group(df_adj, "Whiff%", ["Level", "EraK%"]).reindex(df.index)

    # ERA_adj: lower ERA = better → invert
    df["ERA_adj"] = z_within_group(df_adj, "ERA", ["Level", "EraPPPA"]).reindex(df.index) * -1

    # GB%_adj: higher GB% = more grounders → positive z (no era break)
    df["GB%_adj"] = z_within_group(df_adj, "GB%", ["Level"]).reindex(df.index)

    # PPI_skill_adj: higher = better → positive z
    df["PPI_adj"] = z_within_group(df_adj, "PPI_skill", ["Level", "EraPPPA"]).reindex(df.index)

    print("\nEra-adjusted z-score coverage:")
    for col in ["K%_adj", "BB%_adj", "KBB_adj", "Whiff%_adj", "ERA_adj", "GB%_adj", "PPI_adj"]:
        n = df[col].notna().sum()
        print(f"  {col}: {n:,} rows ({100*n/len(df):.1f}%)")

    # -----------------------------------------------------------------------
    # Column order + save
    # -----------------------------------------------------------------------
    col_order = [
        "PlayerId", "MLBAM_ID", "Season", "Name", "Team", "Level", "League",
        "Age", "Age_Z_SL", "Role",
        "G", "GS", "IP", "BF", "ER", "K", "BB", "IBB", "HRA", "H", "HBP",
        "W", "L", "SV", "SVO", "HLD", "BS", "CG", "SHO", "QS", "WP",
        "ERA", "WHIP", "K9", "BB9", "HR9", "PPI_skill",
        "K%", "BB%", "K-BB%", "Whiff%", "BABIP", "GB%", "LD%", "FB%", "GB/FB",
        "EraK%", "EraHRFB", "EraPPPA",
        "K%_adj", "BB%_adj", "KBB_adj", "Whiff%_adj", "ERA_adj", "GB%_adj", "PPI_adj",
    ]
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(df):,} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
