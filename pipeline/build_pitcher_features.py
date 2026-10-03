"""Build pitcher_features.csv — one row per player-season-level, 2006-2026.

Sources:
  milb_pitching_combined.parquet — counting + rate stats + ProspectSavant metrics
  player_birthdays.csv           — for Age_Z_SL

Derived columns:
  ERA       = 9 × ER / IP
  WHIP      = (H + BB) / IP
  K9        = 9 × K / IP
  BB9       = 9 × BB / IP
  HR9       = 9 × HRA / IP
  PPI_skill = 2×K/IP - 0.5×BB/IP - 1×ER/IP - 2×HRA/IP + 0.75  (controllable rate PPI)
  Role      = PS Role where available; GS/G >= 0.5 fallback
  Age_Z_SL  = age z-scored within Season × Level peers

Era labels (same MLB-derived breaks as hitter model):
  EraK%  : 2015
  EraHRFB: 2016, 2022
  EraPPPA: 2010, 2021

Era-adjusted z-scores (within Level × Era cell, using players with IP >= 20):
  K%_adj, BB%_adj, KBB_adj, Whiff%_adj, ERA_adj, GB%_adj, PPI_adj
  MaxVelo_adj, SpinRate_adj, Chase%_adj, ZContact%_adj (PS tiers, 2023-2026)
  All positive = better (ERA_adj, BB%_adj, ZContact%_adj inverted)
"""

import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR   = Path(__file__).resolve().parent.parent / "data"
PIT_PATH   = DATA_DIR / "api" / "milb_pitching_combined.parquet"
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
    print("Loading milb_pitching_combined.parquet ...")
    df = pd.read_parquet(PIT_PATH)
    df["PlayerId"] = df["PlayerId"].astype(str)
    df["MLBAM_ID"] = df["MLBAM_ID"].astype(str)
    print(f"  {len(df):,} rows")

    print("Loading player_birthdays.csv ...")
    bdays = pd.read_csv(BIRTH_PATH, dtype={"MLBAM_ID": str})
    bdays["MLBAM_ID"] = pd.to_numeric(bdays["MLBAM_ID"], errors="coerce")

    # -----------------------------------------------------------------------
    # Filter minimum IP
    # -----------------------------------------------------------------------
    df = df[df["IP"].fillna(0) >= MIN_IP_FEAT].copy()
    print(f"\nAfter IP >= {MIN_IP_FEAT} filter: {len(df):,} rows")
    print(f"K% coverage: {df['K%'].notna().sum():,} rows")

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
    # Derived stats (computed from counting — override any PS values)
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

    # K%/BB% from counting if not available from advanced
    if "K%" not in df.columns or df["K%"].isna().all():
        df["K%"] = (df["K"] / df["BF"]).where(df["BF"] > 0)
    if "BB%" not in df.columns or df["BB%"].isna().all():
        df["BB%"] = (df["BB"] / df["BF"]).where(df["BF"] > 0)
    if "K-BB%" not in df.columns or df["K-BB%"].isna().all():
        df["K-BB%"] = df["K%"] - df["BB%"]

    # Role: use PS Role where available; fall back to GS/G ratio
    role_computed = np.where(
        df["GS"].fillna(0) / df["G"].clip(lower=1) >= 0.5,
        "SP", "RP"
    )
    if "Role" in df.columns:
        df["Role"] = df["Role"].where(df["Role"].notna(), pd.Series(role_computed, index=df.index))
    else:
        df["Role"] = role_computed

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

    # ── PS-sourced era-adjusted z-scores (2023-2026 where PS data present) ──
    # MaxVelo_adj: higher velo = better → positive z
    if "MaxVelo" in df.columns:
        df["MaxVelo_adj"] = z_within_group(df_adj, "MaxVelo", ["Level", "EraK%"]).reindex(df.index)

    # SpinRate_adj: higher spin = more movement = better → positive z
    if "SpinRate" in df.columns:
        df["SpinRate_adj"] = z_within_group(df_adj, "SpinRate", ["Level", "EraK%"]).reindex(df.index)

    # Chase%_adj: batters chasing = pitcher inducing bad swings = better → positive z
    if "Chase%" in df.columns:
        df["Chase%_adj"] = z_within_group(df_adj, "Chase%", ["Level", "EraK%"]).reindex(df.index)

    # ZContact%_adj: lower in-zone contact = better for pitcher → invert
    if "ZContact%" in df.columns:
        df["ZContact%_adj"] = z_within_group(df_adj, "ZContact%", ["Level", "EraK%"]).reindex(df.index) * -1

    print("\nEra-adjusted z-score coverage:")
    adj_cols = ["K%_adj", "BB%_adj", "KBB_adj", "Whiff%_adj", "ERA_adj", "GB%_adj", "PPI_adj",
                "MaxVelo_adj", "SpinRate_adj", "Chase%_adj", "ZContact%_adj"]
    for col in adj_cols:
        if col in df.columns:
            n = df[col].notna().sum()
            print(f"  {col}: {n:,} rows ({100*n/len(df):.1f}%)")

    # -----------------------------------------------------------------------
    # Player bios — height / weight
    # -----------------------------------------------------------------------
    bios_path = DATA_DIR / "api" / "player_bios.csv"
    if bios_path.exists():
        bios = pd.read_csv(bios_path, usecols=["MLBAM_ID", "Height", "HeightIn", "Weight"],
                           low_memory=False)
        bios["MLBAM_ID"] = pd.to_numeric(bios["MLBAM_ID"], errors="coerce")
        df["MLBAM_ID_num"] = pd.to_numeric(df["MLBAM_ID"], errors="coerce")
        df = df.merge(bios.rename(columns={"MLBAM_ID": "MLBAM_ID_num"}),
                      on="MLBAM_ID_num", how="left")
        df = df.drop(columns=["MLBAM_ID_num"])
        n_ht = df["HeightIn"].notna().sum()
        print(f"  Height populated: {n_ht:,} / {len(df):,} rows")
    else:
        print("  player_bios.csv not found — skipping (run fetch/fetch_player_bios.py)")

    # -----------------------------------------------------------------------
    # Column order + save
    # -----------------------------------------------------------------------
    col_order = [
        "PlayerId", "MLBAM_ID", "Season", "Name", "Team", "Level", "League",
        "Age", "Age_Z_SL", "Role", "Throws", "Height", "HeightIn", "Weight",
        "G", "GS", "IP", "BF", "TBF", "ER", "K", "BB", "IBB", "HRA", "H", "HBP",
        "W", "L", "SV", "SVO", "HLD", "BS", "CG", "SHO", "QS", "WP",
        "ERA", "WHIP", "K9", "BB9", "HR9", "PPI_skill",
        "K%", "BB%", "K-BB%", "Whiff%", "BABIP", "GB%", "LD%", "FB%", "GB/FB",
        # ProspectSavant columns (2023-2026)
        "MaxVelo", "AvgVelo", "EffVelo", "SpinRate", "Extension", "ArmAngle",
        "HBreak_Arm", "IVBreak", "TVBreak",
        "SwStr%", "Chase%", "ZContact%", "Zone%", "ZSwing%", "Strike%",
        "FIP", "xFIP", "xwOBA", "xBA", "xSLG", "wOBA",
        "Barrel%BBE", "Barrel%PA", "HardHit%", "LA", "AvgEV",
        "PSScore",
        # Era labels
        "EraK%", "EraHRFB", "EraPPPA",
        # Era-adjusted z-scores
        "K%_adj", "BB%_adj", "KBB_adj", "Whiff%_adj", "ERA_adj", "GB%_adj", "PPI_adj",
        "MaxVelo_adj", "SpinRate_adj", "Chase%_adj", "ZContact%_adj",
    ]
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(df):,} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
