"""Build ABILITY_Score for every player-season row in prospect_features.parquet.

ABILITY_Score measures demonstrated production, normalized for era and level.

Usage:
  python pipeline/build_ability_score.py                   # personal model (default)
  python pipeline/build_ability_score.py --profile fantrax # Fantrax scoring

Profile differences (Fantrax vs personal):
  Discipline: BB% − 0.5×K%  (vs BB% − 2×K%)  — matches SO=-0.5 scoring weight
  PPPA_Z_SL:  sourced from data/computed_fantrax/minorLeagueData.parquet
  Level discounts: from config/scoring_fantrax.py (same values for now)
  Output:     data/rankings_fantrax/prospect_features.parquet

Component weights (base — power and discipline are dynamic, see below):
  Fantasy Output  47%  -- PPPA_Z_SL with level discount
  Discipline      28%  -- BB% − K_MULT×K%, z-scored within Season+Level (base)
  SB Talent        8%  -- SB_pct × (SB/PA), z-scored within Season+Level
  Game Power      17%  -- 0.5 × HR/FB + 0.5 × HR_AB, z-scored within Season+Level (base)
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR      = Path(__file__).resolve().parent.parent / "data"
AGE_MULT_PATH = DATA_DIR / "computed" / "age_mult_rows.csv"
MIN_PA        = 50    # minimum PA to count toward group z-score params
MIN_ROWS      = 10    # minimum rows in Season+Level cell before falling back to Level-only

# Component weights
W = dict(fantasy=0.47, discipline=0.28, sb=0.08, power=0.17)

AGE_LINEAR      = 0.0054
AGE_KINK        = 0.192
AGE_KINK_THRESH = 1.5

# Personal model defaults
LEVEL_DISCOUNT_DEFAULT = {"AAA": 1.00, "AA": 0.59, "A+": 0.34, "A": 0.23, "R": 0.10}
DISC_K_MULT_DEFAULT    = 2.0   # BB% − 2×K%

LEVEL_DISCOUNT      = LEVEL_DISCOUNT_DEFAULT   # overridden per-profile in main()
DISC_K_MULT         = DISC_K_MULT_DEFAULT      # overridden per-profile in main()
DYNAMIC_POWER_DISC  = True                     # overridden per-profile in main()
# W is also overridden per-profile in main() for Fantrax

# ---------------------------------------------------------------------------
# Z-scoring helpers
# ---------------------------------------------------------------------------

def z_within_sl(df: pd.DataFrame, col: str) -> pd.Series:
    """Z-score col within Season+Level; Level-only fallback for sparse cells.

    Only rows with PA >= MIN_PA and non-null col contribute to group params.
    """
    out   = pd.Series(np.nan, index=df.index, dtype=float)
    valid = df[(df["PA"] >= MIN_PA)].dropna(subset=[col])

    grp_sl = valid.groupby(["Season", "Level"], observed=True)[col].agg(
        ["mean", "std", "count"]
    )
    grp_l = valid.groupby("Level", observed=True)[col].agg(["mean", "std"])

    for (season, level), idx in df.groupby(["Season", "Level"], observed=True).groups.items():
        mu, sig = None, None
        try:
            row = grp_sl.loc[(season, level)]
            if row["count"] >= MIN_ROWS and row["std"] > 1e-9:
                mu, sig = row["mean"], row["std"]
        except KeyError:
            pass

        if mu is None:
            try:
                fb = grp_l.loc[level]
                if fb["std"] > 1e-9:
                    mu, sig = fb["mean"], fb["std"]
            except KeyError:
                continue

        if mu is not None and sig is not None:
            vals = df.loc[idx, col]
            out.loc[idx] = (vals - mu) / sig

    return out.round(4)


def standardize_component(s: pd.Series, clip: float = 3.0) -> pd.Series:
    """Zero-mean, unit-std across non-null entries; winsorize at ±clip; NaN stays NaN."""
    mu, sig = s.mean(), s.std()
    if sig > 0:
        return ((s - mu) / sig).clip(-clip, clip)
    return s - mu


def to_50_10(s: pd.Series) -> pd.Series:
    """Standardize to 50±10, clip at 0."""
    mu, sig = s.mean(), s.std()
    if sig > 0:
        return (50 + 10 * (s - mu) / sig).clip(lower=0).round(2)
    return pd.Series(50.0, index=s.index)


# ---------------------------------------------------------------------------
# Component builders
# ---------------------------------------------------------------------------

def build_fantasy_output(df: pd.DataFrame) -> pd.Series:
    """PPPA_Z_SL multiplied by the empirical level discount factor."""
    discount = df["Level"].map(LEVEL_DISCOUNT).fillna(0.10)   # unknown level → R rate
    return (df["PPPA_Z_SL"] * discount).round(4)


def build_discipline(df: pd.DataFrame) -> pd.Series:
    """BB% − DISC_K_MULT×K%, z-scored within Season+Level.

    Personal: DISC_K_MULT=2.0 → uses precomputed BB_2K column.
    Fantrax:  DISC_K_MULT=0.5 → computed on the fly from BB% and K%.
    """
    if DISC_K_MULT == 2.0 and "BB_2K" in df.columns:
        return z_within_sl(df, "BB_2K")
    tmp = df.copy()
    tmp["_disc_raw"] = df["BB%"] - DISC_K_MULT * df["K%"]
    return z_within_sl(tmp, "_disc_raw")


def build_sb_talent(df: pd.DataFrame) -> pd.Series:
    """SB_pct × (SB/PA), z-scored within Season+Level.

    SB_pct is 0 when SB=0 (regardless of CS count); SB_talent is then 0.
    This represents genuine no-basestealing production, not a missing value.
    """
    sb_pct  = df["SB_pct"].fillna(0)          # NaN (0 SB + 0 CS) → 0
    sb_rate = (df["SB"] / df["PA"]).fillna(0)
    raw     = (sb_pct * sb_rate).round(6)

    tmp = df.copy()
    tmp["_sb_talent"] = raw
    return z_within_sl(tmp, "_sb_talent")


def build_game_power(df: pd.DataFrame) -> pd.Series:
    """0.5 × HR/FB + 0.5 × HR_AB, z-scored within Season+Level.

    PullAir% was replaced: partial r with Career_PPPA_Z = 0.054 vs 0.277 for
    HR/FB or HR_AB (controlling K%+BB2K). Both HR metrics have equal predictive
    power and 100% coverage; blending them averages out single-season noise.

    Uses whichever components are available (blend, hrfb-only, or HR_AB-only).
    """
    hrfb  = df["HR/FB"]
    hr_ab = df["HR_AB"]

    both      = hrfb.notna() & hr_ab.notna()
    hrfb_only = hrfb.notna() & hr_ab.isna()
    hrab_only = hrfb.isna() & hr_ab.notna()

    raw = pd.Series(np.nan, index=df.index, dtype=float)
    raw[both]      = 0.5 * hrfb[both] + 0.5 * hr_ab[both]
    raw[hrfb_only] = hrfb[hrfb_only]
    raw[hrab_only] = hr_ab[hrab_only]

    tmp = df.copy()
    tmp["_game_power"] = raw
    return z_within_sl(tmp, "_game_power")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global LEVEL_DISCOUNT, DISC_K_MULT, DYNAMIC_POWER_DISC, W

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["personal", "fantrax"], default="personal")
    args = parser.parse_args()

    if args.profile == "fantrax":
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from config.scoring_fantrax import (
            LEVEL_DISCOUNT as LD_FX,
            DISC_K_MULT as DKM_FX,
            RANKINGS_DIR, COMPUTED_DIR,
        )
        LEVEL_DISCOUNT     = LD_FX
        DISC_K_MULT        = DKM_FX
        DYNAMIC_POWER_DISC = False  # interaction not significant in Fantrax (p=0.991)
        from config.scoring_fantrax import ABILITY_WEIGHTS
        W = ABILITY_WEIGHTS  # fantasy=0.60, discipline=0.25, sb=0.08, power=0.07
        features_in  = DATA_DIR / "rankings" / "prospect_features.parquet"
        features_out = DATA_DIR / RANKINGS_DIR / "prospect_features.parquet"
        fantrax_pppa_path = DATA_DIR / COMPUTED_DIR / "minorLeagueData.parquet"
    else:
        features_in  = DATA_DIR / "rankings" / "prospect_features.parquet"
        features_out = features_in   # write back in place (personal behavior)
        fantrax_pppa_path = None

    df = pd.read_parquet(features_in)
    print(f"[{args.profile}] Loaded {len(df):,} rows\n")

    # For Fantrax: replace PPPA_Z_SL with Fantrax-scored values.
    # Join on PlayerId + Season + Level (same keys as prospect_features build).
    if fantrax_pppa_path is not None:
        fx_pppa = pd.read_parquet(
            fantrax_pppa_path,
            columns=["PlayerId", "Season", "Level", "PPPA_Z_SL"],
        )
        fx_pppa = fx_pppa.rename(columns={"PPPA_Z_SL": "_PPPA_Z_SL_fx"})
        df = df.merge(fx_pppa, on=["PlayerId", "Season", "Level"], how="left")
        filled = df["_PPPA_Z_SL_fx"].notna().sum()
        print(f"  Fantrax PPPA_Z_SL filled: {filled:,} / {len(df):,} rows")
        df["PPPA_Z_SL"] = df["_PPPA_Z_SL_fx"].fillna(df["PPPA_Z_SL"])
        df = df.drop(columns=["_PPPA_Z_SL_fx"])

    # 1. Raw components
    fantasy = build_fantasy_output(df)
    disc    = build_discipline(df)
    sb      = build_sb_talent(df)
    gp      = build_game_power(df)

    # Save BB_2K z-score before further transformations — used for floor penalty
    # and display flags.  df["BB_2K"] is the raw rate (BB% − 2×K%), not a z-score;
    # the thresholds are calibrated for z-scores.
    bb2k_z = disc.copy()

    print("Non-null coverage before standardization:")
    print(f"  Fantasy Output  {fantasy.notna().sum():,}")
    print(f"  Discipline      {disc.notna().sum():,}")
    print(f"  SB Talent       {sb.notna().sum():,}")
    print(f"  Game Power      {gp.notna().sum():,}\n")

    # 2. Standardize each to mean=0, std=1, winsorize ±3σ
    fantasy = standardize_component(fantasy)
    disc    = standardize_component(disc)
    sb      = standardize_component(sb)
    gp      = standardize_component(gp)

    # 2b. Age multiplier — applied after standardization so age adjusts signal
    # strength within the peer distribution, not the raw stat values.
    # SB excluded: speed is a physical tool, not expected to improve with age.
    #
    # Primary: per-row KNN graduation probability from age_mult_rows.csv,
    # joined on (PlayerId, Season, Level). Each row uses point-in-time features
    # (no look-ahead), so A-ball rows use A-ball age context, not current-level.
    # Fallback (if CSV missing): piecewise formula using this row's Age_Z_SL.
    if AGE_MULT_PATH.exists():
        amt = pd.read_csv(AGE_MULT_PATH, dtype={"PlayerId": str},
                          usecols=["PlayerId", "Season", "Level", "age_mult_pgvb"])
        amt = amt.drop_duplicates(subset=["PlayerId", "Season", "Level"])
        knn_map = {(r.PlayerId, r.Season, r.Level): r.age_mult_pgvb
                   for r in amt.itertuples(index=False)}
        pid_str = df["PlayerId"].astype(str)
        age_mult = pd.Series(
            [knn_map.get((p, s, l), np.nan)
             for p, s, l in zip(pid_str, df["Season"], df["Level"])],
            index=df.index, dtype=float,
        )
        matched = age_mult.notna().sum()
        # age_mult_pgvb was calibrated for 50-scale career adjustment;
        # clip to [0.7, 1.5] so it acts as a per-component signal-strength
        # modifier without creating extreme outliers in the z-score space.
        age_mult = age_mult.clip(0.70, 1.50).fillna(1.0)
        print(f"  KNN age_mult: {matched:,} / {len(age_mult):,} rows matched  "
              f"(mean={age_mult.mean():.3f}  p10={age_mult.quantile(.1):.2f}  "
              f"p90={age_mult.quantile(.9):.2f})")
    else:
        age_z    = (-df["Age_Z_SL"]).clip(-3.0, 3.0).fillna(0.0)
        age_mult = 1.0 + AGE_LINEAR * age_z + AGE_KINK * (age_z - AGE_KINK_THRESH).clip(lower=0)
        print("  age_mult_rows.csv not found — using piecewise fallback")
    fantasy  = fantasy * age_mult
    disc     = disc    * age_mult
    gp       = gp      * age_mult

    # 3. Blend — one-directional power/discipline scaling.
    # For above-average power: power weight rises +0.05/SD, discipline drops −0.05/SD.
    # For average or below: both stay at base. Total always sums to 1.0.
    # gp winsorized at ±3 SD → max shift = +0.15 (power 0.17→0.32, disc 0.28→0.13).
    # Missing power → shift = 0, base weights used.
    # DYNAMIC_POWER_DISC=False (Fantrax): interaction not empirically justified (p=0.991);
    # discipline and power contribute independently at equal strength across all power tiers.
    POWER_SCALE_PER_SD = 0.05
    power_shift = (
        gp.fillna(0).clip(lower=0) * POWER_SCALE_PER_SD
        if DYNAMIC_POWER_DISC else
        pd.Series(0.0, index=gp.index)
    )
    w_power_dyn = W["power"]      + power_shift   # [0.17, 0.32]
    w_disc_dyn  = W["discipline"] - power_shift   # [0.13, 0.28]

    ability_raw = (
        W["fantasy"]  * fantasy.fillna(0)
        + w_disc_dyn  * disc.fillna(0)
        + W["sb"]     * sb.fillna(0)
        + w_power_dyn * gp.fillna(0)
    )

    # 4. Final 50±10 standardization
    ability_score = to_50_10(ability_raw)

    # 5. Write score columns back into prospect_features.csv in place
    df["ABILITY_Score"]   = ability_score
    df["Fantasy_Out"]     = fantasy.round(3)
    df["ABILITY_Disc"]    = disc.round(3)
    df["SB_Talent"]       = sb.round(3)
    df["Game_Power"]      = gp.round(3)
    df["PPPA_Z_SL_disc"]  = (
        df["PPPA_Z_SL"] * df["Level"].map(LEVEL_DISCOUNT).fillna(0.10)
    ).round(4)

    # Discipline floor flags — which threshold(s) fired for this row.
    whiff_z   = df["Whiff%_adj"]
    bb2k_flag = pd.Series("", index=df.index)
    bb2k_flag[bb2k_z.notna() & (bb2k_z <= -0.50)] = "hard"
    bb2k_flag[bb2k_z.notna() & (bb2k_z > -0.50) & (bb2k_z <= -0.29)] = "soft"
    whiff_flag = pd.Series("", index=df.index)
    whiff_flag[whiff_z.notna() & (whiff_z >= 1.0)] = "whiff"
    disc_flag = (bb2k_flag + whiff_flag.apply(lambda w: ("+" if w else "") + w))
    disc_flag = disc_flag.where(bb2k_flag != "", whiff_flag)
    df["Discipline_Flag"] = disc_flag

    features_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(features_out, index=False)
    print(f"[{args.profile}] Wrote {len(df):,} rows -> {features_out}\n")

    print(
        f"ABILITY_Score  mean={ability_score.mean():.1f}  "
        f"std={ability_score.std():.1f}  "
        f"min={ability_score.min():.1f}  "
        f"max={ability_score.max():.1f}\n"
    )

    # Top 20 current prospects (Season >= 2025, most PA row per player)
    current = df[df["Season"] >= 2025].copy()
    best = (
        current.sort_values("PA", ascending=False)
               .drop_duplicates("PlayerId")
               .nlargest(20, "ABILITY_Score")
    )
    print("Top 20 ABILITY (Season >= 2025, one row per player):")
    print(
        best[["Name", "Team", "Level", "Age", "PA", "ABILITY_Score",
              "Fantasy_Out", "ABILITY_Disc", "SB_Talent", "Game_Power"]
             ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
