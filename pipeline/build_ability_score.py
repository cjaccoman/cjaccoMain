"""Build ABILITY_Score for every player-season row in prospect_features.csv.

ABILITY_Score measures demonstrated production, normalized for era and level.
Output: data/rankings/ability_scores.csv (one row per player-season-level)

Component weights:
  Fantasy Output  45%  -- PPPA_Z_SL with level discount
  Discipline      25%  -- BB% − 2×K% (BB_2K), z-scored within Season+Level
  SB Talent       15%  -- SB_pct × (SB/PA), z-scored within Season+Level
  Game Power      15%  -- 0.5 × PullAir% + 0.5 × HR_AB, z-scored within Season+Level

Age adjustment (AGE_ALPHA = 0.20):
  Each component is multiplied by (1 + 0.20 × −Age_Z_SL), clipped to ±2 SD.
  A player 2 SD younger than peers gets a ~40% boost to every component;
  a player 2 SD older gets a ~40% cut. Weights stay proportional — age is
  baked into signal strength here; a standalone Age_Score at 20% weight is
  also added in build_prospect_scores.py for a compounding effect.

PPPA level discount factors (Skill_PPPA full-population study, normalized to AAA=1.0):
  AAA=1.00, AA=0.59, A+=0.34, A=0.23, R=0.10

Z-scoring approach (peer-relative, era-robust by construction):
  PPPA_Z_SL  -- already z-scored within Season+League; apply level discount only.
  BB_2K      -- z-scored within Season+Level; Season+Level peers control for era drift.
  SB_talent  -- SB_pct × (SB/PA), z-scored within Season+Level.
  Game Power -- 0.5×PullAir% + 0.5×HR_AB, z-scored within Season+Level.

Sparse Season+Level cells (< MIN_ROWS qualifying rows) fall back to Level-only
z-scoring. Rows contributing to group params must have PA >= MIN_PA.

All components winsorized at ±3σ before blending.
Missing component → filled with 0 (neutral, peer-average assumption).

Note: non-linear discipline floor penalty is applied post-blend in
build_prospect_scores.py as a gate on Combined_Score, using a career+recent
blend of BB_2K with a slope modifier. It does not fire here.

Final ABILITY_Score standardized to 50±10, clipped at 0.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR    = Path(__file__).resolve().parent.parent / "data"
FEATURES_IN = DATA_DIR / "rankings" / "prospect_features.csv"
OUT_PATH    = DATA_DIR / "rankings" / "ability_scores.csv"
MIN_PA      = 50    # minimum PA to count toward group z-score params
MIN_ROWS    = 10    # minimum rows in Season+Level cell before falling back to Level-only

# Component weights
W = dict(fantasy=0.45, discipline=0.25, sb=0.15, power=0.15)

# Age adjustment: each component is multiplied by (1 + AGE_ALPHA × −Age_Z_SL).
# Age_Z_SL is clipped to ±2 SD before applying so extreme outliers don't dominate.
AGE_ALPHA = 0.20

# PPPA level discount factors (Skill_PPPA full-population study, analysis/skill_pppa_translation.py)
LEVEL_DISCOUNT = {"AAA": 1.00, "AA": 0.59, "A+": 0.34, "A": 0.23, "R": 0.10}



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
    """BB% − 2×K% (BB_2K), z-scored within Season+Level."""
    return z_within_sl(df, "BB_2K")


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
    """0.5 × PullAir% + 0.5 × HR_AB, z-scored within Season+Level.

    Uses whichever components are available (blend, PullAir%-only, or HR_AB-only).
    """
    pull  = df["PullAir%"]
    hr_ab = df["HR_AB"]

    both      = pull.notna() & hr_ab.notna()
    pull_only = pull.notna() & hr_ab.isna()
    hrab_only = pull.isna() & hr_ab.notna()

    raw = pd.Series(np.nan, index=df.index, dtype=float)
    raw[both]      = 0.5 * pull[both] + 0.5 * hr_ab[both]
    raw[pull_only] = pull[pull_only]
    raw[hrab_only] = hr_ab[hrab_only]

    tmp = df.copy()
    tmp["_game_power"] = raw
    return z_within_sl(tmp, "_game_power")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = pd.read_csv(FEATURES_IN)
    print(f"Loaded {len(df):,} rows\n")

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
    # SB excluded: speed is a physical tool, not expected to improve with age —
    # a 23yo swiping 40 bags is as impressive as a 19yo doing the same.
    age_mult = (1.0 + AGE_ALPHA * (-df["Age_Z_SL"]).clip(-2.0, 2.0)).fillna(1.0)
    fantasy  = fantasy * age_mult
    disc     = disc    * age_mult
    gp       = gp      * age_mult

    # 3. Blend — missing component fills with 0 (peer-average neutral)
    ability_raw = (
        W["fantasy"]    * fantasy.fillna(0)
        + W["discipline"] * disc.fillna(0)
        + W["sb"]         * sb.fillna(0)
        + W["power"]      * gp.fillna(0)
    )

    # 4. Final 50±10 standardization
    ability_score = to_50_10(ability_raw)

    # 5. Assemble output
    out = df[["PlayerId", "Season", "Name", "Team", "Level", "Age", "PA",
              "PPPA_Z_SL", "EraSB"]].copy()
    out["ABILITY_Score"] = ability_score
    out["Fantasy_Out"]   = fantasy.round(3)
    out["Discipline"]    = disc.round(3)
    out["SB_Talent"]     = sb.round(3)
    out["Game_Power"]    = gp.round(3)
    # Key raw inputs for transparency
    out["PPPA_Z_SL_disc"] = (
        df["PPPA_Z_SL"] * df["Level"].map(LEVEL_DISCOUNT).fillna(0.10)
    ).round(4)
    out["BB_2K"]  = df["BB_2K"]
    out["SB"]     = df["SB"]
    out["SB_pct"] = df["SB_pct"]
    out["PullAir%"] = df["PullAir%"]
    out["HR_AB"]  = df["HR_AB"]

    # Discipline floor flags — which threshold(s) fired for this row.
    # "hard" = BB_2K_z <= -0.50; "soft" = -0.50 < BB_2K_z <= -0.29;
    # "whiff" = Whiff%_adj >= 1.0.  Multiple flags joined with "+".
    # bb2k_z already computed above (z-score of BB_2K within Season+Level).
    whiff_z = df["Whiff%_adj"]

    bb2k_flag = pd.Series("", index=df.index)
    bb2k_flag[bb2k_z.notna() & (bb2k_z <= -0.50)] = "hard"
    bb2k_flag[bb2k_z.notna() & (bb2k_z > -0.50) & (bb2k_z <= -0.29)] = "soft"

    whiff_flag = pd.Series("", index=df.index)
    whiff_flag[whiff_z.notna() & (whiff_z >= 1.0)] = "whiff"

    disc_flag = (bb2k_flag + whiff_flag.apply(lambda w: ("+" if w else "") + w))
    disc_flag = disc_flag.where(bb2k_flag != "", whiff_flag)
    out["Discipline_Flag"] = disc_flag

    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out):,} rows -> {OUT_PATH}\n")

    print(
        f"ABILITY_Score  mean={ability_score.mean():.1f}  "
        f"std={ability_score.std():.1f}  "
        f"min={ability_score.min():.1f}  "
        f"max={ability_score.max():.1f}\n"
    )

    # Top 20 current prospects (Season >= 2025, most PA row per player)
    current = out[out["Season"] >= 2025].copy()
    best = (
        current.sort_values("PA", ascending=False)
               .drop_duplicates("PlayerId")
               .nlargest(20, "ABILITY_Score")
    )
    print("Top 20 ABILITY (Season >= 2025, one row per player):")
    print(
        best[["Name", "Team", "Level", "Age", "PA", "ABILITY_Score",
              "Fantasy_Out", "Discipline", "SB_Talent", "Game_Power"]
             ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
