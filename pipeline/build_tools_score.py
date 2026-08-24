"""Build TOOLS_Score for every player-season row in prospect_features.csv.

TOOLS_Score measures raw physical skills, normalized for era and level.
Output: data/rankings/tools_scores.csv (one row per player-season-level)

Component weights:
  Discipline   45%
  Power        35%
  Athleticism  20%

Age adjustment (AGE_ALPHA = 0.20):
  Applied to Discipline and Power only — athleticism (speed) is a physical tool
  not expected to improve with age, same rationale as SB in ABILITY_Score.
  Each component is multiplied by (1 + 0.20 × −Age_Z_SL), clipped to ±2 SD.
  A player 2 SD younger than peers gets a ~40% boost; 2 SD older gets a ~40% cut.

Discipline sub-weights:
  Full tier (AAA 2023+, Chase% + Z-Contact% available):
    -Chase%_adj     40%   lower chase = better plate discipline
    ZContact%_adj   35%   higher z-contact = better in-zone contact
    -Whiff%_adj     25%   lower whiff = better overall contact
  Fallback (all other rows):
    -Whiff%_adj    100%

Power sub-weights:
  Full tier (ProspectSavant rows: MaxEV + EV90 available):
    MaxEV_z   35%   ceiling exit velocity
    EV90_z    35%   90th-percentile EV (consistent hard contact)
    HRFB_adj  30%   actual HR production rate (era+level adjusted)
  Fallback (all other rows — no EV data to validate power):
    HRFB_adj × career_shrink  100%
    career_shrink = min(career_FBs_est / FB_CAREER_THRESHOLD, 1.0)
    FB_CAREER_THRESHOLD = 150 FBs (~3 solid MiLB seasons)
    Rationale: single-season HR/FB tops out at r≈0.52 even at 100 FBs in MiLB.
    Without EV validation, extreme single-season HRFB values are dominated by
    small-sample noise.  Shrinking toward neutral by career FB count anchors the
    signal to players with accumulated power evidence rather than one hot streak.

Athleticism sub-weights:
  Full tier (ProspectSavant rows: Spd available):
    Spd_z     50%   measured sprint speed
    3B_PA_adj 50%   in-game speed expression (era+level adjusted)
  Fallback (all other rows):
    3B_PA_adj 100%

Normalization:
  MaxEV, EV90, Spd — raw values; z-scored within Level before blending.
  Era-adjusted _adj columns — already z-scored within Level x era; used directly.
  Each component standardized to mean=0, std=1 across all scored rows.
  Missing component → filled with 0 (era+level average, neutral assumption).
  Final TOOLS_Score standardized to 50±10 across all scored rows, clipped at 0.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR    = Path(__file__).resolve().parent.parent / "data"
FEATURES_IN = DATA_DIR / "rankings" / "prospect_features.csv"
OUT_PATH    = DATA_DIR / "rankings" / "tools_scores.csv"

# Top-level component weights
W = dict(discipline=0.45, power=0.35, athleticism=0.20)

# Age adjustment: Discipline and Power are multiplied by (1 + AGE_ALPHA × −Age_Z_SL).
# Athleticism excluded — speed is a physical tool, not expected to improve with age.
AGE_ALPHA = 0.20

# Discipline sub-weights (full tier)
WD = dict(chase=0.40, zcontact=0.35, whiff=0.25)

# Power sub-weights (full tier)
WP = dict(maxev=0.35, ev90=0.35, hrfb=0.30)

# Athleticism sub-weights (full tier)
WA = dict(spd=0.50, tb3pa=0.50)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def z_within_level(df: pd.DataFrame, col: str) -> pd.Series:
    """Z-score a raw-unit column within each Level (for PS metrics not yet z-scored)."""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for level, idx in df.groupby("Level", observed=True).groups.items():
        vals  = df.loc[idx, col].dropna()
        if len(vals) < 2:
            continue
        mu, sig = vals.mean(), vals.std()
        if sig > 0:
            out.loc[vals.index] = (vals - mu) / sig
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

def build_discipline(df: pd.DataFrame) -> pd.Series:
    inv_whiff   = -df["Whiff%_adj"]
    inv_chase   = -df["Chase%_adj"]
    z_contact   = df["ZContact%_adj"]

    full = (
        df["Chase%_adj"].notna()
        & df["ZContact%_adj"].notna()
        & df["Whiff%_adj"].notna()
    )
    whiff_only = ~full & df["Whiff%_adj"].notna()

    score = pd.Series(np.nan, index=df.index, dtype=float)
    score[full] = (
        WD["chase"]    * inv_chase[full]
        + WD["zcontact"] * z_contact[full]
        + WD["whiff"]    * inv_whiff[full]
    )
    score[whiff_only] = inv_whiff[whiff_only]
    return score


FB_CAREER_THRESHOLD = 150.0   # total career FBs for full HRFB weight in fallback tier


def build_power(
    df: pd.DataFrame, maxev_z: pd.Series, ev90_z: pd.Series
) -> pd.Series:
    hrfb = df["HRFB_adj"]

    full      = maxev_z.notna() & ev90_z.notna() & hrfb.notna()
    hrfb_only = ~(maxev_z.notna() & ev90_z.notna()) & hrfb.notna()

    # Career-FB shrinkage for fallback tier only.  EV data in the full tier provides
    # an independent validation of power, so HRFB at 30% weight there is fine at full
    # strength.  In the fallback tier HRFB carries 100% of Power, and single-season
    # HR/FB is too noisy at typical prospect fly-ball counts; shrink toward neutral
    # until the player has accumulated enough career evidence.
    # Use prior-season FBs only (excludes this row's own contribution) so the
    # current season cannot supply both the extreme HRFB_adj and its own shrinkage weight.
    career_fbs = df["prior_FBs_est"].fillna(0.0)
    career_shrink = (career_fbs / FB_CAREER_THRESHOLD).clip(upper=1.0)

    score = pd.Series(np.nan, index=df.index, dtype=float)
    score[full] = (
        WP["maxev"] * maxev_z[full]
        + WP["ev90"]  * ev90_z[full]
        + WP["hrfb"]  * hrfb[full]
    )
    score[hrfb_only] = (hrfb * career_shrink)[hrfb_only]
    return score


def build_athleticism(df: pd.DataFrame, spd_z: pd.Series) -> pd.Series:
    tb3pa = df["3B_PA_adj"]

    full     = spd_z.notna() & tb3pa.notna()
    pa3_only = ~spd_z.notna() & tb3pa.notna()

    score = pd.Series(np.nan, index=df.index, dtype=float)
    score[full]     = WA["spd"] * spd_z[full] + WA["tb3pa"] * tb3pa[full]
    score[pa3_only] = tb3pa[pa3_only]
    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = pd.read_csv(FEATURES_IN)
    print(f"Loaded {len(df):,} rows\n")

    # 1. Z-score raw ProspectSavant metrics within Level
    maxev_z = z_within_level(df, "MaxEV")
    ev90_z  = z_within_level(df, "EV90")
    spd_z   = z_within_level(df, "Spd")
    print(f"Level-normalized:  MaxEV={maxev_z.notna().sum():,}  "
          f"EV90={ev90_z.notna().sum():,}  Spd={spd_z.notna().sum():,}")

    # 2. Raw component scores
    disc  = build_discipline(df)
    power = build_power(df, maxev_z, ev90_z)
    ath   = build_athleticism(df, spd_z)

    # 3. Standardize each component to mean=0, std=1
    disc  = standardize_component(disc)
    power = standardize_component(power)
    ath   = standardize_component(ath)

    # 3b. Age multiplier on Discipline and Power only.
    # Athleticism excluded — speed is a physical tool, not expected to improve with age.
    age_mult = (1.0 + AGE_ALPHA * (-df["Age_Z_SL"]).clip(-2.0, 2.0)).fillna(1.0)
    disc  = disc  * age_mult
    power = power * age_mult

    # 4. Blend — missing component fills with 0 (era+level average, neutral)
    tools_raw = (
        W["discipline"]    * disc.fillna(0)
        + W["power"]       * power.fillna(0)
        + W["athleticism"] * ath.fillna(0)
    )

    # 5. Final 50±10 standardization
    tools_score = to_50_10(tools_raw)

    # 6. Assemble output
    out = df[["PlayerId", "Season", "Name", "Team", "Level", "Age", "PA",
              "EraK%", "EraHR/FB", "EraSB"]].copy()
    out["TOOLS_Score"]  = tools_score
    out["Discipline"]   = disc.round(3)
    out["Power"]        = power.round(3)
    out["Athleticism"]  = ath.round(3)
    # Key inputs for audit
    out["Whiff%_adj"]   = df["Whiff%_adj"]
    out["Chase%_adj"]   = df["Chase%_adj"]
    out["ZContact%_adj"]= df["ZContact%_adj"]
    out["HRFB_adj"]        = df["HRFB_adj"]
    out["career_FBs_est"]  = df["career_FBs_est"]
    out["prior_FBs_est"]   = df["prior_FBs_est"]
    out["MaxEV_z"]      = maxev_z.round(3)
    out["EV90_z"]       = ev90_z.round(3)
    out["Spd_z"]        = spd_z.round(3)
    out["3B_PA_adj"]    = df["3B_PA_adj"]
    out["Age_Z_SL"]     = df["Age_Z_SL"]

    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out):,} rows -> {OUT_PATH}\n")

    # 7. Summary stats
    print(f"TOOLS_Score  mean={tools_score.mean():.1f}  "
          f"std={tools_score.std():.1f}  "
          f"min={tools_score.min():.1f}  "
          f"max={tools_score.max():.1f}\n")

    # Tier coverage
    full_disc = (
        df["Chase%_adj"].notna() & df["ZContact%_adj"].notna() & df["Whiff%_adj"].notna()
    )
    full_pow = maxev_z.notna() & ev90_z.notna() & df["HRFB_adj"].notna()
    full_ath = spd_z.notna() & df["3B_PA_adj"].notna()

    print("Tier coverage:")
    print(f"  Discipline  full={full_disc.sum():>6,}  "
          f"whiff-only={( ~full_disc & df['Whiff%_adj'].notna()).sum():,}")
    print(f"  Power       full={full_pow.sum():>6,}  "
          f"hrfb-only= {(~(maxev_z.notna()&ev90_z.notna()) & df['HRFB_adj'].notna()).sum():,}")
    print(f"  Athleticism full={full_ath.sum():>6,}  "
          f"3bpa-only= {(~spd_z.notna() & df['3B_PA_adj'].notna()).sum():,}")

    # Top 20 current prospects (Season >= 2025, one row per player)
    current = out[out["Season"] >= 2025].copy()
    best    = (
        current.sort_values(["PA"], ascending=False)
               .drop_duplicates("PlayerId")
               .nlargest(20, "TOOLS_Score")
    )
    print("\nTop 20 TOOLS (Season >= 2025, one row per player):")
    print(
        best[["Name", "Team", "Level", "Age", "TOOLS_Score",
              "Discipline", "Power", "Athleticism"]
             ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
