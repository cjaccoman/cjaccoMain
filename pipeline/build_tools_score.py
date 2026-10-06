"""Build TOOLS_Score for every player-season row in prospect_features.csv.

TOOLS_Score measures raw physical skills, normalized for era and level.
Output: columns written back into prospect_features.csv in place.

Component weights:
  Discipline   45%
  Power        35%
  Athleticism  20%

Age adjustment: none.
  All stats are era+level normalized against peers — adding an age multiplier
  would double-count age. A 19yo in AA is already compared only against other
  19yo AA players; their Chase% z-score already reflects that context.

Discipline sub-weights:
  Full tier (Chase% + Z-Contact% available — AAA 2023+, all levels 2026 via ProspectSavant):
    Without P95_Whiff%:
      -Chase%_adj     40%   lower chase = better plate discipline
      ZContact%_adj   35%   higher z-contact = better in-zone contact
      -Whiff%_adj     25%   lower whiff = better overall contact
    With P95_Whiff% (AAA 2023-2026 + A-ball Trackman parks):
      -Chase%_adj     35%
      ZContact%_adj   30%
      -Whiff%_adj     20%
      -P95_Whiff%_z   15%   velocity handling — whiff rate vs 95+ mph pitches
  Fallback (all other rows — no Chase%/Z-Contact% data):
    Without P95_Whiff%:
      -Whiff%_adj     60%   contact avoidance
      BB%_adj         40%   pitch recognition (walk rate, era+level adjusted)
    With P95_Whiff%:
      -Whiff%_adj     50%
      BB%_adj         35%
      -P95_Whiff%_z   15%
    Whiff%-only + P95_Whiff% (no BB% available):
      -Whiff%_adj     85%
      -P95_Whiff%_z   15%

Power sub-weights:
  Full tier (ProspectSavant rows: MaxEV + EV90 available — AAA 2023-2026):
    MaxEV_z   35%   ceiling exit velocity
    EV90_z    35%   90th-percentile EV (consistent hard contact)
    HRFB_adj  30%   actual HR production rate (era+level adjusted)
  MaxEV-only tier (game-feed rows with MaxEV but no EV90 — A-ball FSL parks 2021+):
    MaxEV_z   70%   ceiling exit velocity (higher weight to compensate for missing EV90)
    HRFB_adj  30%   actual HR production rate (no shrinkage — MaxEV validates power)
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
FEATURES_IN = DATA_DIR / "rankings" / "prospect_features.parquet"

# Top-level component weights
W = dict(discipline=0.45, power=0.35, athleticism=0.20)

# Discipline sub-weights
WD = dict(chase=0.40, zcontact=0.35, whiff=0.25)          # full tier, no P95
WD_P95 = dict(chase=0.35, zcontact=0.30, whiff=0.20, p95w=0.15)  # full tier + P95

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

def build_discipline(df: pd.DataFrame, p95_whiff_z: pd.Series) -> pd.Series:
    inv_whiff   = -df["Whiff%_adj"]
    inv_chase   = -df["Chase%_adj"]
    z_contact   = df["ZContact%_adj"]
    inv_p95w    = -p95_whiff_z  # higher P95_Whiff% = worse velocity handling

    full = (
        df["Chase%_adj"].notna()
        & df["ZContact%_adj"].notna()
        & df["Whiff%_adj"].notna()
    )
    fallback = ~full & df["Whiff%_adj"].notna()
    bb_adj   = df["BB%_adj"]

    # P95 sub-masks within each tier
    full_p95    = full     & inv_p95w.notna()
    full_nop95  = full     & inv_p95w.isna()
    fb_p95_bb   = fallback & inv_p95w.notna() & bb_adj.notna()
    fb_nop95_bb = fallback & inv_p95w.isna()  & bb_adj.notna()
    fb_p95_w    = fallback & inv_p95w.notna() & bb_adj.isna()
    fb_nop95_w  = fallback & inv_p95w.isna()  & bb_adj.isna()

    score = pd.Series(np.nan, index=df.index, dtype=float)

    # Full tier + P95
    score[full_p95] = (
        WD_P95["chase"]    * inv_chase[full_p95]
        + WD_P95["zcontact"] * z_contact[full_p95]
        + WD_P95["whiff"]    * inv_whiff[full_p95]
        + WD_P95["p95w"]     * inv_p95w[full_p95]
    )
    # Full tier without P95
    score[full_nop95] = (
        WD["chase"]    * inv_chase[full_nop95]
        + WD["zcontact"] * z_contact[full_nop95]
        + WD["whiff"]    * inv_whiff[full_nop95]
    )
    # Fallback: Whiff% + BB% + P95
    score[fb_p95_bb] = (
        0.50 * inv_whiff[fb_p95_bb]
        + 0.35 * bb_adj[fb_p95_bb]
        + 0.15 * inv_p95w[fb_p95_bb]
    )
    # Fallback: Whiff% + BB%, no P95
    score[fb_nop95_bb] = 0.60 * inv_whiff[fb_nop95_bb] + 0.40 * bb_adj[fb_nop95_bb]
    # Fallback: Whiff% + P95 only (no BB%)
    score[fb_p95_w] = 0.85 * inv_whiff[fb_p95_w] + 0.15 * inv_p95w[fb_p95_w]
    # Fallback: Whiff% only
    score[fb_nop95_w] = inv_whiff[fb_nop95_w]

    return score


FB_CAREER_THRESHOLD = 150.0   # total career FBs for full HRFB weight in fallback tier


def build_power(
    df: pd.DataFrame, maxev_z: pd.Series, ev90_z: pd.Series
) -> pd.Series:
    hrfb = df["HRFB_adj"]

    full       = maxev_z.notna() & ev90_z.notna() & hrfb.notna()
    maxev_only = maxev_z.notna() & ev90_z.isna()  & hrfb.notna()
    hrfb_only  = maxev_z.isna()                   & hrfb.notna()

    # Career-FB shrinkage for fallback (hrfb_only) tier only.  Any tier with EV data
    # provides independent validation of power, so HRFB at 30% weight there is fine
    # at full strength.  In the fallback tier HRFB carries 100% of Power, and single-
    # season HR/FB is too noisy at typical prospect fly-ball counts; shrink toward
    # neutral until the player has accumulated enough career evidence.
    # Use prior-season FBs only (excludes this row's own contribution) so the
    # current season cannot supply both the extreme HRFB_adj and its own shrinkage weight.
    career_fbs = df["prior_FBs_est"].fillna(0.0)
    career_shrink = (career_fbs / FB_CAREER_THRESHOLD).clip(upper=1.0)

    score = pd.Series(np.nan, index=df.index, dtype=float)
    # Full tier: MaxEV + EV90 + HRFB (ProspectSavant AAA 2023-2026)
    score[full] = (
        WP["maxev"] * maxev_z[full]
        + WP["ev90"]  * ev90_z[full]
        + WP["hrfb"]  * hrfb[full]
    )
    # MaxEV-only tier: MaxEV + HRFB (game-feed A-ball FSL parks 2021+)
    score[maxev_only] = (
        0.70 * maxev_z[maxev_only]
        + 0.30 * hrfb[maxev_only]
    )
    # Fallback tier: HRFB with career-FB shrinkage (no EV validation)
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
    df = pd.read_parquet(FEATURES_IN)
    print(f"Loaded {len(df):,} rows\n")

    # 1. Z-score raw metrics within Level
    maxev_z    = z_within_level(df, "MaxEV")
    ev90_z     = z_within_level(df, "EV90")
    spd_z      = z_within_level(df, "Spd")
    p95_whiff_z = z_within_level(df, "P95_Whiff%")
    print(f"Level-normalized:  MaxEV={maxev_z.notna().sum():,}  "
          f"EV90={ev90_z.notna().sum():,}  Spd={spd_z.notna().sum():,}  "
          f"P95_Whiff%={p95_whiff_z.notna().sum():,}")

    # 2. Raw component scores
    disc  = build_discipline(df, p95_whiff_z)
    power = build_power(df, maxev_z, ev90_z)
    ath   = build_athleticism(df, spd_z)

    # 3. Standardize each component to mean=0, std=1
    disc  = standardize_component(disc)
    power = standardize_component(power)
    ath   = standardize_component(ath)

    # 4. Blend — missing component fills with 0 (era+level average, neutral)
    tools_raw = (
        W["discipline"]    * disc.fillna(0)
        + W["power"]       * power.fillna(0)
        + W["athleticism"] * ath.fillna(0)
    )

    # 5. Final 50±10 standardization
    tools_score = to_50_10(tools_raw)

    # 6. Write score columns back into prospect_features.csv in place
    df["TOOLS_Score"]    = tools_score
    df["TOOLS_Disc"]     = disc.round(3)
    df["TOOLS_Power"]    = power.round(3)
    df["TOOLS_Ath"]      = ath.round(3)
    df["MaxEV_z"]        = maxev_z.round(3)
    df["EV90_z"]         = ev90_z.round(3)
    df["Spd_z"]          = spd_z.round(3)
    df["P95_Whiff%_z"]   = p95_whiff_z.round(3)

    df.to_parquet(FEATURES_IN, index=False)
    print(f"Wrote {len(df):,} rows -> {FEATURES_IN}\n")

    # 7. Summary stats
    print(f"TOOLS_Score  mean={tools_score.mean():.1f}  "
          f"std={tools_score.std():.1f}  "
          f"min={tools_score.min():.1f}  "
          f"max={tools_score.max():.1f}\n")

    # Tier coverage
    full_disc = (
        df["Chase%_adj"].notna() & df["ZContact%_adj"].notna() & df["Whiff%_adj"].notna()
    )
    full_pow     = maxev_z.notna() & ev90_z.notna() & df["HRFB_adj"].notna()
    maxev_only   = maxev_z.notna() & ev90_z.isna()  & df["HRFB_adj"].notna()
    hrfb_only    = maxev_z.isna()                   & df["HRFB_adj"].notna()
    full_ath     = spd_z.notna() & df["3B_PA_adj"].notna()
    has_p95      = p95_whiff_z.notna()

    print("Tier coverage:")
    print(f"  Discipline  full+P95={( full_disc & has_p95).sum():>5,}  "
          f"full={full_disc.sum():>6,}  "
          f"fallback+P95={(~full_disc & df['Whiff%_adj'].notna() & has_p95).sum():,}  "
          f"whiff-only={(~full_disc & df['Whiff%_adj'].notna() & ~has_p95).sum():,}")
    print(f"  Power       full={full_pow.sum():>6,}  "
          f"maxev-only={maxev_only.sum():>5,}  "
          f"hrfb-only={hrfb_only.sum():,}")
    print(f"  Athleticism full={full_ath.sum():>6,}  "
          f"3bpa-only= {(~spd_z.notna() & df['3B_PA_adj'].notna()).sum():,}")

    # Top 20 current prospects (Season >= 2025, one row per player)
    current = df[df["Season"] >= 2025].copy()
    best    = (
        current.sort_values(["PA"], ascending=False)
               .drop_duplicates("PlayerId")
               .nlargest(20, "TOOLS_Score")
    )
    print("\nTop 20 TOOLS (Season >= 2025, one row per player):")
    print(
        best[["Name", "Team", "Level", "Age", "TOOLS_Score",
              "TOOLS_Disc", "TOOLS_Power", "TOOLS_Ath"]
             ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
