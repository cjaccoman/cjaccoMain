"""Aggregate per-season TOOLS and ABILITY scores to one value per current prospect.

Eligibility:
  Most recent season >= 2025 (active 2025 or 2026)
  Age at most recent season <= 24
  Career MLB PA < 50 (via hist_mlb_data.csv, PlayerId + name fallback)

Methodology:
  For each player, compute a PA × level_weight career average of TOOLS_Score and
  ABILITY_Score. Shrinkage is applied per level before averaging: each level's
  PA-weighted average is shrunk toward the pool mean (50) based on that level's
  total weighted PA vs a fixed threshold. Small-sample levels contribute near-neutral
  values; full-season data contributes at face value.

  level_weight = LEVEL_DISCOUNT factors (AAA=1.00 → R=0.39).
  Per-level shrinkage:
    level_den    = sum(PA × level_wt) for that player × level
    shrink       = min(level_den / threshold, 1.0)
    shrunk_score = 50 + shrink × (level_avg − 50)
  Thresholds (fixed, in AAA-equivalent PA):
    TOOLS:   250  (tools metrics noisier → more data needed)
    ABILITY: 175

Combined_Score = 0.50 × Current_Score + 0.50 × OVR_Score
  Current_Score  = 0.40 × TOOLS_Score + 0.40 × ABILITY_Score + 0.20 × Age_Score
    (all three standardized to 50±10 within current pool)
  Age_Score = −Age_Z_SL standardized to 50±10 (younger than peers = higher score).
    Uses most-recent season Age_Z_SL. Age enters the model twice: here as a
    standalone 20% component, and inside TOOLS/ABILITY via AGE_ALPHA=0.20
    per-component multiplier (a player 2 SD older takes a ~40% cut to each
    component before blending).
  OVR_Score = Combined_Score from prospect_scores_ovr.csv
    (0.40 × TOOLS + 0.40 × ABILITY + 0.20 × Slope_Score, standardized
    within the full historical pool; Slope_Score is the PA-weighted
    PPPA_Z_SL trajectory across levels, making OVR genuinely different
    from Current_Score and the blend meaningful)
  Players missing from OVR pool receive neutral 50 for that component.

Prerequisite: build_prospect_scores_ovr.py must run before this script.

Output: data/rankings/prospect_scores.csv
"""

import re
import unicodedata
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR      = Path(__file__).resolve().parent.parent / "data"
TOOLS_PATH    = DATA_DIR / "rankings" / "tools_scores.csv"
ABILITY_PATH  = DATA_DIR / "rankings" / "ability_scores.csv"
FEATURES_PATH = DATA_DIR / "rankings" / "prospect_features.csv"
MLB_PATH      = DATA_DIR / "historical" / "hist_mlb_data.csv"
OVR_PATH      = DATA_DIR / "rankings" / "prospect_scores_ovr.csv"
OUT_PATH      = DATA_DIR / "rankings" / "prospect_scores.csv"
HIT_PATH      = DATA_DIR / "api" / "milb_hitting.csv"
POS_PATH      = DATA_DIR / "api" / "player_positions.csv"

# Discipline gate — applied post-blend to Combined_Score.
# Thresholds are percentile cutoffs applied to disc_composite_z (z-scored within pool).
# Penalties are in Combined_Score points (scale: top prospects range ~70-85).
# Slope modifier fires when recent_BB2K differs from career_BB2K by >= TREND_THRESHOLD
# in raw BB_2K rate (BB% − 2×K%); positive trend = improving discipline.
DISC_HARD_FLOOR   = -1.00   # bottom ~16% of pool composite
DISC_SOFT_FLOOR   = -0.67   # bottom ~25% of pool composite
DISC_HARD_PENALTY =  3.0    # pts deducted from Combined_Score
DISC_SOFT_PENALTY =  1.5
DISC_PENALTY_CAP  =  4.0    # max total deduction
TREND_THRESHOLD   =  0.02   # raw BB_2K rate shift to trigger slope modifier
TREND_IMPROVE_MUL =  0.60   # soften penalty when visibly improving
TREND_WORSEN_MUL  =  1.25   # harden penalty when visibly worsening

MAX_PROSPECT_AGE  = 24
MIN_SEASON        = 2025
MLB_PA_EXCL       = 50

LEVEL_DISCOUNT = {"AAA": 1.00, "AA": 0.84, "A+": 0.71, "A": 0.57, "R": 0.39}

TOOLS_PA_THRESH    = 250
ABILITY_PA_THRESH  = 175
ARCHETYPE_PATH     = DATA_DIR / "rankings" / "archetype_labels.csv"

W_TOOLS   = 0.40
W_ABILITY = 0.40
W_AGE     = 0.20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(s) -> str:
    s = re.sub(r"\s+jr$", "", str(s).lower().strip().replace(".", ""))
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn").strip()


def _wt_avg_shrunk(df: pd.DataFrame, score_col: str, threshold: float) -> pd.Series:
    valid = df[["PlayerId", "Level", score_col, "wt"]].dropna(subset=[score_col]).copy()
    valid["_wtd"] = valid[score_col] * valid["wt"]

    lgrp      = valid.groupby(["PlayerId", "Level"], observed=True)
    level_den = lgrp["wt"].sum()
    level_num = lgrp["_wtd"].sum()
    level_avg = level_num / level_den

    level_shrink = (level_den / threshold).clip(upper=1.0)
    level_shrunk = 50 + level_shrink * (level_avg - 50)

    final_num = (level_shrunk * level_den).groupby(level=0).sum()
    final_den = level_den.groupby(level=0).sum()
    return (final_num / final_den).rename(score_col)


def to_50_10(s: pd.Series) -> pd.Series:
    mu, sig = s.mean(), s.std()
    if sig > 0:
        return (50 + 10 * (s - mu) / sig).clip(lower=0, upper=100).round(2)
    return pd.Series(50.0, index=s.index)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    tools = pd.read_csv(
        TOOLS_PATH,
        usecols=["PlayerId", "Season", "Name", "Team", "Level", "Age", "PA",
                 "TOOLS_Score", "Age_Z_SL"],
    )
    ability = pd.read_csv(
        ABILITY_PATH,
        usecols=["PlayerId", "Season", "Level", "ABILITY_Score", "Discipline_Flag"],
    )
    scores = tools.merge(ability, on=["PlayerId", "Season", "Level"], how="left")
    scores["level_wt"] = scores["Level"].map(LEVEL_DISCOUNT).fillna(0.39)
    scores["wt"]       = scores["PA"] * scores["level_wt"]
    print(f"Loaded {len(scores):,} player-season rows")

    # MLB exclusion sets
    mlb = pd.read_csv(MLB_PATH, usecols=["PlayerId", "Name", "PA"])
    mlb["_norm"]   = mlb["Name"].apply(_norm)
    mlb_id_pa      = mlb.groupby("PlayerId")["PA"].sum()
    mlb_name_pa    = mlb.groupby("_norm")["PA"].sum()
    excl_ids       = set(mlb_id_pa[mlb_id_pa >= MLB_PA_EXCL].index.astype(str))
    excl_norms     = set(mlb_name_pa[mlb_name_pa >= MLB_PA_EXCL].index)

    # Most-recent row per player — within the same season prefer highest level,
    # then most PA as a final tiebreaker.
    LEVEL_ORDER = {"AAA": 5, "AA": 4, "A+": 3, "A": 2, "R": 1}
    scores["_level_ord"] = scores["Level"].map(LEVEL_ORDER).fillna(0)
    latest = (
        scores
        .sort_values(["Season", "_level_ord", "PA"], ascending=[False, False, False])
        .groupby("PlayerId", sort=False)
        .first()
        .reset_index()
        .drop(columns="_level_ord")
    )
    scores = scores.drop(columns="_level_ord")

    # Eligibility: season floor + age cap
    pool = latest[
        (latest["Season"] >= MIN_SEASON) & (latest["Age"] <= MAX_PROSPECT_AGE)
    ].copy()
    print(f"Eligible (Season>={MIN_SEASON}, Age<={MAX_PROSPECT_AGE}): {len(pool):,}")

    # MLB exclusion
    pool["_norm"] = pool["Name"].apply(_norm)
    pool = pool[
        ~pool["PlayerId"].astype(str).isin(excl_ids) & ~pool["_norm"].isin(excl_norms)
    ].drop(columns="_norm")
    print(f"After MLB exclusion (>={MLB_PA_EXCL} career PA): {len(pool):,}")

    # Career-average scores with per-level shrinkage
    hist        = scores[scores["PlayerId"].isin(pool["PlayerId"])]
    tools_avg   = _wt_avg_shrunk(hist, "TOOLS_Score",   TOOLS_PA_THRESH)
    ability_avg = _wt_avg_shrunk(hist, "ABILITY_Score", ABILITY_PA_THRESH)
    total_wpa   = hist.groupby("PlayerId")["wt"].sum().rename("Total_Weighted_PA")
    career_pa   = hist.groupby("PlayerId")["PA"].sum().rename("Career_PA")
    last_season = hist.groupby("PlayerId")["Season"].max().rename("Last_Season")

    pool["TOOLS_Score"]       = pool["PlayerId"].map(tools_avg)
    pool["ABILITY_Score"]     = pool["PlayerId"].map(ability_avg)
    pool["Total_Weighted_PA"] = pool["PlayerId"].map(total_wpa).round(2)
    pool["Career_PA"]         = pool["PlayerId"].map(career_pa).astype(int)
    pool["Last_Season"]       = pool["PlayerId"].map(last_season).astype(int)

    # Discipline_Flag from most-recent Season>=2025 row per player.
    # Shows which thresholds fired in their latest qualifying season.
    recent_flags = (
        scores[scores["Season"] >= 2025]
        .sort_values("Season", ascending=False)
        .drop_duplicates("PlayerId")[["PlayerId", "Discipline_Flag"]]
        .set_index("PlayerId")["Discipline_Flag"]
        .fillna("")
    )
    pool["Discipline_Flag"] = pool["PlayerId"].map(recent_flags).fillna("")

    # Career discipline flag — PA-weighted BB_2K and Whiff%_adj across non-AAA seasons.
    # AAA excluded to avoid survivorship bias: players who reach AAA are already a
    # filtered group; their AAA discipline looks artificially clean vs. true development.
    feats = pd.read_csv(
        FEATURES_PATH,
        usecols=["PlayerId", "Season", "Level", "PA", "BB_2K", "Whiff%_adj"],
    )
    # Career BB_2K: PA-weighted avg of raw rate, excluding AAA to avoid survivorship bias
    sub_career = feats[feats["Level"] != "AAA"].dropna(subset=["BB_2K"])
    career_bb2k = (
        (sub_career["BB_2K"] * sub_career["PA"]).groupby(sub_career["PlayerId"]).sum()
        / sub_career.groupby("PlayerId")["PA"].sum()
    ).rename("Career_BB2K")

    pool["Career_BB2K"] = pool["PlayerId"].map(career_bb2k)

    # Discipline slope: PA-weighted OLS of BB_2K on Season across all career rows.
    # Positive = improving discipline over time. Requires >= 2 qualifying seasons
    # (PA >= 50, non-null BB_2K) to compute; NaN otherwise (no slope modifier fires).
    def _bb2k_slope(grp: pd.DataFrame) -> float:
        valid = grp[(grp["PA"] >= 50) & grp["BB_2K"].notna()]
        if len(valid) < 2:
            return np.nan
        x = valid["Season"].values.astype(float)
        y = valid["BB_2K"].values
        w = valid["PA"].values.astype(float)
        xbar = np.average(x, weights=w)
        ybar = np.average(y, weights=w)
        num = np.sum(w * (x - xbar) * (y - ybar))
        den = np.sum(w * (x - xbar) ** 2)
        return float(num / den) if den > 1e-9 else np.nan

    disc_slopes = (
        feats.groupby("PlayerId", group_keys=False)
        .apply(_bb2k_slope, include_groups=False)
    )
    pool["Disc_Slope"] = pool["PlayerId"].map(disc_slopes).round(4)

    # Composite: career BB_2K only — stable, noise-resistant. Slope captures
    # trajectory separately; no need for a noisy recent-season component here.
    pool["Disc_Composite"] = pool["Career_BB2K"]
    dc_vals = pool["Disc_Composite"].dropna()
    if len(dc_vals) > 1 and dc_vals.std() > 0:
        mu, sig = dc_vals.mean(), dc_vals.std()
        pool["Disc_Composite_Z"] = ((pool["Disc_Composite"] - mu) / sig).round(4)
    else:
        pool["Disc_Composite_Z"] = np.nan

    # Career Whiff flag (informational only — Whiff signal lives in TOOLS)
    wh_rows = feats[feats["Level"] != "AAA"].dropna(subset=["Whiff%_adj"])
    career_whiff = (
        (wh_rows["Whiff%_adj"] * wh_rows["PA"]).groupby(wh_rows["PlayerId"]).sum()
        / wh_rows.groupby("PlayerId")["PA"].sum()
    )
    pool["Career_Whiff_Z"] = pool["PlayerId"].map(career_whiff)
    pool["Career_Disc_Flag"] = ""   # populated below after gate computation

    # Standardize TOOLS and ABILITY within current pool to 50±10
    pool["TOOLS_Score"]   = to_50_10(pool["TOOLS_Score"].fillna(50))
    pool["ABILITY_Score"] = to_50_10(pool["ABILITY_Score"].fillna(50))

    # Age_Score: Age_Z_SL inverted (younger than peers = positive), standardized to 50±10.
    # Uses most-recent season Age_Z_SL from the latest row per player.
    recent_age_z = (
        scores.sort_values("Season", ascending=False)
        .drop_duplicates("PlayerId")
        .set_index("PlayerId")["Age_Z_SL"]
    )
    pool["Age_Z_SL"] = pool["PlayerId"].map(recent_age_z)
    pool["Age_Score"] = to_50_10((-pool["Age_Z_SL"]).fillna(0))  # invert: younger = higher

    # Current model score (TOOLS + ABILITY + Age blend)
    pool["Current_Score"] = (
        W_TOOLS   * pool["TOOLS_Score"]
        + W_ABILITY * pool["ABILITY_Score"]
        + W_AGE    * pool["Age_Score"]
    ).round(2)

    # OVR score — career arc relative to all historical prospects
    # Prerequisite: build_prospect_scores_ovr.py must have run first
    ovr = pd.read_csv(OVR_PATH, usecols=["PlayerId", "Combined_Score"])
    ovr = ovr.rename(columns={"Combined_Score": "OVR_Score"})
    pool = pool.merge(ovr, on="PlayerId", how="left")
    pool["OVR_Score"] = pool["OVR_Score"].fillna(50.0)   # not in OVR pool → neutral
    n_missing_ovr = (pool["OVR_Score"] == 50.0).sum()
    if n_missing_ovr:
        print(f"  {n_missing_ovr} players not in OVR pool -> OVR_Score set to 50")

    # Archetype labels (requires build_archetypes.py to have run first)
    if ARCHETYPE_PATH.exists():
        arch = pd.read_csv(ARCHETYPE_PATH, usecols=["PlayerId", "Archetype"])
        pool = pool.merge(arch, on="PlayerId", how="left")
        pool["Archetype"] = pool["Archetype"].fillna("")
        n_arch = (pool["Archetype"] != "").sum()
        print(f"Archetype coverage: {n_arch:,} / {len(pool):,}")
    else:
        pool["Archetype"] = ""

    # Archetype level-shift adjustment.
    # Power/K-Risk receives +3: calibrated to career PPPA_Z residual (+0.109),
    # estimated slope beta ~0.046 PPPA_Z per score point -> 2.4 pts, rounded to 3.
    pool["Archetype_Adj"] = 0.0
    pool.loc[pool["Archetype"] == "Power/K-Risk", "Archetype_Adj"] = 3.0
    n_adj = (pool["Archetype_Adj"] != 0).sum()
    print(f"Archetype_Adj applied: {n_adj:,} players (+3.0 Power/K-Risk)")

    # Final blend
    pool["Combined_Score"] = (
        0.50 * pool["Current_Score"]
        + 0.50 * pool["OVR_Score"]
        + pool["Archetype_Adj"]
    ).round(2)

    # Post-blend discipline gate — applied to Combined_Score.
    # Uses Disc_Composite_Z (50/50 career+recent BB_2K, z-scored within pool).
    # Slope modifier uses PA-weighted OLS of BB_2K on Season (Disc_Slope); softens
    # the penalty for improving trajectories, hardens for worsening ones.
    dcz   = pool["Disc_Composite_Z"]
    trend = pool["Disc_Slope"]

    disc_pen = pd.Series(0.0, index=pool.index)
    has_dc = dcz.notna()

    disc_pen[has_dc & (dcz <= DISC_HARD_FLOOR)]                              += DISC_HARD_PENALTY
    disc_pen[has_dc & (dcz > DISC_HARD_FLOOR) & (dcz <= DISC_SOFT_FLOOR)]   += DISC_SOFT_PENALTY

    # Slope modifier: only fires when trend is meaningful AND a penalty exists
    improving = trend.notna() & (trend >= TREND_THRESHOLD) & (disc_pen > 0)
    worsening = trend.notna() & (trend <= -TREND_THRESHOLD) & (disc_pen > 0)
    disc_pen[improving] *= TREND_IMPROVE_MUL
    disc_pen[worsening] *= TREND_WORSEN_MUL
    disc_pen = disc_pen.clip(upper=DISC_PENALTY_CAP)

    pool["Combined_Score"] = (pool["Combined_Score"] - disc_pen).round(2)

    # Career_Disc_Flag: label which threshold(s) fired for transparency
    gate_flag = pd.Series("", index=pool.index)
    gate_flag[has_dc & (dcz <= DISC_HARD_FLOOR)] = "hard"
    gate_flag[has_dc & (dcz > DISC_HARD_FLOOR) & (dcz <= DISC_SOFT_FLOOR)] = "soft"
    gate_flag[improving & (gate_flag != "")] += "+improving"
    gate_flag[worsening & (gate_flag != "")] += "+worsening"
    pool["Career_Disc_Flag"] = gate_flag

    n_gated = (disc_pen > 0).sum()
    print(f"Discipline gate fired: {n_gated:,} / {len(pool):,} players")
    print(f"  {pool['Career_Disc_Flag'].value_counts().to_dict()}")
    print(f"  Slope data available for {pool['Disc_Slope'].notna().sum():,} / {len(pool):,} players")

    pool["Combined_Rank"] = pool["Combined_Score"].rank(ascending=False, method="min").astype(int)

    # Position from MLB Stats API via MLBAM_ID crosswalk
    pid_to_mlbam = (
        pd.read_csv(HIT_PATH, usecols=["PlayerId", "MLBAM_ID"])
        .drop_duplicates("PlayerId")
        .set_index("PlayerId")["MLBAM_ID"]
        .dropna().astype(int)
    )
    api_pos = pd.read_csv(POS_PATH).set_index("MLBAM_ID")["Position"]
    pool["MLBAM_ID"] = pool["PlayerId"].map(pid_to_mlbam)
    pool["Pos"] = pool["MLBAM_ID"].map(api_pos).fillna("")
    pool.loc[pool["Pos"].isin(["P", "TWP", "X"]), "Pos"] = ""
    pos_covered = (pool["Pos"] != "").sum()
    print(f"Position coverage: {pos_covered:,} / {len(pool):,} ({pos_covered/len(pool)*100:.1f}%)")

    out_cols = [
        "Combined_Rank", "PlayerId", "Name", "Pos", "Team", "Level", "Age",
        "Last_Season", "Career_PA", "Total_Weighted_PA",
        "TOOLS_Score", "ABILITY_Score", "Age_Score", "Current_Score", "OVR_Score",
        "Archetype", "Archetype_Adj", "Combined_Score",
        "Discipline_Flag", "Career_Disc_Flag",
        "Disc_Composite_Z", "Disc_Slope",
    ]
    out = pool[out_cols].sort_values("Combined_Rank").reset_index(drop=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out):,} prospects -> {OUT_PATH}")
    print(f"  TOOLS_Score    mean={out['TOOLS_Score'].mean():.1f}  std={out['TOOLS_Score'].std():.1f}")
    print(f"  ABILITY_Score  mean={out['ABILITY_Score'].mean():.1f}  std={out['ABILITY_Score'].std():.1f}")
    print(f"  Current_Score  mean={out['Current_Score'].mean():.1f}  std={out['Current_Score'].std():.1f}")
    print(f"  OVR_Score      mean={out['OVR_Score'].mean():.1f}  std={out['OVR_Score'].std():.1f}")
    print(f"\nTop 25:")
    print(out.head(25)[["Combined_Rank","Name","Pos","Team","Level","Age",
                         "Last_Season","Career_PA","TOOLS_Score","ABILITY_Score",
                         "Age_Score","Current_Score","OVR_Score","Combined_Score"]].to_string(index=False))


if __name__ == "__main__":
    main()
