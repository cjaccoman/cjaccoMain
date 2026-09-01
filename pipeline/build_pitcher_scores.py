"""Aggregate per-season STUFF and PERFORMANCE scores to one value per pitcher prospect.

Eligibility (current pool):
  Most recent season >= 2025
  Age at most recent season <= 25
  Career MLB IP < 30 (approximated via MLB_IP column if present; else no exclusion)

STUFF_Score (raw physical skills — pitcher analog of TOOLS):
  K%_adj + Whiff%_adj composite : 45%
  -BB%_adj                       : 35%
  GB%_adj                        : 20%
  Age adjustment: each component × (1 + 0.20 × -Age_Z_SL), clipped ±2 SD

PERFORMANCE_Score (demonstrated production — pitcher analog of ABILITY):
  ERA_adj                        : 40%
  KBB_adj                        : 30%
  PPI_adj                        : 30%
  Same age adjustment as STUFF.

Career aggregation:
  IP × level_weight career average per level, with per-level shrinkage.
  Level discounts: AAA=1.00, AA=0.59, A+=0.34, A=0.23, R=0.10 (same as hitters for now).
  IP shrinkage thresholds (raw IP):  STUFF=167, PERFORMANCE=120  (~1 full MiLB season)

Combined_Score:
  0.50 × Current_Score + 0.50 × OVR_Score
  Current_Score = 0.30 × STUFF + 0.50 × PERFORMANCE + 0.20 × Age_Score
  OVR_Score     = 0.40 × STUFF + 0.40 × PERFORMANCE + 0.20 × PPI_Slope_Score

Separate SP and RP rankings (Role determined by most recent qualifying season).

Output: data/rankings/pitcher_scores.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR      = Path(__file__).resolve().parent.parent / "data"
FEATURES_PATH = DATA_DIR / "rankings" / "pitcher_features.csv"
OUT_PATH      = DATA_DIR / "rankings" / "pitcher_scores.csv"

MAX_PROSPECT_AGE = 25
MIN_SEASON       = 2025
MIN_IP_CAREER    = 10   # minimum career IP to be included in current pool

LEVEL_DISCOUNT = {"AAA": 1.00, "AA": 0.59, "A+": 0.34, "A": 0.23, "R": 0.10}

# Raw-IP shrinkage thresholds — 1 full MiLB season equivalent
STUFF_IP_THRESH       = 167.0
PERFORMANCE_IP_THRESH = 120.0

AGE_ALPHA  = 0.20   # per-component age multiplier
AGE_CLIP   = 2.0    # ±2 SD clip on age adjustment

W_STUFF   = 0.30
W_PERF    = 0.50
W_AGE     = 0.20


def to_50_10(s: pd.Series) -> pd.Series:
    """Standardize to mean=50, std=10; clip at 0 floor."""
    s = s.dropna()
    if s.std(ddof=1) < 1e-9:
        return pd.Series(50.0, index=s.index)
    z = (s - s.mean()) / s.std(ddof=1)
    return (50 + 10 * z).clip(lower=0)


def age_adj(component: pd.Series, age_z: pd.Series) -> pd.Series:
    """Apply age multiplier: component × (1 + alpha × -age_z), clipped."""
    adj_factor = (1.0 + AGE_ALPHA * (-age_z.clip(-AGE_CLIP, AGE_CLIP))).fillna(1.0)
    return component * adj_factor


def ppi_slope(df: pd.DataFrame) -> pd.Series:
    """PA-weighted OLS slope of PPI_skill on Season per player."""
    slope_map = {}
    for pid, grp in df[df["PPI_skill"].notna()].groupby("PlayerId"):
        if len(grp) < 2:
            continue
        seas = grp["Season"].values.astype(float)
        ppi = grp["PPI_skill"].values
        wts = grp["IP"].values.astype(float)
        sc = seas - np.average(seas, weights=wts)
        denom = np.dot(wts, sc ** 2)
        if denom < 1e-9:
            continue
        slope_map[pid] = np.dot(wts, sc * ppi) / denom
    return df["PlayerId"].map(slope_map)


def wt_avg_shrunk(df: pd.DataFrame, score_col: str, threshold: float,
                  wt_col: str = "wt") -> pd.Series:
    """IP-weighted career average per player, shrunk toward 50 by raw IP per level."""
    results = {}
    for pid, grp in df.groupby("PlayerId"):
        level_avgs = []
        level_wts = []
        for level, lgrp in grp.groupby("Level"):
            raw_ip = lgrp["IP"].sum()
            wt_sum = lgrp[wt_col].sum()
            if wt_sum <= 0:
                continue
            avg = (lgrp[score_col] * lgrp[wt_col]).sum() / wt_sum
            shrink = min(raw_ip / threshold, 1.0)
            shrunk = 50.0 + shrink * (avg - 50.0)
            level_avgs.append(shrunk)
            level_wts.append(wt_sum)
        if not level_avgs:
            continue
        total_wt = sum(level_wts)
        results[pid] = sum(a * w for a, w in zip(level_avgs, level_wts)) / total_wt
    return pd.Series(results)   # index = PlayerId


def main() -> None:
    print("=== build_pitcher_scores.py ===\n")

    df = pd.read_csv(FEATURES_PATH, dtype={"PlayerId": str})
    print(f"Loaded {len(df):,} pitcher-season rows")

    # -----------------------------------------------------------------------
    # Per-row STUFF and PERFORMANCE scores (pre-aggregation)
    # -----------------------------------------------------------------------
    df["level_wt"] = df["Level"].map(LEVEL_DISCOUNT).fillna(0)
    df["wt"] = df["IP"] * df["level_wt"]

    # Fill missing adj z-scores with 0 (neutral)
    adj_cols = ["K%_adj", "BB%_adj", "KBB_adj", "Whiff%_adj", "ERA_adj", "GB%_adj", "PPI_adj"]
    for c in adj_cols:
        if c not in df.columns:
            df[c] = 0.0
        else:
            df[c] = df[c].fillna(0.0)

    # Age adjustment
    az = df["Age_Z_SL"].fillna(0.0)
    df["K%_a"]    = age_adj(df["K%_adj"],    az)
    df["BB%_a"]   = age_adj(df["BB%_adj"],   az)
    df["KBB_a"]   = age_adj(df["KBB_adj"],   az)
    df["Whiff%_a"] = age_adj(df["Whiff%_adj"], az)
    df["ERA_a"]   = age_adj(df["ERA_adj"],   az)
    df["GB%_a"]   = age_adj(df["GB%_adj"],   az)
    df["PPI_a"]   = age_adj(df["PPI_adj"],   az)

    # STUFF: K/Whiff (45%) + Command (35%) + GB (20%)
    k_whiff = 0.5 * df["K%_a"] + 0.5 * df["Whiff%_a"]
    df["STUFF_raw"] = (
        0.45 * k_whiff
        + 0.35 * df["BB%_a"]
        + 0.20 * df["GB%_a"]
    )

    # PERFORMANCE: ERA (40%) + KBB (30%) + PPI (30%)
    df["PERF_raw"] = (
        0.40 * df["ERA_a"]
        + 0.30 * df["KBB_a"]
        + 0.30 * df["PPI_a"]
    )

    # Standardize to 50±10 across all rows
    df["STUFF_Score"]       = to_50_10(df["STUFF_raw"]).reindex(df.index).fillna(50)
    df["PERFORMANCE_Score"] = to_50_10(df["PERF_raw"]).reindex(df.index).fillna(50)

    print(f"  STUFF_Score: mean={df['STUFF_Score'].mean():.1f}, "
          f"std={df['STUFF_Score'].std():.1f}")
    print(f"  PERFORMANCE_Score: mean={df['PERFORMANCE_Score'].mean():.1f}, "
          f"std={df['PERFORMANCE_Score'].std():.1f}")

    # -----------------------------------------------------------------------
    # PPI trajectory slope (for OVR Slope_Score)
    # -----------------------------------------------------------------------
    df["PPI_Slope"] = ppi_slope(df)

    # -----------------------------------------------------------------------
    # OVR scores (all historical players)
    # -----------------------------------------------------------------------
    stuff_ovr = wt_avg_shrunk(df, "STUFF_Score",       STUFF_IP_THRESH)
    perf_ovr  = wt_avg_shrunk(df, "PERFORMANCE_Score", PERFORMANCE_IP_THRESH)

    # Slope_Score: one value per player (already constant within player from ppi_slope)
    slope_vals = df.groupby("PlayerId")["PPI_Slope"].first()
    slope_std = slope_vals.std(ddof=1)
    if slope_std > 1e-9:
        slope_z = (slope_vals - slope_vals.mean()) / slope_std
        slope_50 = (50 + 10 * slope_z).clip(0, 100)
    else:
        slope_50 = pd.Series(50.0, index=slope_vals.index)

    player_ovr = pd.DataFrame({
        "STUFF_OVR": to_50_10(stuff_ovr.dropna()),
        "PERF_OVR":  to_50_10(perf_ovr.dropna()),
        "Slope_Score": slope_50,
    })
    player_ovr["OVR_Score"] = (
        0.40 * player_ovr["STUFF_OVR"].fillna(50)
        + 0.40 * player_ovr["PERF_OVR"].fillna(50)
        + 0.20 * player_ovr["Slope_Score"].fillna(50)
    )

    # -----------------------------------------------------------------------
    # Current prospect pool
    # -----------------------------------------------------------------------
    # Identify most recent season per player
    last_sea = df.groupby("PlayerId")["Season"].max()

    eligible = (last_sea >= MIN_SEASON)
    eligible_pids = eligible[eligible].index.tolist()
    curr = df[df["PlayerId"].isin(eligible_pids)].copy()

    # Age at most recent season
    age_at_last = (
        df.sort_values("Season")
        .groupby("PlayerId")
        .last()[["Age", "Age_Z_SL", "Role", "Name", "Team", "Level"]]
    )
    age_at_last = age_at_last[age_at_last["Age"].fillna(99) <= MAX_PROSPECT_AGE]
    eligible_pids = list(set(eligible_pids) & set(age_at_last.index))
    curr = curr[curr["PlayerId"].isin(eligible_pids)]

    print(f"\nCurrent pool: {len(eligible_pids)} pitchers "
          f"(age <= {MAX_PROSPECT_AGE}, most recent season >= {MIN_SEASON})")

    # Career IP filter
    career_ip = curr.groupby("PlayerId")["IP"].sum()
    eligible_pids = [p for p in eligible_pids if career_ip.get(p, 0) >= MIN_IP_CAREER]
    curr = curr[curr["PlayerId"].isin(eligible_pids)]
    print(f"  After >= {MIN_IP_CAREER} career IP filter: {len(eligible_pids)} pitchers")

    # Career-aggregated scores for current pool
    stuff_curr = wt_avg_shrunk(curr, "STUFF_Score",       STUFF_IP_THRESH)
    perf_curr  = wt_avg_shrunk(curr, "PERFORMANCE_Score", PERFORMANCE_IP_THRESH)

    # Age_Score (current pool): age_z is in z-score units, convert to 50±10
    age_z_curr = age_at_last.loc[eligible_pids, "Age_Z_SL"]

    pool_idx = pd.Index(eligible_pids)

    # Career averages are already on 50±10 scale from per-row standardization.
    # Avoid pool re-standardization here — it would double-standardize and
    # amplify outliers whose career averages are compressed by shrinkage.
    stuff_s = stuff_curr.reindex(pool_idx).fillna(50)
    perf_s  = perf_curr.reindex(pool_idx).fillna(50)
    age_s   = to_50_10(-age_z_curr.dropna()).reindex(pool_idx).fillna(50)

    current_score = W_STUFF * stuff_s + W_PERF * perf_s + W_AGE * age_s

    # OVR_Score for pool members
    ovr_s = player_ovr.loc[player_ovr.index.isin(eligible_pids), "OVR_Score"].reindex(pool_idx).fillna(50)

    combined = 0.50 * current_score + 0.50 * ovr_s

    # -----------------------------------------------------------------------
    # Build output DataFrame
    # -----------------------------------------------------------------------
    meta = age_at_last.loc[eligible_pids]
    career_ip_s = curr.groupby("PlayerId")["IP"].sum().reindex(pool_idx)
    career_g    = curr.groupby("PlayerId")["G"].sum().reindex(pool_idx)
    career_gs   = curr.groupby("PlayerId")["GS"].sum().reindex(pool_idx)

    out = pd.DataFrame({
        "PlayerId":         pool_idx,
        "Name":             meta["Name"].reindex(pool_idx).values,
        "Team":             meta["Team"].reindex(pool_idx).values,
        "Level":            meta["Level"].reindex(pool_idx).values,
        "Age":              meta["Age"].reindex(pool_idx).values,
        "Role":             meta["Role"].reindex(pool_idx).values,
        "Career_IP":        career_ip_s.values,
        "Career_G":         career_g.values,
        "Career_GS":        career_gs.values,
        "STUFF_Score":      stuff_s.values,
        "PERFORMANCE_Score": perf_s.values,
        "Age_Score":        age_s.values,
        "Current_Score":    current_score.values,
        "OVR_Score":        ovr_s.values,
        "Combined_Score":   combined.values,
    })

    out = out.sort_values("Combined_Score", ascending=False).reset_index(drop=True)
    out["Combined_Rank"] = out.index + 1

    # Separate SP / RP ranks
    sp_mask = out["Role"] == "SP"
    rp_mask = out["Role"] == "RP"
    out.loc[sp_mask, "SP_Rank"] = out.loc[sp_mask, "Combined_Score"].rank(
        ascending=False, method="min").astype(int)
    out.loc[rp_mask, "RP_Rank"] = out.loc[rp_mask, "Combined_Score"].rank(
        ascending=False, method="min").astype(int)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(out):,} rows -> {OUT_PATH}")
    print(f"  SP: {sp_mask.sum()} | RP: {rp_mask.sum()}")
    print("\nTop 10 SP:")
    print(out[sp_mask].head(10)[["Name", "Team", "Level", "Age", "Combined_Score", "SP_Rank"]].to_string(index=False))
    print("\nTop 10 RP:")
    print(out[rp_mask].head(10)[["Name", "Team", "Level", "Age", "Combined_Score", "RP_Rank"]].to_string(index=False))


if __name__ == "__main__":
    main()
