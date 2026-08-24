"""Aggregate per-season TOOLS and ABILITY scores to one value per historical prospect.

OVR (all-time) model — independent of build_prospect_scores.py so weights,
thresholds, and eligibility rules can be tuned separately.

Eligibility:
  All years 2006+ (no season floor)
  Age at most recent MiLB season <= 24
  MLB exclusion SKIPPED — successful graduates must remain in the pool so the
  historical ranking is meaningful. Without this, every pre-2026 year is stripped
  of its best prospects (Trout 2011, Harper 2011, etc.) leaving only busts.
  Players whose first season in our data is 2006 AND age >= 18 in 2006 are
  excluded — anyone 18+ in 2006 was signed at 16 in 2004 or earlier and almost
  certainly had pre-2006 professional PA not visible in our data window.

Methodology:
  Same PA × level_weight shrinkage approach as build_prospect_scores.py.
  level_weight = LEVEL_DISCOUNT factors (AAA=1.00 → R=0.39).
  Per-level shrinkage thresholds (AAA-equivalent PA):
    TOOLS:   250
    ABILITY: 175

Combined_Score = 0.40 × TOOLS_Score + 0.40 × ABILITY_Score + 0.20 × Slope_Score
  TOOLS_Score   — PA × level_weight career average (per-level shrinkage, threshold=250)
  ABILITY_Score — PA × level_weight career average (per-level shrinkage, threshold=175)
  Slope_Score   — PA-weighted PPPA_Z_SL trajectory across levels, standardized to 50±10.
                  Positive slope = production improves as player advances through levels.
                  Players with < 2 qualifying levels (>=80 PA each) receive neutral 50.

Post-processing (run separately after this script):
  add_mlbt100.py    — appends MLBt100 column
  add_trajectory.py — reorders columns; Slope_Score already computed here

Output: data/rankings/prospect_scores_ovr.csv
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
OUT_PATH      = DATA_DIR / "rankings" / "prospect_scores_ovr.csv"

# Discipline floor thresholds (same as build_ability_score.py)
BB2K_HARD_FLOOR = -0.50
BB2K_SOFT_FLOOR = -0.29
WHIFF_FLOOR     =  1.0

MAX_PROSPECT_AGE  = 24
MLB_PA_EXCL       = 50   # kept for reference; exclusion is skipped in OVR mode

LEVEL_DISCOUNT = {"AAA": 1.00, "AA": 0.84, "A+": 0.71, "A": 0.57, "R": 0.39}
LEVEL_NUM      = {"R": 1, "A": 2, "A+": 3, "AA": 4, "AAA": 5}

TOOLS_PA_THRESH   = 250
ABILITY_PA_THRESH = 175
SLOPE_PA_THRESH   = 80    # min PA at a level to count toward slope
MIN_LEVELS_SLOPE  = 2     # distinct levels needed to compute slope

W_TOOLS   = 0.40
W_ABILITY = 0.40
W_SLOPE   = 0.20

# Players first appearing in 2006 at this age or older almost certainly had
# pre-2006 professional PA (signed at 16 in 2004 or earlier).
PRE2006_AGE_CUTOFF = 18


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


def _pa_weighted_slope(level_nums, pppa_vals, pa_weights):
    """PA-weighted linear regression slope of pppa_vals ~ level_nums."""
    w = np.array(pa_weights, dtype=float)
    x = np.array(level_nums, dtype=float)
    y = np.array(pppa_vals, dtype=float)
    w_sum = w.sum()
    if w_sum == 0:
        return np.nan
    x_bar = np.dot(w, x) / w_sum
    y_bar = np.dot(w, y) / w_sum
    num = np.dot(w, (x - x_bar) * (y - y_bar))
    den = np.dot(w, (x - x_bar) ** 2)
    return num / den if den > 1e-9 else np.nan


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
        usecols=["PlayerId", "Season", "Level", "ABILITY_Score", "PPPA_Z_SL", "Discipline_Flag"],
    )
    scores = tools.merge(ability, on=["PlayerId", "Season", "Level"], how="left")
    scores["level_wt"] = scores["Level"].map(LEVEL_DISCOUNT).fillna(0.39)
    scores["wt"]       = scores["PA"] * scores["level_wt"]
    print(f"Loaded {len(scores):,} player-season rows")

    # Pre-2006 career exclusion
    first_info = (
        scores.sort_values("Season")
              .groupby("PlayerId", sort=False)
              .first()[["Season", "Age"]]
    )
    pre2006_pids = first_info[
        (first_info["Season"] == 2006) & (first_info["Age"] >= PRE2006_AGE_CUTOFF)
    ].index
    scores = scores[~scores["PlayerId"].isin(pre2006_pids)]
    print(f"Excluded {len(pre2006_pids):,} players with likely pre-2006 career")

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

    # Eligibility: age cap only (no season floor, no MLB exclusion)
    pool = latest[latest["Age"] <= MAX_PROSPECT_AGE].copy()
    print(f"Eligible (Age<={MAX_PROSPECT_AGE}): {len(pool):,}")
    print("MLB exclusion skipped (OVR historical mode)")

    # Career-average scores with per-level shrinkage
    hist         = scores[scores["PlayerId"].isin(pool["PlayerId"])]
    tools_avg    = _wt_avg_shrunk(hist, "TOOLS_Score",   TOOLS_PA_THRESH)
    ability_avg  = _wt_avg_shrunk(hist, "ABILITY_Score", ABILITY_PA_THRESH)
    total_wpa    = hist.groupby("PlayerId")["wt"].sum().rename("Total_Weighted_PA")
    career_pa    = hist.groupby("PlayerId")["PA"].sum().rename("Career_PA")
    first_season = hist.groupby("PlayerId")["Season"].min().rename("First_Season")
    last_season  = hist.groupby("PlayerId")["Season"].max().rename("Last_Season")
    milb_seasons = hist.groupby("PlayerId")["Season"].nunique().rename("MiLB_Seasons")

    # PA-weighted career MLB PPPA from hist_mlb_data.csv — PlayerId join only.
    # Name fallback omitted: normalized names drop suffixes (Jr./Sr.) and create
    # false matches between different players sharing a family name.
    mlb = pd.read_csv(MLB_PATH, usecols=["PlayerId", "PA", "PPPA"])
    mlb_by_id = mlb.groupby("PlayerId").apply(
        lambda g: (g["PPPA"] * g["PA"]).sum() / g["PA"].sum(), include_groups=False
    ).rename("Career_PPPA").round(3)
    pool["Career_PPPA"] = pool["PlayerId"].map(mlb_by_id)

    pool["TOOLS_Score"]       = pool["PlayerId"].map(tools_avg)
    pool["ABILITY_Score"]     = pool["PlayerId"].map(ability_avg)
    pool["Total_Weighted_PA"] = pool["PlayerId"].map(total_wpa).round(2)
    pool["Career_PA"]         = pool["PlayerId"].map(career_pa).astype(int)
    pool["First_Season"]      = pool["PlayerId"].map(first_season).astype(int)
    pool["Last_Season"]       = pool["PlayerId"].map(last_season).astype(int)
    pool["MiLB_Seasons"]      = pool["PlayerId"].map(milb_seasons).fillna(1).astype(int)

    # Standardize TOOLS and ABILITY within pool to 50±10
    pool["TOOLS_Score"]   = to_50_10(pool["TOOLS_Score"].fillna(50))
    pool["ABILITY_Score"] = to_50_10(pool["ABILITY_Score"].fillna(50))

    # Slope_Score: PA-weighted PPPA_Z_SL trajectory across levels
    hs = hist[["PlayerId", "Level", "PA", "PPPA_Z_SL"]].copy()
    hs["level_num"] = hs["Level"].map(LEVEL_NUM)
    hs = hs.dropna(subset=["level_num", "PPPA_Z_SL"])
    hs["_wtd"] = hs["PPPA_Z_SL"] * hs["PA"]
    lv = hs.groupby(["PlayerId", "Level", "level_num"]).agg(
        PA_sum=("PA", "sum"),
        wtd_sum=("_wtd", "sum"),
    ).reset_index()
    lv["PPPA_Z_lv"] = lv["wtd_sum"] / lv["PA_sum"]
    lv = lv[lv["PA_sum"] >= SLOPE_PA_THRESH]

    slopes, n_levels = {}, {}
    for pid, grp in lv.groupby("PlayerId"):
        grp = grp.sort_values("level_num")
        n = grp["level_num"].nunique()
        n_levels[pid] = n
        if n < MIN_LEVELS_SLOPE:
            slopes[pid] = np.nan
            continue
        slopes[pid] = _pa_weighted_slope(
            grp["level_num"].values,
            grp["PPPA_Z_lv"].values,
            grp["PA_sum"].values,
        )

    # Discipline_Flag from most-recent season per player (any year — OVR is historical).
    recent_flags = (
        scores.sort_values("Season", ascending=False)
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
    sub = feats[feats["Level"] != "AAA"].copy()

    bb2k_rows = sub.dropna(subset=["BB_2K"])
    career_bb2k = (
        (bb2k_rows["BB_2K"] * bb2k_rows["PA"]).groupby(bb2k_rows["PlayerId"]).sum()
        / bb2k_rows.groupby("PlayerId")["PA"].sum()
    ).rename("Career_BB2K")

    wh_rows = sub.dropna(subset=["Whiff%_adj"])
    career_whiff = (
        (wh_rows["Whiff%_adj"] * wh_rows["PA"]).groupby(wh_rows["PlayerId"]).sum()
        / wh_rows.groupby("PlayerId")["PA"].sum()
    ).rename("Career_Whiff_Z")

    pool["Career_BB2K"]    = pool["PlayerId"].map(career_bb2k)
    pool["Career_Whiff_Z"] = pool["PlayerId"].map(career_whiff)

    bb2k_vals = pool["Career_BB2K"].dropna()
    if len(bb2k_vals) > 1 and bb2k_vals.std() > 0:
        mu, sig = bb2k_vals.mean(), bb2k_vals.std()
        pool["Career_BB2K_Z"] = ((pool["Career_BB2K"] - mu) / sig).round(4)
    else:
        pool["Career_BB2K_Z"] = np.nan

    cbb = pool["Career_BB2K_Z"]
    cwh = pool["Career_Whiff_Z"]
    cbb_flag = pd.Series("", index=pool.index)
    cbb_flag[cbb.notna() & (cbb <= BB2K_HARD_FLOOR)] = "hard"
    cbb_flag[cbb.notna() & (cbb > BB2K_HARD_FLOOR) & (cbb <= BB2K_SOFT_FLOOR)] = "soft"
    cwh_flag = pd.Series("", index=pool.index)
    cwh_flag[cwh.notna() & (cwh >= WHIFF_FLOOR)] = "whiff"
    career_flag = cbb_flag + cwh_flag.apply(lambda w: ("+" if w else "") + w)
    career_flag = career_flag.where(cbb_flag != "", cwh_flag)
    pool["Career_Disc_Flag"] = career_flag

    career_flagged = (pool["Career_Disc_Flag"].notna() & (pool["Career_Disc_Flag"] != "")).sum()
    print(f"Career_Disc_Flag (non-AAA): {career_flagged:,} / {len(pool):,} flagged")

    pool["PPPA_Slope"] = pool["PlayerId"].map(slopes).round(3)
    pool["N_Levels"]   = pool["PlayerId"].map(n_levels).fillna(0).astype(int)

    slope_raw = pool["PPPA_Slope"]
    if slope_raw.notna().sum() >= 2:
        mu, sig = slope_raw.mean(), slope_raw.std()
        pool["Slope_Score"] = (
            (50 + 10 * (slope_raw - mu) / sig).clip(lower=0, upper=100).round(2)
            if sig > 0 else pd.Series(50.0, index=pool.index)
        )
    else:
        pool["Slope_Score"] = 50.0
    pool["Slope_Score"] = pool["Slope_Score"].fillna(50.0)

    pool["Combined_Score"] = (
        W_TOOLS   * pool["TOOLS_Score"]
        + W_ABILITY * pool["ABILITY_Score"]
        + W_SLOPE   * pool["Slope_Score"]
    ).round(2)
    pool["Combined_Rank"] = pool["Combined_Score"].rank(ascending=False, method="min").astype(int)

    out_cols = [
        "Combined_Rank", "PlayerId", "Name", "Team", "Level", "Age",
        "First_Season", "Last_Season", "Career_PA", "Career_PPPA", "Total_Weighted_PA",
        "TOOLS_Score", "ABILITY_Score", "Slope_Score", "Combined_Score",
        "Discipline_Flag", "Career_Disc_Flag", "PPPA_Slope", "N_Levels", "MiLB_Seasons",
    ]
    out = pool[out_cols].sort_values("Combined_Rank").reset_index(drop=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out):,} prospects -> {OUT_PATH}")
    print(f"  TOOLS_Score   mean={out['TOOLS_Score'].mean():.1f}  std={out['TOOLS_Score'].std():.1f}")
    print(f"  ABILITY_Score mean={out['ABILITY_Score'].mean():.1f}  std={out['ABILITY_Score'].std():.1f}")
    print(f"  Slope_Score   mean={out['Slope_Score'].mean():.1f}  std={out['Slope_Score'].std():.1f}")
    print(f"  Slope coverage: {out['PPPA_Slope'].notna().sum():,} / {len(out):,} players")
    print(f"\nTop 25 (OVR):")
    print(out.head(25)[["Combined_Rank","Name","Team","Level","Age",
                         "Career_PA","MiLB_Seasons","N_Levels",
                         "TOOLS_Score","ABILITY_Score","Slope_Score","Combined_Score",
                         "PPPA_Slope"]].to_string(index=False))


if __name__ == "__main__":
    main()
