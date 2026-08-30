"""Build per-player-season BABIP and HR/FB luck tracker.

For each player-season, computes deviation from that player's own career
baseline to flag seasons where production was inflated or suppressed by luck
(high BABIP, elevated HR/FB) vs. genuine skill.

BABIP = (1B + 2B + 3B) / (PA - BB - IBB - SO - HR)
   — approximates (H - HR) / (AB - SO - HR + SF); ignores HBP/SF/SH (~small)

HR/FB sourced from missing_milb_data.csv (FanGraphs). HR/FB > 1.0 nulled
(physically impossible, small-sample artifact).

Career baseline is the leave-one-out PA-weighted average — current season
excluded so the baseline isn't contaminated by the season being evaluated.

Luck_Score = 0.60 × BABIP_delta_z + 0.40 × HRFB_delta_z
  Positive = lucky (better than career baseline + peers)
  Negative = unlucky

Minimum PA filter: MIN_PA = 150 per season — below this, single-season BABIP
  is too noisy (r < 0.20 at 100 PA, improves toward ~0.50 at 400+ PA).

Baseline reliability shrinkage: Luck_Score is multiplied by
  min(Prior_Career_PA / PRIOR_PA_THRESH, 1.0), where Prior_Career_PA is the
  sum of PA in all other qualifying seasons. When most of a player's career is
  the current season, the leave-one-out baseline is thin and unreliable — the
  shrinkage discounts the Luck_Score toward 0 rather than reporting a false
  high/low. PRIOR_PA_THRESH = 300 (roughly 2 solid seasons).

  Luck_Score_raw is the pre-shrinkage score; Luck_Score is the final value.

Output: data/computed/babip_luck.csv
  One row per player-season. Sorted by PlayerId, Season.

Run:
  python analysis/build_babip_luck.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA    = Path(__file__).resolve().parent.parent / "data"
MLD     = DATA / "computed" / "minorLeagueData.csv"
ADV     = DATA / "fangraphs" / "missing_milb_data.csv"
ADV2    = DATA / "fangraphs" / "ml_updated_data.csv"   # 2026 override
OUT     = DATA / "computed"  / "babip_luck.csv"

MIN_PA           = 150   # minimum PA per season — below this BABIP is too noisy
MIN_CAREER_SEAS  = 2    # need at least 2 qualifying seasons to compute a baseline delta
PRIOR_PA_THRESH  = 300  # prior-season PA for full baseline reliability (≈ 2 solid seasons)

# Luck-adjusted PA weighting
# Discount effective PA for lucky seasons so they pull the career average less.
# Luck_Score is already baseline_rel-shrunk, so thin-prior-PA seasons barely adjust.
# α = 0.20: +2.5 SD lucky → 0.50x PA weight; +4 SD lucky → floor of 0.25x
# Unlucky seasons (negative Luck_Score) get upweighted, capped at 1.50x
LUCK_PA_ALPHA    = 0.20
LUCK_PA_FLOOR    = 0.25  # very lucky seasons still contribute ≥25% of actual PA
LUCK_PA_CEIL     = 1.50  # unlucky seasons upweighted by at most 50%

# PPPA conversion constants
# Hits from BABIP luck: weighted avg of 1B(2pts), 2B(4pts), 3B(6pts) in scoring system
# Using typical MiLB distribution ~65% singles, ~28% doubles, ~7% triples → ≈2.85; rounded to 2.8
HIT_PTS = 2.8
# Extra HRs from HR/FB luck: HR(+4) + TB(+4) = 8 deterministic pts (excludes context-dependent R/RBI)
HR_PTS  = 8


def _babip(df: pd.DataFrame) -> pd.Series:
    """BABIP = (1B+2B+3B) / (PA - BB - IBB - SO - HR); NaN when denominator <= 0."""
    num   = df["1B"] + df["2B"] + df["3B"]
    denom = df["PA"] - df["BB"] - df["IBB"] - df["SO"] - df["HR"]
    return num.where(denom > 0) / denom.where(denom > 0)


def _leave_one_out_avg(group: pd.DataFrame, col: str) -> pd.Series:
    """PA-weighted career average excluding the current row (leave-one-out)."""
    total_num = (group[col] * group["PA"]).sum()
    total_den = group["PA"].sum()
    row_num   = group[col] * group["PA"]
    row_den   = group["PA"]
    loo_num   = total_num - row_num
    loo_den   = total_den - row_den
    return (loo_num / loo_den.where(loo_den > 0)).where(loo_den > 0)


def _z_within(df: pd.DataFrame, col: str, group_cols: list[str]) -> pd.Series:
    """Z-score col within each group; NaN when group std == 0."""
    grp  = df.groupby(group_cols, observed=True)[col]
    mu   = grp.transform("mean")
    std  = grp.transform("std")
    return (df[col] - mu) / std.where(std > 0)


def main() -> None:
    # ── Load base stats ───────────────────────────────────────────────────────
    mld = pd.read_csv(MLD)
    mld = mld[mld["PA"] >= MIN_PA].copy()

    mld["BABIP"] = _babip(mld)

    # ── Load HR/FB ────────────────────────────────────────────────────────────
    adv_cols = ["PlayerId", "Season", "Level", "HR/FB"]
    adv = pd.read_csv(ADV, usecols=adv_cols)
    # 2026 override
    adv2 = pd.read_csv(ADV2, usecols=[c for c in adv_cols if c in
                                       pd.read_csv(ADV2, nrows=0).columns])
    adv = pd.concat([adv[~adv["Season"].isin(adv2["Season"].unique())], adv2],
                    ignore_index=True)

    # Null physically impossible / extreme small-sample HR/FB values.
    # > 1.0 is a FanGraphs denominator miscounting artifact.
    # = 1.0 (100% HR/FB) is physically possible but only at tiny FB counts
    # where it's pure noise; treating it as missing is safer.
    if "HR/FB" in adv.columns:
        adv.loc[adv["HR/FB"] >= 1.0, "HR/FB"] = np.nan

    # Coerce PlayerId to string for join (mld uses int, adv uses str for recent players)
    mld["PlayerId"] = mld["PlayerId"].astype(str)
    adv["PlayerId"] = adv["PlayerId"].astype(str)

    import unicodedata
    def _norm(s: pd.Series) -> pd.Series:
        return s.apply(lambda x: unicodedata.normalize("NFD", str(x))
                       .encode("ascii", "ignore").decode().lower().strip()
                       if pd.notna(x) else x)

    # Primary join: PlayerId + Season + Level
    hrfb_src = adv[["PlayerId", "Season", "Level", "HR/FB"]].dropna(subset=["HR/FB"])
    mld = mld.merge(hrfb_src, on=["PlayerId", "Season", "Level"], how="left")

    # Fallback: Name + Season + Level for rows still missing HR/FB
    missing = mld["HR/FB"].isna()
    if missing.any():
        hrfb_name = hrfb_src.copy()
        adv_full = pd.read_csv(ADV, usecols=["PlayerId", "Name", "Season", "Level", "HR/FB"])
        adv2_cols = [c for c in ["PlayerId","Name","Season","Level","HR/FB"]
                     if c in pd.read_csv(ADV2, nrows=0).columns]
        adv2_full = pd.read_csv(ADV2, usecols=adv2_cols)
        hrfb_name = pd.concat([
            adv_full[~adv_full["Season"].isin(adv2_full["Season"].unique())],
            adv2_full], ignore_index=True).dropna(subset=["HR/FB"])
        hrfb_name["_nname"] = _norm(hrfb_name["Name"])
        mld["_nname"] = _norm(mld["Name"])
        fill_src = hrfb_name[["_nname","Season","Level","HR/FB"]].drop_duplicates(
            subset=["_nname","Season","Level"])
        fill = mld[missing][["_nname","Season","Level"]].merge(
            fill_src, on=["_nname","Season","Level"], how="left")
        mld.loc[missing, "HR/FB"] = fill["HR/FB"].values
        mld.drop(columns=["_nname"], inplace=True)

    # ── Career leave-one-out baselines (grouped by PlayerId) ─────────────────
    # Cross-level career average: all qualifying seasons regardless of level.
    # Level BABIP norms shift universally (better pitching/defense up the ladder),
    # so the trend affects everyone equally and doesn't distort player comparisons.
    # Using cross-level maximises coverage vs. same-level-only grouping.
    # Requires >= MIN_CAREER_SEAS qualifying seasons total.
    valid_babip = mld[mld["BABIP"].notna()]
    seas_count  = valid_babip.groupby("PlayerId")["Season"].count()
    multi_pid   = seas_count[seas_count >= MIN_CAREER_SEAS].index

    babip_base = (
        valid_babip[valid_babip["PlayerId"].isin(multi_pid)]
        .groupby("PlayerId", group_keys=False)
        .apply(lambda g: _leave_one_out_avg(g, "BABIP"))
    )
    mld["BABIP_career"] = babip_base.reindex(mld.index)
    mld["BABIP_delta"]  = mld["BABIP"] - mld["BABIP_career"]

    # HR/FB baseline (same grouping)
    valid_hrfb = mld[mld["HR/FB"].notna()]
    seas_count_hrfb = valid_hrfb.groupby("PlayerId")["Season"].count()
    multi_pid_hrfb  = seas_count_hrfb[seas_count_hrfb >= MIN_CAREER_SEAS].index

    hrfb_base = (
        valid_hrfb[valid_hrfb["PlayerId"].isin(multi_pid_hrfb)]
        .groupby("PlayerId", group_keys=False)
        .apply(lambda g: _leave_one_out_avg(g, "HR/FB"))
    )
    mld["HRFB_career"] = hrfb_base.reindex(mld.index)
    mld["HRFB_delta"]  = mld["HR/FB"] - mld["HRFB_career"]

    # ── BABIP_delta slope: PA-weighted OLS of (BABIP - Career_BABIP) on Season ─
    # Positive = gap trending wider (BABIP pulling ahead of career baseline over time)
    # Negative = gap shrinking (BABIP regressing toward or below career baseline)
    # Requires >= 2 seasons with a non-null BABIP_delta.
    slope_map = {}
    for pid, grp in mld[mld["BABIP_delta"].notna()].groupby("PlayerId"):
        if len(grp) < 2:
            continue
        seas = grp["Season"].values.astype(float)
        delta = grp["BABIP_delta"].values
        pa = grp["PA"].values.astype(float)
        sc = seas - seas.mean()
        denom = np.dot(pa, sc ** 2)
        if denom < 1e-9:
            continue
        slope_map[pid] = np.dot(pa, sc * delta) / denom
    mld["BABIP_Delta_Slope"] = mld["PlayerId"].map(slope_map)

    # ── Z-score deltas within Season+Level ────────────────────────────────────
    mld["BABIP_delta_z"] = _z_within(mld, "BABIP_delta", ["Season", "Level"])
    mld["HRFB_delta_z"]  = _z_within(mld, "HRFB_delta",  ["Season", "Level"])

    # ── Composite Luck_Score ──────────────────────────────────────────────────
    # Use whichever components are available; weight accordingly
    has_babip = mld["BABIP_delta_z"].notna()
    has_hrfb  = mld["HRFB_delta_z"].notna()
    both      = has_babip & has_hrfb
    only_bab  = has_babip & ~has_hrfb
    only_hrfb = ~has_babip & has_hrfb

    luck = pd.Series(np.nan, index=mld.index)
    luck[both]      = 0.60 * mld.loc[both, "BABIP_delta_z"] + 0.40 * mld.loc[both, "HRFB_delta_z"]
    luck[only_bab]  = mld.loc[only_bab, "BABIP_delta_z"]
    luck[only_hrfb] = mld.loc[only_hrfb, "HRFB_delta_z"]
    mld["Luck_Score_raw"] = luck

    # ── Baseline reliability shrinkage ────────────────────────────────────────
    total_pa           = mld.groupby("PlayerId")["PA"].transform("sum")
    mld["Prior_Career_PA"] = (total_pa - mld["PA"]).clip(lower=0)
    baseline_rel       = (mld["Prior_Career_PA"] / PRIOR_PA_THRESH).clip(upper=1.0)
    mld["Luck_Score"]  = mld["Luck_Score_raw"] * baseline_rel

    # ── Luck-adjusted effective PA ────────────────────────────────────────────
    # PA_luck_weight discounts lucky seasons so they contribute less when career
    # averages are computed downstream. Unlucky seasons get a modest upweight.
    # Where Luck_Score is null (< 2 career seasons), PA_luck_weight = PA (no change).
    luck_factor = (1.0 - LUCK_PA_ALPHA * mld["Luck_Score"]).clip(
        lower=LUCK_PA_FLOOR, upper=LUCK_PA_CEIL
    )
    mld["PA_luck_weight"] = (mld["PA"] * luck_factor).where(
        mld["Luck_Score"].notna(), mld["PA"]
    )

    # ── PPPA conversion ───────────────────────────────────────────────────────
    # Express luck in PPPA units so magnitude is intuitive.
    # BIP_rate = balls in play per PA
    bip      = (mld["PA"] - mld["BB"] - mld["IBB"] - mld["SO"] - mld["HR"]).clip(lower=0)
    bip_rate = bip / mld["PA"]

    # BABIP component: extra hits per PA * hit scoring value * reliability weight
    babip_pppa = mld["BABIP_delta"].fillna(0) * bip_rate * HIT_PTS * baseline_rel

    # HRFB component: extra HRs per PA * HR scoring value, only where HR/FB exists
    hrfb_has  = mld["HR/FB"].notna() & (mld["HR/FB"] > 0)
    fb_est    = mld["HR"].where(hrfb_has, 0) / mld["HR/FB"].where(hrfb_has, 1)
    hrfb_pppa = mld["HRFB_delta"].fillna(0) * (fb_est / mld["PA"]) * HR_PTS * baseline_rel
    hrfb_pppa = hrfb_pppa.where(hrfb_has & mld["HRFB_delta"].notna(), 0)

    mld["Luck_PPPA"]     = (babip_pppa + hrfb_pppa).where(mld["BABIP_delta"].notna())
    mld["Luck_PPPA_pct"] = (
        (mld["Luck_PPPA"] / mld["PPPA"] * 100)
        .where(mld["PPPA"].notna() & (mld["PPPA"].abs() > 0.01))
        .clip(lower=-200, upper=200)
    )

    # ── Output ────────────────────────────────────────────────────────────────
    out_cols = [
        "PlayerId", "Name", "Season", "Level", "Team", "Age", "PA",
        "Prior_Career_PA", "PA_luck_weight",
        "BABIP", "BABIP_career", "BABIP_delta", "BABIP_delta_z",
        "BABIP_Delta_Slope",
        "HR/FB", "HRFB_career", "HRFB_delta", "HRFB_delta_z",
        "Luck_Score_raw", "Luck_Score",
        "Luck_PPPA", "Luck_PPPA_pct",
        "PPPA", "PPPA_Z_SL",
    ]
    out = mld[[c for c in out_cols if c in mld.columns]].sort_values(
        ["PlayerId", "Season"]
    )

    # Round floats
    for col in ["BABIP", "BABIP_career", "BABIP_delta", "HR/FB", "HRFB_career",
                "HRFB_delta", "PPPA"]:
        if col in out.columns:
            out[col] = out[col].round(3)
    for col in ["Luck_PPPA"]:
        if col in out.columns:
            out[col] = out[col].round(4)
    for col in ["Luck_PPPA_pct"]:
        if col in out.columns:
            out[col] = out[col].round(1)
    for col in ["BABIP_Delta_Slope"]:
        if col in out.columns:
            out[col] = out[col].round(4)
    for col in ["BABIP_delta_z", "HRFB_delta_z", "Luck_Score_raw", "Luck_Score", "PPPA_Z_SL"]:
        if col in out.columns:
            out[col] = out[col].round(2)

    out.to_csv(OUT, index=False)
    print(f"Wrote {len(out):,} rows -> babip_luck.csv")
    print(f"  BABIP populated    : {out['BABIP'].notna().sum():,}")
    print(f"  BABIP_delta populated: {out['BABIP_delta'].notna().sum():,}")
    print(f"  HRFB_delta populated : {out['HRFB_delta'].notna().sum():,}")
    print(f"  Luck_Score_raw populated : {out['Luck_Score_raw'].notna().sum():,}")
    print(f"  Luck_Score populated     : {out['Luck_Score'].notna().sum():,}")

    # Sample: most lucky and most unlucky single seasons (min 200 PA)
    qualified = out[out["PA"] >= 200].dropna(subset=["Luck_Score"])
    print(f"\nTop 10 luckiest seasons (PA >= 200):")
    top = qualified.nlargest(10, "Luck_Score")[
        ["Name", "Season", "Level", "PA", "BABIP", "BABIP_career",
         "HR/FB", "HRFB_career", "Luck_Score"]
    ]
    print(top.to_string(index=False))

    print(f"\nTop 10 unluckiest seasons (PA >= 200):")
    bot = qualified.nsmallest(10, "Luck_Score")[
        ["Name", "Season", "Level", "PA", "BABIP", "BABIP_career",
         "HR/FB", "HRFB_career", "Luck_Score"]
    ]
    print(bot.to_string(index=False))


if __name__ == "__main__":
    main()
