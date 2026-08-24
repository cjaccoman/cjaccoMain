"""Full pipeline runner. Run this whenever minorLeagueData.csv is updated.

Steps:
  1. compute_stats.py     — recompute derived cols, z-scores, player_scores.csv
  2. ovr_hist_data.csv    — rebuild combined historical MiLB dataset
  3. regression_season_league.csv — descriptive OLS per Season+League (reference only)
  4. regression_predictive.csv   — WLS predictive regression per Season_T
  5. skill_scores.csv     — Phase 1 skill score using era-specific coefficients
  6. regression_mlb_outcomes.csv — Phase 2 WLS: MiLB skills → first-year MLB PPPA_Z
  7. prospect_mlb_proj.csv       — apply Phase 2 coefficients to current prospects
  8. build_prospect_pool  — Phase 3 unified ranking (Skill + MLB_Proj + BW)
"""

import sys
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd
import statsmodels.api as sm

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "pipeline"))
import compute_stats
import build_prospect_pool

DATA_DIR     = _ROOT / "data"
HIST_DIR     = DATA_DIR / "historical"
FG_DIR       = DATA_DIR / "fangraphs"
COMPUTED_DIR = DATA_DIR / "computed"

LEVEL_MAP = {"R": 1, "A": 2, "A+": 3, "AA": 4, "AAA": 5}
JOIN_KEY = ["PlayerId", "Season", "Team", "Level", "PA"]
SKILL_PREDS    = ["BB/K", "K%", "GB/FB", "SwStr%", "Age_Z_SL", "ISO"]
ALL_PREDS      = ["BB/K", "K%", "ISO", "wRC+", "GB/FB", "SwStr%", "Age_Z_SL"]
PRED_REGRESSION = ["BB/K", "K%", "GB/FB", "SwStr%", "Age_Z_SL", "ISO", "level_diff"]
MLB_PREDS      = ["BB/K", "K%", "GB/FB", "Age_Z_SL", "wRC+", "ISO"]
MLB_TRAJ       = ["K%"]                  # stats for trajectory (delta) in Phase 2

# Phase 4 arc feature constants
PHASE4_BEST    = ["PPPA_Z_SL"]           # peer-relative PPPA best-season; controls for SB environment inflation
PHASE4_ARC_COLS = ["PPPA_Z_SL_best"]
PHASE4_MIN_PA  = 80                       # min PA in a season to qualify
PHASE4_MIN_S   = 3                        # min qualifying seasons for slope+SD


# ---------------------------------------------------------------------------
# Step 3: ovr_hist_data.csv
# ---------------------------------------------------------------------------

STD_LEVELS = {"R", "A", "A+", "AA", "AAA"}
REMAP_LEVELS = {"DSL": "R", "CPX": "R", "A-": "R"}
MISS_KEY   = ["PlayerId", "Season", "Team", "Level"]
_NAME_KEY  = ["Name", "Season", "Level"]


def _strip_accents(s: pd.Series) -> pd.Series:
    return s.str.lower().str.strip().apply(
        lambda x: "".join(c for c in unicodedata.normalize("NFD", str(x))
                          if unicodedata.category(c) != "Mn")
    )

_NORM_KEY = ["_nn", "Season", "Level"]


def _name_fill(out: pd.DataFrame, source: pd.DataFrame, stats: list) -> pd.DataFrame:
    """For rows where ALL stats are null (PlayerId join missed), retry on normalized Name+Season+Level."""
    no_match = out[stats].isna().all(axis=1)
    if not no_match.any():
        return out
    src = source.copy()
    src["_nn"] = _strip_accents(src["Name"])
    agg = src.groupby(_NORM_KEY)[stats].mean().round(2).reset_index()
    out["_nn"] = _strip_accents(out["Name"])
    fill = (out.loc[no_match, _NORM_KEY].reset_index()
               .merge(agg, on=_NORM_KEY, how="left")
               .set_index("index"))
    for col in stats:
        out.loc[no_match, col] = fill[col]
    out = out.drop(columns="_nn")
    return out


def build_ovr_hist_data() -> pd.DataFrame:
    print("  Building ovr_hist_data.csv...")
    BASE_COLS = ["PlayerId", "Season", "Name", "Level", "League", "Age"]
    MISS_STATS = ["wRC+", "BB/K", "K%", "ISO", "SwStr%", "GB/FB"]
    ADV_STATS  = ["BB/K", "K%", "ISO"]
    BAT_STATS  = ["GB/FB", "SwStr%"]

    ml  = pd.read_csv(COMPUTED_DIR / "minorLeagueData.csv", dtype={"PlayerId": str})
    ml["Level"] = ml["Level"].replace(REMAP_LEVELS)

    adv = pd.read_csv(HIST_DIR / "historical_ml_advanced.csv",
                      usecols=[*JOIN_KEY, "Name", "BB/K", "K%", "ISO"]).drop_duplicates(subset=JOIN_KEY)
    adv["Level"] = adv["Level"].replace(REMAP_LEVELS)
    adv = adv.rename(columns={"BB/K": "BB/K_adv", "K%": "K%_adv", "ISO": "ISO_adv"})

    bat = pd.read_csv(HIST_DIR / "historical_ml_batted.csv",
                      usecols=[*JOIN_KEY, "Name", "GB/FB", "SwStr%"]).drop_duplicates(subset=JOIN_KEY)
    bat["Level"] = bat["Level"].replace(REMAP_LEVELS)
    bat = bat.rename(columns={"GB/FB": "GB/FB_bat", "SwStr%": "SwStr%_bat"})

    # missing_milb_data.csv is the primary source for all rate stats.
    # Historical CSVs serve as fallbacks only.
    miss_raw = pd.read_csv(FG_DIR / "missing_milb_data.csv",
                           usecols=[*MISS_KEY, "Name", *MISS_STATS],
                           dtype={"PlayerId": str})
    miss_raw["Level"] = miss_raw["Level"].replace(REMAP_LEVELS)
    miss_raw = miss_raw[miss_raw["Level"].isin(STD_LEVELS)].copy()
    miss_agg = (miss_raw.groupby(MISS_KEY)[MISS_STATS].mean().round(2).reset_index())

    out = (ml
           .merge(adv.drop(columns="Name"), on=JOIN_KEY, how="left")
           .merge(bat.drop(columns="Name"), on=JOIN_KEY, how="left")
           .merge(miss_agg, on=MISS_KEY, how="left")
           .drop_duplicates(subset=[*JOIN_KEY, "League"]))

    # Name fallback for miss: fills rows where PlayerId didn't match (ID system mismatch)
    out = _name_fill(out, miss_raw[_NAME_KEY + MISS_STATS], MISS_STATS)

    # Prefer missing_milb values; fill remaining nulls from historical CSV fallbacks
    out["BB/K"]   = out["BB/K"].fillna(out["BB/K_adv"])
    out["K%"]     = out["K%"].fillna(out["K%_adv"])
    out["ISO"]    = out["ISO"].fillna(out["ISO_adv"])
    out["SwStr%"] = out["SwStr%"].fillna(out["SwStr%_bat"])
    out["GB/FB"]  = out["GB/FB"].fillna(out["GB/FB_bat"])

    # Name fallback for adv/bat: any stats still null after miss fallback
    adv_src = adv.rename(columns={"BB/K_adv": "BB/K", "K%_adv": "K%", "ISO_adv": "ISO"})
    bat_src = bat.rename(columns={"GB/FB_bat": "GB/FB", "SwStr%_bat": "SwStr%"})
    out = _name_fill(out, adv_src[_NAME_KEY + ADV_STATS], ADV_STATS)
    out = _name_fill(out, bat_src[_NAME_KEY + BAT_STATS], BAT_STATS)

    out = out.drop(columns=["BB/K_adv", "K%_adv", "ISO_adv", "SwStr%_bat", "GB/FB_bat"])

    # ml_updated_data.csv: override 2026 rate stats with the most current season data
    upd = pd.read_csv(FG_DIR / "ml_updated_data.csv", dtype={"PlayerId": str})
    upd = upd.drop(columns=["BB/K.1"], errors="ignore")
    upd["Level"] = upd["Level"].replace(REMAP_LEVELS)
    upd = upd[upd["Level"].isin(STD_LEVELS)].copy()
    upd["_nn"] = _strip_accents(upd["Name"])
    out["_nn"] = _strip_accents(out["Name"])
    upd_agg = upd.groupby(_NORM_KEY)[MISS_STATS].mean().round(2).reset_index()
    upd_agg = upd_agg.rename(columns={c: f"_upd_{c}" for c in MISS_STATS})
    out = out.merge(upd_agg, on=_NORM_KEY, how="left").drop(columns="_nn")
    upd_mask = (out["Season"] == 2026) & out[[f"_upd_{c}" for c in MISS_STATS]].notna().any(axis=1)
    for col in MISS_STATS:
        mask = (out["Season"] == 2026) & out[f"_upd_{col}"].notna()
        out.loc[mask, col] = out.loc[mask, f"_upd_{col}"]
    out = out.drop(columns=[f"_upd_{c}" for c in MISS_STATS])
    print(f"    Overrode {upd_mask.sum()} 2026 rows from ml_updated_data.csv")

    # ProspectSavant AAA fallback: fill 2026 AAA rows still missing rate stats.
    # K% and SwStr% are percentage-scale in PS (0-100); ISO is decimal. GB/FB excluded (PS returns zeros).
    # wRC+ always zero from PS API so excluded.
    PS_DIR = DATA_DIR / "prospectSavant"
    ps_path = PS_DIR / "ps_AAA_2026.csv"
    PS_FILL = ["BB/K", "K%", "ISO", "SwStr%"]
    if ps_path.exists():
        ps_aaa = pd.read_csv(ps_path, usecols=["Name", "BB%", "K%", "ISO", "SwStr%"])
        ps_aaa["_nn"]   = _strip_accents(ps_aaa["Name"])
        ps_aaa["BB/K"]  = (ps_aaa["BB%"] / ps_aaa["K%"].replace(0, float("nan"))).round(2)
        ps_aaa["K%"]    = (ps_aaa["K%"]    / 100).round(4)
        ps_aaa["SwStr%"]= (ps_aaa["SwStr%"] / 100).round(4)
        ps_agg = ps_aaa.groupby("_nn")[PS_FILL].mean().round(4).reset_index()
        ps_agg.columns = ["_nn"] + [f"_ps_{c}" for c in PS_FILL]
        out["_nn"] = _strip_accents(out["Name"])
        out = out.merge(ps_agg, on="_nn", how="left").drop(columns="_nn")
        aaa_2026 = (out["Season"] == 2026) & (out["Level"] == "AAA")
        n_filled = 0
        for col in PS_FILL:
            mask = aaa_2026 & out[col].isna() & out[f"_ps_{col}"].notna()
            out.loc[mask, col] = out.loc[mask, f"_ps_{col}"]
            n_filled += mask.sum()
        out = out.drop(columns=[f"_ps_{c}" for c in PS_FILL])
        print(f"    PS AAA fallback: filled {n_filled} missing rate-stat cells for 2026 AAA rows")

    # TBC historical BA rankings — join by normalized Name + Season
    # Gives each player-season a BA_Rank if they appeared on the pre-season list that year.
    tbc_path = DATA_DIR / "rankings" / "tbc_rankings.csv"
    if tbc_path.exists():
        tbc = pd.read_csv(tbc_path)
        ba_hist = tbc[tbc["source"] == "BA"][["year", "player", "rank"]].copy()
        ba_hist["_nn"]   = _strip_accents(ba_hist["player"])
        ba_hist["Season"] = ba_hist["year"].astype(int)
        ba_hist = ba_hist.rename(columns={"rank": "BA_Rank"})
        out["_nn"] = _strip_accents(out["Name"])
        out = out.merge(ba_hist[["_nn", "Season", "BA_Rank"]], on=["_nn", "Season"], how="left")
        out = out.drop(columns="_nn")
        print(f"    BA_Rank joined: {out['BA_Rank'].notna().sum()} player-season rows ranked")
    else:
        out["BA_Rank"] = None

    out = out[BASE_COLS + ["PA", "SB", "BB/K", "K%", "ISO", "wRC+", "GB/FB", "SwStr%",
                            "PPPA", "PPPA_Z_SL", "Age_Z_SL", "BA_Rank"]]

    float_cols = out.select_dtypes(include="float").columns
    out[float_cols] = out[float_cols].round(2)

    out.to_csv(HIST_DIR / "ovr_hist_data.csv", index=False)
    print(f"    Wrote {len(out)} rows")
    return out


# ---------------------------------------------------------------------------
# Step 4: regression_season_league.csv
# ---------------------------------------------------------------------------

def build_regression_season_league(df: pd.DataFrame) -> None:
    print("  Building regression_season_league.csv...")
    MIN_ROWS = len(ALL_PREDS) + 5
    records = []

    for (season, league), grp in df.groupby(["Season", "League"]):
        clean = grp[["PPPA"] + ALL_PREDS].dropna()
        if len(clean) < MIN_ROWS:
            continue
        X = sm.add_constant(clean[ALL_PREDS])
        res = sm.OLS(clean["PPPA"], X).fit()

        row = {
            "Season": season, "League": league, "N": len(clean),
            "R2": round(res.rsquared, 3), "Adj_R2": round(res.rsquared_adj, 3),
            "F_stat": round(res.fvalue, 3), "F_pval": round(res.f_pvalue, 3),
        }
        for pred in ALL_PREDS:
            row[f"{pred}_coef"] = round(res.params[pred], 3)
            row[f"{pred}_pval"] = round(res.pvalues[pred], 3)
        records.append(row)

    out = pd.DataFrame(records).sort_values(["Season", "League"]).reset_index(drop=True)
    out.to_csv(HIST_DIR / "regression_season_league.csv", index=False)
    print(f"    Ran {len(records)} regressions")


# ---------------------------------------------------------------------------
# Step 5: regression_predictive.csv
# ---------------------------------------------------------------------------

def build_regression_predictive(df: pd.DataFrame) -> pd.DataFrame:
    print("  Building regression_predictive.csv...")
    MIN_ROWS = 30

    df = df.copy()
    df["level_num"] = df["Level"].map(LEVEL_MAP)

    t1 = df[["PlayerId", "Season", "Level", "PA", "PPPA", "level_num"]].copy()
    t1["Season"] = t1["Season"] - 1

    pairs = df.merge(
        t1.rename(columns={"Level": "Level_next", "PA": "PA_next",
                            "PPPA": "PPPA_next", "level_num": "level_num_next"}),
        on=["PlayerId", "Season"], how="inner",
    )
    pairs["level_diff"] = pairs["level_num_next"] - pairs["level_num"]
    pairs["weight"] = pairs[["PA", "PA_next"]].min(axis=1)

    records = []
    for season, grp in pairs.groupby("Season"):
        clean = grp[["PPPA_next"] + PRED_REGRESSION + ["weight"]].dropna()
        if len(clean) < MIN_ROWS:
            continue
        X = sm.add_constant(clean[PRED_REGRESSION])
        res = sm.WLS(clean["PPPA_next"], X, weights=clean["weight"]).fit()

        row = {
            "Season_T": season, "N": len(clean),
            "R2": round(res.rsquared, 3), "Adj_R2": round(res.rsquared_adj, 3),
            "F_stat": round(res.fvalue, 3), "F_pval": round(res.f_pvalue, 3),
        }
        for pred in PRED_REGRESSION:
            row[f"{pred}_coef"] = round(res.params[pred], 3)
            row[f"{pred}_pval"] = round(res.pvalues[pred], 3)
        records.append(row)

    out = pd.DataFrame(records).sort_values("Season_T").reset_index(drop=True)
    out.to_csv(HIST_DIR / "regression_predictive.csv", index=False)
    print(f"    Ran {len(records)} regressions")
    return out


# ---------------------------------------------------------------------------
# Step 6: skill_scores.csv
# ---------------------------------------------------------------------------

def build_skill_scores(df: pd.DataFrame, reg: pd.DataFrame) -> None:
    print("  Building skill_scores.csv...")
    coef_cols = {p: f"{p}_coef" for p in SKILL_PREDS}

    rows = df.dropna(subset=SKILL_PREDS).copy()
    rows = rows[rows["Season"].isin(reg["Season_T"])].copy()

    reg_coefs = reg[["Season_T"] + list(coef_cols.values())].rename(columns={"Season_T": "Season"})
    rows = rows.merge(reg_coefs, on="Season", how="left")

    rows["_raw_skill"] = sum(rows[p] * rows[coef_cols[p]] for p in SKILL_PREDS)

    grp = rows.groupby(["Season", "League"])["_raw_skill"]
    rows["_mean"] = grp.transform("mean")
    rows["_std"]  = grp.transform("std")
    rows["Skill_Z"] = ((rows["_raw_skill"] - rows["_mean"]) / rows["_std"]).round(3)
    rows = rows.dropna(subset=["Skill_Z"])

    rows["_weighted"] = rows["Skill_Z"] * rows["PA"]
    grouped = rows.groupby("PlayerId").agg(
        _weighted_sum=("_weighted", "sum"),
        Total_PA=("PA", "sum"),
        Seasons=("Season", "nunique"),
    )
    raw = grouped["_weighted_sum"] / grouped["Total_PA"]
    grouped["Skill_Score"] = (50 + 10 * raw).clip(lower=0).round(2)

    latest_name  = rows.sort_values("Season").groupby("PlayerId")["Name"].last()
    season_range = (
        rows.groupby("PlayerId")["Season"]
        .agg(lambda s: f"{s.min()}" if s.min() == s.max() else f"{s.min()}-{s.max()}")
        .rename("Season_Range")
    )

    out = grouped.join(latest_name).join(season_range).reset_index()
    out = out[["PlayerId", "Name", "Seasons", "Season_Range", "Total_PA", "Skill_Score"]]
    out = out.sort_values("Skill_Score", ascending=False).reset_index(drop=True)
    out.to_csv(HIST_DIR / "skill_scores.csv", index=False)
    print(f"    Scored {len(out)} players")


# ---------------------------------------------------------------------------
# Step 7: regression_mlb_outcomes.csv
# ---------------------------------------------------------------------------

def _milb_traj_agg(milb: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (player-level PA-weighted averages, trajectory deltas) for a milb slice."""
    traj_cols = [f"{p}_delta" for p in MLB_TRAJ]

    for p in MLB_PREDS:
        milb[f"_w_{p}"] = milb[p] * milb["PA"]

    # Season-level PA-weighted averages (needed for trajectory)
    season_agg = (
        milb.groupby(["PlayerId", "Season"])
        .agg(**{f"_wsum_{p}": (f"_w_{p}", "sum") for p in MLB_PREDS},
             _total_pa=("PA", "sum"))
        .reset_index()
    )
    for p in MLB_PREDS:
        season_agg[p] = season_agg[f"_wsum_{p}"] / season_agg["_total_pa"]
    season_agg = season_agg.sort_values(["PlayerId", "Season"])

    # Player-level PA-weighted averages across all included seasons
    player_agg = (
        milb.groupby("PlayerId")
        .agg(**{f"_wsum_{p}": (f"_w_{p}", "sum") for p in MLB_PREDS},
             _total_pa=("PA", "sum"))
        .reset_index()
    )
    for p in MLB_PREDS:
        player_agg[p] = player_agg[f"_wsum_{p}"] / player_agg["_total_pa"]
    avgs = player_agg[["PlayerId"] + MLB_PREDS]

    # Trajectory: delta between earliest and latest of the (up to 2) seasons
    early = season_agg.groupby("PlayerId").nth(0).reset_index()[["PlayerId"] + MLB_TRAJ]
    late  = season_agg.groupby("PlayerId").nth(-1).reset_index()[["PlayerId"] + MLB_TRAJ]
    traj  = early.merge(late, on="PlayerId", suffixes=("_e", "_l"))
    for p in MLB_TRAJ:
        traj[f"{p}_delta"] = traj[f"{p}_l"] - traj[f"{p}_e"]
    traj = traj[["PlayerId"] + traj_cols]

    return avgs, traj


def _arc_features(milb: pd.DataFrame) -> pd.DataFrame:
    """Phase 4: best-season K%/PPPA_Z_SL and K% slope+SD from all available seasons.

    Best-season: per-stat, seasons with PA >= PHASE4_MIN_PA qualify.
      K%_best       = lowest K% across qualifying seasons (fewer Ks = better)
      PPPA_Z_SL_best = highest peer-relative PPPA across qualifying seasons
                       (z-scored vs Season+League peers — controls for SB environment)
    Slope+SD: OLS slope and residual SD of K% across 3+ qualifying seasons.
      K%_slope   = K% change per MiLB season (negative = improving)
      K%_arc_sd  = residual SD around slope (higher = more volatile arc)
    Players with <3 qualifying seasons get NaN for slope/SD (filled with 0 by callers).
    """
    all_players = pd.DataFrame({"PlayerId": milb["PlayerId"].unique()})

    # Rate stats — PA-weighted season average, then best across qualifying seasons
    best_dir = {"K%": "min"}  # lower K% = better; all others max
    for stat in PHASE4_BEST:
        if stat not in milb.columns:
            continue
        valid = milb[["PlayerId", "Season", "PA", stat]].dropna(subset=[stat]).copy()
        valid["_wv"] = valid[stat] * valid["PA"]
        s = (valid.groupby(["PlayerId", "Season"])
             .agg(_wv=("_wv", "sum"), _pa=("PA", "sum"))
             .reset_index())
        s[stat] = s["_wv"] / s["_pa"]
        qual = s[s["_pa"] >= PHASE4_MIN_PA]
        agg_fn = "min" if best_dir.get(stat) == "min" else "max"
        best = (qual.groupby("PlayerId")[stat]
                .agg(agg_fn)
                .rename(f"{stat}_best")
                .reset_index())
        all_players = all_players.merge(best, on="PlayerId", how="left")

    # K% slope + arc SD (PHASE4_MIN_S+ qualifying seasons only)
    if "K%" in milb.columns:
        valid_k = milb[["PlayerId", "Season", "PA", "K%"]].dropna(subset=["K%"]).copy()
        valid_k["_wv"] = valid_k["K%"] * valid_k["PA"]
        sk = (valid_k.groupby(["PlayerId", "Season"])
              .agg(_wv=("_wv", "sum"), _pa=("PA", "sum"))
              .reset_index())
        sk["K%"] = sk["_wv"] / sk["_pa"]
        sk_qual = sk[sk["_pa"] >= PHASE4_MIN_PA].sort_values(["PlayerId", "Season"])

        slope_records = []
        for pid, grp in sk_qual.groupby("PlayerId"):
            if len(grp) < PHASE4_MIN_S:
                continue
            idx  = np.arange(len(grp), dtype=float)
            vals = grp["K%"].values.astype(float)
            slope, intercept = np.polyfit(idx, vals, 1)
            resid_sd = float(np.std(vals - (slope * idx + intercept)))
            slope_records.append({
                "PlayerId":  pid,
                "K%_slope":  round(slope, 5),
                "K%_arc_sd": round(resid_sd, 5),
            })

        slope_df = (pd.DataFrame(slope_records) if slope_records
                    else pd.DataFrame(columns=["PlayerId", "K%_slope", "K%_arc_sd"]))
        all_players = all_players.merge(slope_df, on="PlayerId", how="left")

    return all_players


def build_mlb_outcomes_regression(ovr: pd.DataFrame) -> None:
    print("  Building regression_mlb_outcomes.csv...")
    traj_cols = [f"{p}_delta" for p in MLB_TRAJ]

    mlb = pd.read_csv(HIST_DIR / "hist_mlb_data.csv", dtype={"PlayerId": str})
    first_mlb = (
        mlb.sort_values("Season")
        .groupby("PlayerId")
        .first()
        .reset_index()[["PlayerId", "Season", "PA", "PPPA_Z"]]
        .rename(columns={"Season": "MLB_Season", "PA": "MLB_PA", "PPPA_Z": "MLB_PPPA_Z"})
    )

    # All pre-debut MiLB rows — used for Phase 4 arc features
    milb_all = ovr.merge(first_mlb[["PlayerId", "MLB_Season"]], on="PlayerId", how="inner")
    milb_all = milb_all[milb_all["Season"] < milb_all["MLB_Season"]].copy()

    # Phase 4: arc features from full-season ball only (A and above)
    # R level includes remapped DSL/CPX/A- — too noisy to anchor best-season projections
    arc = _arc_features(milb_all[milb_all["Level"] != "R"])
    arc_cols = [c for c in PHASE4_ARC_COLS if c in arc.columns]

    # Keep only the 2 most recent pre-debut seasons for level averages + delta
    recent_seasons = (
        milb_all[["PlayerId", "Season"]]
        .drop_duplicates()
        .sort_values(["PlayerId", "Season"], ascending=[True, False])
        .groupby("PlayerId")
        .head(2)
    )
    milb = milb_all.merge(recent_seasons, on=["PlayerId", "Season"], how="inner")
    milb = milb.dropna(subset=MLB_PREDS + ["PA"]).copy()

    avgs, traj = _milb_traj_agg(milb)
    data = avgs.merge(first_mlb, on="PlayerId", how="inner")
    data = data.merge(traj, on="PlayerId", how="left")
    for col in traj_cols:
        data[col] = data[col].fillna(0)

    # Phase 4: join only the arc columns in use (filtering avoids NaN bleed from
    # other _arc_features outputs like K%_slope/K%_arc_sd that are no longer predictors)
    arc_merge = arc[["PlayerId"] + arc_cols]
    data = data.merge(arc_merge, on="PlayerId", how="left")
    for col in arc_cols:
        data[col] = data[col].fillna(data[col].mean())

    data = data.dropna()

    all_preds = MLB_PREDS + traj_cols + arc_cols
    X = sm.add_constant(data[all_preds])
    res = sm.WLS(data["MLB_PPPA_Z"], X, weights=data["MLB_PA"]).fit()

    row = {
        "N": len(data),
        "R2": round(res.rsquared, 3),
        "Adj_R2": round(res.rsquared_adj, 3),
        "F_stat": round(res.fvalue, 3),
        "F_pval": round(res.f_pvalue, 3),
    }
    for p in all_preds:
        row[f"{p}_coef"] = round(res.params[p], 4)
        row[f"{p}_pval"] = round(res.pvalues[p], 4)

    pd.DataFrame([row]).to_csv(HIST_DIR / "regression_mlb_outcomes.csv", index=False)
    print(f"    N={len(data)}, R²={row['R2']}, predictors={len(all_preds)}")


# ---------------------------------------------------------------------------
# Step 7: prospect_mlb_proj.csv
# ---------------------------------------------------------------------------

def build_prospect_mlb_proj(ovr: pd.DataFrame) -> None:
    """Apply Phase 2 regression coefficients to all players in ovr_hist_data.
    build_prospect_pool.py joins on PlayerId to pull scores for pool members."""
    print("  Building prospect_mlb_proj.csv...")
    traj_cols = [f"{p}_delta" for p in MLB_TRAJ]

    reg = pd.read_csv(HIST_DIR / "regression_mlb_outcomes.csv")
    # Determine predictor list from saved regression (includes Phase 4 arc cols if present)
    all_preds = [p for p in MLB_PREDS + traj_cols + PHASE4_ARC_COLS
                 if f"{p}_coef" in reg.columns]

    milb_all = ovr.dropna(subset=MLB_PREDS + ["PA"]).copy()

    # Phase 4: arc features from full-season ball only (A and above)
    # R level includes remapped DSL/CPX/A- — too noisy to anchor best-season projections
    arc = _arc_features(milb_all[milb_all["Level"] != "R"])
    arc_cols = [c for c in arc.columns if c != "PlayerId" and c in all_preds]

    # Most recent 2 seasons per player for level averages + delta
    recent = (
        milb_all[["PlayerId", "Season"]]
        .drop_duplicates()
        .sort_values(["PlayerId", "Season"], ascending=[True, False])
        .groupby("PlayerId")
        .head(2)
    )
    milb = milb_all.merge(recent, on=["PlayerId", "Season"], how="inner")

    avgs, traj = _milb_traj_agg(milb)
    data = avgs.merge(traj, on="PlayerId", how="left")
    for col in traj_cols:
        data[col] = data[col].fillna(0)

    # Track qualifying PA (from the 2 most recent complete seasons) for shrinkage
    qual_pa = milb.groupby("PlayerId")["PA"].sum().reset_index().rename(columns={"PA": "_qual_pa"})
    data = data.merge(qual_pa, on="PlayerId", how="left")

    # Phase 4: join arc features; slope+SD default to 0, best-season to column mean
    data = data.merge(arc, on="PlayerId", how="left")
    for col in arc_cols:
        if col in ("K%_slope", "K%_arc_sd"):
            data[col] = data[col].fillna(0)
        else:
            data[col] = data[col].fillna(data[col].mean())

    data = data.dropna(subset=all_preds)

    # Apply coefficients (intercept excluded — standardization removes its effect)
    # Age_Z_SL is boosted 3× — regression underestimates age due to survivorship bias
    AGE_BOOST = 3.0
    data["_raw"] = sum(
        data[p] * float(reg[f"{p}_coef"].iloc[0]) * (AGE_BOOST if p == "Age_Z_SL" else 1.0)
        for p in all_preds
    )

    # PA shrinkage: thin samples are pulled toward the mean
    # Threshold scales by level quality (MLE-grounded): higher discount levels need more PA
    LEVEL_PA_FULL = {
        "AAA": 150, "AA": 150,
        "A+": 250, "A": 250,
        "R": 400, "A-": 400,
        "DSL": 550, "CPX": 550,
    }
    ml_levels = (
        pd.read_csv(COMPUTED_DIR / "minorLeagueData.csv", usecols=["PlayerId", "Season", "Level", "PA"],
                    dtype={"PlayerId": str})
        .sort_values(["PlayerId", "Season", "PA"], ascending=[True, False, False])
        .drop_duplicates(subset="PlayerId", keep="first")[["PlayerId", "Level"]]
    )
    data = data.merge(ml_levels, on="PlayerId", how="left")
    data["_pa_full"] = data["Level"].map(LEVEL_PA_FULL).fillna(250)
    mean_r = data["_raw"].mean()
    shrink = (data["_qual_pa"].fillna(0) / data["_pa_full"]).clip(upper=1.0)
    data["_raw"] = mean_r + shrink * (data["_raw"] - mean_r)

    std_r = data["_raw"].std()
    data["MLB_Proj_Score"] = (
        (50 + 10 * (data["_raw"] - mean_r) / std_r).clip(lower=0).round(2)
    )

    data[["PlayerId", "MLB_Proj_Score"]].to_csv(HIST_DIR / "prospect_mlb_proj.csv", index=False)

    print(f"    Scored {len(data)} players")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Step 1: compute_stats.py")
    compute_stats.main()

    print("Steps 2-5: historical pipeline")
    ovr = build_ovr_hist_data()
    build_regression_season_league(ovr)
    reg = build_regression_predictive(ovr)
    build_skill_scores(ovr, reg)

    print("Step 6: Phase 2 MLB outcomes regression")
    build_mlb_outcomes_regression(ovr)

    print("Step 7: Apply Phase 2 to current prospects")
    build_prospect_mlb_proj(ovr)

    print("Step 8: Phase 3 unified prospect rankings")
    build_prospect_pool.main()

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
