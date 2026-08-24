"""MiLB -> MLB stat impact analysis using XGBoost + SHAP.

Two outcomes (PPPA = fantasy points per plate appearance):
  1. first_year  — PPPA in first MLB season (MLB PA >= 100)
  2. career      — PA-weighted PPPA across all MLB seasons (career PA >= 400, seasons >= 2)

Features: PA-weighted MiLB averages across all pre-debut seasons + trajectory + context.
Method:   XGBoost regression, 5-fold CV for R², SHAP for feature importance.

Outputs:
  data/historical/stat_impact_firstyear.csv
  data/historical/stat_impact_career.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HIST_DIR = DATA_DIR / "historical"

JOIN_KEY  = ["PlayerId", "Season", "Team", "Level", "PA"]
LEVEL_MAP = {"R": 1, "A": 2, "A+": 3, "AA": 4, "AAA": 5}

MIN_PA_FIRST  = 100   # min first-year MLB PA to include
MIN_PA_CAREER = 400   # min career MLB PA to include
MIN_SEASONS   = 2     # min MLB seasons for career outcome

ADV_FEATS  = ["BB%", "K%", "ISO", "AVG", "OBP", "SLG", "BABIP", "wOBA", "wRC+"]
BAT_FEATS  = ["GB%", "LD%", "FB%", "IFFB%", "HR/FB", "GB/FB",
              "Pull%", "Cent%", "Oppo%", "SwStr%"]
BASE_FEATS = ["PPPA", "Age_Z_SL"]
TRAJ_STATS = ["K%", "PPPA", "BB%", "ISO", "wRC+"]   # priority order: K%/PPPA first

# Phase 4 arc feature constants
PHASE4_BEST   = ["K%", "PPPA"]   # rate stats for best-season features
PHASE4_MIN_PA = 80               # min PA in a season to qualify
PHASE4_MIN_S  = 3                # min qualifying seasons for slope+SD

XGB_PARAMS = dict(
    n_estimators=2000,       # capped by early stopping
    max_depth=3,             # shallow — prevents memorisation with 30 feats / ~900 obs
    learning_rate=0.02,
    subsample=0.7,
    colsample_bytree=0.6,
    min_child_weight=20,     # require 20+ weighted samples per leaf
    reg_alpha=1.0,           # L1
    reg_lambda=5.0,          # L2
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
    early_stopping_rounds=50,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_outcomes() -> tuple[pd.DataFrame, pd.DataFrame]:
    mlb = pd.read_csv(HIST_DIR / "hist_mlb_data.csv", dtype={"PlayerId": str})

    first_mlb = (
        mlb.sort_values("Season")
        .groupby("PlayerId").first().reset_index()
        [["PlayerId", "Season", "PA", "PPPA"]]
        .rename(columns={"Season": "MLB_Season", "PA": "MLB_PA_1st", "PPPA": "PPPA_1st"})
    )

    career_pa   = mlb.groupby("PlayerId")["PA"].sum().rename("Career_PA")
    career_seas = mlb.groupby("PlayerId")["Season"].nunique().rename("MLB_Seasons")
    career_pppa = (
        mlb.assign(_w=mlb["PPPA"] * mlb["PA"])
        .groupby("PlayerId")
        .apply(lambda g: g["_w"].sum() / g["PA"].sum(), include_groups=False)
        .rename("Career_PPPA")
    )
    career = pd.concat([career_pa, career_seas, career_pppa], axis=1).reset_index()

    return first_mlb, career


def _pa_weighted_means(milb: pd.DataFrame, stat_cols: list) -> pd.DataFrame:
    """Vectorised PA-weighted mean per player for each stat column."""
    all_players = pd.DataFrame({"PlayerId": milb["PlayerId"].unique()})
    for col in stat_cols:
        valid = milb[["PlayerId", "PA", col]].dropna(subset=[col]).copy()
        valid["_wv"] = valid[col] * valid["PA"]
        grp = (valid.groupby("PlayerId")
               .agg(_wv=("_wv", "sum"), _pa=("PA", "sum"))
               .reset_index())
        grp[col] = grp["_wv"] / grp["_pa"]
        all_players = all_players.merge(grp[["PlayerId", col]], on="PlayerId", how="left")
    return all_players


def _trajectory(milb: pd.DataFrame, stat_cols: list) -> pd.DataFrame:
    """Delta (latest season - earliest season) of each stat per player.
    Single-season players get delta = 0."""
    sorted_seasons = (
        milb[["PlayerId", "Season"]].drop_duplicates()
        .sort_values(["PlayerId", "Season"])
    )
    first_s = sorted_seasons.groupby("PlayerId").first().reset_index().rename(
        columns={"Season": "first_s"})
    last_s  = sorted_seasons.groupby("PlayerId").last().reset_index().rename(
        columns={"Season": "last_s"})
    fl = first_s.merge(last_s, on="PlayerId")

    for col in stat_cols:
        valid = milb[["PlayerId", "Season", "PA", col]].dropna(subset=[col]).copy()
        valid["_wv"] = valid[col] * valid["PA"]
        s = (valid.groupby(["PlayerId", "Season"])
             .agg(_wv=("_wv", "sum"), _pa=("PA", "sum"))
             .reset_index())
        s[col] = s["_wv"] / s["_pa"]

        early = fl[["PlayerId", "first_s"]].merge(
            s.rename(columns={"Season": "first_s"}),
            on=["PlayerId", "first_s"], how="left"
        )[col].values

        late = fl[["PlayerId", "last_s"]].merge(
            s.rename(columns={"Season": "last_s"}),
            on=["PlayerId", "last_s"], how="left"
        )[col].values

        fl[f"{col}_delta"] = late - early
        single = fl["first_s"] == fl["last_s"]
        fl.loc[single, f"{col}_delta"] = 0.0

    traj_cols = [f"{c}_delta" for c in stat_cols]
    return fl[["PlayerId"] + traj_cols]


def _best_season(milb: pd.DataFrame, stat_cols: list) -> pd.DataFrame:
    """Phase 4: best single-season value per player for each stat (PA >= PHASE4_MIN_PA).

    K% uses min (fewer strikeouts = better); all other stats use max.
    Only seasons with PA >= PHASE4_MIN_PA qualify. Players with no qualifying season get NaN.
    """
    all_players = pd.DataFrame({"PlayerId": milb["PlayerId"].unique()})
    best_dir = {"K%": "min"}

    for col in stat_cols:
        if col not in milb.columns:
            continue
        valid = milb[["PlayerId", "Season", "PA", col]].dropna(subset=[col]).copy()
        valid["_wv"] = valid[col] * valid["PA"]
        s = (valid.groupby(["PlayerId", "Season"])
             .agg(_wv=("_wv", "sum"), _pa=("PA", "sum"))
             .reset_index())
        s[col] = s["_wv"] / s["_pa"]
        qual = s[s["_pa"] >= PHASE4_MIN_PA]
        agg_fn = "min" if best_dir.get(col) == "min" else "max"
        best = (qual.groupby("PlayerId")[col]
                .agg(agg_fn)
                .rename(f"{col}_best")
                .reset_index())
        all_players = all_players.merge(best, on="PlayerId", how="left")

    return all_players


def _slope_sd_k(milb: pd.DataFrame) -> pd.DataFrame:
    """Phase 4: OLS slope and residual SD of K% across qualifying seasons.

    Season index is 0,1,2,... so slope is K% change per MiLB season (negative = improving).
    Only players with PHASE4_MIN_S+ qualifying seasons (PA >= PHASE4_MIN_PA) are computed;
    others receive NaN (filled with 0 by callers — no trend, no volatility).
    """
    valid = milb[["PlayerId", "Season", "PA", "K%"]].dropna(subset=["K%"]).copy()
    valid["_wv"] = valid["K%"] * valid["PA"]
    s = (valid.groupby(["PlayerId", "Season"])
         .agg(_wv=("_wv", "sum"), _pa=("PA", "sum"))
         .reset_index())
    s["K%"] = s["_wv"] / s["_pa"]
    qual = s[s["_pa"] >= PHASE4_MIN_PA].sort_values(["PlayerId", "Season"])

    records = []
    for pid, grp in qual.groupby("PlayerId"):
        if len(grp) < PHASE4_MIN_S:
            continue
        idx  = np.arange(len(grp), dtype=float)
        vals = grp["K%"].values.astype(float)
        slope, intercept = np.polyfit(idx, vals, 1)
        resid_sd = float(np.std(vals - (slope * idx + intercept)))
        records.append({
            "PlayerId":  pid,
            "K%_slope":  round(slope, 5),
            "K%_arc_sd": round(resid_sd, 5),
        })

    all_players = pd.DataFrame({"PlayerId": milb["PlayerId"].unique()})
    if records:
        all_players = all_players.merge(pd.DataFrame(records), on="PlayerId", how="left")
    else:
        all_players["K%_slope"]  = np.nan
        all_players["K%_arc_sd"] = np.nan

    return all_players


def load_milb_features(first_mlb: pd.DataFrame) -> pd.DataFrame:
    print("  Loading MiLB feature sources...")
    all_feat_cols = ADV_FEATS + BAT_FEATS + BASE_FEATS

    adv = pd.read_csv(HIST_DIR / "historical_ml_advanced.csv",
                      usecols=[*JOIN_KEY, *ADV_FEATS],
                      dtype={"PlayerId": str}).drop_duplicates(subset=JOIN_KEY)
    bat = pd.read_csv(HIST_DIR / "historical_ml_batted.csv",
                      usecols=[*JOIN_KEY, *BAT_FEATS],
                      dtype={"PlayerId": str}).drop_duplicates(subset=JOIN_KEY)
    # ovr_hist_data has no Team column; join on PlayerId+Season+Level+PA
    OVR_KEY = ["PlayerId", "Season", "Level", "PA"]
    ovr = pd.read_csv(HIST_DIR / "ovr_hist_data.csv",
                      usecols=[*OVR_KEY, "SB", "PPPA", "Age_Z_SL"],
                      dtype={"PlayerId": str})

    milb = (adv
            .merge(bat, on=JOIN_KEY, how="outer")
            .merge(ovr, on=OVR_KEY, how="left"))

    # Pre-debut seasons only, standard levels only
    milb = milb.merge(first_mlb[["PlayerId", "MLB_Season"]], on="PlayerId", how="inner")
    milb = milb[milb["Season"] < milb["MLB_Season"]].drop(columns="MLB_Season")
    milb = milb[milb["Level"].isin(LEVEL_MAP)].copy()
    milb["level_num"] = milb["Level"].map(LEVEL_MAP)

    print(f"    Pre-debut rows: {len(milb):,}  |  Players: {milb['PlayerId'].nunique():,}")

    print("    Computing PA-weighted averages...")
    avgs = _pa_weighted_means(milb, all_feat_cols)

    print("    Computing trajectory deltas...")
    traj = _trajectory(milb, TRAJ_STATS)

    # Phase 4: best-season features (K%, PPPA, SB_rate)
    print("    Computing Phase 4 best-season features...")
    if "SB" in milb.columns:
        milb["SB_rate"] = milb["SB"] / milb["PA"]
        best_stats = PHASE4_BEST + ["SB_rate"]
    else:
        best_stats = PHASE4_BEST
    best = _best_season(milb, best_stats)

    # Phase 4: K% slope + arc SD (players with 3+ qualifying seasons)
    print("    Computing Phase 4 K% slope + arc SD...")
    arc_sd = _slope_sd_k(milb)
    # Fill NaN: slope/SD default to 0 (no trend / no volatility for <3-season players)
    arc_sd["K%_slope"]  = arc_sd["K%_slope"].fillna(0)
    arc_sd["K%_arc_sd"] = arc_sd["K%_arc_sd"].fillna(0)

    # Context features
    ctx = (milb.groupby("PlayerId")
           .agg(
               Total_MiLB_PA =("PA",         "sum"),
               MiLB_Seasons  =("Season",     "nunique"),
               Max_Level     =("level_num",  "max"),
           )
           .reset_index())

    # Age at debut from first_mlb
    debut_age = first_mlb[["PlayerId", "MLB_Season"]].copy()
    debut_age = debut_age.rename(columns={"MLB_Season": "Debut_Year"})

    feats = (avgs
             .merge(traj,      on="PlayerId", how="left")
             .merge(best,      on="PlayerId", how="left")
             .merge(arc_sd,    on="PlayerId", how="left")
             .merge(ctx,       on="PlayerId", how="left")
             .merge(debut_age, on="PlayerId", how="left"))

    print(f"    Feature matrix: {feats.shape[0]} players × {feats.shape[1]-1} features")
    return feats


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def run_model(X: pd.DataFrame, y: pd.Series, weights: pd.Series,
              label: str) -> pd.DataFrame:
    print(f"\n  [{label}]  N={len(X)}  features={X.shape[1]}")

    X_arr = X.values.astype(float)
    y_arr = y.values.astype(float)
    w_arr = weights.values.astype(float)

    # 5-fold CV with early stopping — records optimal n_estimators per fold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2, best_iters = [], []
    oof_preds = np.full(len(y_arr), np.nan)

    for train_idx, val_idx in kf.split(X_arr):
        m = xgb.XGBRegressor(**XGB_PARAMS)
        m.fit(X_arr[train_idx], y_arr[train_idx],
              sample_weight=w_arr[train_idx],
              eval_set=[(X_arr[val_idx], y_arr[val_idx])],
              verbose=False)
        pred = m.predict(X_arr[val_idx])
        oof_preds[val_idx] = pred
        cv_r2.append(r2_score(y_arr[val_idx], pred, sample_weight=w_arr[val_idx]))
        best_iters.append(m.best_iteration)

    median_iters = int(np.median(best_iters))
    oof_r2 = r2_score(y_arr, oof_preds, sample_weight=w_arr)
    print(f"    5-fold CV  R² = {np.mean(cv_r2):.3f} ± {np.std(cv_r2):.3f}  "
          f"(OOF R²={oof_r2:.3f}, best_iter median={median_iters})")

    # Final model with median early-stopped n_estimators — no early stopping
    final_params = {k: v for k, v in XGB_PARAMS.items()
                    if k not in ("early_stopping_rounds",)}
    final_params["n_estimators"] = median_iters
    model = xgb.XGBRegressor(**final_params)
    model.fit(X_arr, y_arr, sample_weight=w_arr, verbose=False)
    train_r2 = r2_score(y_arr, model.predict(X_arr), sample_weight=w_arr)
    print(f"    Train      R² = {train_r2:.3f}")

    # SHAP on held-out OOF predictions gives more honest directions
    # but TreeExplainer needs the trained model; we use OOF R² as honesty check
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_arr)
    shap_df   = pd.DataFrame(shap_vals, columns=X.columns)

    importance = pd.DataFrame({
        "Feature":       X.columns,
        "Mean_Abs_SHAP": np.abs(shap_df).mean().values,
        "Mean_SHAP":     shap_df.mean().values,
        "Std_SHAP":      shap_df.std().values,
    }).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

    importance["Rank"]   = range(1, len(importance) + 1)
    importance["OOF_R2"] = round(float(oof_r2), 3)

    for col in ["Mean_Abs_SHAP", "Mean_SHAP", "Std_SHAP"]:
        importance[col] = importance[col].round(5)

    return importance


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== MiLB -> MLB Stat Impact Analysis ===\n")

    first_mlb, career = load_outcomes()
    feats = load_milb_features(first_mlb)

    feat_cols = [c for c in feats.columns if c != "PlayerId"]

    # ── Outcome 1: First-year PPPA ──────────────────────────────────────────
    print("\nOutcome 1: First-year MLB PPPA")
    d1 = (feats
          .merge(first_mlb, on="PlayerId")
          .query("MLB_PA_1st >= @MIN_PA_FIRST")
          .dropna(subset=["PPPA_1st"]))
    imp1 = run_model(d1[feat_cols], d1["PPPA_1st"], d1["MLB_PA_1st"], "First-year")
    imp1.to_csv(HIST_DIR / "stat_impact_firstyear.csv", index=False)
    print("    -> stat_impact_firstyear.csv")

    # ── Outcome 2: Career PPPA ───────────────────────────────────────────────
    print("\nOutcome 2: Career MLB PPPA")
    d2 = (feats
          .merge(career, on="PlayerId")
          .query("Career_PA >= @MIN_PA_CAREER and MLB_Seasons >= @MIN_SEASONS")
          .dropna(subset=["Career_PPPA"]))
    imp2 = run_model(d2[feat_cols], d2["Career_PPPA"], d2["Career_PA"], "Career")
    imp2.to_csv(HIST_DIR / "stat_impact_career.csv", index=False)
    print("    -> stat_impact_career.csv")

    # ── Side-by-side summary ─────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"{'Rank':<5} {'Feature':<20} {'1st|SHAP':>9} {'Dir':>6}  {'Car|SHAP':>9} {'Dir':>6}")
    print("-" * 72)
    r2_map = imp2.set_index("Feature")["Mean_Abs_SHAP"]
    dir_map = imp2.set_index("Feature")["Mean_SHAP"]
    for _, row in imp1.head(20).iterrows():
        feat = row["Feature"]
        car_abs = r2_map.get(feat, float("nan"))
        car_dir = dir_map.get(feat, float("nan"))
        dir1 = "+" if row["Mean_SHAP"] > 0 else "-"
        dir2 = "+" if car_dir > 0 else "-"
        print(f"{int(row['Rank']):<5} {feat:<20} {row['Mean_Abs_SHAP']:>9.4f} {dir1:>6}  "
              f"{car_abs:>9.4f} {dir2:>6}")
    print(f"\n  First-year OOF R²: {imp1['OOF_R2'].iloc[0]:.3f}  |  "
          f"Career OOF R²: {imp2['OOF_R2'].iloc[0]:.3f}")


if __name__ == "__main__":
    main()
