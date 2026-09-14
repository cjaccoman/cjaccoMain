"""Two-tier KNN age graduation model.

Tier 1 (KNN graduation probability):
  Features: current_age_z, age_z_slope, career_pa_eq (AAA-equivalent)
  Standardized within Level; weights 1.0 / 1.0 / 0.5.
  K=50 nearest historical players at same Level. P(grad) = fraction graduated.

Tier 2 (Conditional PPPA | graduated):
  WLS on graduated players: Career_PPPA_Z ~ age_z_pos + age_kink
  Gives E[Career_PPPA_Z | graduated, age_z].

Output: data/computed/age_grad_model.csv
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

LEVEL_ORDER    = {"R": 1, "A": 2, "A+": 3, "AA": 4, "AAA": 5}
LEVEL_DISCOUNT = {"AAA": 1.00, "AA": 0.59, "A+": 0.34, "A": 0.23, "R": 0.10}
MIN_PA         = 50
K_NEIGHBORS    = 50
AGE_KINK_THRESH = 1.5
W_AGE_Z, W_SLOPE, W_PA_EQ = 1.0, 1.0, 0.5


def pa_wt_avg(grp: pd.DataFrame, col: str) -> float:
    return float(np.average(grp[col], weights=grp["PA"]))


def wls_slope(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    x_bar = np.average(x, weights=w)
    denom  = np.dot(w, (x - x_bar) ** 2)
    return float(np.dot(w, (x - x_bar) * y) / denom) if denom > 1e-9 else np.nan


def snapshot_at(grp: pd.DataFrame, target_level: str, min_level_pa: int = MIN_PA) -> dict | None:
    """Feature snapshot for a player as of reaching target_level."""
    tgt = LEVEL_ORDER[target_level]

    # current_age_z: most recent season at target level with PA >= min_level_pa
    at_lvl = grp[(grp["Level"] == target_level) & (grp["PA"] >= min_level_pa)].sort_values("Season")
    if len(at_lvl) == 0:
        return None
    current_age_z = float(at_lvl.iloc[-1]["Age_Z_SL"])

    # Qualifying rows at or below target level for slope + career_pa_eq
    valid = grp["Level"].map(LEVEL_ORDER).notna()
    qual  = grp[valid & (grp["Level"].map(LEVEL_ORDER) <= tgt) & (grp["PA"] >= MIN_PA)].copy()
    qual["_lnum"] = qual["Level"].map(LEVEL_ORDER)

    career_pa_eq = float((qual["PA"] * qual["Level"].map(LEVEL_DISCOUNT)).sum()) if len(qual) > 0 else 0.0

    # age_z_slope: per-level PA-weighted avg, then OLS
    slope = np.nan
    if len(qual) > 0:
        lnums, agez_avgs, pa_sums = [], [], []
        for lvl, lgrp in qual.groupby("Level"):
            lnums.append(LEVEL_ORDER[lvl])
            agez_avgs.append(pa_wt_avg(lgrp, "Age_Z_SL"))
            pa_sums.append(float(lgrp["PA"].sum()))
        if len(lnums) >= 2:
            slope = wls_slope(np.array(lnums, dtype=float),
                              np.array(agez_avgs, dtype=float),
                              np.array(pa_sums, dtype=float))

    return {"current_age_z": current_age_z, "age_z_slope": slope, "career_pa_eq": career_pa_eq}


def build_reference_table(feat: pd.DataFrame, grad_map: dict) -> pd.DataFrame:
    rows = []
    for pid, grp in feat.groupby("PlayerId"):
        graduated = grad_map.get(str(pid), False)
        for level in LEVEL_ORDER:
            snap = snapshot_at(grp, level)
            if snap is None:
                continue
            rows.append({"PlayerId": str(pid), "Level": level, "graduated": graduated, **snap})
    return pd.DataFrame(rows)


def fit_tier2(feat: pd.DataFrame, comps: pd.DataFrame, career_pppa_disc: pd.Series):
    """Survivors-only WLS with pppa_disc control to isolate the marginal age effect.

    Including pppa_disc prevents the age coefficients from absorbing the production-
    mediated path (young players at a level also tend to produce better, which predicts
    better MLB outcomes). Without this control the linear term is inflated ~3x and the
    kink is compressed.  Returns (model, mean_pppa_disc); predictions should set
    pppa_disc=mean to derive a pure age-only multiplier.
    """
    grads = comps[comps["graduated"].astype(bool)][["PlayerId", "Career_PPPA_Z"]].dropna(subset=["Career_PPPA_Z"])
    age_z = (
        feat[feat["PA"] >= MIN_PA].sort_values("Season")
        .groupby("PlayerId").last()[["Age_Z_SL"]].reset_index()
    )
    age_z["PlayerId"] = age_z["PlayerId"].astype(str)
    df = grads.merge(age_z, on="PlayerId", how="inner")
    df["pppa_disc"] = df["PlayerId"].map(career_pppa_disc)
    df = df.dropna(subset=["Career_PPPA_Z", "Age_Z_SL", "pppa_disc"])
    df["age_z_pos"] = (-df["Age_Z_SL"]).clip(-3, 3)
    df["age_kink"]  = (df["age_z_pos"] - AGE_KINK_THRESH).clip(lower=0)
    model = smf.wls("Career_PPPA_Z ~ age_z_pos + age_kink + pppa_disc", data=df,
                    weights=pd.Series(1.0, index=df.index)).fit()
    # Use the within-graduate mean so the baseline reflects a typical graduate,
    # not the all-players mean (which includes non-graduates with lower pppa_disc).
    mean_disc_grads = float(df["pppa_disc"].mean())
    print(f"  Tier 2 (N={len(df):,}):  intercept={model.params['Intercept']:.4f}  "
          f"age_z_pos={model.params['age_z_pos']:.4f}  "
          f"age_kink={model.params['age_kink']:.4f}  "
          f"pppa_disc={model.params['pppa_disc']:.4f}  R²={model.rsquared:.4f}")
    print(f"  mean pppa_disc (graduates): {mean_disc_grads:.4f}")
    return model, mean_disc_grads


def knn_per_level(ref: pd.DataFrame, query: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for level in LEVEL_ORDER:
        ref_l = ref[ref["Level"] == level].copy().reset_index(drop=True)
        qry_l = query[query["Level"] == level].copy().reset_index(drop=True)
        if len(qry_l) == 0:
            continue

        base_rate = float(ref_l["graduated"].mean()) if len(ref_l) > 0 else np.nan

        ref_l["slope_imp"] = ref_l["age_z_slope"].fillna(0.0)
        qry_l["slope_imp"] = qry_l["age_z_slope"].fillna(0.0)

        feat_cols = ["current_age_z", "slope_imp", "career_pa_eq"]
        ref_mat = ref_l[feat_cols].values.astype(float)
        qry_mat = qry_l[feat_cols].values.astype(float)

        mu  = ref_mat.mean(axis=0)
        sig = ref_mat.std(axis=0)
        sig[sig < 1e-9] = 1.0
        ref_w = ((ref_mat - mu) / sig) * np.array([W_AGE_Z, W_SLOPE, W_PA_EQ])
        qry_w = ((qry_mat - mu) / sig) * np.array([W_AGE_Z, W_SLOPE, W_PA_EQ])

        ref_pids = ref_l["PlayerId"].values
        ref_grad = ref_l["graduated"].values.astype(float)

        for i, qrow in qry_l.iterrows():
            pid  = str(qrow["PlayerId"])
            mask = ref_pids != pid
            cw   = ref_w[mask]
            cg   = ref_grad[mask]
            actual_k = min(K_NEIGHBORS, len(cw))

            if actual_k == 0:
                p_grad = base_rate
            else:
                dists  = np.sqrt(((cw - qry_w[i]) ** 2).sum(axis=1))
                nn_idx = np.argpartition(dists, actual_k - 1)[:actual_k]
                p_grad = float(cg[nn_idx].mean())

            out_rows.append({
                "PlayerId":          pid,
                "Level":             level,
                "current_age_z":     round(qrow["current_age_z"], 3),
                "age_z_slope":       round(qrow["age_z_slope"], 3) if not np.isnan(qrow["age_z_slope"]) else np.nan,
                "has_slope":         not np.isnan(qrow["age_z_slope"]),
                "career_pa_eq":      round(qrow["career_pa_eq"], 1),
                "p_grad_knn":        round(p_grad, 4),
                "n_neighbors":       actual_k,
                "level_base_p_grad": round(base_rate, 4),
            })

    return pd.DataFrame(out_rows)


def main() -> None:
    print("=== build_age_grad_model.py ===\n")

    feat  = pd.read_csv(DATA_DIR / "rankings" / "prospect_features.csv", dtype={"PlayerId": str})
    comps = pd.read_csv(DATA_DIR / "computed"  / "player_comps.csv",      dtype={"PlayerId": str})
    pool  = pd.read_csv(DATA_DIR / "rankings"  / "prospect_scores.csv",   dtype={"PlayerId": str})

    comps["graduated"] = comps["graduated"].fillna(False).astype(bool)
    grad_map = dict(zip(comps["PlayerId"].astype(str), comps["graduated"]))

    print(f"Features:     {len(feat):,} rows  ({feat['PlayerId'].nunique():,} players)")
    print(f"Comps:        {int(comps['graduated'].sum()):,} graduates / {len(comps):,} total")
    print(f"Current pool: {len(pool):,} prospects\n")

    # Reference table
    print("Building reference table...")
    ref = build_reference_table(feat, grad_map)
    print(f"Reference:    {len(ref):,} snapshots  ({ref['PlayerId'].nunique():,} players)")
    grad_by_level = ref.groupby("Level")["graduated"].agg(["sum", "count", "mean"])
    print(f"\n  Grad rates by level:\n{grad_by_level.reindex(LEVEL_ORDER).round(3).to_string()}\n")

    # Career-average pppa_disc for all players (PA-weighted PPPA_Z_SL × level_discount)
    f = feat[feat["PA"] >= MIN_PA].copy()
    f["pppa_disc_row"] = f["PPPA_Z_SL"] * f["Level"].map(LEVEL_DISCOUNT).fillna(0)
    career_pppa_disc = (
        f.groupby("PlayerId")
        .apply(lambda g: float(np.average(g["pppa_disc_row"], weights=g["PA"])))
    )
    career_pppa_disc.index = career_pppa_disc.index.astype(str)
    print(f"pppa_disc computed for {len(career_pppa_disc):,} players "
          f"(mean={career_pppa_disc.mean():.3f}, std={career_pppa_disc.std():.3f})\n")

    # Tier 2 regression
    print("Tier 2 regression:")
    t2, mean_disc = fit_tier2(feat, comps, career_pppa_disc)
    baseline_cond = float(t2.predict(
        pd.DataFrame({"age_z_pos": [0.0], "age_kink": [0.0], "pppa_disc": [mean_disc]})
    ).iloc[0])
    print(f"  E[PPPA_Z | grad, age_z=0, pppa_disc=mean] = {baseline_cond:.4f}\n")

    # Query snapshots for current pool
    curr_level = pool.set_index("PlayerId")["Level"].to_dict()
    pool_pids  = set(pool["PlayerId"])

    qry_rows = []
    for pid, grp in feat[feat["PlayerId"].isin(pool_pids)].groupby("PlayerId"):
        level = curr_level.get(pid)
        if level not in LEVEL_ORDER:
            continue
        snap = snapshot_at(grp, level)
        if snap is None:
            # Fallback: try lower PA threshold at current level
            snap = snapshot_at(grp, level, min_level_pa=10)
        if snap is None:
            continue
        qry_rows.append({"PlayerId": pid, "Level": level, **snap})

    query_df = pd.DataFrame(qry_rows)
    print(f"Query set: {len(query_df):,} of {len(pool):,} prospects have valid snapshots\n")

    # KNN
    print("Running KNN...")
    knn_out = knn_per_level(ref, query_df)

    # Tier 2 predictions
    knn_out["age_z_pos"]   = (-knn_out["current_age_z"]).clip(-3, 3)
    knn_out["age_kink"]    = (knn_out["age_z_pos"] - AGE_KINK_THRESH).clip(lower=0)
    knn_out["pppa_disc"]   = knn_out["PlayerId"].map(career_pppa_disc).fillna(mean_disc)

    # Age-only prediction: pppa_disc fixed at mean — isolates pure age signal for implied_mult
    pred_ageonly = knn_out[["age_z_pos", "age_kink"]].copy()
    pred_ageonly["pppa_disc"] = mean_disc
    knn_out["cond_pppa_z_ageonly"] = t2.predict(pred_ageonly).round(4).values

    # Controlled prediction: player's actual pppa_disc — best estimate of expected PPPA
    knn_out["cond_pppa_z"] = t2.predict(
        knn_out[["age_z_pos", "age_kink", "pppa_disc"]]
    ).round(4).values

    knn_out["expected_pppa_z"]     = (knn_out["p_grad_knn"] * knn_out["cond_pppa_z"]).round(4)
    knn_out["level_base_expected"] = (knn_out["level_base_p_grad"] * baseline_cond).round(4)
    knn_out["p_grad_vs_baseline"]  = (knn_out["p_grad_knn"] / knn_out["level_base_p_grad"]).round(3)

    # implied_mult: pure age signal only (pppa_disc=mean), so it's comparable to current_mult
    expected_ageonly          = knn_out["p_grad_knn"] * knn_out["cond_pppa_z_ageonly"]
    knn_out["implied_mult_2tier"] = (expected_ageonly / knn_out["level_base_expected"]).round(3)

    # p_grad_vs_baseline clipped to a usable multiplier range.
    # Floor 0.20: prevents extreme old-age profiles from zeroing out ABILITY.
    # Ceiling 4.0: prevents extreme youth from dominating; can tune down later.
    knn_out["age_mult_pgvb"] = knn_out["p_grad_vs_baseline"].clip(lower=0.20, upper=4.0).round(4)

    # current piecewise multiplier for comparison
    knn_out["current_mult"] = (
        1.0 + 0.0054 * knn_out["age_z_pos"] + 0.192 * knn_out["age_kink"]
    ).round(4)

    # Merge pool metadata
    meta = pool[["PlayerId", "Name", "Team", "Age", "ABILITY_Score", "TOOLS_Score",
                 "Combined_Score", "Combined_Rank"]].copy()
    out = knn_out.merge(meta, on="PlayerId", how="left")
    out = out.sort_values("Combined_Rank").reset_index(drop=True)

    out_cols = [
        "PlayerId", "Name", "Team", "Level", "Age",
        "current_age_z", "age_z_slope", "has_slope", "career_pa_eq",
        "p_grad_knn", "n_neighbors", "level_base_p_grad", "p_grad_vs_baseline",
        "pppa_disc", "cond_pppa_z_ageonly", "cond_pppa_z",
        "level_base_expected", "expected_pppa_z",
        "implied_mult_2tier", "age_mult_pgvb", "current_mult",
        "Combined_Rank", "Combined_Score", "ABILITY_Score", "TOOLS_Score",
    ]
    out_path = DATA_DIR / "computed" / "age_grad_model.csv"
    out[out_cols].to_csv(out_path, index=False)
    print(f"\nWrote {len(out):,} rows -> {out_path}")

    # Level summary
    print("\n=== Level Summary ===")
    lvl_sum = out.groupby("Level").agg(
        n=("PlayerId", "count"),
        base_grad=("level_base_p_grad", "first"),
        mean_p_grad=("p_grad_knn", "mean"),
        min_p=("p_grad_knn", "min"),
        max_p=("p_grad_knn", "max"),
        pct_with_slope=("has_slope", "mean"),
        mean_imp_mult=("implied_mult_2tier", "mean"),
    ).reindex([l for l in LEVEL_ORDER if l in out["Level"].values])
    print(lvl_sum.round(3).to_string())

    # Top 30
    print("\n=== Top 30 Prospects ===")
    print(out.head(30)[[
        "Combined_Rank", "Name", "Level", "Age",
        "current_age_z", "age_z_slope",
        "p_grad_knn", "level_base_p_grad", "p_grad_vs_baseline",
        "age_mult_pgvb", "current_mult",
    ]].to_string(index=False))

    # Distribution of age_mult_pgvb by level
    print("\n=== age_mult_pgvb distribution by level ===")
    for lvl in [l for l in LEVEL_ORDER if l in out["Level"].values]:
        g = out[out["Level"] == lvl]["age_mult_pgvb"]
        print(f"{lvl:4s}  p10={g.quantile(.10):.2f}  p25={g.quantile(.25):.2f}  "
              f"p50={g.quantile(.50):.2f}  p75={g.quantile(.75):.2f}  "
              f"p90={g.quantile(.90):.2f}  clipped_at_floor={( g == 0.20).sum()}"
              f"  clipped_at_ceil={(g == 4.0).sum()}")


if __name__ == "__main__":
    main()
