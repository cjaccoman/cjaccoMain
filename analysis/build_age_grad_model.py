"""Two-tier KNN age graduation model — per-row edition.

Computes age_mult_pgvb for every row in prospect_features.csv using
point-in-time features: each row's snapshot uses only data available up
to and including that row's Season (no look-ahead).

Tier 1 (KNN graduation probability):
  Features: current_age_z, age_z_slope, career_pa_eq (AAA-equivalent)
  Standardized within Level; weights 1.0 / 1.0 / 0.5.
  K=50 nearest historical players at same Level. P(grad) = fraction graduated.

Tier 2 (Conditional PPPA | graduated):
  WLS on graduated players: Career_PPPA_Z ~ age_z_pos + age_kink
  Gives E[Career_PPPA_Z | graduated, age_z]. Used for implied_mult_2tier only.

Primary output: data/computed/age_mult_rows.csv
  One row per player-season-level (all rows in prospect_features with PA >= MIN_PA).
  Columns: PlayerId, Season, Level, Name, Age, PA, current_age_z, age_z_slope,
           career_pa_eq, p_grad_knn, n_neighbors, level_base_p_grad,
           p_grad_vs_baseline, age_mult_pgvb

Secondary output: data/computed/age_grad_model.csv
  Current prospect pool snapshot (analysis reference only — not used by pipeline).
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
    """Feature snapshot for a player as of reaching target_level (reference table use)."""
    tgt = LEVEL_ORDER[target_level]
    at_lvl = grp[(grp["Level"] == target_level) & (grp["PA"] >= min_level_pa)].sort_values("Season")
    if len(at_lvl) == 0:
        return None
    current_age_z = float(at_lvl.iloc[-1]["Age_Z_SL"])

    valid = grp["Level"].map(LEVEL_ORDER).notna()
    qual  = grp[valid & (grp["Level"].map(LEVEL_ORDER) <= tgt) & (grp["PA"] >= MIN_PA)].copy()
    career_pa_eq = float((qual["PA"] * qual["Level"].map(LEVEL_DISCOUNT)).sum()) if len(qual) > 0 else 0.0

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


def build_query_table_perrow(feat: pd.DataFrame) -> pd.DataFrame:
    """Per-row features with point-in-time truncation for all qualifying rows.

    For each row (PlayerId, Season, Level) with PA >= MIN_PA:
      current_age_z: the row's own Age_Z_SL
      age_z_slope  : PA-weighted OLS slope of Age_Z_SL vs level_num, using all
                     of this player's rows with Season <= this row's Season
                     and level_num <= this row's level_num and PA >= MIN_PA
      career_pa_eq : sum(PA * level_discount) for same filtered set
    """
    feat = feat.copy()
    feat["_lnum"] = feat["Level"].map(LEVEL_ORDER)
    feat["_pa_eq"] = feat["PA"] * feat["Level"].map(LEVEL_DISCOUNT).fillna(0)

    result_rows = []
    player_groups = list(feat.groupby("PlayerId"))
    n_players = len(player_groups)

    print(f"  Building per-row query table for {n_players:,} players...")

    for i, (pid, grp) in enumerate(player_groups):
        if i % 2000 == 0:
            print(f"    {i:,} / {n_players:,}")

        grp = grp.sort_values("Season").reset_index(drop=True)
        grp_valid = grp[grp["PA"] >= MIN_PA]

        for _, row in grp.iterrows():
            if row["PA"] < MIN_PA:
                continue
            if pd.isna(row.get("_lnum")):
                continue

            season = row["Season"]
            level  = row["Level"]
            lnum   = row["_lnum"]

            prior = grp_valid[
                (grp_valid["Season"] <= season) &
                (grp_valid["_lnum"] <= lnum)
            ]

            age_z_val = row["Age_Z_SL"]
            current_age_z = float(age_z_val) if pd.notna(age_z_val) else 0.0
            career_pa_eq  = float(prior["_pa_eq"].sum())

            slope = np.nan
            by_level = prior.groupby("Level")
            if len(by_level) >= 2:
                lnums_s, agez_s, pa_s = [], [], []
                for lvl, lgrp in by_level:
                    if lvl not in LEVEL_ORDER:
                        continue
                    total_pa = float(lgrp["PA"].sum())
                    if total_pa > 0:
                        lnums_s.append(LEVEL_ORDER[lvl])
                        agez_s.append(float((lgrp["Age_Z_SL"] * lgrp["PA"]).sum() / total_pa))
                        pa_s.append(total_pa)
                if len(lnums_s) >= 2:
                    slope = wls_slope(np.array(lnums_s, dtype=float),
                                      np.array(agez_s, dtype=float),
                                      np.array(pa_s, dtype=float))

            result_rows.append({
                "PlayerId":      str(pid),
                "Season":        int(season),
                "Level":         level,
                "Name":          row.get("Name", ""),
                "Age":           row.get("Age", np.nan),
                "PA":            int(row["PA"]),
                "current_age_z": round(current_age_z, 3),
                "age_z_slope":   round(float(slope), 3) if pd.notna(slope) else np.nan,
                "career_pa_eq":  round(career_pa_eq, 1),
            })

    return pd.DataFrame(result_rows)


def fit_tier2(feat: pd.DataFrame, comps: pd.DataFrame, career_pppa_disc: pd.Series):
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
    mean_disc_grads = float(df["pppa_disc"].mean())
    print(f"  Tier 2 (N={len(df):,}):  intercept={model.params['Intercept']:.4f}  "
          f"age_z_pos={model.params['age_z_pos']:.4f}  "
          f"age_kink={model.params['age_kink']:.4f}  R²={model.rsquared:.4f}")
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

        print(f"  Level {level}: {len(qry_l):,} query rows vs {len(ref_l):,} reference rows")

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
                "Season":            int(qrow.get("Season", 0)),
                "Level":             level,
                "Name":              qrow.get("Name", ""),
                "Age":               qrow.get("Age", np.nan),
                "PA":                int(qrow.get("PA", 0)),
                "current_age_z":     round(qrow["current_age_z"], 3),
                "age_z_slope":       round(qrow["age_z_slope"], 3) if not np.isnan(qrow["age_z_slope"]) else np.nan,
                "has_slope":         not np.isnan(qrow["age_z_slope"]),
                "career_pa_eq":      round(qrow["career_pa_eq"], 1),
                "p_grad_knn":        round(p_grad, 4),
                "n_neighbors":       actual_k,
                "level_base_p_grad": round(base_rate, 4),
            })

    return pd.DataFrame(out_rows)


def apply_level_normalization(knn_out: pd.DataFrame) -> pd.DataFrame:
    """Normalize age_mult_pgvb so median across all rows at each level = 1.0.
    R-ball is forced to 1.0 (graduation signal too sparse).
    """
    knn_out = knn_out.copy()
    knn_out["p_grad_vs_baseline"] = (
        knn_out["p_grad_knn"] / knn_out["level_base_p_grad"]
    ).round(3)

    non_r = knn_out["Level"] != "R"
    level_medians = knn_out[non_r].groupby("Level")["p_grad_vs_baseline"].median()

    def _pgvb_mult(row):
        if row["Level"] == "R":
            return 1.0
        med = level_medians.get(row["Level"], 1.0)
        if med <= 0:
            return 1.0
        return float(np.clip(row["p_grad_vs_baseline"] / med, 0.20, 4.0))

    knn_out["age_mult_pgvb"] = knn_out.apply(_pgvb_mult, axis=1).round(4)
    return knn_out


def main() -> None:
    print("=== build_age_grad_model.py (per-row edition) ===\n")

    feat  = pd.read_parquet(DATA_DIR / "rankings" / "prospect_features.parquet")
    comps = pd.read_csv(DATA_DIR / "computed"  / "player_comps.csv",      dtype={"PlayerId": str})

    comps["graduated"] = comps["graduated"].fillna(False).astype(bool)
    grad_map = dict(zip(comps["PlayerId"].astype(str), comps["graduated"]))

    print(f"Features:     {len(feat):,} rows  ({feat['PlayerId'].nunique():,} players)")
    print(f"Comps:        {int(comps['graduated'].sum()):,} graduates / {len(comps):,} total\n")

    # Reference table (terminal snapshot per player per level, for KNN)
    print("Building reference table...")
    ref = build_reference_table(feat, grad_map)
    print(f"Reference:    {len(ref):,} snapshots  ({ref['PlayerId'].nunique():,} players)")
    grad_by_level = ref.groupby("Level")["graduated"].agg(["sum", "count", "mean"])
    print(f"\n  Grad rates by level:\n{grad_by_level.reindex(LEVEL_ORDER).round(3).to_string()}\n")

    # Career-average pppa_disc (for Tier 2)
    f = feat[feat["PA"] >= MIN_PA].copy()
    f["pppa_disc_row"] = f["PPPA_Z_SL"] * f["Level"].map(LEVEL_DISCOUNT).fillna(0)
    career_pppa_disc = (
        f.groupby("PlayerId")
        .apply(lambda g: float(np.average(g["pppa_disc_row"], weights=g["PA"])))
    )
    career_pppa_disc.index = career_pppa_disc.index.astype(str)

    # Tier 2 regression
    print("Tier 2 regression:")
    t2, mean_disc = fit_tier2(feat, comps, career_pppa_disc)
    baseline_cond = float(t2.predict(
        pd.DataFrame({"age_z_pos": [0.0], "age_kink": [0.0], "pppa_disc": [mean_disc]})
    ).iloc[0])
    print(f"  E[PPPA_Z | grad, age_z=0, pppa_disc=mean] = {baseline_cond:.4f}\n")

    # Per-row query table
    print("Building per-row query table (point-in-time)...")
    query_df = build_query_table_perrow(feat)
    print(f"\nQuery rows: {len(query_df):,}  "
          f"(players: {query_df['PlayerId'].nunique():,})\n")

    # KNN
    print("Running KNN (per-row)...")
    knn_out = knn_per_level(ref, query_df)
    print(f"\nKNN complete: {len(knn_out):,} rows\n")

    # Level normalization
    knn_out = apply_level_normalization(knn_out)

    # Tier 2 predictions (for analysis columns only — not used in pipeline)
    knn_out["age_z_pos"] = (-knn_out["current_age_z"]).clip(-3, 3)
    knn_out["age_kink"]  = (knn_out["age_z_pos"] - AGE_KINK_THRESH).clip(lower=0)
    knn_out["pppa_disc"] = knn_out["PlayerId"].map(career_pppa_disc).fillna(mean_disc)

    pred_ageonly = knn_out[["age_z_pos", "age_kink"]].copy()
    pred_ageonly["pppa_disc"] = mean_disc
    knn_out["cond_pppa_z_ageonly"] = t2.predict(pred_ageonly).round(4).values
    knn_out["cond_pppa_z"] = t2.predict(
        knn_out[["age_z_pos", "age_kink", "pppa_disc"]]
    ).round(4).values
    knn_out["expected_pppa_z"]     = (knn_out["p_grad_knn"] * knn_out["cond_pppa_z"]).round(4)
    knn_out["level_base_expected"] = (knn_out["level_base_p_grad"] * baseline_cond).round(4)
    expected_ageonly               = knn_out["p_grad_knn"] * knn_out["cond_pppa_z_ageonly"]
    knn_out["implied_mult_2tier"]  = (expected_ageonly / knn_out["level_base_expected"]).round(3)

    # current piecewise multiplier for comparison
    knn_out["current_mult"] = (
        1.0 + 0.0054 * knn_out["age_z_pos"] + 0.192 * knn_out["age_kink"]
    ).round(4)

    # Primary output: per-row multipliers for build_ability_score.py
    rows_out_cols = [
        "PlayerId", "Season", "Level", "Name", "Age", "PA",
        "current_age_z", "age_z_slope", "career_pa_eq",
        "p_grad_knn", "n_neighbors", "level_base_p_grad",
        "p_grad_vs_baseline", "age_mult_pgvb",
    ]
    rows_path = DATA_DIR / "computed" / "age_mult_rows.csv"
    knn_out[rows_out_cols].to_csv(rows_path, index=False)
    print(f"Wrote {len(knn_out):,} rows -> {rows_path}")

    # Distribution by level
    print("\n=== age_mult_pgvb distribution by level ===")
    for lvl in [l for l in LEVEL_ORDER if l in knn_out["Level"].values]:
        g = knn_out[knn_out["Level"] == lvl]["age_mult_pgvb"]
        print(f"{lvl:4s}  p10={g.quantile(.10):.2f}  p25={g.quantile(.25):.2f}  "
              f"p50={g.quantile(.50):.2f}  p75={g.quantile(.75):.2f}  "
              f"p90={g.quantile(.90):.2f}  "
              f"floor={(g == 0.20).sum()}  ceil={(g == 4.0).sum()}")

    # Secondary output: current prospect pool snapshot (analysis only)
    try:
        pool = pd.read_csv(DATA_DIR / "rankings" / "prospect_scores.csv", dtype={"PlayerId": str})
        pool_pids = set(pool["PlayerId"])
        curr_level = pool.set_index("PlayerId")["Level"].to_dict()

        # For each current prospect, use their most recent qualifying row
        curr_rows = (
            knn_out[knn_out["PlayerId"].isin(pool_pids)]
            .sort_values(["Season", "PA"], ascending=False)
            .drop_duplicates("PlayerId")
            .copy()
        )
        meta = pool[["PlayerId", "Name", "Team", "Age", "ABILITY_Score",
                     "TOOLS_Score", "Combined_Score", "Combined_Rank"]].copy()
        analysis_out = curr_rows.merge(meta.drop(columns=["Name", "Age"], errors="ignore"),
                                       on="PlayerId", how="left")
        analysis_out = analysis_out.sort_values("Combined_Rank", na_position="last")

        model_path = DATA_DIR / "computed" / "age_grad_model.csv"
        analysis_out.to_csv(model_path, index=False)
        print(f"\nWrote {len(analysis_out):,} rows -> {model_path} (analysis reference)")

        print("\n=== Top 25 Current Prospects (most-recent row) ===")
        top = analysis_out.head(25)
        if "Combined_Rank" in top.columns:
            print(top[["Combined_Rank", "Name", "Level", "Age",
                        "current_age_z", "age_z_slope",
                        "p_grad_knn", "level_base_p_grad", "p_grad_vs_baseline",
                        "age_mult_pgvb", "current_mult"]].to_string(index=False))
    except FileNotFoundError:
        print("  (prospect_scores.csv not found — skipping analysis output)")


if __name__ == "__main__":
    main()
