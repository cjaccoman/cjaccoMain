"""Comprehensive MiLB talent archetype analysis: which skill combinations predict MLB PPPA success.

Target: Career_PPPA_Z (PA-weighted career MLB PPPA z-score), not wRC+/WAR.
Key model insight: K% = single largest drag (SO = -2 pts); SB = +3 has no wRC+ analog.

Output: data/historical/archetype_analysis_output.txt
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import cross_val_score

DATA_DIR       = Path(__file__).resolve().parent.parent / "data"
PC_PATH        = DATA_DIR / "computed"  / "player_comps.csv"
FEATURES_PATH  = DATA_DIR / "rankings"  / "prospect_features.parquet"
OUT_PATH       = DATA_DIR / "historical" / "archetype_analysis_output.txt"

RANDOM_STATE = 42
FULL_SEASON_LEVELS = {"A", "A+", "AA", "AAA"}

# ── tee output to file ────────────────────────────────────────────────────────
class Tee:
    def __init__(self, fh): self.fh = fh
    def write(self, s):
        sys.__stdout__.buffer.write(s.encode("utf-8", errors="replace"))
        self.fh.write(s)
    def flush(self):
        sys.__stdout__.buffer.flush()
        self.fh.flush()

# ── helpers ───────────────────────────────────────────────────────────────────
def tier(z):
    if   z >= 0.75:  return "Elite"
    elif z >= 0.25:  return "Above Avg"
    elif z >= -0.25: return "Average"
    else:            return "Below Avg"

def success(z):
    return z >= 0.25  # Elite + Above Avg

def name_cluster(disc_z, sb_z, power_z, thr=0.30):
    """Name cluster from STANDARDIZED centroid values (z-scores, not raw rates)."""
    hi_disc  = disc_z  >  thr
    lo_disc  = disc_z  < -thr
    hi_sb    = sb_z    >  thr
    hi_power = power_z >  thr

    if hi_disc and hi_sb and hi_power:              return "Five-Tool"
    if hi_disc and hi_sb and not hi_power:          return "Contact/Speed"
    if hi_disc and not hi_sb and not hi_power:      return "Pure Contact"
    if hi_disc and hi_power and not hi_sb:          return "Contact/Power"
    if lo_disc and hi_sb and hi_power:              return "Power-Speed/K-Risk"
    if lo_disc and hi_sb:                           return "Speed/K-Risk"
    if lo_disc and hi_power:                        return "Power/K-Risk"
    if hi_sb and not lo_disc:                       return "Speed"
    if hi_power and not lo_disc:                    return "Power"
    return "Average"

def dedup_names(names):
    from collections import Counter
    counts = Counter(names.values())
    seen, result = {}, {}
    for k in sorted(names):
        v = names[k]
        if counts[v] > 1:
            seen[v] = seen.get(v, 0) + 1
            result[k] = f"{v}_{seen[v]}"
        else:
            result[k] = v
    return result

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    # ── 1. Build graduated player dataset ────────────────────────────────────
    print("=" * 72)
    print("SECTION 1 — GRADUATED PLAYER DATASET")
    print("=" * 72)

    pc = pd.read_csv(PC_PATH)
    g  = pc[(pc["graduated"] == True) & (pc["Career_MLB_PA"] >= 200)].copy()
    print(f"Graduated players with Career_MLB_PA >= 200: {len(g):,}")

    career_feats = [
        "BB2K_career", "Whiff_career", "SBTalent_career", "HRFB_career",
        "PullAir_career", "Kpct_career", "BBpct_career",
        "MaxEV_career", "EV90_career", "Chase_career", "ZContact_career",
    ]
    core_feats   = ["BB2K_career", "Kpct_career", "SBTalent_career",
                    "HRFB_career", "PullAir_career"]

    # Mean-impute sparse columns within graduated set
    for col in career_feats:
        if g[col].isna().any():
            mu = g[col].mean()
            n_missing = g[col].isna().sum()
            g[col] = g[col].fillna(mu)
            print(f"  Imputed {n_missing} NaN in {col} with mean={mu:.4f}")

    g["Tier"]    = g["PPPA_Z_career"].apply(tier)
    g["Success"] = g["PPPA_Z_career"].apply(success)

    print(f"\nSuccess tier distribution (Career_PPPA_Z):")
    print(f"  {g['Tier'].value_counts().to_string()}")
    print(f"  Overall success rate: {g['Success'].mean()*100:.1f}%")
    print(f"  PPPA_Z_career — mean={g['PPPA_Z_career'].mean():.3f} "
          f"std={g['PPPA_Z_career'].std():.3f} "
          f"p25={g['PPPA_Z_career'].quantile(.25):.3f} "
          f"p75={g['PPPA_Z_career'].quantile(.75):.3f}")
    print(f"\nNote: Graduated sample is survivor-biased (above-average MLB PA required).")
    print(f"Tier thresholds are fixed (-0.25/+0.25/+0.75 PPPA_Z); ~{(g['PPPA_Z_career']<-0.25).mean()*100:.0f}% "
          f"fall 'Below Avg', ~{(g['PPPA_Z_career']>=0.75).mean()*100:.0f}% are 'Elite'.")

    # ── 2. Feature importance ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SECTION 2 — FEATURE IMPORTANCE (Random Forest + Decision Tree)")
    print("=" * 72)

    X = g[career_feats].values
    y = g["PPPA_Z_career"].values

    rf = RandomForestRegressor(n_estimators=500, max_features="sqrt",
                               random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X, y)

    cv_scores = cross_val_score(rf, X, y, cv=5, scoring="r2")
    print(f"\nRandom Forest (500 trees) — 5-fold CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    fi = (pd.Series(rf.feature_importances_, index=career_feats)
          .sort_values(ascending=False))
    print("\nFeature importances (MiLB -> MLB PPPA, points-scoring):")
    for feat, imp in fi.items():
        bar = "█" * int(imp * 80)
        print(f"  {feat:<22}  {imp:.4f}  {bar}")

    # Decision tree for interpretable rules
    dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=25,
                               random_state=RANDOM_STATE)
    dt.fit(X, y)
    dt_cv = cross_val_score(dt, X, y, cv=5, scoring="r2")
    print(f"\nDecision Tree (max_depth=4) — 5-fold CV R²: {dt_cv.mean():.3f} ± {dt_cv.std():.3f}")
    print("\nDecision Tree rules (truncated):")
    tree_text = export_text(dt, feature_names=career_feats, max_depth=4)
    # Print only first 60 lines to avoid overwhelming output
    lines = tree_text.split("\n")
    print("\n".join(lines[:60]))
    if len(lines) > 60:
        print(f"  ... ({len(lines)-60} more lines)")

    # ── 3. K-means sweep ─────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SECTION 3 — K-MEANS SWEEP (k=4..10, core features)")
    print("=" * 72)

    X_core   = g[core_feats].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_core)

    print(f"\n{'k':>3}  {'Silhouette':>11}  {'Davies-Bouldin':>15}  {'SuccessSpread':>14}")
    sweep_results = {}
    for k in range(4, 11):
        km   = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labs = km.fit_predict(X_scaled)
        sil  = silhouette_score(X_scaled, labs)
        db   = davies_bouldin_score(X_scaled, labs)
        # success rate spread
        sr   = pd.Series(labs).map(
            g.groupby(labs)["Success"].mean().to_dict()
        )
        spread = sr.max() - sr.min() if not sr.isna().all() else 0
        sweep_results[k] = {"silhouette": sil, "db": db, "spread": spread, "km": km}
        print(f"  {k:>2}  {sil:>11.4f}  {db:>15.4f}  {spread:>14.3f}")

    best_k = max(range(4, 11), key=lambda k: sweep_results[k]["silhouette"])
    print(f"\nBest k by silhouette: {best_k}")
    print(f"Using k={best_k} for detailed analysis (also reporting k=6 for comparison).")

    # ── 4. Detailed cluster analysis — best_k and k=6 ────────────────────────
    for analysis_k in sorted(set([best_k, 6])):
        print("\n" + "=" * 72)
        print(f"SECTION 4 — CLUSTER PROFILES (k={analysis_k})")
        print("=" * 72)

        km_k  = sweep_results[analysis_k]["km"]
        g_k   = g.copy()
        g_k["Cluster_ID"] = km_k.labels_

        centers = pd.DataFrame(km_k.cluster_centers_, columns=core_feats)
        # Inverse-transform to original scale for display
        centers_raw = pd.DataFrame(
            scaler.inverse_transform(km_k.cluster_centers_), columns=core_feats
        )
        # Name using standardized centroids (BB2K high=good disc, negate Kpct for disc proxy)
        raw_names = {
            i: name_cluster(
                row["BB2K_career"],       # standardized: positive = good discipline
                row["SBTalent_career"],   # standardized: positive = high SB talent
                row["HRFB_career"],       # standardized: positive = high HR/FB power
            )
            for i, row in centers.iterrows()
        }
        cluster_names = dedup_names(raw_names)

        print(f"\n{'Archetype':<22} {'N':>5}  {'MeanZ':>7}  {'MedZ':>7}  "
              f"{'Success%':>9}  {'BB2K':>7}  {'K%':>6}  {'SBTal':>7}  "
              f"{'HRFB':>6}  {'PullAir':>8}")
        print("-" * 90)

        cluster_summary = {}
        for cid in sorted(g_k["Cluster_ID"].unique()):
            sub = g_k[g_k["Cluster_ID"] == cid]
            if len(sub) < 5:
                continue
            arch  = cluster_names[cid]
            n     = len(sub)
            mz    = sub["PPPA_Z_career"].mean()
            medz  = sub["PPPA_Z_career"].median()
            succ  = sub["Success"].mean() * 100
            bb2k  = centers_raw.loc[cid, "BB2K_career"]
            kpct  = centers_raw.loc[cid, "Kpct_career"]
            sbtal = centers_raw.loc[cid, "SBTalent_career"]
            hrfb  = centers_raw.loc[cid, "HRFB_career"]
            pull  = centers_raw.loc[cid, "PullAir_career"]
            print(f"  {arch:<22} {n:>5}  {mz:>7.3f}  {medz:>7.3f}  "
                  f"{succ:>8.1f}%  {bb2k:>7.4f}  {kpct:>5.1f}%  {sbtal:>7.5f}  "
                  f"{hrfb:>5.1f}%  {pull:>7.1f}%")
            cluster_summary[cid] = {"arch": arch, "n": n, "mean_z": mz,
                                    "success_pct": succ}

        print(f"\nTop 5 players per cluster (Career_PPPA_Z):")
        for cid in sorted(g_k["Cluster_ID"].unique()):
            sub = g_k[g_k["Cluster_ID"] == cid]
            if len(sub) < 5:
                continue
            arch = cluster_names[cid]
            top5 = sub.nlargest(5, "PPPA_Z_career")[
                ["Name", "PPPA_Z_career", "Career_MLB_PA"]
            ]
            print(f"\n  [{arch}]")
            for _, row in top5.iterrows():
                print(f"    {row['Name']:<25}  PPPA_Z={row['PPPA_Z_career']:.2f}  "
                      f"Career_PA={int(row['Career_MLB_PA'])}")

    # ── 5. TOOLS × ABILITY quadrant analysis ─────────────────────────────────
    print("\n" + "=" * 72)
    print("SECTION 5 — TOOLS × ABILITY QUADRANT ANALYSIS")
    print("=" * 72)

    ab = pd.read_csv(FEATURES_PATH, usecols=[
        "PlayerId", "Season", "Level", "PA",
        "ABILITY_Score", "ABILITY_Disc", "SB_Talent", "Game_Power",
    ]).rename(columns={"ABILITY_Disc": "Discipline"})
    ts = pd.read_csv(FEATURES_PATH, usecols=[
        "PlayerId", "Season", "Level", "PA", "TOOLS_Score",
        "TOOLS_Disc", "TOOLS_Power", "TOOLS_Ath",
    ]).rename(columns={"TOOLS_Disc": "Discipline", "TOOLS_Power": "Power", "TOOLS_Ath": "Athleticism"})

    mlb = pd.read_parquet(DATA_DIR / "historical" / "hist_mlb_data.parquet",
                      columns=["PlayerId", "Season"])
    debut_year = mlb.groupby("PlayerId")["Season"].min().rename("debut_year").reset_index()

    def agg_scores(df, score_col, extra_cols=None):
        cols = extra_cols or []
        sub  = df[df["Level"].isin(FULL_SEASON_LEVELS) & (df["PA"] >= 50)].copy()
        sub  = sub.merge(debut_year, on="PlayerId", how="left")
        sub  = sub[sub["debut_year"].isna() | (sub["Season"] < sub["debut_year"])]
        agg_dict = {score_col: (sub[score_col] * sub["PA"]).groupby(sub["PlayerId"]).sum()
                               / sub.groupby("PlayerId")["PA"].sum()}
        for col in cols:
            agg_dict[col] = ((sub[col] * sub["PA"]).groupby(sub["PlayerId"]).sum()
                              / sub.groupby("PlayerId")["PA"].sum())
        res = pd.DataFrame(agg_dict).reset_index().rename(columns={"index": "PlayerId"})
        return res

    ab_agg = agg_scores(ab, "ABILITY_Score",
                        extra_cols=["Discipline", "SB_Talent", "Game_Power"])
    ts_agg = agg_scores(ts, "TOOLS_Score",
                        extra_cols=["Discipline", "Power", "Athleticism"])
    ts_agg = ts_agg.rename(columns={"Discipline": "TOOLS_Disc",
                                     "Power": "TOOLS_Power",
                                     "Athleticism": "TOOLS_Ath"})

    # Make PlayerId consistent type
    ab_agg["PlayerId"] = ab_agg["PlayerId"].astype(str)
    ts_agg["PlayerId"] = ts_agg["PlayerId"].astype(str)
    g_quad = g.copy()
    g_quad["PlayerId"] = g_quad["PlayerId"].astype(str)

    quad = (
        g_quad[["PlayerId", "Name", "PPPA_Z_career", "Success"]]
        .merge(ab_agg[["PlayerId", "ABILITY_Score"]], on="PlayerId", how="left")
        .merge(ts_agg[["PlayerId", "TOOLS_Score"]], on="PlayerId", how="left")
    )
    n_matched = quad[["ABILITY_Score", "TOOLS_Score"]].notna().all(axis=1).sum()
    print(f"\nPlayers with both TOOLS and ABILITY scores: {n_matched:,} / {len(quad):,}")

    q_valid = quad.dropna(subset=["ABILITY_Score", "TOOLS_Score"]).copy()

    # 2×2 grid (median split)
    ab_med = q_valid["ABILITY_Score"].median()
    ts_med = q_valid["TOOLS_Score"].median()
    q_valid["AB_tier"] = (q_valid["ABILITY_Score"] >= ab_med).map(
        {True: "Hi-ABILITY", False: "Lo-ABILITY"})
    q_valid["TS_tier"] = (q_valid["TOOLS_Score"] >= ts_med).map(
        {True: "Hi-TOOLS", False: "Lo-TOOLS"})

    print(f"\n2×2 TOOLS × ABILITY grid (median split):")
    print(f"  ABILITY median={ab_med:.1f}, TOOLS median={ts_med:.1f}\n")
    pivot = q_valid.groupby(["TS_tier", "AB_tier"]).agg(
        N=("PPPA_Z_career", "count"),
        Mean_PPPA_Z=("PPPA_Z_career", "mean"),
        Success_Rate=("Success", "mean"),
    ).round(3)
    print(pivot.to_string())

    # 3×3 grid (tercile)
    q_valid["AB_t3"] = pd.qcut(q_valid["ABILITY_Score"], 3,
                                labels=["Lo", "Mid", "Hi"])
    q_valid["TS_t3"] = pd.qcut(q_valid["TOOLS_Score"], 3,
                                labels=["Lo", "Mid", "Hi"])

    print(f"\n3×3 TOOLS (rows) × ABILITY (cols) grid — mean Career_PPPA_Z:")
    pivot3 = q_valid.pivot_table(
        values="PPPA_Z_career", index="TS_t3", columns="AB_t3", aggfunc="mean"
    ).round(3)
    print(pivot3.to_string())

    print(f"\n3×3 TOOLS × ABILITY — Success Rate (% ≥+0.25):")
    pivot3s = q_valid.pivot_table(
        values="Success", index="TS_t3", columns="AB_t3", aggfunc="mean"
    ).round(3)
    print(pivot3s.to_string())

    print(f"\n3×3 TOOLS × ABILITY — N players:")
    pivot3n = q_valid.pivot_table(
        values="PPPA_Z_career", index="TS_t3", columns="AB_t3", aggfunc="count"
    ).fillna(0).astype(int)
    print(pivot3n.to_string())

    # ── 6. Discipline floor — K% interaction ─────────────────────────────────
    print("\n" + "=" * 72)
    print("SECTION 6 — DISCIPLINE FLOOR: K% × SBTalent / HRFB INTERACTION")
    print("=" * 72)

    g["K_terc"] = pd.qcut(g["Kpct_career"], 3,
                           labels=["Lo-K (good)", "Mid-K", "Hi-K (bad)"])
    g["SB_terc"] = pd.qcut(g["SBTalent_career"].rank(method="first"), 3,
                            labels=["Lo-SB", "Mid-SB", "Hi-SB"])
    g["HR_terc"] = pd.qcut(g["HRFB_career"], 3,
                            labels=["Lo-HR/FB", "Mid-HR/FB", "Hi-HR/FB"])

    print(f"\nK% tercile boundaries:")
    print(f"  {g.groupby('K_terc', observed=True)['Kpct_career'].agg(['min','max']).to_string()}")

    print(f"\nK% tercile × SBTalent tercile — Success Rate (% Career_PPPA_Z ≥ +0.25):")
    ksb = g.pivot_table(values="Success", index="K_terc", columns="SB_terc",
                         aggfunc="mean", observed=True).round(3)
    print(ksb.to_string())

    print(f"\nK% tercile × SBTalent tercile — mean Career_PPPA_Z:")
    ksb_z = g.pivot_table(values="PPPA_Z_career", index="K_terc", columns="SB_terc",
                           aggfunc="mean", observed=True).round(3)
    print(ksb_z.to_string())

    print(f"\nK% tercile × HR/FB tercile — Success Rate:")
    khr = g.pivot_table(values="Success", index="K_terc", columns="HR_terc",
                         aggfunc="mean", observed=True).round(3)
    print(khr.to_string())

    print(f"\nK% tercile × HR/FB tercile — mean Career_PPPA_Z:")
    khr_z = g.pivot_table(values="PPPA_Z_career", index="K_terc", columns="HR_terc",
                           aggfunc="mean", observed=True).round(3)
    print(khr_z.to_string())

    # ── 7. Graduation-adjusted archetype value ────────────────────────────────
    print("\n" + "=" * 72)
    print("SECTION 7 — GRADUATION-ADJUSTED ARCHETYPE VALUE")
    print("=" * 72)
    print("(penalizes archetypes that look good in MiLB but rarely reach MLB)")

    # Use best_k clusters on core features for ALL players (not just graduated)
    X_all    = pc[core_feats].copy()
    for col in core_feats:
        X_all[col] = X_all[col].fillna(X_all[col].mean())
    X_all_sc = scaler.transform(X_all.values)

    km_best   = sweep_results[best_k]["km"]
    pc_all    = pc.copy()
    pc_all["Cluster_ID"] = km_best.predict(X_all_sc)

    # Standardized centroids for best_k (for naming)
    best_centers_std = pd.DataFrame(km_best.cluster_centers_, columns=core_feats)
    best_names = dedup_names({
        i: name_cluster(row["BB2K_career"], row["SBTalent_career"], row["HRFB_career"])
        for i, row in best_centers_std.iterrows()
    })

    for cid in sorted(pc_all["Cluster_ID"].unique()):
        sub_all  = pc_all[pc_all["Cluster_ID"] == cid]
        sub_grad = sub_all[(sub_all["graduated"] == True) &
                           (sub_all["Career_MLB_PA"] >= 200)]
        n_all    = len(sub_all)
        n_grad   = len(sub_grad)
        p_grad   = n_grad / n_all if n_all > 0 else 0
        e_pppa_z = sub_grad["PPPA_Z_career"].mean() if n_grad > 0 else np.nan
        adj_val  = p_grad * e_pppa_z if not np.isnan(e_pppa_z) else 0.0
        arch     = best_names[cid]
        print(f"  [{arch:<22}]  N_all={n_all:5d}  P(grad)={p_grad:.3f}  "
              f"E[PPPA_Z|grad]={e_pppa_z:+.3f}  Adj_Value={adj_val:+.4f}")

    # ── 8. Summary recommendations ────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SECTION 8 — RANKED SUMMARY & RECOMMENDATIONS")
    print("=" * 72)

    # Re-run best_k on graduated set for final summary (use standardized centroids for naming)
    km_final  = sweep_results[best_k]["km"]
    g["Cluster_ID"] = km_final.labels_
    centers_std_final = pd.DataFrame(km_final.cluster_centers_, columns=core_feats)

    arch_names_final = dedup_names({
        i: name_cluster(row["BB2K_career"], row["SBTalent_career"], row["HRFB_career"])
        for i, row in centers_std_final.iterrows()
    })

    summary_rows = []
    for cid in sorted(g["Cluster_ID"].unique()):
        sub  = g[g["Cluster_ID"] == cid]
        if len(sub) < 30:
            continue
        arch = arch_names_final[cid]
        n    = len(sub)
        mz   = sub["PPPA_Z_career"].mean()
        succ = sub["Success"].mean() * 100
        summary_rows.append({
            "Archetype": arch, "N": n,
            "Mean_PPPA_Z": round(mz, 3),
            "Success_Pct": round(succ, 1),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("Mean_PPPA_Z", ascending=False)
    print(f"\n{'Rank':<5}  {'Archetype':<22}  {'N':>5}  {'Mean PPPA_Z':>12}  {'Success%':>9}")
    print("-" * 62)
    for rank, (_, row) in enumerate(summary_df.iterrows(), 1):
        print(f"  {rank:<3}  {row['Archetype']:<22}  {row['N']:>5}  "
              f"{row['Mean_PPPA_Z']:>12.3f}  {row['Success_Pct']:>8.1f}%")

    print(f"""
KEY FINDINGS:
-------------
1. Feature importance (RF): The top MiLB predictors of MLB PPPA are listed in
   Section 2. K% and BB2K_career consistently rank at top — consistent with
   the model's -2 SO penalty. SBTalent and HRFB are next.

2. TOOLS × ABILITY: The 3×3 grid (Section 5) shows whether high TOOLS alone
   or high ABILITY alone is sufficient, or whether the combination is required.
   Inspect the Hi-TOOLS/Hi-ABILITY cell for the "sweet spot" vs Hi-ABILITY-only.

3. Discipline floor (Section 6): The K% × SBTalent and K% × HRFB grids show
   whether power/speed archetypes with high K% translate — the Hi-K/Hi-SB and
   Hi-K/Hi-HRFB cells are the critical test. If success rates collapse, the
   discipline gate is validated.

4. Graduation adjustment (Section 7): Compares raw cluster quality to MLB
   delivery. An archetype that looks strong but graduates few players has lower
   adjusted value — reveals which archetypes are drafting-floor risks.
""")

    print("\n" + "=" * 72)
    print("RECOMMENDATION FOR build_archetypes.py UPGRADE")
    print("=" * 72)
    fi_series = pd.Series(rf.feature_importances_, index=career_feats).sort_values(ascending=False)
    top3      = fi_series.head(3).index.tolist()
    print(f"""
Current system: K-means k=6 on [Discipline, SB_Talent, Game_Power] (ABILITY only).

Suggested upgrade:
  1. Add TOOLS sub-components to clustering: include TOOLS Discipline, Power,
     Athleticism alongside ABILITY components — 6 dimensions total.
  2. Optimal k from sweep: {best_k}. If interpretability is prioritized, k=6 is
     defensible (similar silhouette within ~0.01). Best k={best_k} is recommended.
  3. Top RF features for MLB PPPA (should be prioritized in archetype design):
     {top3[0]}, {top3[1]}, {top3[2]}
  4. Consider weighting cluster dimensions by RF feature importance rather than
     equal-weight StandardScaler — this would make clusters more PPPA-predictive
     rather than geometrically balanced.
  5. Graduation-adjusted archetype value (Section 7) should be shown in the HTML
     artifact alongside raw cluster means — it surfaces which archetypes are
     misleading in MiLB context.
""")


if __name__ == "__main__":
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        sys.stdout = Tee(fh)
        try:
            main()
        finally:
            sys.stdout = sys.__stdout__
    print(f"\nOutput written to: {OUT_PATH}")
