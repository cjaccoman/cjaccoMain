"""Build player archetypes from PA-weighted TOOLS + ABILITY sub-components.

K-means (k=6) on 7 dimensions — 3 TOOLS (Discipline/Power/Athleticism) and
4 ABILITY (BB-rate / K-rate separate, SB_Talent, Game_Power) — identifies
natural player profiles.  BB% and K% are kept as independent dimensions
(rather than combined into BB_2K) so Three-True-Outcomes profiles (high BB +
high K + power) form their own cluster instead of collapsing into "Average".
Fantasy_Out is excluded from clustering (it's a composite of everything) and
used as within-archetype validation only.

Dimensions are StandardScaler-normalized then re-weighted by RF feature importances
trained on the historical graduated sample (Career_PPPA_Z outcome), so the
clustering optimises toward dimensions that are actually predictive of fantasy
production — not just geometrically balanced.

Pre-debut window (historical players with known MLB debut):
  sub-components from seasons BEFORE first MLB appearance,
  restricted to full-season levels (A/A+/AA/AAA), PA >= 50 per row.

Current prospects (no MLB debut):
  all available seasons at A/A+/AA/AAA.

Minimum 100 total qualifying PA required to assign an archetype.

Outputs:
  data/rankings/archetype_labels.csv  -- one row per player:
      PlayerId, Archetype, Cluster_ID, Cluster_PA,
      TOOLS_Disc, TOOLS_Power, TOOLS_Ath,
      AB_BB, AB_K, AB_SB, AB_Power, Fantasy_Out
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

DATA_DIR      = Path(__file__).resolve().parent.parent / "data"
FEATURES_PATH = DATA_DIR / "rankings" / "prospect_features.csv"
MLB_PATH      = DATA_DIR / "historical" / "hist_mlb_data.csv"
OVR_PATH      = DATA_DIR / "rankings" / "prospect_scores_ovr.csv"
OUT_PATH      = DATA_DIR / "rankings" / "archetype_labels.csv"

FULL_SEASON_LEVELS = {"A", "A+", "AA", "AAA"}
MIN_PA_ROW          = 50    # min PA per season-level row
MIN_CLUSTER_PA      = 100   # min total qualifying PA to assign an archetype
N_CLUSTERS          = 6     # k=6: enough resolution to capture TTO + richer profiles
RANDOM_STATE        = 42
MLB_PA_MIN          = 100   # min first-year MLB PA for calibration
CAREER_MLB_PA_MIN   = 400
CAREER_MLB_SEAS_MIN = 2

# 7 cluster dimensions: 3 TOOLS + 4 ABILITY
# BB% and K% are separate dims (not combined as BB_2K) so TTO profiles are visible.
CLUSTER_DIMS = [
    "TOOLS_Disc", "TOOLS_Power", "TOOLS_Ath",
    "AB_BB",  "AB_K",  "AB_SB",  "AB_Power",
]


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

def _name_cluster(
    t_disc:  float, t_power: float, t_ath: float,
    ab_bb:   float, ab_k:    float,
    ab_sb:   float, ab_pow:  float,
    thr: float = 0.28,
) -> str:
    """Name a cluster from its centroid coordinates in standardised space.

    ab_bb = BB% z-score (positive = more walks)
    ab_k  = K%  z-score (positive = more strikeouts — not inverted)
    """
    hi_t_disc = t_disc  >  thr
    lo_t_disc = t_disc  < -thr
    hi_t_pow  = t_power >  thr
    hi_t_ath  = t_ath   >  thr
    hi_bb     = ab_bb   >  thr    # high walk rate
    lo_bb     = ab_bb   < -thr    # low walk rate
    hi_k      = ab_k    >  thr    # high strikeout rate
    lo_k      = ab_k    < -thr    # low strikeout rate
    hi_ab_sb  = ab_sb   >  thr
    hi_ab_pow = ab_pow  >  thr

    # TTO: high BB + high K + power (BB and K both elevated but cancel in BB_2K)
    is_tto    = hi_bb and hi_k and (hi_t_pow or hi_ab_pow)
    # Disciplined: good tools discipline, OR high walks + low Ks
    good_disc = (hi_t_disc and lo_k) or hi_bb
    # True K-risk: low walks AND high Ks AND poor tools discipline
    bad_disc  = lo_bb and hi_k and lo_t_disc
    # Moderate K-risk: high Ks without walk compensation, even if tools OK
    k_risk    = hi_k and not hi_bb
    has_power = hi_t_pow  or hi_ab_pow
    has_speed = hi_t_ath  or hi_ab_sb

    if is_tto:                                       return "Three True Outcomes"
    if good_disc and has_speed and has_power:         return "All-Around"
    if good_disc and has_speed:                      return "Contact/Speed"
    if good_disc and has_power:                      return "Contact/Power"
    if good_disc:                                    return "Pure Contact"
    if bad_disc  and has_power and has_speed:         return "Power-Speed/K-Risk"
    if bad_disc  and has_power:                      return "Power/K-Risk"
    if bad_disc  and has_speed:                      return "Speed/K-Risk"
    if k_risk   and has_power:                       return "Power/K-Risk"
    if k_risk   and has_speed:                       return "Speed/K-Risk"
    if has_speed:                                    return "Speed"
    if has_power:                                    return "Raw Power"
    return "Average"


def _z_within_sl(df: pd.DataFrame, col: str,
                  min_pa: int = 50, min_rows: int = 10) -> pd.Series:
    """Z-score col within Season+Level; Level-only fallback for sparse cells.
    Only rows with PA >= min_pa and non-null col contribute to group params.
    """
    out   = pd.Series(np.nan, index=df.index, dtype=float)
    valid = df[df["PA"] >= min_pa].dropna(subset=[col])

    grp_sl = valid.groupby(["Season", "Level"], observed=True)[col].agg(
        ["mean", "std", "count"]
    )
    grp_l = valid.groupby("Level", observed=True)[col].agg(["mean", "std"])

    for (season, level), idx in df.groupby(["Season", "Level"], observed=True).groups.items():
        mu, sig = None, None
        try:
            row = grp_sl.loc[(season, level)]
            if row["count"] >= min_rows and row["std"] > 1e-9:
                mu, sig = row["mean"], row["std"]
        except KeyError:
            pass
        if mu is None:
            try:
                fb = grp_l.loc[level]
                if fb["std"] > 1e-9:
                    mu, sig = fb["mean"], fb["std"]
            except KeyError:
                continue
        if mu is not None and sig is not None:
            out.loc[idx] = (df.loc[idx, col] - mu) / sig

    return out.round(4)


def _dedup_names(names: dict) -> dict:
    from collections import Counter
    counts = Counter(names.values())
    seen   = {}
    result = {}
    for k in sorted(names):
        v = names[k]
        if counts[v] > 1:
            seen[v] = seen.get(v, 0) + 1
            result[k] = f"{v}_{seen[v]}"
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Load all per-row scores from the consolidated prospect_features.csv
    pf = pd.read_csv(
        FEATURES_PATH,
        usecols=["PlayerId", "Season", "Level", "PA",
                 "TOOLS_Disc", "TOOLS_Power", "TOOLS_Ath",
                 "SB_Talent", "Game_Power", "Fantasy_Out",
                 "BB%", "K%"],
    )
    # BB% and K% as separate dimensions (not combined as BB_2K).
    # Z-scored within Season+Level so era drift doesn't contaminate clustering.
    pf["BB%_z"] = _z_within_sl(pf, "BB%")
    pf["K%_z"]  = _z_within_sl(pf, "K%")

    rows = pf.rename(columns={
        "SB_Talent":  "AB_SB",
        "Game_Power": "AB_Power",
        "BB%_z":      "AB_BB",
        "K%_z":       "AB_K",
    })
    print(f"Loaded {len(rows):,} joined rows")
    print(f"  TOOLS coverage:  {rows['TOOLS_Disc'].notna().sum():,} / {len(rows):,} rows")
    print(f"  AB_BB coverage:  {rows['AB_BB'].notna().sum():,} / {len(rows):,} rows")
    print(f"  AB_K  coverage:  {rows['AB_K'].notna().sum():,} / {len(rows):,} rows")

    mlb = pd.read_csv(MLB_PATH, usecols=["PlayerId", "Season", "PA", "PPPA_Z"])
    debut_year = mlb.groupby("PlayerId")["Season"].min().rename("debut_year")

    # Full-season levels, minimum PA per row
    fs = rows[
        rows["Level"].isin(FULL_SEASON_LEVELS) & (rows["PA"] >= MIN_PA_ROW)
    ].copy()
    fs = fs.merge(debut_year.reset_index(), on="PlayerId", how="left")

    # Pre-debut window
    cluster_rows = fs[fs["debut_year"].isna() | (fs["Season"] < fs["debut_year"])].copy()
    print(f"  Full-season pre-debut rows: {len(cluster_rows):,}")

    # PA-weighted career average per player for each dimension
    all_dims = CLUSTER_DIMS + ["Fantasy_Out"]
    agg = {}
    for col in all_dims:
        valid = cluster_rows.dropna(subset=[col])
        num   = (valid[col] * valid["PA"]).groupby(valid["PlayerId"]).sum()
        den   = valid.groupby("PlayerId")["PA"].sum()
        agg[col] = (num / den).rename(col)
    total_pa = cluster_rows.groupby("PlayerId")["PA"].sum().rename("Cluster_PA")

    features = pd.concat(list(agg.values()) + [total_pa], axis=1)
    features = features[features["Cluster_PA"] >= MIN_CLUSTER_PA].copy()
    print(f"  Players with >= {MIN_CLUSTER_PA} qualifying PA: {len(features):,}")

    # Split historical vs current
    hist_mask = features.index.isin(debut_year.index)
    hist_feat = features[hist_mask]
    curr_feat = features[~hist_mask]
    print(f"  Historical: {len(hist_feat):,}  Current: {len(curr_feat):,}")

    # ------------------------------------------------------------------
    # RF-importance weighting on historical graduated sample
    # ------------------------------------------------------------------
    mlb_career = (
        mlb.groupby("PlayerId")
        .apply(lambda g: pd.Series({
            "Career_PPPA_Z": (g["PPPA_Z"] * g["PA"]).sum() / g["PA"].sum(),
            "Career_PA":     g["PA"].sum(),
        }), include_groups=False)
        .reset_index()
    )
    mlb_career = mlb_career[mlb_career["Career_PA"] >= 200]

    rf_data = hist_feat[CLUSTER_DIMS].join(
        mlb_career.set_index("PlayerId")["Career_PPPA_Z"]
    ).dropna()
    print(f"\n  RF training set (historical + MLB outcome): {len(rf_data):,} players")

    importances = np.ones(len(CLUSTER_DIMS))   # uniform fallback
    if len(rf_data) >= 50:
        X_rf = rf_data[CLUSTER_DIMS].fillna(0).values
        y_rf = rf_data["Career_PPPA_Z"].values
        rf = RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X_rf, y_rf)
        importances = rf.feature_importances_
        # Normalise so average weight = 1.0 (preserves scale of StandardScaler)
        importances = importances / importances.mean()
        print("  RF feature importances (normalised, mean=1.0):")
        for dim, imp in zip(CLUSTER_DIMS, importances):
            bar = "#" * int(imp * 10)
            print(f"    {dim:<15} {imp:.3f}  {bar}")

    # ------------------------------------------------------------------
    # K-means on historical players
    # ------------------------------------------------------------------
    X_hist   = hist_feat[CLUSTER_DIMS].fillna(0).values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_hist) * importances   # importance weighting

    km          = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=20)
    hist_labels = km.fit_predict(X_scaled)

    # Back-transform centroids to original standardised space for naming
    centers_scaled = pd.DataFrame(km.cluster_centers_ / importances, columns=CLUSTER_DIMS)
    raw_names = {
        i: _name_cluster(
            row.TOOLS_Disc, row.TOOLS_Power, row.TOOLS_Ath,
            row.AB_BB,      row.AB_K,
            row.AB_SB,      row.AB_Power,
        )
        for i, row in centers_scaled.iterrows()
    }
    cluster_names = _dedup_names(raw_names)

    print(f"\nCluster centroids (standardised, importance-weighted space):")
    label_counts = pd.Series(hist_labels).value_counts()
    hdr = (f"  {'Archetype':<24} {'TDisc':>6} {'TPow':>6} {'TAth':>6}"
           f"  {'ABB':>6} {'AK':>6} {'ASB':>6} {'APow':>6}  {'N':>5}")
    print(hdr)
    for i, row in centers_scaled.iterrows():
        n = label_counts.get(i, 0)
        print(
            f"  {cluster_names[i]:<24}"
            f" {row.TOOLS_Disc:+6.2f} {row.TOOLS_Power:+6.2f} {row.TOOLS_Ath:+6.2f}"
            f"  {row.AB_BB:+6.2f} {row.AB_K:+6.2f} {row.AB_SB:+6.2f} {row.AB_Power:+6.2f}"
            f"  {n:5d}"
        )

    # Predict current prospects
    X_curr      = curr_feat[CLUSTER_DIMS].fillna(0).values
    curr_labels = km.predict(scaler.transform(X_curr) * importances)

    # Assemble output
    hist_out = hist_feat.copy()
    hist_out["Cluster_ID"] = hist_labels
    hist_out["Archetype"]  = [cluster_names[l] for l in hist_labels]

    curr_out = curr_feat.copy()
    curr_out["Cluster_ID"] = curr_labels
    curr_out["Archetype"]  = [cluster_names[l] for l in curr_labels]

    all_out = (
        pd.concat([hist_out, curr_out])
        .rename_axis("PlayerId")
        .reset_index()
    )

    # Post-clustering individual-level overrides.
    # K-means centroids sometimes misclassify extreme players because distance to
    # the nearest centroid is dominated by one dimension, ignoring offsetting signals.

    # TTO override: high BB + high K + positive power.
    # Catches players whose elevated BB% and K% cancel in BB_2K, making them look
    # "average" in discipline when they're actually a specific high-variance profile.
    tto_mask = (
        (all_out["AB_BB"]    > 0.40) &   # above-average walk rate
        (all_out["AB_K"]     > 0.40) &   # above-average strikeout rate
        (all_out["AB_Power"] > 0.00)      # any positive demonstrated power
    )
    n_tto = tto_mask.sum()
    all_out.loc[tto_mask, "Archetype"] = "Three True Outcomes"
    print(f"  TTO: {n_tto:,} players relabeled as 'Three True Outcomes'")

    # Contact/Power rescue: players pulled into Power/K-Risk by extreme power
    # but who actually have good plate discipline (high BB%, normal-or-low K%).
    # AB_K < 0.40 means their K% is not elevated relative to peers — they don't
    # belong in a K-risk cluster regardless of how high their power is.
    cp_rescue_mask = (
        (all_out["Archetype"] == "Power/K-Risk") &
        (all_out["AB_BB"] > 0.50) &    # genuinely good walk rate
        (all_out["AB_K"]  < 0.40)      # K% near or below average — not K-risk
    )
    n_cp = cp_rescue_mask.sum()
    all_out.loc[cp_rescue_mask, "Archetype"] = "Contact/Power"
    print(f"  Contact/Power rescue: {n_cp:,} Power/K-Risk players moved "
          f"(high BB, non-elevated K)")

    print()

    out_cols = ["PlayerId", "Archetype", "Cluster_ID", "Cluster_PA",
                "TOOLS_Disc", "TOOLS_Power", "TOOLS_Ath",
                "AB_BB", "AB_K", "AB_SB", "AB_Power", "Fantasy_Out"]
    all_out[out_cols].to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(all_out):,} rows -> {OUT_PATH}")
    print(f"\nArchetype distribution:")
    print(all_out["Archetype"].value_counts().to_string())

    # ------------------------------------------------------------------
    # Calibration: first-year and career residuals vs OVR_Score
    # ------------------------------------------------------------------
    debut_mlb = mlb.merge(debut_year.reset_index(), on="PlayerId", how="left")
    debut_mlb = debut_mlb[
        (debut_mlb["Season"] == debut_mlb["debut_year"]) & (debut_mlb["PA"] >= MLB_PA_MIN)
    ][["PlayerId", "PPPA_Z"]].rename(columns={"PPPA_Z": "MLB_PPPA_Z"})

    ovr = pd.read_csv(OVR_PATH, usecols=["PlayerId", "Combined_Score"])
    ovr = ovr.rename(columns={"Combined_Score": "OVR_Score"})

    cal = (
        debut_mlb
        .merge(ovr,                                on="PlayerId", how="inner")
        .merge(all_out[["PlayerId", "Archetype"]], on="PlayerId", how="inner")
    )
    print(f"\nCalibration set: {len(cal):,} players (>= {MLB_PA_MIN} first-year MLB PA)")

    if len(cal) >= 10:
        X = np.column_stack([np.ones(len(cal)), cal["OVR_Score"].values])
        y = cal["MLB_PPPA_Z"].values
        coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        cal["Expected_Z"] = coefs[0] + coefs[1] * cal["OVR_Score"]
        cal["Residual"]   = cal["MLB_PPPA_Z"] - cal["Expected_Z"]

        arch_cal = (
            cal.groupby("Archetype")
            .agg(N=("Residual","count"), Mean_OVR=("OVR_Score","mean"),
                 Mean_MLB_Z=("MLB_PPPA_Z","mean"), Mean_Residual=("Residual","mean"))
            .round(3).sort_values("Mean_Residual", ascending=False)
        )
        print("\nFirst-year calibration residuals (positive = model under-scores):")
        print(arch_cal.to_string())

        corr_by_arch = (
            cal.groupby("Archetype")
            .apply(lambda g: g["OVR_Score"].corr(g["MLB_PPPA_Z"]) if len(g) >= 10 else np.nan,
                   include_groups=False)
            .rename("First_Year_r").round(3)
        )
        print("\nWithin-archetype OVR_Score <-> first-year MLB_PPPA_Z (r):")
        print(corr_by_arch.to_string())

    # Career calibration
    mlb_full = pd.read_csv(MLB_PATH, usecols=["PlayerId", "Season", "PA", "PPPA_Z"])
    career_stats = (
        mlb_full.groupby("PlayerId")
        .apply(lambda g: pd.Series({
            "Career_MLB_PA":  g["PA"].sum(),
            "MLB_Seasons":    g["Season"].nunique(),
            "Career_PPPA_Z":  (g["PPPA_Z"] * g["PA"]).sum() / g["PA"].sum(),
        }), include_groups=False)
        .reset_index()
    )
    career_qual = career_stats[
        (career_stats["Career_MLB_PA"] >= CAREER_MLB_PA_MIN) &
        (career_stats["MLB_Seasons"]   >= CAREER_MLB_SEAS_MIN)
    ][["PlayerId", "Career_PPPA_Z", "Career_MLB_PA", "MLB_Seasons"]]

    cal_career = (
        career_qual
        .merge(ovr,                                on="PlayerId", how="inner")
        .merge(all_out[["PlayerId", "Archetype"]], on="PlayerId", how="inner")
    )
    print(f"\n--- Career calibration (>= {CAREER_MLB_PA_MIN} MLB PA, >= {CAREER_MLB_SEAS_MIN} seasons) ---")
    print(f"N = {len(cal_career):,}")

    if len(cal_career) >= 10:
        X_c = np.column_stack([np.ones(len(cal_career)), cal_career["OVR_Score"].values])
        y_c = cal_career["Career_PPPA_Z"].values
        coefs_c, _, _, _ = np.linalg.lstsq(X_c, y_c, rcond=None)
        cal_career["Expected_Z"] = coefs_c[0] + coefs_c[1] * cal_career["OVR_Score"]
        cal_career["Residual"]   = cal_career["Career_PPPA_Z"] - cal_career["Expected_Z"]

        arch_career = (
            cal_career.groupby("Archetype")
            .agg(N=("Residual","count"), Mean_OVR=("OVR_Score","mean"),
                 Mean_Career_PA=("Career_MLB_PA","mean"),
                 Mean_Career_Z=("Career_PPPA_Z","mean"),
                 Mean_Residual=("Residual","mean"))
            .round(3).sort_values("Mean_Residual", ascending=False)
        )
        print("\nCareer calibration residuals (positive = model under-scores):")
        print(arch_career.to_string())

        corr_career = (
            cal_career.groupby("Archetype")
            .apply(lambda g: g["OVR_Score"].corr(g["Career_PPPA_Z"]) if len(g) >= 10 else np.nan,
                   include_groups=False)
            .rename("Career_r").round(3)
        )
        print("\nWithin-archetype OVR_Score <-> Career_PPPA_Z (r):")
        print(corr_career.to_string())

        first_res  = arch_cal["Mean_Residual"].rename("First_Year_Resid")
        career_res = arch_career["Mean_Residual"].rename("Career_Resid")
        comparison = pd.concat([first_res, career_res], axis=1).round(3)
        comparison["Delta"] = (comparison["Career_Resid"] - comparison["First_Year_Resid"]).round(3)
        print("\nFirst-year vs career residual comparison:")
        print(comparison.sort_values("Career_Resid", ascending=False).to_string())


if __name__ == "__main__":
    main()
