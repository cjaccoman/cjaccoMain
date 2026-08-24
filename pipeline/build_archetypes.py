"""Build player archetypes from PA-weighted ABILITY sub-components.

K-means (k=6) on [Discipline, SB_Talent, Game_Power] z-scores identifies
natural player profiles. Cluster labels are assigned by centroid signature.
Fantasy_Out is excluded from clustering (it's a composite of the three
type dimensions) and used as within-archetype validation only.

Pre-debut window (historical players with known MLB debut):
  ability sub-components from seasons BEFORE first MLB appearance,
  restricted to full-season levels (A/A+/AA/AAA), PA >= 50 per row.

Current prospects (no MLB debut):
  all available seasons at A/A+/AA/AAA.

Minimum 100 total qualifying PA required to assign an archetype.

Outputs:
  data/rankings/archetype_labels.csv  -- one row per player:
      PlayerId, Archetype, Cluster_ID, Cluster_PA,
      Discipline, SB_Talent, Game_Power, Fantasy_Out

Also prints calibration: OVR_Score residuals vs first-year MLB PPPA_Z
per archetype (requires >= 100 PA in debut MLB season).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA_DIR     = Path(__file__).resolve().parent.parent / "data"
ABILITY_PATH = DATA_DIR / "rankings" / "ability_scores.csv"
MLB_PATH     = DATA_DIR / "historical" / "hist_mlb_data.csv"
OVR_PATH     = DATA_DIR / "rankings" / "prospect_scores_ovr.csv"
OUT_PATH     = DATA_DIR / "rankings" / "archetype_labels.csv"

FULL_SEASON_LEVELS = {"A", "A+", "AA", "AAA"}
MIN_PA_ROW     = 50    # min PA per season-level row to contribute to cluster features
MIN_CLUSTER_PA = 100   # min total qualifying PA to assign an archetype
N_CLUSTERS     = 6
RANDOM_STATE   = 42
MLB_PA_MIN          = 100   # min first-year MLB PA for calibration
CAREER_MLB_PA_MIN   = 400   # min career MLB PA for career calibration
CAREER_MLB_SEAS_MIN = 2     # min MLB seasons for career calibration

CLUSTER_DIMS = ["Discipline", "SB_Talent", "Game_Power"]


def _name_cluster(disc: float, sb: float, power: float, threshold: float = 0.20) -> str:
    hi_disc  = disc  >  threshold
    lo_disc  = disc  < -threshold
    hi_sb    = sb    >  threshold
    hi_power = power >  threshold

    if hi_disc and hi_sb and not hi_power:       return "Contact/Speed"
    if hi_disc and not hi_sb and not hi_power:   return "Pure Contact"
    if hi_disc and hi_power:                     return "Contact/Power"
    if lo_disc and hi_sb:                        return "Speed/K-Risk"
    if lo_disc and hi_power:                     return "Power/K-Risk"
    if hi_sb:                                    return "Speed"
    return "Average"


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


def main() -> None:
    ability = pd.read_csv(
        ABILITY_PATH,
        usecols=["PlayerId", "Season", "Level", "PA",
                 "Fantasy_Out", "Discipline", "SB_Talent", "Game_Power"],
    )
    print(f"Loaded {len(ability):,} ability rows")

    mlb = pd.read_csv(MLB_PATH, usecols=["PlayerId", "Season", "PA", "PPPA_Z"])
    debut_year = mlb.groupby("PlayerId")["Season"].min().rename("debut_year")

    # Full-season levels only, minimum PA per row
    fs = ability[
        ability["Level"].isin(FULL_SEASON_LEVELS) & (ability["PA"] >= MIN_PA_ROW)
    ].copy()
    fs = fs.merge(debut_year.reset_index(), on="PlayerId", how="left")

    # Pre-debut window: seasons before first MLB appearance; all seasons for current prospects
    cluster_rows = fs[fs["debut_year"].isna() | (fs["Season"] < fs["debut_year"])].copy()
    print(f"  Full-season pre-debut rows: {len(cluster_rows):,}")

    # PA-weighted mean per player for each cluster dimension + Fantasy_Out
    agg = {}
    for col in CLUSTER_DIMS + ["Fantasy_Out"]:
        valid = cluster_rows.dropna(subset=[col])
        num   = (valid[col] * valid["PA"]).groupby(valid["PlayerId"]).sum()
        den   = valid.groupby("PlayerId")["PA"].sum()
        agg[col] = (num / den).rename(col)
    total_pa = cluster_rows.groupby("PlayerId")["PA"].sum().rename("Cluster_PA")

    features = pd.concat(list(agg.values()) + [total_pa], axis=1)
    features = features[features["Cluster_PA"] >= MIN_CLUSTER_PA].copy()
    print(f"  Players with >= {MIN_CLUSTER_PA} qualifying PA: {len(features):,}")

    # Separate historical (debut known) from current prospects
    hist_mask = features.index.isin(debut_year.index)
    hist_feat = features[hist_mask]
    curr_feat = features[~hist_mask]
    print(f"  Historical (debut known): {len(hist_feat):,}  Current: {len(curr_feat):,}")

    # Fit K-means on historical players (validated distribution)
    X_hist   = hist_feat[CLUSTER_DIMS].fillna(0).values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_hist)

    km          = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=20)
    hist_labels = km.fit_predict(X_scaled)

    # Name each cluster from its centroid signature
    centers      = pd.DataFrame(km.cluster_centers_, columns=CLUSTER_DIMS)
    raw_names    = {i: _name_cluster(row.Discipline, row.SB_Talent, row.Game_Power)
                    for i, row in centers.iterrows()}
    cluster_names = _dedup_names(raw_names)

    print("\nCluster centroids (standardized units):")
    print(f"  {'Archetype':<20} {'Disc':>6} {'SB':>6} {'Power':>6}  {'N':>5}")
    label_counts = pd.Series(hist_labels).value_counts()
    for i, row in centers.iterrows():
        n = label_counts.get(i, 0)
        print(f"  {cluster_names[i]:<20} {row.Discipline:+6.2f} {row.SB_Talent:+6.2f} {row.Game_Power:+6.2f}  {n:5d}")

    # Predict archetypes for current prospects
    X_curr      = curr_feat[CLUSTER_DIMS].fillna(0).values
    curr_labels = km.predict(scaler.transform(X_curr))

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
    out_cols = ["PlayerId", "Archetype", "Cluster_ID", "Cluster_PA",
                "Discipline", "SB_Talent", "Game_Power", "Fantasy_Out"]
    all_out[out_cols].to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(all_out):,} rows -> {OUT_PATH}")
    print(f"\nArchetype distribution:")
    print(all_out["Archetype"].value_counts().to_string())

    # ------------------------------------------------------------------
    # Calibration: OVR_Score residuals vs first-year MLB PPPA_Z
    # ------------------------------------------------------------------
    debut_mlb = mlb.merge(debut_year.reset_index(), on="PlayerId", how="left")
    debut_mlb = debut_mlb[
        (debut_mlb["Season"] == debut_mlb["debut_year"]) & (debut_mlb["PA"] >= MLB_PA_MIN)
    ][["PlayerId", "PPPA_Z"]].rename(columns={"PPPA_Z": "MLB_PPPA_Z"})

    ovr = pd.read_csv(OVR_PATH, usecols=["PlayerId", "Combined_Score"])
    ovr = ovr.rename(columns={"Combined_Score": "OVR_Score"})

    cal = (
        debut_mlb
        .merge(ovr,                               on="PlayerId", how="inner")
        .merge(all_out[["PlayerId", "Archetype"]], on="PlayerId", how="inner")
    )
    print(f"\nCalibration set: {len(cal):,} players (>= {MLB_PA_MIN} first-year MLB PA)")

    if len(cal) < 10:
        print("  Too few players for calibration.")
        return

    # Global linear fit: OVR_Score -> MLB_PPPA_Z (controls for overall model level)
    X = np.column_stack([np.ones(len(cal)), cal["OVR_Score"].values])
    y = cal["MLB_PPPA_Z"].values
    coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    cal["Expected_Z"] = coefs[0] + coefs[1] * cal["OVR_Score"]
    cal["Residual"]   = cal["MLB_PPPA_Z"] - cal["Expected_Z"]

    arch_cal = (
        cal.groupby("Archetype")
        .agg(
            N=("Residual", "count"),
            Mean_OVR=("OVR_Score", "mean"),
            Mean_MLB_Z=("MLB_PPPA_Z", "mean"),
            Mean_Residual=("Residual", "mean"),
        )
        .round(3)
        .sort_values("Mean_Residual", ascending=False)
    )
    print("\nCalibration residuals by archetype (positive = model under-scores vs actual MLB):")
    print(arch_cal.to_string())

    corr_by_arch = (
        cal.groupby("Archetype")
        .apply(
            lambda g: g["OVR_Score"].corr(g["MLB_PPPA_Z"]) if len(g) >= 10 else np.nan,
            include_groups=False,
        )
        .rename("Within_Arch_r")
        .round(3)
    )
    print("\nWithin-archetype OVR_Score <-> MLB_PPPA_Z correlation (r):")
    print(corr_by_arch.to_string())

    # ------------------------------------------------------------------
    # Career calibration: OVR_Score residuals vs PA-weighted career PPPA_Z
    # Matches milb_impact_analysis.py standard: >= 400 career MLB PA, >= 2 seasons
    # Uses PA-weighted career PPPA_Z (averaging per-season z-scores) to control
    # for era drift across multi-season careers.
    # ------------------------------------------------------------------
    mlb_full = pd.read_csv(MLB_PATH, usecols=["PlayerId", "Season", "PA", "PPPA_Z"])

    career_stats = (
        mlb_full.groupby("PlayerId")
        .apply(
            lambda g: pd.Series({
                "Career_MLB_PA":   g["PA"].sum(),
                "MLB_Seasons":     g["Season"].nunique(),
                "Career_PPPA_Z":   (g["PPPA_Z"] * g["PA"]).sum() / g["PA"].sum(),
            }),
            include_groups=False,
        )
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
    print(f"\n--- Career calibration ---")
    print(f"N = {len(cal_career):,} players (>= {CAREER_MLB_PA_MIN} career MLB PA, >= {CAREER_MLB_SEAS_MIN} seasons)")

    if len(cal_career) >= 10:
        X_c = np.column_stack([np.ones(len(cal_career)), cal_career["OVR_Score"].values])
        y_c = cal_career["Career_PPPA_Z"].values
        coefs_c, _, _, _ = np.linalg.lstsq(X_c, y_c, rcond=None)
        cal_career["Expected_Z"] = coefs_c[0] + coefs_c[1] * cal_career["OVR_Score"]
        cal_career["Residual"]   = cal_career["Career_PPPA_Z"] - cal_career["Expected_Z"]

        arch_career = (
            cal_career.groupby("Archetype")
            .agg(
                N=("Residual", "count"),
                Mean_OVR=("OVR_Score", "mean"),
                Mean_Career_PA=("Career_MLB_PA", "mean"),
                Mean_Career_Z=("Career_PPPA_Z", "mean"),
                Mean_Residual=("Residual", "mean"),
            )
            .round(3)
            .sort_values("Mean_Residual", ascending=False)
        )
        print("\nCareer calibration residuals (positive = model under-scores vs career MLB output):")
        print(arch_career.to_string())

        corr_career = (
            cal_career.groupby("Archetype")
            .apply(
                lambda g: g["OVR_Score"].corr(g["Career_PPPA_Z"]) if len(g) >= 10 else np.nan,
                include_groups=False,
            )
            .rename("Career_r")
            .round(3)
        )
        print("\nWithin-archetype OVR_Score <-> Career_PPPA_Z correlation (r):")
        print(corr_career.to_string())

        # Side-by-side comparison of first-year vs career residuals
        first_res = arch_cal["Mean_Residual"].rename("First_Year_Resid")
        career_res = arch_career["Mean_Residual"].rename("Career_Resid")
        comparison = pd.concat([first_res, career_res], axis=1).round(3)
        comparison["Delta"] = (comparison["Career_Resid"] - comparison["First_Year_Resid"]).round(3)
        print("\nFirst-year vs career residual comparison (Delta = career - first-year):")
        print(comparison.sort_values("Career_Resid", ascending=False).to_string())


if __name__ == "__main__":
    main()
