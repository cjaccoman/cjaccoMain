"""Build historical player comparison dataset and prospect comparator tool.

Dataset (data/computed/player_comps.csv):
  One row per player. For each MiLB level (AAA, AA, A+, A, R):
    PPPA_Z_{level}  -- PA-weighted mean PPPA_Z_SL across all seasons at that level
    Age_{level}     -- PA-weighted mean age at that level
    PA_{level}      -- total PA at that level

  MLB outcomes:
    graduated       -- reached 100+ PA in MLB
    FirstYr_PPPA_Z  -- PPPA_Z in first MLB season with PA >= 100
    Career_PPPA_Z   -- PA-weighted career MLB PPPA_Z (seasons PA >= 50)
    Career_MLB_PA   -- total career MLB PA (seasons PA >= 50)
    MLB_Season      -- year of MLB debut

  ID columns: MLBAM_ID, PlayerId (FanGraphs), Name

Comparator:
  find_comps(player_name_or_id, n=10, min_shared_levels=1)

  For each candidate in the historical pool, compute a weighted Euclidean
  distance over shared MiLB levels:
    - Primary features: PPPA_Z (weight 1.0) and Age (weight 0.5), both
      z-scored within level across the pool
    - Secondary skill signals (see SKILL_WEIGHTS): BB_2K, Whiff%, HRFB,
      PullAir%, SBTalent — each z-scored within level; only included when
      both players have data at that level. Combined weight ≈ 0.53.
    - Level weight: LEVEL_DISCOUNT (AAA=1.00 → R=0.10) — higher levels
      count more toward similarity
    - PA weight: min(PA_query, PA_candidate) / PA_THRESHOLD, capped at 1.
      More data = more reliable comp
    - Distance only computed on levels where both players have PA >= MIN_COMP_PA

  Returns top-N comps with their MLB outcomes, so you can see what similar
  profiles historically produced.

Usage:
  python analysis/build_player_comps.py            # build dataset + print sample comps
  python analysis/build_player_comps.py --name "Max Clark"
  python analysis/build_player_comps.py --name "Josue De Paula" --n 15
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

DATA  = Path(__file__).resolve().parent.parent / "data"
OUT   = DATA / "computed" / "player_comps.csv"

LEVELS         = ["AAA", "AA", "A+", "A", "R"]
LEVEL_DISCOUNT = {"AAA": 1.00, "AA": 0.59, "A+": 0.34, "A": 0.23, "R": 0.10}
MIN_COMP_PA    = 80    # minimum PA at a level to count in distance calculation
PA_THRESHOLD   = 300   # PA at which level gets full weight in distance
AGE_WEIGHT     = 0.5   # relative weight of age vs. PPPA_Z in distance per level

# Secondary skill signals added to distance. Weights are relative to PPPA_Z=1.0.
# Combined skill contribution ≈ 0.53 vs. PPPA_Z=1.0, Age=0.5 → skills ≈ 26% of signal.
SKILL_WEIGHTS = {
    "BB2K":     0.15,  # BB% − 2×K% — single strongest discipline signal
    "Whiff":    0.10,  # Whiff% — contact quality complement to BB2K
    "HRFB":     0.10,  # HR/FB — raw power production
    "PullAir":  0.08,  # pull-air batted ball profile — power shape
    "SBTalent": 0.10,  # SB_pct × SB/PA — speed/baserunning talent
    "MaxEV":    0.08,  # max exit velocity — raw power ceiling (sparse: AAA/A 2023+, R 2026)
    "EV90":     0.08,  # 90th-pct EV — consistent hard contact (same coverage as MaxEV)
}


# ---------------------------------------------------------------------------
# Build dataset
# ---------------------------------------------------------------------------

def build_dataset() -> pd.DataFrame:
    ovr  = pd.read_csv(DATA / "historical" / "ovr_hist_data.csv")
    milb = pd.read_csv(DATA / "api" / "milb_hitting.csv")
    mlb  = pd.read_csv(DATA / "historical" / "hist_mlb_data.csv")

    # ID bridge: ovr.PlayerId (FG) -> milb.PlayerId -> milb.MLBAM_ID -> mlb.MLBAMID
    id_bridge = milb[["PlayerId", "MLBAM_ID", "Name"]].drop_duplicates("PlayerId")

    # Aggregate MiLB per player per level — PPPA_Z and Age from ovr_hist_data;
    # BB_2K, Whiff%, SB_talent, HR/FB from prospect_features (has MLBAM_ID directly)
    pf = pd.read_csv(DATA / "rankings" / "prospect_features.csv")

    # PA-weighted aggregator helper
    SKILL_COLS = ["BB_2K", "Whiff%", "SB_pct", "HR/FB"]

    def pa_weighted_agg(src, extra_cols):
        src = src[src["PA"] >= 10].copy()
        for col in extra_cols:
            src[f"PA_x_{col}"] = src["PA"] * src[col]           # NaN where col is NaN
            src[f"PAn_{col}"]  = src["PA"].where(src[col].notna(), 0)  # 0 where col is NaN
        agg_dict = {"PA": "sum", "Name": "first"}
        for c in extra_cols:
            agg_dict[f"PA_x_{c}"] = "sum"
            agg_dict[f"PAn_{c}"]  = "sum"
        grp = src.groupby(["PlayerId", "Level"], observed=True).agg(agg_dict).reset_index()
        grp.rename(columns={"PA": "PA_total"}, inplace=True)
        for col in extra_cols:
            valid_pa = grp[f"PAn_{col}"]
            # NaN when no non-null values exist in this group; otherwise weighted avg
            grp[f"{col}_wt"] = (grp[f"PA_x_{col}"] / valid_pa).where(valid_pa > 0)
            grp.drop(columns=[f"PAn_{col}"], inplace=True)
        return grp

    ovr_valid = ovr[ovr["PA"] >= 10].copy()
    ovr_valid["PA_x_PPPA_Z"] = ovr_valid["PA"] * ovr_valid["PPPA_Z_SL"]
    ovr_valid["PA_x_Age"]    = ovr_valid["PA"] * ovr_valid["Age"]
    level_agg = (
        ovr_valid.groupby(["PlayerId", "Level"], observed=True)
        .agg(PA_total=("PA","sum"), PA_x_PPPA_Z=("PA_x_PPPA_Z","sum"),
             PA_x_Age=("PA_x_Age","sum"), Name=("Name","first"))
        .reset_index()
    )
    level_agg["PPPA_Z_wt"] = level_agg["PA_x_PPPA_Z"] / level_agg["PA_total"]
    level_agg["Age_wt"]    = level_agg["PA_x_Age"]    / level_agg["PA_total"]

    # Skill metrics from prospect_features (keyed on MLBAM_ID)
    pf_valid = pf[pf["PA"] >= 10].copy()
    # SB_talent raw = SB_pct × SB/PA (matches build_ability_score.py)
    pf_valid["SB_pct"] = pf_valid["SB_pct"].fillna(0)
    pf_valid["SB_talent"] = pf_valid["SB_pct"] * (pf_valid["SB"] / pf_valid["PA"])
    pf_skill_cols = ["BB_2K", "Whiff%", "SB_talent", "HR/FB", "PullAir%", "K%", "BB%", "MaxEV", "EV90"]
    pf_agg = pa_weighted_agg(pf_valid, pf_skill_cols)
    # Bridge pf PlayerId → MLBAM_ID for later join; also keep for merge with ovr
    pf_bridge = pf[["PlayerId", "MLBAM_ID"]].drop_duplicates("PlayerId")

    def make_wide(src_agg, val_col, prefix):
        w = (src_agg[src_agg["Level"].isin(LEVELS)]
             .pivot(index="PlayerId", columns="Level", values=val_col))
        w.columns = [f"{prefix}_{c.replace('+','plus')}" for c in w.columns]
        return w

    pppa_wide = make_wide(level_agg, "PPPA_Z_wt", "PPPA_Z")
    age_wide  = make_wide(level_agg, "Age_wt",    "Age")
    pa_wide   = make_wide(level_agg, "PA_total",  "PA")
    bb2k_wide     = make_wide(pf_agg, "BB_2K_wt",      "BB2K")
    whiff_wide    = make_wide(pf_agg, "Whiff%_wt",     "Whiff")
    sbt_wide      = make_wide(pf_agg, "SB_talent_wt",  "SBTalent")
    hrfb_wide     = make_wide(pf_agg, "HR/FB_wt",      "HRFB")
    pullair_wide  = make_wide(pf_agg, "PullAir%_wt",   "PullAir")
    kpct_wide     = make_wide(pf_agg, "K%_wt",         "Kpct")
    bbpct_wide    = make_wide(pf_agg, "BB%_wt",        "BBpct")
    maxev_wide    = make_wide(pf_agg, "MaxEV_wt",      "MaxEV")
    ev90_wide     = make_wide(pf_agg, "EV90_wt",       "EV90")

    name_ser = level_agg.drop_duplicates("PlayerId").set_index("PlayerId")["Name"]

    # Career averages — PA × level-discount weighted across all levels
    def career_avg_ser(src_agg, val_col, out_name):
        tmp = src_agg.copy()
        tmp["disc"] = tmp["Level"].map(LEVEL_DISCOUNT)
        tmp["wt"]   = tmp["PA_total"] * tmp["disc"]
        tmp["wtv"]  = tmp["wt"] * tmp[val_col]          # NaN where val_col is NaN
        # Only count weight for rows where val_col is non-null
        tmp["wt_valid"] = tmp["wt"].where(tmp[val_col].notna(), 0)
        g = tmp.groupby("PlayerId")[["wt_valid", "wtv"]].sum()
        return (g["wtv"] / g["wt_valid"]).where(g["wt_valid"] > 0).rename(out_name)

    ca_pppa    = career_avg_ser(level_agg, "PPPA_Z_wt",    "PPPA_Z_career")
    ca_bb2k    = career_avg_ser(pf_agg,    "BB_2K_wt",     "BB2K_career")
    ca_whiff   = career_avg_ser(pf_agg,    "Whiff%_wt",    "Whiff_career")
    ca_sbt     = career_avg_ser(pf_agg,    "SB_talent_wt", "SBTalent_career")
    ca_hrfb    = career_avg_ser(pf_agg,    "HR/FB_wt",     "HRFB_career")
    ca_pullair = career_avg_ser(pf_agg,    "PullAir%_wt",  "PullAir_career")
    ca_kpct    = career_avg_ser(pf_agg,    "K%_wt",        "Kpct_career")
    ca_bbpct   = career_avg_ser(pf_agg,    "BB%_wt",       "BBpct_career")
    ca_maxev   = career_avg_ser(pf_agg,    "MaxEV_wt",     "MaxEV_career")
    ca_ev90    = career_avg_ser(pf_agg,    "EV90_wt",      "EV90_career")

    wide = pppa_wide.join([age_wide, pa_wide,
                           bb2k_wide, whiff_wide, sbt_wide, hrfb_wide, pullair_wide,
                           kpct_wide, bbpct_wide, maxev_wide, ev90_wide,
                           ca_pppa, ca_bb2k, ca_whiff, ca_sbt,
                           ca_hrfb, ca_pullair, ca_kpct, ca_bbpct,
                           ca_maxev, ca_ev90,
                           name_ser]).reset_index()

    # Attach MLBAM_ID — prefer pf_bridge (has MLBAM_ID for all pf players),
    # fall back to id_bridge from milb_hitting
    wide = wide.merge(pf_bridge, on="PlayerId", how="left")
    missing_mlbam = wide["MLBAM_ID"].isna()
    if missing_mlbam.any():
        wide.loc[missing_mlbam, "MLBAM_ID"] = (
            wide.loc[missing_mlbam, "PlayerId"]
            .map(id_bridge.set_index("PlayerId")["MLBAM_ID"])
        )

    # MLB outcomes keyed on MLBAM_ID -> mlb.MLBAMID
    mlb_valid = mlb[mlb["PA"] >= 50].copy()
    career_agg = (
        mlb_valid.groupby("MLBAMID")
        .apply(lambda g: pd.Series({
            "Career_PPPA_Z":  (g["PPPA_Z"] * g["PA"]).sum() / g["PA"].sum(),
            "Career_MLB_PA":  g["PA"].sum(),
        }))
        .reset_index()
        .rename(columns={"MLBAMID": "MLBAM_ID"})
    )

    first_yr = (
        mlb[mlb["PA"] >= 100]
        .sort_values("Season")
        .groupby("MLBAMID")
        .first()[["PPPA_Z", "PA", "Season"]]
        .reset_index()
        .rename(columns={"MLBAMID": "MLBAM_ID", "PPPA_Z": "FirstYr_PPPA_Z",
                         "PA": "FirstYr_PA", "Season": "MLB_Season"})
    )

    # Career MiLB span — min/max Season with PA >= 10
    span = (
        ovr[ovr["PA"] >= 10]
        .groupby("PlayerId")["Season"]
        .agg(MiLB_First="min", MiLB_Last="max")
        .reset_index()
    )

    wide = wide.merge(career_agg, on="MLBAM_ID", how="left")
    wide = wide.merge(first_yr,   on="MLBAM_ID", how="left")
    wide = wide.merge(span,       on="PlayerId",  how="left")
    wide["graduated"] = wide["FirstYr_PPPA_Z"].notna()

    # MLB EV fallback: for graduated players with no MiLB EV, use career MLB MaxEV
    mlb_ev_path = DATA / "api" / "mlb_statcast_ev.csv"
    if mlb_ev_path.exists():
        mlb_ev = pd.read_csv(mlb_ev_path)[["MLBAM_ID", "MaxEV_mlb"]]
        wide = wide.merge(mlb_ev, on="MLBAM_ID", how="left")
        missing_ev = wide["MaxEV_career"].isna() & wide["graduated"]
        wide.loc[missing_ev, "MaxEV_career"] = wide.loc[missing_ev, "MaxEV_mlb"]
        wide.drop(columns=["MaxEV_mlb"], inplace=True)

    # Ensure all level columns exist
    for lvl in LEVELS:
        lvl_key = lvl.replace("+", "plus")
        for prefix in ["PPPA_Z", "Age", "PA", "BB2K", "Whiff", "SBTalent", "HRFB", "PullAir", "Kpct", "BBpct", "MaxEV", "EV90"]:
            col = f"{prefix}_{lvl_key}"
            if col not in wide.columns:
                wide[col] = np.nan

    col_order = (
        ["PlayerId", "MLBAM_ID", "Name"]
        + [f"PPPA_Z_{l.replace('+','plus')}"   for l in LEVELS]
        + [f"Age_{l.replace('+','plus')}"       for l in LEVELS]
        + [f"PA_{l.replace('+','plus')}"        for l in LEVELS]
        + [f"BB2K_{l.replace('+','plus')}"      for l in LEVELS]
        + [f"Whiff_{l.replace('+','plus')}"     for l in LEVELS]
        + [f"SBTalent_{l.replace('+','plus')}"  for l in LEVELS]
        + [f"HRFB_{l.replace('+','plus')}"      for l in LEVELS]
        + [f"PullAir_{l.replace('+','plus')}"   for l in LEVELS]
        + [f"Kpct_{l.replace('+','plus')}"      for l in LEVELS]
        + [f"BBpct_{l.replace('+','plus')}"     for l in LEVELS]
        + [f"MaxEV_{l.replace('+','plus')}"     for l in LEVELS]
        + [f"EV90_{l.replace('+','plus')}"      for l in LEVELS]
        + ["PPPA_Z_career", "BB2K_career", "Whiff_career", "SBTalent_career",
           "HRFB_career", "PullAir_career", "Kpct_career", "BBpct_career",
           "MaxEV_career", "EV90_career"]
        + ["MiLB_First", "MiLB_Last", "graduated", "MLB_Season",
           "FirstYr_PPPA_Z", "FirstYr_PA", "Career_PPPA_Z", "Career_MLB_PA"]
    )
    wide = wide[[c for c in col_order if c in wide.columns]]

    # Round to readable precision
    lk_list = [l.replace("+", "plus") for l in LEVELS]
    for lk in lk_list:
        for col in [f"PPPA_Z_{lk}", f"FirstYr_PPPA_Z", f"Career_PPPA_Z"]:
            if col in wide.columns:
                wide[col] = wide[col].round(2)
        if f"Age_{lk}" in wide.columns:
            wide[f"Age_{lk}"] = wide[f"Age_{lk}"].round(2)
        if f"PA_{lk}" in wide.columns:
            wide[f"PA_{lk}"] = wide[f"PA_{lk}"].round(0).astype("Int64")
        for col in [f"BB2K_{lk}", f"Whiff_{lk}", f"HRFB_{lk}", f"PullAir_{lk}",
                    f"Kpct_{lk}", f"BBpct_{lk}"]:
            if col in wide.columns:
                wide[col] = wide[col].round(3)
        for col in [f"MaxEV_{lk}", f"EV90_{lk}"]:
            if col in wide.columns:
                wide[col] = wide[col].round(1)
        if f"SBTalent_{lk}" in wide.columns:
            wide[f"SBTalent_{lk}"] = wide[f"SBTalent_{lk}"].round(4)
    for col in ["FirstYr_PPPA_Z", "Career_PPPA_Z", "PPPA_Z_career"]:
        if col in wide.columns:
            wide[col] = wide[col].round(2)
    for col in ["BB2K_career", "Whiff_career", "HRFB_career", "PullAir_career",
                "Kpct_career", "BBpct_career"]:
        if col in wide.columns:
            wide[col] = wide[col].round(3)
    for col in ["MaxEV_career", "EV90_career"]:
        if col in wide.columns:
            wide[col] = wide[col].round(1)
    if "SBTalent_career" in wide.columns:
        wide["SBTalent_career"] = wide["SBTalent_career"].round(4)
    if "Career_MLB_PA" in wide.columns:
        wide["Career_MLB_PA"] = wide["Career_MLB_PA"].round(0)
    if "FirstYr_PA" in wide.columns:
        wide["FirstYr_PA"] = wide["FirstYr_PA"].round(0)

    return wide


# ---------------------------------------------------------------------------
# Comparator
# ---------------------------------------------------------------------------

def _prep_pool(df: pd.DataFrame):
    """Pre-compute level-wise z-score params for all distance features."""
    params = {}
    all_feats = ["PPPA_Z", "Age"] + list(SKILL_WEIGHTS.keys())
    for lvl in LEVELS:
        lk = lvl.replace("+", "plus")
        pa_col = f"PA_{lk}"
        mask = df[pa_col].fillna(0) >= MIN_COMP_PA
        for feat in all_feats:
            col = f"{feat}_{lk}"
            if col not in df.columns:
                continue
            vals = df.loc[mask, col].dropna()
            params[(lvl, feat)] = (vals.mean(), vals.std()) if len(vals) >= 5 else (0.0, 1.0)
    return params


def find_comps(query_name_or_id, pool: pd.DataFrame, n: int = 10,
               min_shared_levels: int = 1, exclude_self: bool = True):
    """Return top-N historical comps for a player.

    query_name_or_id: Name string (partial match OK) or numeric PlayerId/MLBAM_ID.
    """
    # Resolve query row
    if isinstance(query_name_or_id, str):
        mask = pool["Name"].str.contains(query_name_or_id, case=False, na=False)
        matches = pool[mask]
        if len(matches) == 0:
            raise ValueError(f"No player matching '{query_name_or_id}'")
        if len(matches) > 1:
            # Pick player with most total career PA — almost always the right
            # choice when the same name appears for multiple different players
            # or when a partial string matches unrelated names.
            pa_cols = [c for c in matches.columns if c.startswith("PA_")]
            matches = matches.copy()
            matches["_total_pa"] = matches[pa_cols].sum(axis=1, skipna=True)
            best = matches.sort_values("_total_pa", ascending=False)
            if len(best["Name"].unique()) > 1 or best.iloc[0]["_total_pa"] != best.iloc[1]["_total_pa"]:
                query = best.iloc[0]
                if best.iloc[0]["Name"] != best.iloc[1]["Name"]:
                    print(f"  Multiple name matches — selected '{query['Name']}' "
                          f"(PlayerId={query['PlayerId']}, most career PA)")
            else:
                query = best.iloc[0]
        else:
            query = matches.iloc[0]
    else:
        pid = int(query_name_or_id)
        row = pool[pool["PlayerId"] == pid]
        if len(row) == 0:
            row = pool[pool["MLBAM_ID"] == pid]
        if len(row) == 0:
            raise ValueError(f"PlayerId/MLBAM_ID {pid} not found")
        query = row.iloc[0]

    params = _prep_pool(pool)

    def _z(val, mu, sig):
        return (val - mu) / sig if sig > 1e-9 else 0.0

    # Build feature vector for query
    q_feats = {}
    for lvl in LEVELS:
        lk = lvl.replace("+", "plus")
        pa = query.get(f"PA_{lk}", np.nan)
        if pd.isna(pa) or pa < MIN_COMP_PA:
            continue
        pz = query.get(f"PPPA_Z_{lk}", np.nan)
        ag = query.get(f"Age_{lk}", np.nan)
        if pd.notna(pz) and pd.notna(ag):
            mu_z, sig_z = params.get((lvl, "PPPA_Z"), (0.0, 1.0))
            mu_a, sig_a = params.get((lvl, "Age"),    (0.0, 1.0))
            skills_z = {}
            for sk in SKILL_WEIGHTS:
                sv = query.get(f"{sk}_{lk}", np.nan)
                if pd.notna(sv):
                    mu_s, sig_s = params.get((lvl, sk), (0.0, 1.0))
                    skills_z[sk] = _z(sv, mu_s, sig_s)
            q_feats[lvl] = {
                "pppa_z":   _z(pz, mu_z, sig_z),
                "age_z":    _z(ag, mu_a, sig_a),
                "skills_z": skills_z,
                "pa":       pa,
            }

    if not q_feats:
        raise ValueError("Query player has no qualifying level data (PA >= MIN_COMP_PA)")

    results = []
    for _, cand in pool.iterrows():
        if exclude_self and cand["PlayerId"] == query["PlayerId"]:
            continue

        dist_sq   = 0.0
        weight_sum = 0.0
        shared    = 0

        for lvl, qf in q_feats.items():
            lk = lvl.replace("+", "plus")
            c_pa = cand.get(f"PA_{lk}", np.nan)
            if pd.isna(c_pa) or c_pa < MIN_COMP_PA:
                continue
            c_pz = cand.get(f"PPPA_Z_{lk}", np.nan)
            c_ag = cand.get(f"Age_{lk}", np.nan)
            if pd.isna(c_pz) or pd.isna(c_ag):
                continue

            mu_z, sig_z = params.get((lvl, "PPPA_Z"), (0.0, 1.0))
            mu_a, sig_a = params.get((lvl, "Age"),    (0.0, 1.0))

            d_pppa = _z(c_pz, mu_z, sig_z) - qf["pppa_z"]
            d_age  = _z(c_ag, mu_a, sig_a) - qf["age_z"]

            skill_term = 0.0
            for sk, sw in SKILL_WEIGHTS.items():
                q_sk = qf["skills_z"].get(sk)
                c_sv = cand.get(f"{sk}_{lk}", np.nan)
                if q_sk is not None and pd.notna(c_sv):
                    mu_s, sig_s = params.get((lvl, sk), (0.0, 1.0))
                    skill_term += sw * (_z(c_sv, mu_s, sig_s) - q_sk) ** 2

            # Equal weight per level — qualification is binary (both ≥ MIN_COMP_PA)
            wt = 1.0

            dist_sq    += wt * (d_pppa**2 + AGE_WEIGHT * d_age**2 + skill_term)
            weight_sum += wt
            shared     += 1

        if shared < min_shared_levels or weight_sum == 0:
            continue

        results.append({
            "PlayerId":       cand["PlayerId"],
            "Name":           cand["Name"],
            "dist":           np.sqrt(dist_sq / weight_sum),
            "shared_levels":  shared,
            "graduated":      cand.get("graduated", False),
            "MLB_Season":     cand.get("MLB_Season", np.nan),
            "FirstYr_PPPA_Z": cand.get("FirstYr_PPPA_Z", np.nan),
            "Career_PPPA_Z":  cand.get("Career_PPPA_Z", np.nan),
            "Career_MLB_PA":  cand.get("Career_MLB_PA", np.nan),
        })

    if not results:
        return query, pd.DataFrame()
    comps = (pd.DataFrame(results)
             .sort_values("dist")
             .head(n)
             .reset_index(drop=True))
    comps.index += 1
    return query, comps


# Career-average features and weights (mirrors SKILL_WEIGHTS + PPPA_Z primary)
CAREER_FEATURES = {
    "PPPA_Z_career": 1.00,
    "BB2K_career":   0.15,
    "Whiff_career":  0.10,
    "HRFB_career":   0.10,
    "PullAir_career":0.08,
    "SBTalent_career":0.10,
    "Kpct_career":   0.10,
    "BBpct_career":  0.08,
    "MaxEV_career":  0.08,
    "EV90_career":   0.08,
}


def find_career_comps(query_name_or_id, pool: pd.DataFrame, n: int = 10,
                      exclude_self: bool = True):
    """Return top-N comps based on career-average skill profile.

    No shared-level requirement — every player with career averages is eligible.
    Distance is weighted Euclidean in z-score space over CAREER_FEATURES.
    """
    # Resolve query (reuse same name/ID logic as find_comps)
    if isinstance(query_name_or_id, str):
        mask = pool["Name"].str.contains(query_name_or_id, case=False, na=False)
        matches = pool[mask]
        if len(matches) == 0:
            raise ValueError(f"No player matching '{query_name_or_id}'")
        if len(matches) > 1:
            pa_cols = [c for c in matches.columns if c.startswith("PA_")]
            matches = matches.copy()
            matches["_total_pa"] = matches[pa_cols].sum(axis=1, skipna=True)
            query = matches.sort_values("_total_pa", ascending=False).iloc[0]
        else:
            query = matches.iloc[0]
    else:
        pid = int(query_name_or_id)
        row = pool[pool["PlayerId"] == pid]
        if len(row) == 0:
            row = pool[pool["MLBAM_ID"] == pid]
        if len(row) == 0:
            raise ValueError(f"PlayerId/MLBAM_ID {pid} not found")
        query = row.iloc[0]

    # Required features — must be non-null for both query and candidate
    # Optional features (sparse coverage) — included only when both have data
    REQUIRED_CAREER = [f for f in CAREER_FEATURES if f not in ("MaxEV_career", "EV90_career")]
    OPTIONAL_CAREER = ["MaxEV_career", "EV90_career"]

    feat_cols = list(CAREER_FEATURES.keys())
    eligible = pool.dropna(subset=REQUIRED_CAREER)
    params = {f: (eligible[f].mean(), eligible[f].std()) for f in feat_cols if f in eligible.columns}

    def _z(val, f):
        mu, sig = params[f]
        return (val - mu) / sig if sig > 1e-9 else 0.0

    q_missing = [f for f in REQUIRED_CAREER if pd.isna(query.get(f))]
    if q_missing:
        raise ValueError(f"Query player missing career features: {q_missing}")

    q_vec = {f: _z(query[f], f) for f in REQUIRED_CAREER if pd.notna(query.get(f))}

    results = []
    for _, cand in eligible.iterrows():
        if exclude_self and cand["PlayerId"] == query["PlayerId"]:
            continue
        dist_sq = 0.0
        weight_sum = 0.0
        for f, w in CAREER_FEATURES.items():
            q_val = query.get(f)
            c_val = cand.get(f)
            if pd.isna(q_val) or pd.isna(c_val):
                continue  # skip optional features when either side is missing
            if f not in params:
                continue
            dist_sq    += w * (_z(c_val, f) - _z(q_val, f)) ** 2
            weight_sum += w
        if weight_sum == 0:
            continue
        results.append({
            "PlayerId":       cand["PlayerId"],
            "Name":           cand["Name"],
            "dist":           np.sqrt(dist_sq / weight_sum),
            "graduated":      cand.get("graduated", False),
            "MLB_Season":     cand.get("MLB_Season", np.nan),
            "FirstYr_PPPA_Z": cand.get("FirstYr_PPPA_Z", np.nan),
            "Career_PPPA_Z":  cand.get("Career_PPPA_Z", np.nan),
            "Career_MLB_PA":  cand.get("Career_MLB_PA", np.nan),
        })

    if not results:
        return query, pd.DataFrame()
    comps = (pd.DataFrame(results)
             .sort_values("dist")
             .head(n)
             .reset_index(drop=True))
    comps.index += 1
    return query, comps


def print_comps(query, comps, n_shown=10):
    name   = query["Name"]
    levels_shown = []
    for lvl in LEVELS:
        lk = lvl.replace("+", "plus")
        pa = query.get(f"PA_{lk}", np.nan)
        if pd.notna(pa) and pa >= MIN_COMP_PA:
            pz = query.get(f"PPPA_Z_{lk}", np.nan)
            ag = query.get(f"Age_{lk}", np.nan)
            levels_shown.append(f"{lvl}: PPPA_Z={pz:.2f} Age={ag:.1f} PA={pa:.0f}")

    print(f"\n{'='*65}")
    print(f"Query: {name}")
    for l in levels_shown:
        print(f"  {l}")
    graduated = comps["graduated"].sum()
    print(f"\nTop {len(comps)} comps  ({graduated}/{len(comps)} graduated to MLB)")
    grad_z = comps.loc[comps["graduated"], "Career_PPPA_Z"]
    if len(grad_z):
        print(f"  Among graduates — Career_PPPA_Z: "
              f"mean={grad_z.mean():.2f}  median={grad_z.median():.2f}")
    print()
    print(f"  {'#':>2}  {'Name':<25} {'Shared':>7}  {'Dist':>5}  "
          f"{'Grad':>5}  {'1stYr_Z':>8}  {'CareerZ':>8}  {'MLB_PA':>7}")
    for i, row in comps.iterrows():
        grad_str = "YES" if row["graduated"] else "no"
        fyr  = f"{row['FirstYr_PPPA_Z']:+.2f}" if pd.notna(row["FirstYr_PPPA_Z"]) else "  N/A"
        carz = f"{row['Career_PPPA_Z']:+.2f}"  if pd.notna(row["Career_PPPA_Z"])  else "  N/A"
        mpa  = f"{row['Career_MLB_PA']:.0f}"   if pd.notna(row["Career_MLB_PA"])  else "  N/A"
        print(f"  {i:>2}  {row['Name']:<25} {row['shared_levels']:>7}  "
              f"{row['dist']:>5.3f}  {grad_str:>5}  {fyr:>8}  {carz:>8}  {mpa:>7}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--n",    type=int, default=10)
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild even if player_comps.csv exists")
    args = parser.parse_args()

    if not OUT.exists() or args.rebuild:
        print("Building player_comps.csv …")
        df = build_dataset()
        df.to_csv(OUT, index=False)
        print(f"Wrote {len(df):,} rows -> {OUT}")
        print(f"  Graduated: {df['graduated'].sum():,} / {len(df):,}")
    else:
        df = pd.read_csv(OUT)
        print(f"Loaded {len(df):,} rows from {OUT}")

    # Sample comps
    query_names = (
        [args.name] if args.name
        else ["Max Clark", "Josue De Paula", "Eduardo Quintero", "Jesus Made"]
    )

    for name in query_names:
        try:
            query, comps = find_comps(name, df, n=args.n)
            print_comps(query, comps)
        except ValueError as e:
            print(f"\n  Error for '{name}': {e}")


if __name__ == "__main__":
    main()
