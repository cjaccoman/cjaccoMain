"""Build comp-based projections for current prospects.

For each current prospect in prospect_scores.csv, finds the 30 closest historical
comps in player_comps.csv (career-average feature space) and computes:

  Career_PA_Proj   — match-pct-weighted mean of Career_MLB_PA across all 30 comps;
                     non-graduates contribute 0. Naturally folds graduation
                     probability into the projection without a separate model.
  Career_PA_P25    — 25th percentile of the comp Career_MLB_PA distribution
  Career_PA_P75    — 75th percentile
  MaxEV_proj       — weighted mean of MaxEV_career for comps that have EV data.
                     Only present when >= MIN_EV_COMPS comps have MaxEV_career.
  MaxEV_proj_n     — count of comps with MaxEV data used in projection
  MaxEV_proj_conf  — mean match_pct of those EV comps (quality indicator)

Output: data/rankings/prospect_projections.csv

Distance metric mirrors find_career_comps() in build_player_comps.py:
  Required features: PPPA_Z_career, BB2K, Whiff, HRFB, PullAir, SBTalent, Kpct, BBpct
  Optional features: MaxEV_career, EV90_career, Chase_career, ZContact_career
    (included only when both prospect and comp have data)
  Weights from CAREER_FEATURES dict.

Run:
  python analysis/build_prospect_projections.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA      = Path(__file__).resolve().parent.parent / "data"
POOL_PATH = DATA / "computed"  / "player_comps.csv"
PS_PATH   = DATA / "rankings"  / "prospect_scores.csv"
OUT_PATH  = DATA / "rankings"  / "prospect_projections.csv"

N_COMPS      = 30   # comps per prospect
MIN_EV_COMPS = 5    # minimum comps with MaxEV data to report MaxEV_proj

# Must match CAREER_FEATURES in build_player_comps.py
CAREER_FEATURES = {
    "PPPA_Z_career":   1.00,
    "BB2K_career":     0.15,
    "Whiff_career":    0.10,
    "HRFB_career":     0.10,
    "PullAir_career":  0.08,
    "SBTalent_career": 0.10,
    "Kpct_career":     0.10,
    "BBpct_career":    0.08,
    "MaxEV_career":    0.08,
    "EV90_career":     0.08,
    "Chase_career":    0.08,
    "ZContact_career": 0.08,
}

REQUIRED = [f for f in CAREER_FEATURES
            if f not in ("MaxEV_career", "EV90_career", "Chase_career", "ZContact_career")]
OPTIONAL = ["MaxEV_career", "EV90_career", "Chase_career", "ZContact_career"]

# MiLB PA shrinkage: Career_PA_Proj_adj = Career_PA_Proj × min(career_milb_pa / threshold, 1.0)
# Shrinks toward 0 for thin-resume players — a 17yo with 200 PA has 40% weight vs full credit.
MILB_PA_SHRINK_THRESH = 500


def main() -> None:
    pool = pd.read_csv(POOL_PATH)
    ps   = pd.read_csv(PS_PATH)

    # Pool eligible: must have all required career features
    eligible = pool.dropna(subset=REQUIRED).reset_index(drop=True)
    print(f"Pool: {len(pool):,} total  ->  {len(eligible):,} with required career features")

    # Compute z-score params from eligible pool
    req_mu  = eligible[REQUIRED].mean().values.copy()       # (n_req,)
    req_std = eligible[REQUIRED].std().values.copy()        # (n_req,)
    req_std[req_std < 1e-9] = 1.0
    req_wt  = np.array([CAREER_FEATURES[f] for f in REQUIRED])  # (n_req,)

    # Pool z-score matrix for required features: (n_pool, n_req)
    pool_z_req = (eligible[REQUIRED].values - req_mu) / req_std

    # Pre-compute optional feature z-score params and arrays
    opt_params = {}
    opt_pool_z = {}
    for f in OPTIONAL:
        vals = eligible[f].values
        mu_f  = np.nanmean(vals)
        std_f = np.nanstd(vals)
        if std_f < 1e-9:
            std_f = 1.0
        opt_params[f] = (mu_f, std_f)
        opt_pool_z[f] = (vals - mu_f) / std_f  # NaN where pool has no data

    pool_career_pa = np.where(
        eligible["graduated"].fillna(False).values,
        eligible["Career_MLB_PA"].fillna(0).values,
        0.0,
    )
    pool_maxev  = eligible["MaxEV_career"].values  # NaN for most non-AAA historical
    pool_ids    = eligible["PlayerId"].values

    # Current prospect PlayerId set — look up each in the pool
    prospect_pids = ps["PlayerId"].values
    in_pool_mask  = np.isin(prospect_pids, pool_ids)
    print(f"Current prospects: {len(ps):,}  ->  {in_pool_mask.sum():,} found in comp pool")

    pool_pid_to_idx = {pid: i for i, pid in enumerate(pool_ids)}

    records = []
    skipped = 0

    for _, row in ps.iterrows():
        pid  = row["PlayerId"]
        name = row["Name"]

        if pid not in pool_pid_to_idx:
            skipped += 1
            continue

        # Query career feature values
        pool_row = eligible.iloc[pool_pid_to_idx[pid]]
        q_req = np.array([pool_row.get(f, np.nan) for f in REQUIRED], dtype=float)
        if np.any(np.isnan(q_req)):
            skipped += 1
            continue

        # Required feature distances — fully vectorized (n_pool,)
        q_req_z = (q_req - req_mu) / req_std
        diff_req = pool_z_req - q_req_z[np.newaxis, :]     # (n_pool, n_req)
        dist_sq   = (req_wt * diff_req ** 2).sum(axis=1)   # (n_pool,)
        wt_sum    = np.full(len(eligible), req_wt.sum())

        # Optional feature distances — added per-feature where both have data
        for f, w in [(f, CAREER_FEATURES[f]) for f in OPTIONAL]:
            q_val = pool_row.get(f)
            if pd.isna(q_val):
                continue
            mu_f, std_f = opt_params[f]
            q_z_opt  = (q_val - mu_f) / std_f
            p_z_opt  = opt_pool_z[f]
            has_data = ~np.isnan(p_z_opt)
            dist_sq[has_data]  += w * (p_z_opt[has_data] - q_z_opt) ** 2
            wt_sum[has_data]   += w

        dist = np.sqrt(dist_sq / np.maximum(wt_sum, 1e-9))

        # Exclude self
        self_idx = pool_pid_to_idx[pid]
        dist[self_idx] = np.inf

        # Top N comps
        top_idx   = np.argpartition(dist, N_COMPS)[:N_COMPS]
        top_idx   = top_idx[np.argsort(dist[top_idx])]
        top_dists = dist[top_idx]
        top_mpct  = 100.0 * np.exp(-top_dists)     # match_pct per comp

        # ── Career_PA_Proj ────────────────────────────────────────────────────
        top_cpa   = pool_career_pa[top_idx]         # 0 for non-grads
        total_wt  = top_mpct.sum()
        career_pa_proj = float((top_mpct * top_cpa).sum() / total_wt) if total_wt > 0 else np.nan

        # Percentile bands on the weighted distribution (unweighted for simplicity)
        career_pa_p25 = float(np.percentile(top_cpa, 25))
        career_pa_p75 = float(np.percentile(top_cpa, 75))

        # MiLB-PA shrinkage: discount projection for thin-resume players.
        # A 17yo with 200 PA gets 40% weight; ≥500 PA gets full credit.
        milb_pa      = float(row.get("Career_PA", 0) or 0)
        milb_shrink  = min(milb_pa / MILB_PA_SHRINK_THRESH, 1.0)
        career_pa_proj_adj = career_pa_proj * milb_shrink if pd.notna(career_pa_proj) else np.nan

        # ── MaxEV_proj ────────────────────────────────────────────────────────
        top_maxev   = pool_maxev[top_idx]
        ev_mask     = ~np.isnan(top_maxev)
        maxev_proj  = maxev_n = maxev_conf = np.nan
        if ev_mask.sum() >= MIN_EV_COMPS:
            ev_mpct   = top_mpct[ev_mask]
            ev_vals   = top_maxev[ev_mask]
            ev_wt     = ev_mpct.sum()
            maxev_proj = float((ev_mpct * ev_vals).sum() / ev_wt) if ev_wt > 0 else np.nan
            maxev_n    = int(ev_mask.sum())
            maxev_conf = float(ev_mpct.mean())

        records.append({
            "PlayerId":           pid,
            "Name":               name,
            "Career_PA_Proj":     round(career_pa_proj, 0) if pd.notna(career_pa_proj) else np.nan,
            "Career_PA_Proj_Adj": round(career_pa_proj_adj, 0) if pd.notna(career_pa_proj_adj) else np.nan,
            "Career_PA_P25":      round(career_pa_p25, 0),
            "Career_PA_P75":      round(career_pa_p75, 0),
            "MaxEV_proj":         round(maxev_proj, 1) if pd.notna(maxev_proj) else np.nan,
            "MaxEV_proj_n":       maxev_n,
            "MaxEV_proj_conf":    round(maxev_conf, 1) if pd.notna(maxev_conf) else np.nan,
        })

    out = pd.DataFrame(records)
    out.to_csv(OUT_PATH, index=False)

    print(f"\nWrote {len(out):,} prospects  ({skipped} skipped — missing from pool or required features)")
    print(f"  Career_PA_Proj non-null : {out['Career_PA_Proj'].notna().sum():,}")
    print(f"  MaxEV_proj non-null     : {out['MaxEV_proj'].notna().sum():,}")

    # Sample output — top 25 by prospect rank
    ps_ranked = ps[["PlayerId", "Combined_Rank"]].merge(out, on="PlayerId")
    sample = ps_ranked.sort_values("Combined_Rank").head(25)
    print(f"\nTop 25 prospects — projections:")
    print(f"  {'Rank':>5}  {'Name':<25}  {'CareerPA':>8}  {'P25':>6}  {'P75':>6}  {'MaxEV_proj':>10}  {'EV_n':>5}")
    for _, r in sample.iterrows():
        pa_str = f"{r['Career_PA_Proj']:>8.0f}" if pd.notna(r['Career_PA_Proj']) else "     N/A"
        p25    = f"{r['Career_PA_P25']:>6.0f}"
        p75    = f"{r['Career_PA_P75']:>6.0f}"
        ev_str = f"{r['MaxEV_proj']:>10.1f}" if pd.notna(r['MaxEV_proj']) else "       N/A"
        ev_n   = f"{r['MaxEV_proj_n']:>5.0f}" if pd.notna(r['MaxEV_proj_n']) else "  N/A"
        name   = ps[ps['PlayerId'] == r['PlayerId']]['Name'].iloc[0]
        print(f"  {r['Combined_Rank']:>5}  {name:<25}  {pa_str}  {p25}  {p75}  {ev_str}  {ev_n}")


if __name__ == "__main__":
    main()
