"""Build prospect_comps.json for the HTML artifact comps tab.

Computes top-10 comps + projection for the top TOP_N prospects by
Combined_Rank. Embeds per-level profile stats for both the query
player and each comp so the artifact can show them without a round-trip.

Output: data/rankings/prospect_comps.json
  {
    "<PlayerId>": {
      "name": "...", "rank": 1, "team": "...", "level": "AA", "age": 20,
      "projection": {"grad_pct": 0.8, "n_grads": 8,
                     "ceiling": 1.10, "median": 0.40, "floor": -0.03},
      "profile": [{"level":"AA","pppa_z":1.13,"age":20.0,"pa":551,
                   "bb2k":0.041,"kpct":0.148,"hrfb":0.087,"sbtalent":0.003},...],
      "comps": [{"name":"...","match_pct":65.2,"shared":3,
                 "graduated":true,"career_z":0.45,"firstyr_z":0.40,"career_pa":6579,
                 "profile":[...]}, ...]
    }, ...
  }
"""

import json
import sys
import math
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.build_player_comps import (
    LEVELS, LEVEL_DISCOUNT, MIN_COMP_PA, PA_THRESHOLD,
    AGE_WEIGHT, SKILL_WEIGHTS, TRAJ_WEIGHT,
    _prep_pool, project_from_comps, dist_to_pct,
)

DATA = ROOT / "data"
COMPS_CSV  = DATA / "computed"  / "player_comps.csv"
SCORES_CSV = DATA / "rankings"  / "prospect_scores.csv"
OUT_JSON   = DATA / "rankings"  / "prospect_comps.json"

TOP_N   = 250   # compute for top-N ranked prospects
N_COMPS = 10    # comps per prospect

# ── Stat columns to include in player profiles ────────────────────────────
PROFILE_LEVEL_COLS = {
    "pppa_z":   ("PPPA_Z_{lk}",   2),
    "age":      ("Age_{lk}",      1),
    "pa":       ("PA_{lk}",       0),
    "bb2k":     ("BB2K_{lk}",     3),
    "kpct":     ("Kpct_{lk}",     3),
    "hrfb":     ("HRFB_{lk}",     3),
    "sbtalent": ("SBTalent_{lk}", 4),
}


def _nan(v):
    if v is None:
        return None
    try:
        if math.isnan(float(v)):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _profile(row, levels_with_pa=None):
    """Extract per-level profile stats from a player_comps row (Series or dict)."""
    result = []
    for lvl in LEVELS:
        lk = lvl.replace("+", "plus")
        pa_key = f"PA_{lk}"
        pa_raw = row.get(pa_key) if isinstance(row, dict) else row.get(pa_key, None)
        pa = _nan(pa_raw)
        if pa is None or float(pa) < MIN_COMP_PA:
            continue
        if levels_with_pa is not None and lvl not in levels_with_pa:
            continue
        entry = {"level": lvl}
        for stat, (col_tmpl, rnd) in PROFILE_LEVEL_COLS.items():
            col = col_tmpl.format(lk=lk)
            v = row.get(col) if isinstance(row, dict) else row.get(col, None)
            v = _nan(v)
            entry[stat] = round(float(v), rnd) if v is not None else None
        result.append(entry)
    return result


def _find_comps_fast(query_rec, pool_records, pool_df, params, n=N_COMPS):
    """Vectorized comp finder using pre-converted pool_records (list of dicts).

    ~10× faster than pool.iterrows() for batch computation.
    """
    q_feats = {}
    for lvl in LEVELS:
        lk = lvl.replace("+", "plus")
        pa = query_rec.get(f"PA_{lk}", None)
        if pa is None or float(pa) < MIN_COMP_PA:
            continue
        pz = query_rec.get(f"PPPA_Z_{lk}")
        ag = query_rec.get(f"Age_{lk}")
        if pz is None or ag is None or math.isnan(float(pz)) or math.isnan(float(ag)):
            continue

        def _z(val, lvl_=lvl, feat_=None):
            mu, sig = params.get((lvl_, feat_), (0.0, 1.0))
            return (float(val) - mu) / sig if sig > 1e-9 else 0.0

        skills_z = {}
        for sk in SKILL_WEIGHTS:
            sv = query_rec.get(f"{sk}_{lk}")
            if sv is not None and not math.isnan(float(sv)):
                skills_z[sk] = _z(sv, lvl, sk)

        q_feats[lvl] = {
            "pppa_z": _z(pz, lvl, "PPPA_Z"),
            "age_z":  _z(ag, lvl, "Age"),
            "skills_z": skills_z,
            "pa": float(pa),
        }

    if not q_feats:
        return pd.DataFrame()

    # Trajectory
    q_traj_z = None
    q_traj_raw = query_rec.get("PPPA_Z_trajectory")
    if q_traj_raw is not None and not math.isnan(float(q_traj_raw)):
        mu_t, sig_t = params.get(("career", "trajectory"), (0.0, 1.0))
        q_traj_z = (float(q_traj_raw) - mu_t) / sig_t if sig_t > 1e-9 else 0.0

    q_pid = query_rec.get("PlayerId")
    results = []

    for cand in pool_records:
        if cand.get("PlayerId") == q_pid:
            continue

        dist_sq = 0.0
        weight_sum = 0.0
        shared = 0

        for lvl, qf in q_feats.items():
            lk = lvl.replace("+", "plus")
            c_pa_raw = cand.get(f"PA_{lk}")
            if c_pa_raw is None or float(c_pa_raw) < MIN_COMP_PA:
                continue
            c_pz = cand.get(f"PPPA_Z_{lk}")
            c_ag = cand.get(f"Age_{lk}")
            if c_pz is None or c_ag is None:
                continue
            try:
                c_pz_f = float(c_pz);  c_ag_f = float(c_ag)
            except (TypeError, ValueError):
                continue
            if math.isnan(c_pz_f) or math.isnan(c_ag_f):
                continue

            mu_pz, sig_pz = params.get((lvl, "PPPA_Z"), (0.0, 1.0))
            mu_ag, sig_ag = params.get((lvl, "Age"),    (0.0, 1.0))

            def _zc(val, mu, sig):
                return (float(val) - mu) / sig if sig > 1e-9 else 0.0

            d_pppa = _zc(c_pz_f, mu_pz, sig_pz) - qf["pppa_z"]
            d_age  = _zc(c_ag_f, mu_ag, sig_ag)  - qf["age_z"]

            skill_term = 0.0
            for sk, sw in SKILL_WEIGHTS.items():
                q_sk = qf["skills_z"].get(sk)
                if q_sk is None:
                    continue
                c_sv = cand.get(f"{sk}_{lk}")
                if c_sv is None:
                    continue
                try:
                    c_sv_f = float(c_sv)
                except (TypeError, ValueError):
                    continue
                if math.isnan(c_sv_f):
                    continue
                mu_s, sig_s = params.get((lvl, sk), (0.0, 1.0))
                skill_term += sw * (_zc(c_sv_f, mu_s, sig_s) - q_sk) ** 2

            pa_eff = min(qf["pa"], float(c_pa_raw), PA_THRESHOLD) / PA_THRESHOLD
            wt = LEVEL_DISCOUNT[lvl] * pa_eff

            dist_sq    += wt * (d_pppa**2 + AGE_WEIGHT * d_age**2 + skill_term)
            weight_sum += wt
            shared     += 1

        if shared < 1 or weight_sum == 0:
            continue

        if q_traj_z is not None:
            c_traj_raw = cand.get("PPPA_Z_trajectory")
            if c_traj_raw is not None:
                try:
                    c_traj_f = float(c_traj_raw)
                    if not math.isnan(c_traj_f):
                        mu_t, sig_t = params.get(("career", "trajectory"), (0.0, 1.0))
                        c_traj_z = (c_traj_f - mu_t) / sig_t if sig_t > 1e-9 else 0.0
                        dist_sq    += weight_sum * TRAJ_WEIGHT * (c_traj_z - q_traj_z) ** 2
                        weight_sum += weight_sum * TRAJ_WEIGHT
                except (TypeError, ValueError):
                    pass

        d = math.sqrt(dist_sq / weight_sum)
        results.append({
            "name":        cand.get("Name", ""),
            "match_pct":   dist_to_pct(d),
            "shared":      shared,
            "graduated":   bool(cand.get("graduated", False)),
            "career_z":    _nan(cand.get("Career_PPPA_Z")),
            "firstyr_z":   _nan(cand.get("FirstYr_PPPA_Z")),
            "career_pa":   _nan(cand.get("Career_MLB_PA")),
            "_dist":       d,
            "_cand":       cand,
        })

    if not results:
        return pd.DataFrame()

    results.sort(key=lambda x: x["_dist"])
    top = results[:n]

    # Build projection from a minimal DataFrame
    comp_df = pd.DataFrame([{
        "graduated":      r["graduated"],
        "Career_PPPA_Z":  r["career_z"],
        "match_pct":      r["match_pct"],
    } for r in top])
    proj = project_from_comps(comp_df)

    return top, proj


def main():
    print("Loading data…")
    pool_df  = pd.read_csv(COMPS_CSV)
    scores   = pd.read_csv(SCORES_CSV)

    params       = _prep_pool(pool_df)
    pool_records = pool_df.to_dict("records")
    pool_by_pid  = {str(r["PlayerId"]): r for r in pool_records}

    top_scores = (
        scores[scores["Combined_Rank"] <= TOP_N]
        .sort_values("Combined_Rank")
        .copy()
    )
    print(f"Computing comps for {len(top_scores)} prospects…")

    output = {}
    for i, (_, score_row) in enumerate(top_scores.iterrows(), 1):
        pid    = str(score_row["PlayerId"])
        name   = score_row["Name"]
        rank   = int(score_row["Combined_Rank"])

        query_rec = pool_by_pid.get(pid)
        if query_rec is None:
            # Try name match as fallback
            matches = [r for r in pool_records
                       if r.get("Name","").strip().lower() == name.strip().lower()]
            query_rec = matches[0] if matches else None

        if query_rec is None:
            print(f"  [{i:>3}] {name}: not found in player_comps.csv — skipped")
            continue

        result = _find_comps_fast(query_rec, pool_records, pool_df, params)
        if isinstance(result, pd.DataFrame):  # empty
            print(f"  [{i:>3}] {name}: no comps found — skipped")
            continue

        top_comps, proj = result

        # Levels the query player has data at (for display)
        q_profile = _profile(query_rec)
        q_levels  = {e["level"] for e in q_profile}

        comp_list = []
        for c in top_comps:
            cand_rec  = c["_cand"]
            # Only show comp profile at levels the query also played
            cand_prof = _profile(cand_rec, levels_with_pa=q_levels)
            comp_list.append({
                "name":      c["name"],
                "match_pct": c["match_pct"],
                "shared":    c["shared"],
                "graduated": c["graduated"],
                "career_z":  c["career_z"],
                "firstyr_z": c["firstyr_z"],
                "career_pa": int(c["career_pa"]) if c["career_pa"] is not None else None,
                "profile":   cand_prof,
            })

        output[pid] = {
            "name":       name,
            "rank":       rank,
            "team":       str(score_row.get("Team", "")),
            "level":      str(score_row.get("Level", "")),
            "age":        _nan(score_row.get("Age")),
            "projection": {
                "grad_pct": round(proj["grad_pct"], 2),
                "n_grads":  proj["n_grads"],
                "ceiling":  _nan(proj["ceiling"]),
                "median":   _nan(proj["median"]),
                "floor":    _nan(proj["floor"]),
            },
            "profile":  q_profile,
            "comps":    comp_list,
        }

        if i % 25 == 0:
            print(f"  {i}/{len(top_scores)}…")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, separators=(",", ":"))
    size_kb = OUT_JSON.stat().st_size // 1024
    print(f"Wrote {len(output)} prospects → {OUT_JSON}  ({size_kb} KB)")


if __name__ == "__main__":
    main()
