"""Fantrax Skill_PPPA MiLB -> MLB Translation Study

Skill_PPPA (Fantrax) = -0.5*K% + 4*HR/PA + 2*SB/PA - 1*CS/PA

Weights follow the Fantrax competitive dynasty scoring formula:
  SO=-0.5, HR=+4, SB=+2, CS=-1 (same HR as personal; K and SB weights differ).

Methodology is identical to skill_pppa_translation.py (Approach B — full
population, non-graduates get MLB_Skill = 0). Approach B is the authoritative
one: removes selection bias and answers "expected MLB value per unit of MiLB
production at each level."

Outputs printed to console; call update_config() to write results to
config/scoring_fantrax.py automatically.
"""

import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

DATA    = Path(__file__).resolve().parent.parent / "data"
ROOT    = Path(__file__).resolve().parent.parent
LEVELS  = ["AAA", "AA", "A+", "A", "R"]

# Current placeholders (copied from personal model)
CURRENT_FX = {"AAA": 1.00, "AA": 0.59, "A+": 0.34, "A": 0.23, "R": 0.10}
PERSONAL   = {"AAA": 1.00, "AA": 0.59, "A+": 0.34, "A": 0.23, "R": 0.10}

# ---------------------------------------------------------------------------
# Scoring formulas
# ---------------------------------------------------------------------------

def skill_pppa_fantrax(df):
    pa = df["PA"]
    return (
        -0.5 * (df["SO"] / pa)
        +  4.0 * (df["HR"] / pa)
        +  2.0 * (df["SB"] / pa)
        -  1.0 * (df["CS"] / pa)
    )

def skill_pppa_personal(df):
    pa = df["PA"]
    return (
        -2.0 * (df["SO"] / pa)
        +  4.0 * (df["HR"] / pa)
        +  3.0 * (df["SB"] / pa)
        -  1.5 * (df["CS"] / pa)
    )

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
mlb  = pd.read_parquet(DATA / "historical" / "hist_mlb_data.parquet")
milb = pd.read_csv(DATA / "api" / "milb_hitting.csv")

# Active Fantrax prospects excluded (haven't had time to graduate)
prospects_fx = pd.read_csv(DATA / "rankings_fantrax" / "prospect_scores.csv")
active_fg = set(prospects_fx["PlayerId"].dropna().astype(int))

# ---------------------------------------------------------------------------
# Build mlb_debut keyed on MLBAM_ID
# ---------------------------------------------------------------------------
mlb_valid = mlb[mlb["PA"] >= 100].copy()
mlb_valid["MLB_Skill_FX"]  = skill_pppa_fantrax(mlb_valid)
mlb_valid["MLB_Skill_PER"] = skill_pppa_personal(mlb_valid)

mlb_debut = (
    mlb_valid.sort_values("Season")
    .groupby("MLBAMID").first()
    .reset_index()
    [["MLBAMID", "MLB_Skill_FX", "MLB_Skill_PER", "Season", "PA"]]
)
mlb_debut.columns = ["MLBAM_ID", "MLB_Skill_FX", "MLB_Skill_PER", "MLB_Season", "MLB_PA"]
grad_mlbam_ids = set(mlb_debut["MLBAM_ID"].astype(int))

milb["Skill_FX"]  = skill_pppa_fantrax(milb)
milb["Skill_PER"] = skill_pppa_personal(milb)

# ---------------------------------------------------------------------------
# Approach B: full population, non-graduates = 0
# Best (highest PA) season per player-level among non-active prospects
# ---------------------------------------------------------------------------
milb_b = milb[~milb["PlayerId"].isin(active_fg)].copy()
milb_b = milb_b[milb_b["PA"] >= 50]
milb_best = (
    milb_b.sort_values("PA", ascending=False)
    .drop_duplicates(["MLBAM_ID", "Level"])
    .copy()
)
mlb_debut["MLBAM_ID"] = mlb_debut["MLBAM_ID"].astype(milb_best["MLBAM_ID"].dtype)
milb_best = milb_best.merge(
    mlb_debut[["MLBAM_ID", "MLB_Skill_FX", "MLB_Skill_PER"]],
    on="MLBAM_ID", how="left"
)
milb_best["graduated"]    = milb_best["MLBAM_ID"].isin(grad_mlbam_ids)
milb_best["MLB_Skill_FX"]  = milb_best["MLB_Skill_FX"].fillna(0)
milb_best["MLB_Skill_PER"] = milb_best["MLB_Skill_PER"].fillna(0)

# ---------------------------------------------------------------------------
# Approach B: OLS regression slopes  (works regardless of Skill_PPPA sign)
#
# Why not mean-ratios: with K% weighted at -0.5, mean MiLB Skill_PPPA can be
# near zero or negative at lower levels where HR/PA is tiny. A near-zero
# denominator makes mean(MLB) / mean(MiLB) blow up or flip sign.
#
# OLS slope = Cov(MLB_Skill, MiLB_Skill) / Var(MiLB_Skill) gives the
# expected MLB Skill_PPPA per unit of MiLB Skill_PPPA. Sign-invariant,
# and conceptually identical to the mean-ratio when the ratio is well-defined.
# Normalized to AAA=1.0 to produce level discounts.
# ---------------------------------------------------------------------------
print("=" * 70)
print("Fantrax Skill_PPPA = -0.5*K% + 4*HR/PA + 2*SB/PA - 1*CS/PA")
print("Approach B: full population, non-graduates get MLB_Skill = 0")
print("OLS regression slope (sign-invariant; required because Fantrax")
print("K weight is small enough that mean MiLB Skill_PPPA can be negative)")
print(f"Active Fantrax prospects excluded: {len(active_fg):,}")
print(f"Unique non-prospect players: {milb_best['MLBAM_ID'].nunique():,}")
print(f"Graduates (MLBAM match, PA>=100): {milb_best['graduated'].sum():,}")
print("=" * 70)

def ols_slope(x, y):
    """OLS slope of y ~ x (with intercept); returns NaN if degenerate."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 10:
        return np.nan
    xbar, ybar = x.mean(), y.mean()
    denom = ((x - xbar) ** 2).sum()
    return float(((x - xbar) * (y - ybar)).sum() / denom) if denom > 1e-12 else np.nan

slopes_fx, slopes_per = {}, {}
for lvl in LEVELS:
    g = milb_best[milb_best["Level"] == lvl]
    slopes_fx[lvl]  = ols_slope(g["Skill_FX"],  g["MLB_Skill_FX"])
    slopes_per[lvl] = ols_slope(g["Skill_PER"], g["MLB_Skill_PER"])

print()
print(f"{'Level':4s}  {'N':>6}  {'Grad':>6}  {'Grad%':>6}  "
      f"{'FX_slope':>9}  {'Per_slope':>10}  "
      f"{'FX_norm':>8}  {'Per_norm':>9}  "
      f"{'FX_curr':>8}  {'vs_curr':>8}")

aaa_fx_s  = slopes_fx.get("AAA", 1.0) or 1.0
aaa_per_s = slopes_per.get("AAA", 1.0) or 1.0
results = {}
for lvl in LEVELS:
    if lvl not in slopes_fx or slopes_fx[lvl] is None or np.isnan(slopes_fx[lvl]):
        continue
    g    = milb_best[milb_best["Level"] == lvl]
    n    = len(g)
    ng   = int(g["graduated"].sum())
    fx_n  = slopes_fx[lvl]  / aaa_fx_s
    per_n = slopes_per[lvl] / aaa_per_s
    curr  = CURRENT_FX.get(lvl, float("nan"))
    results[lvl] = fx_n
    print(f"{lvl:4s}  {n:6d}  {ng:6d}  {ng/n:6.1%}  "
          f"{slopes_fx[lvl]:9.4f}  {slopes_per[lvl]:10.4f}  "
          f"{fx_n:8.3f}  {per_n:9.3f}  "
          f"{curr:8.2f}  {fx_n-curr:+8.3f}")

# ---------------------------------------------------------------------------
# Side-by-side Approach A vs B for Fantrax
# ---------------------------------------------------------------------------
mlb_debut_a = mlb_debut.rename(columns={"MLBAM_ID": "MLBAM_ID"})
milb_a = milb.merge(mlb_debut_a[["MLBAM_ID", "MLB_Season"]], on="MLBAM_ID")
milb_a = milb_a[(milb_a["Season"] < milb_a["MLB_Season"]) & (milb_a["PA"] >= 50)]
milb_last_a = (
    milb_a.sort_values("Season", ascending=False)
    .drop_duplicates(["MLBAM_ID", "Level"])
    .copy()
)
paired_a = milb_last_a.merge(
    mlb_debut_a[["MLBAM_ID", "MLB_Skill_FX"]],
    on="MLBAM_ID"
)

ratios_a = {}
for lvl in LEVELS:
    g = paired_a[paired_a["Level"] == lvl].dropna(subset=["Skill_FX", "MLB_Skill_FX"])
    if len(g) >= 10 and g["Skill_FX"].mean() != 0:
        ratios_a[lvl] = g["MLB_Skill_FX"].mean() / g["Skill_FX"].mean()

print()
print("=" * 70)
print("Approach A vs. B (Fantrax formula, normalized AAA=1.0)")
print("=" * 70)
print(f"{'Level':4s}  {'Approach A':>11}  {'Approach B':>11}  {'Current FX':>11}  {'Personal':>9}")
aaa_a   = ratios_a.get("AAA", 1.0)
aaa_b_s = slopes_fx.get("AAA", 1.0) or 1.0
for lvl in LEVELS:
    na = f"{ratios_a[lvl]/aaa_a:.3f}" if lvl in ratios_a else "    N/A"
    nb = f"{slopes_fx[lvl]/aaa_b_s:.3f}" if lvl in slopes_fx and not np.isnan(slopes_fx.get(lvl, float('nan'))) else "    N/A"
    curr = CURRENT_FX.get(lvl, float("nan"))
    per  = PERSONAL.get(lvl, float("nan"))
    print(f"{lvl:4s}  {na:>11}  {nb:>11}  {curr:>11.2f}  {per:>9.2f}")

# ---------------------------------------------------------------------------
# Component-level breakdown: how each stat changes across levels (Approach A)
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("Component breakdown by level (Approach A, Fantrax formula)")
print("K_ratio = MLB K% / MiLB K%;  HR_ratio = MLB HR/PA / MiLB HR/PA")
print("netSB_ratio = MLB (SB-0.5CS)/PA / MiLB (SB-0.5CS)/PA")
print("=" * 70)
print(f"{'Level':4s}  {'N':>5}  {'K_ratio':>8}  {'HR_ratio':>9}  {'netSB_ratio':>12}")
for lvl in LEVELS:
    g = paired_a[paired_a["Level"] == lvl].dropna(subset=["Skill_FX", "MLB_Skill_FX"])
    if len(g) < 10:
        continue
    mlb_sub = mlb[mlb["MLBAMID"].isin(g["MLBAM_ID"])].copy()
    mlb_sub = mlb_sub.merge(mlb_debut_a[["MLBAM_ID", "MLB_Season"]], left_on="MLBAMID", right_on="MLBAM_ID")
    mlb_sub = mlb_sub[mlb_sub["Season"] == mlb_sub["MLB_Season"]].drop_duplicates("MLBAMID")

    m_k   = (g["SO"] / g["PA"]).mean()
    m_hr  = (g["HR"] / g["PA"]).mean()
    m_nsb = ((g["SB"] - 0.5*g["CS"]) / g["PA"]).mean()
    mlb_k   = (mlb_sub["SO"] / mlb_sub["PA"]).mean() if len(mlb_sub) else float("nan")
    mlb_hr  = (mlb_sub["HR"] / mlb_sub["PA"]).mean() if len(mlb_sub) else float("nan")
    mlb_nsb = ((mlb_sub["SB"] - 0.5*mlb_sub["CS"]) / mlb_sub["PA"]).mean() if len(mlb_sub) else float("nan")

    print(f"{lvl:4s}  {len(g):5d}  "
          f"{mlb_k/m_k if m_k>0 else float('nan'):8.3f}  "
          f"{mlb_hr/m_hr if m_hr>0 else float('nan'):9.3f}  "
          f"{mlb_nsb/m_nsb if m_nsb>0 else float('nan'):12.3f}")

# ---------------------------------------------------------------------------
# Recommended values (Approach B, rounded to 2dp)
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("RECOMMENDED Fantrax level discounts (Approach B, normalized AAA=1.0)")
print("=" * 70)
recommended = {}
for lvl in LEVELS:
    if lvl in results:
        val = round(results[lvl], 2)
        recommended[lvl] = val
        curr = CURRENT_FX.get(lvl, float("nan"))
        flag = "  <-- CHANGE" if abs(val - curr) >= 0.03 else ""
        print(f"  {lvl:4s}  {val:.2f}  (was {curr:.2f}){flag}")

print()
print("To update config/scoring_fantrax.py, re-run with --update flag:")
print("  python analysis/skill_pppa_translation_fantrax.py --update")

# ---------------------------------------------------------------------------
# Optional: update config/scoring_fantrax.py in place
# ---------------------------------------------------------------------------
if "--update" in sys.argv:
    config_path = ROOT / "config" / "scoring_fantrax.py"
    text = config_path.read_text(encoding="utf-8")
    old_block = re.search(
        r'LEVEL_DISCOUNT\s*=\s*\{[^}]+\}', text, re.DOTALL
    )
    if old_block and recommended:
        items = ", ".join(f'"{k}": {recommended[k]:.2f}' for k in LEVELS if k in recommended)
        new_block = f"LEVEL_DISCOUNT = {{{items}}}"
        text = text[:old_block.start()] + new_block + text[old_block.end():]
        config_path.write_text(text, encoding="utf-8")
        print(f"\nUpdated LEVEL_DISCOUNT in {config_path}")
        print(f"  {new_block}")
    else:
        print("\nCould not find LEVEL_DISCOUNT block to update.")
