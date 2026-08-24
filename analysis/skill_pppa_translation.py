"""Skill_PPPA MiLB -> MLB Translation Study

Skill_PPPA = -2*K% + 4*HR/PA + 3*SB/PA - 1.5*CS/PA

The three stats with the highest PPPA formula weights AND strongest MiLB->MLB
signal (Phase 2 ablation). Strips lineup noise (R, RBI).

NOTE: Skill_PPPA is negative for most players (K% drag dominates).
This is expected — we care about relative differences, not the sign.

Two population approaches:

  Approach A — graduates only (selection-biased):
    Pair each player's last pre-debut MiLB season per level with their
    first-year MLB Skill_PPPA (PA >= 100). Conditioned on graduation;
    inflates lower-level ratios because only elite prospects graduate
    from Rookie/A ball.

  Approach B — full population, zero for non-graduates:
    All MiLB players except current active prospects (not yet had time
    to graduate). Players who never reached MLB get MLB_Skill = 0.
    Removes selection bias; answers "expected MLB value per unit of
    MiLB production" — what the level discount actually needs to capture.

Key fix: join on MLBAM_ID (milb_hitting) <-> MLBAMID (hist_mlb_data).
The PlayerId columns use different ID systems across files.

Outputs printed to console; no CSV written (research only).
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

CURRENT = {"AAA": 1.00, "AA": 0.87, "A+": 0.80, "A": 0.80, "R": 0.75}
OLD     = {"AAA": 1.00, "AA": 0.84, "A+": 0.71, "A": 0.57, "R": 0.39}
LEVELS  = ["AAA", "AA", "A+", "A", "R"]

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
mlb  = pd.read_csv(DATA / "historical" / "hist_mlb_data.csv")
milb = pd.read_csv(DATA / "api" / "milb_hitting.csv")
prospects = pd.read_csv(DATA / "rankings" / "prospect_scores.csv")

def skill_pppa(df):
    pa = df["PA"]
    return (
        -2.0 * (df["SO"] / pa)
        + 4.0 * (df["HR"] / pa)
        + 3.0 * (df["SB"] / pa)
        - 1.5 * (df["CS"] / pa)
    )

# ---------------------------------------------------------------------------
# Build mlb_debut — keyed on MLBAMID (correct MLBAM ID column in hist_mlb)
# ---------------------------------------------------------------------------
mlb_valid = mlb[mlb["PA"] >= 100].copy()
mlb_valid["Skill_PPPA"] = skill_pppa(mlb_valid)
mlb_debut = (
    mlb_valid.sort_values("Season")
    .groupby("MLBAMID").first()
    .reset_index()
    [["MLBAMID", "Skill_PPPA", "Season", "PA"]]
)
mlb_debut.columns = ["MLBAM_ID", "MLB_Skill", "MLB_Season", "MLB_PA"]
# Set of MLBAM IDs that graduated (PA >= 100 in MLB)
grad_mlbam_ids = set(mlb_debut["MLBAM_ID"].astype(int))

milb["Skill_PPPA"] = skill_pppa(milb)

# ---------------------------------------------------------------------------
# Approach A: graduates only
# milb_hitting.MLBAM_ID -> mlb_debut.MLBAM_ID (both are MLBAM IDs)
# ---------------------------------------------------------------------------
milb_pre = milb.merge(mlb_debut[["MLBAM_ID", "MLB_Season"]], on="MLBAM_ID")
milb_pre = milb_pre[(milb_pre["Season"] < milb_pre["MLB_Season"]) & (milb_pre["PA"] >= 50)]
milb_last = (
    milb_pre.sort_values("Season", ascending=False)
    .drop_duplicates(["MLBAM_ID", "Level"])
    .copy()
)
paired = milb_last.merge(
    mlb_debut[["MLBAM_ID", "MLB_Skill", "MLB_Season"]],
    on="MLBAM_ID", suffixes=("", "_debut")
)

# ---------------------------------------------------------------------------
# Approach B: full population, non-graduates get MLB_Skill = 0
# Exclude active prospects (haven't had time to graduate).
# Graduates identified by MLBAM_ID match, NOT by MLB_Skill sign.
# ---------------------------------------------------------------------------
active_fg = set(prospects["PlayerId"].dropna().astype(int))
milb_b = milb[~milb["PlayerId"].isin(active_fg)].copy()
milb_b = milb_b[milb_b["PA"] >= 50]
# Best (highest PA) season per player-level
milb_best = (
    milb_b.sort_values("PA", ascending=False)
    .drop_duplicates(["MLBAM_ID", "Level"])
    .copy()
)
milb_best = milb_best.merge(
    mlb_debut[["MLBAM_ID", "MLB_Skill"]],
    on="MLBAM_ID", how="left"
)
# graduated flag based on ID match, not sign of Skill_PPPA
milb_best["graduated"] = milb_best["MLBAM_ID"].isin(grad_mlbam_ids)
milb_best["MLB_Skill"] = milb_best["MLB_Skill"].fillna(0)  # non-graduates = 0

# ---------------------------------------------------------------------------
# 1. Approach A — correlation by level
# ---------------------------------------------------------------------------
print("=" * 65)
print("1. Approach A (graduates only): MiLB -> MLB Skill_PPPA correlation")
print("   Skill_PPPA = -2*K% + 4*HR/PA + 3*SB/PA - 1.5*CS/PA")
print("=" * 65)
for lvl in LEVELS:
    g = paired[paired["Level"] == lvl].dropna(subset=["Skill_PPPA", "MLB_Skill"])
    if len(g) < 20:
        print(f"  {lvl:4s}  N<20, skipped")
        continue
    r, p = stats.pearsonr(g["Skill_PPPA"], g["MLB_Skill"])
    print(f"  {lvl:4s}  N={len(g):5d}  r={r:.3f}  p={p:.4f}")

# ---------------------------------------------------------------------------
# 2. Approach B — full population ratios
# ---------------------------------------------------------------------------
print()
print("=" * 65)
print("2. Approach B (full population, non-graduates=0)")
print("   ratio = mean(MLB_Skill) / mean(MiLB_Skill), normalized AAA=1.0")
print("=" * 65)
n_active = len(active_fg)
n_total  = milb_best["MLBAM_ID"].nunique()
n_grad   = milb_best["graduated"].sum()
print(f"  Active prospects excluded: {n_active:,}")
print(f"  Unique non-prospect players: {n_total:,}  |  graduates: {n_grad:,}")
print()

ratios_b = {}
for lvl in LEVELS:
    g = milb_best[milb_best["Level"] == lvl]
    if len(g) >= 20 and g["Skill_PPPA"].mean() != 0:
        ratios_b[lvl] = g["MLB_Skill"].mean() / g["Skill_PPPA"].mean()

if "AAA" in ratios_b:
    aaa_r = ratios_b["AAA"]
    print(f"  {'Level':4s}  {'N':>6}  {'graduates':>10}  {'grad%':>6}  "
          f"{'norm':>6}  {'current':>8}  {'old':>6}  {'vs_curr':>8}")
    for lvl in LEVELS:
        if lvl not in ratios_b:
            continue
        g    = milb_best[milb_best["Level"] == lvl]
        n    = len(g)
        ng   = g["graduated"].sum()
        norm = ratios_b[lvl] / aaa_r
        curr = CURRENT.get(lvl, float("nan"))
        old  = OLD.get(lvl, float("nan"))
        print(f"  {lvl:4s}  {n:6d}  {ng:10d}  {ng/n:6.1%}  "
              f"{norm:6.3f}  {curr:8.2f}  {old:6.2f}  {norm-curr:+8.3f}")

# ---------------------------------------------------------------------------
# 3. Side-by-side: Approach A vs. B vs. current
# ---------------------------------------------------------------------------
print()
print("=" * 65)
print("3. Approach A vs. B vs. current model (both normalized AAA=1.0)")
print("=" * 65)
ratios_a = {}
for lvl in LEVELS:
    g = paired[paired["Level"] == lvl].dropna(subset=["Skill_PPPA", "MLB_Skill"])
    if len(g) >= 10 and g["Skill_PPPA"].mean() != 0:
        ratios_a[lvl] = g["MLB_Skill"].mean() / g["Skill_PPPA"].mean()

if "AAA" in ratios_a and "AAA" in ratios_b:
    aaa_a = ratios_a["AAA"]
    aaa_b = ratios_b["AAA"]
    print(f"  {'Level':4s}  {'Approach A':>11}  {'Approach B':>11}  {'current':>8}  {'old':>6}")
    for lvl in LEVELS:
        na = f"{ratios_a[lvl]/aaa_a:.3f}" if lvl in ratios_a else "    N/A"
        nb = f"{ratios_b[lvl]/aaa_b:.3f}" if lvl in ratios_b else "    N/A"
        curr = CURRENT.get(lvl, float("nan"))
        old  = OLD.get(lvl, float("nan"))
        print(f"  {lvl:4s}  {na:>11}  {nb:>11}  {curr:8.2f}  {old:6.2f}")

# ---------------------------------------------------------------------------
# 4. Component breakdown (Approach A, graduates-only)
# K ratio > 1 means players strike out more in MLB (expected)
# HR ratio: varies — selection bias at lower levels
# netSB ratio: most stable across levels
# ---------------------------------------------------------------------------
print()
print("=" * 65)
print("4. Component breakdown by level (Approach A — graduates only)")
print("   K ratio > 1 = more K% in MLB; HR/netSB ratios = MLB vs. MiLB mean")
print("=" * 65)
print(f"  {'Level':4s}  {'N':>5}  {'K_ratio':>8}  {'HR_ratio':>9}  {'netSB_ratio':>12}")
for lvl in LEVELS:
    g = paired[paired["Level"] == lvl].dropna(subset=["Skill_PPPA", "MLB_Skill"])
    if len(g) < 10:
        continue
    mlb_full = mlb[mlb["MLBAMID"].isin(g["MLBAM_ID"])].copy()
    mlb_full = mlb_full.merge(mlb_debut[["MLBAM_ID", "MLB_Season"]], left_on="MLBAMID", right_on="MLBAM_ID")
    mlb_full = mlb_full[mlb_full["Season"] == mlb_full["MLB_Season"]].drop_duplicates("MLBAMID")

    m_k     = (g["SO"] / g["PA"]).mean()
    mlb_k   = (mlb_full["SO"] / mlb_full["PA"]).mean() if len(mlb_full) else float("nan")
    m_hr    = (g["HR"] / g["PA"]).mean()
    mlb_hr  = (mlb_full["HR"] / mlb_full["PA"]).mean() if len(mlb_full) else float("nan")
    m_nsb   = ((g["SB"] - 0.5*g["CS"]) / g["PA"]).mean()
    mlb_nsb = ((mlb_full["SB"] - 0.5*mlb_full["CS"]) / mlb_full["PA"]).mean() if len(mlb_full) else float("nan")

    print(f"  {lvl:4s}  {len(g):5d}  "
          f"{mlb_k/m_k if m_k>0 else float('nan'):8.3f}  "
          f"{mlb_hr/m_hr if m_hr>0 else float('nan'):9.3f}  "
          f"{mlb_nsb/m_nsb if m_nsb>0 else float('nan'):12.3f}")
