"""SB/Speed MiLB -> MLB Translation Study

Research questions:
  1. How well does MiLB SB/PA predict first-year MLB SB/PA, by level?
  2. Has the 2023 rule change shifted the translation ratio?
  3. Do current level discount factors (AAA=1.00, AA=0.84, A+=0.71, A=0.57, R=0.39) hold?
  4. Does ProspectSavant Sprint Speed add signal over MiLB SB/PA?
  5. Where does Spd provide unique value vs. where is it redundant?

Outputs printed to console; no CSV written (research only).
"""

import glob
import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from scipy import stats
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
mlb  = pd.read_csv(DATA / "historical" / "hist_mlb_data.csv")
milb = pd.read_csv(DATA / "api" / "milb_hitting.csv")

# First MLB season with PA >= 100 per player
mlb_debut = (mlb[mlb["PA"] >= 100]
             .sort_values("Season")
             .groupby("PlayerId").first()
             .reset_index()[["PlayerId", "Name", "Season", "PA", "SB", "CS"]])
mlb_debut.columns = ["PlayerId", "Name", "MLB_Season", "MLB_PA", "MLB_SB", "MLB_CS"]
mlb_debut["MLB_SB_PA"] = mlb_debut["MLB_SB"] / mlb_debut["MLB_PA"]

# Pre-debut MiLB seasons (PA >= 50), last season per player-level
milb_pre = milb.merge(mlb_debut[["PlayerId", "MLB_Season"]], on="PlayerId")
milb_pre = milb_pre[(milb_pre["Season"] < milb_pre["MLB_Season"]) & (milb_pre["PA"] >= 50)]
milb_last = (milb_pre.sort_values("Season", ascending=False)
             .drop_duplicates(["PlayerId", "Level"])
             .copy())
milb_last["MiLB_SB_PA"] = milb_last["SB"] / milb_last["PA"]

paired = milb_last.merge(
    mlb_debut[["PlayerId", "MLB_SB_PA", "MLB_Season"]],
    on="PlayerId", suffixes=("", "_debut")
)

# ---------------------------------------------------------------------------
# ProspectSavant sprint speed (AAA 2023-2026)
# ---------------------------------------------------------------------------
ps_files = glob.glob(str(DATA / "prospectSavant" / "ps_AAA_*.csv"))
ps_aaa = pd.concat([pd.read_csv(f) for f in ps_files], ignore_index=True)
ps_aaa = ps_aaa[ps_aaa["Spd"].gt(0)].copy()
ps_aaa["Spd"] = pd.to_numeric(ps_aaa["Spd"], errors="coerce")
ps_aaa["MLBAMId"] = pd.to_numeric(ps_aaa["MLBAMId"], errors="coerce")
milb_ids = milb[["PlayerId", "MLBAM_ID"]].drop_duplicates()
ps_aaa = ps_aaa.merge(milb_ids, left_on="MLBAMId", right_on="MLBAM_ID", how="left")

ps_pre = ps_aaa.merge(mlb_debut[["PlayerId", "MLB_Season"]], on="PlayerId", how="inner")
ps_pre = ps_pre[ps_pre["Season"] < ps_pre["MLB_Season"]]
ps_latest = ps_pre.sort_values("Season", ascending=False).drop_duplicates("PlayerId")

speed_paired = ps_latest.merge(
    mlb_debut[["PlayerId", "MLB_SB_PA", "MLB_PA", "MLB_SB"]], on="PlayerId"
).dropna(subset=["Spd"])

aaa_milb = (milb_pre[milb_pre["Level"] == "AAA"]
            .sort_values("Season", ascending=False)
            .drop_duplicates("PlayerId")
            .assign(MiLB_SB_PA=lambda d: d["SB"] / d["PA"]))
both = speed_paired.merge(aaa_milb[["PlayerId", "MiLB_SB_PA"]], on="PlayerId")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def r2(y, yhat):
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


def ols_r2(X, y):
    coef, *_ = lstsq(X, y, rcond=None)
    return r2(y, X @ coef), coef


# ---------------------------------------------------------------------------
# 1. Correlation by level
# ---------------------------------------------------------------------------
print("=" * 60)
print("1. MiLB SB/PA -> first-year MLB SB/PA  (corr by level)")
print("=" * 60)
CURRENT = {"AAA": 1.00, "AA": 0.84, "A+": 0.71, "A": 0.57, "R": 0.39}
for lvl in ["AAA", "AA", "A+", "A", "R"]:
    g = paired[paired["Level"] == lvl]
    if len(g) < 20:
        continue
    r, p = stats.pearsonr(g["MiLB_SB_PA"], g["MLB_SB_PA"])
    print(f"  {lvl:4s}  N={len(g):5d}  r={r:.3f}  p={p:.4f}")

# ---------------------------------------------------------------------------
# 2. Level translation ratios vs. current discounts
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("2. SB translation ratios — all-time vs. 2023+ debuts")
print("   (ratio = mean MLB_SB_PA / mean MiLB_SB_PA, norm to AAA=1.0)")
print("=" * 60)
post23 = paired[paired["MLB_Season_debut"] >= 2023]

for label, df in [("All-time", paired), ("2023+ debuts", post23)]:
    ratios = {}
    for lvl in ["AAA", "AA", "A+", "A", "R"]:
        g = df[df["Level"] == lvl]
        if len(g) >= 10 and g["MiLB_SB_PA"].mean() > 0:
            ratios[lvl] = g["MLB_SB_PA"].mean() / g["MiLB_SB_PA"].mean()
    aaa_r = ratios.get("AAA", 1.0)
    print(f"\n  [{label}]")
    for lvl in ["AAA", "AA", "A+", "A", "R"]:
        if lvl not in ratios:
            continue
        n = df[df["Level"] == lvl].shape[0]
        norm = ratios[lvl] / aaa_r
        curr = CURRENT.get(lvl, "?")
        diff = norm - curr
        print(f"    {lvl:4s}  N={n:4d}  norm={norm:.3f}  current={curr:.2f}  diff={diff:+.3f}")

# ---------------------------------------------------------------------------
# 3. Pre/post 2023 — AAA deep dive
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("3. AAA: pre-2023 vs. 2023+ translation")
print("=" * 60)
aaa_p = paired[paired["Level"] == "AAA"]
for label, g in [("pre-2023", aaa_p[aaa_p["MLB_Season_debut"] < 2023]),
                 ("2023+",    aaa_p[aaa_p["MLB_Season_debut"] >= 2023])]:
    r, _ = stats.pearsonr(g["MiLB_SB_PA"], g["MLB_SB_PA"])
    ratio = g["MLB_SB_PA"].mean() / g["MiLB_SB_PA"].mean()
    print(f"  {label:<12}  N={len(g):3d}  r={r:.3f}  "
          f"MLB/MiLB={ratio:.3f}  "
          f"MiLB_mean={g['MiLB_SB_PA'].mean():.4f}  "
          f"MLB_mean={g['MLB_SB_PA'].mean():.4f}")

# ---------------------------------------------------------------------------
# 4. Sprint Speed incremental value
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("4. Sprint Speed incremental value over AAA MiLB SB/PA")
print(f"   (N={len(both)} players with both Spd and AAA pre-debut SB/PA)")
print("=" * 60)

y = both["MLB_SB_PA"].values
r_milb, _ = stats.pearsonr(both["MiLB_SB_PA"], y)
r_spd,  _ = stats.pearsonr(both["Spd"], y)

X1 = np.column_stack([np.ones(len(both)), both["MiLB_SB_PA"]])
X2 = np.column_stack([np.ones(len(both)), both["Spd"]])
X3 = np.column_stack([np.ones(len(both)), both["MiLB_SB_PA"], both["Spd"]])

r2_milb, c1 = ols_r2(X1, y)
r2_spd,  c2 = ols_r2(X2, y)
r2_comb, c3 = ols_r2(X3, y)

print(f"  MiLB_SB_PA alone:          r={r_milb:.3f}  R²={r2_milb:.3f}")
print(f"  Spd alone:                 r={r_spd:.3f}  R²={r2_spd:.3f}")
print(f"  MiLB_SB_PA + Spd:                     R²={r2_comb:.3f}  (gain={r2_comb-r2_milb:+.3f})")
print(f"  Coefs: intercept={c3[0]:.4f}  MiLB_SB_PA={c3[1]:.3f}  Spd={c3[2]:.5f}")

# Spd value for low-PA players (< 200 career PA in MiLB)
low_pa = both[both["PA"] < 200]
hi_pa  = both[both["PA"] >= 200]
print(f"\n  Spd corr with MLB_SB_PA:")
print(f"    Low career PA (<200):  N={len(low_pa):3d}  r={low_pa['Spd'].corr(low_pa['MLB_SB_PA']):.3f}")
print(f"    High career PA (>=200): N={len(hi_pa):3d}  r={hi_pa['Spd'].corr(hi_pa['MLB_SB_PA']):.3f}")

# ---------------------------------------------------------------------------
# 5. Year-by-year SB/PA trends
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("5. Year-by-year mean SB/PA — pre-debut AAA vs. first-year MLB")
print("=" * 60)
aaa_trend = (milb_pre[milb_pre["Level"] == "AAA"]
             .assign(SB_PA=lambda d: d["SB"] / d["PA"])
             .groupby("Season")["SB_PA"].mean())
mlb_trend = mlb_debut.groupby("MLB_Season")["MLB_SB_PA"].mean()

print(f"  {'Season':>6}  {'AAA MiLB':>10}  {'1st-yr MLB':>10}  {'ratio':>6}")
for yr in range(2017, 2027):
    aaa_v = aaa_trend.get(yr, np.nan)
    mlb_v = mlb_trend.get(yr, np.nan)
    ratio = mlb_v / aaa_v if (aaa_v and mlb_v and aaa_v > 0) else np.nan
    print(f"  {yr:>6}  {aaa_v:>10.4f}  {mlb_v:>10.4f}  {ratio:>6.3f}")
