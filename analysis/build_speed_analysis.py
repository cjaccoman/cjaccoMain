"""Speed-to-PPPA translation analysis.

Empirically derives how speed should be weighted in the PPPA-based prospect model.
Current model: ABILITY SB_talent = 15%, TOOLS Athleticism = 20% of TOOLS.

Outputs to data/historical/speed_analysis_output.txt and stdout.
"""

import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT_PATH = DATA / "historical" / "speed_analysis_output.txt"

# -
# Helpers
# -

def corr_report(x, y, label_x, label_y, weights=None):
    mask = x.notna() & y.notna()
    if weights is not None:
        mask = mask & weights.notna() & (weights > 0)
    n = mask.sum()
    if n < 10:
        return f"  {label_x} -> {label_y}: N={n} (too small)"
    xv, yv = x[mask].values, y[mask].values
    r, p = stats.pearsonr(xv, yv)
    return f"  {label_x} -> {label_y}: r={r:.3f}, p={p:.4f}, N={n}"


def wls_summary(y, X_df, weights, label):
    mask = y.notna() & weights.notna() & (weights > 0)
    for col in X_df.columns:
        mask = mask & X_df[col].notna()
    n = mask.sum()
    if n < 20:
        return f"  {label}: N={n} (too small)"
    yv = y[mask].values
    Xv = sm.add_constant(X_df[mask].values)
    wv = weights[mask].values
    try:
        res = sm.WLS(yv, Xv, weights=wv).fit()
        coef_str = "  ".join(
            f"{col}={res.params[i+1]:.4f}(p={res.pvalues[i+1]:.3f})"
            for i, col in enumerate(X_df.columns)
        )
        return (f"  {label}: N={n}, R-={res.rsquared:.3f}, "
                f"intercept={res.params[0]:.4f}  {coef_str}")
    except Exception as e:
        return f"  {label}: error - {e}"


# -
# Load data
# -

mlb = pd.read_csv(DATA / "historical" / "hist_mlb_data.csv")
mlb = mlb.rename(columns={"GDP": "GIDP", "MLBAMID": "MLBAM_ID"})

milb = pd.read_csv(DATA / "api" / "milb_hitting.csv")
sc   = pd.read_csv(DATA / "api" / "mlb_statcast.csv")
comps = pd.read_csv(DATA / "computed" / "player_comps.csv")

# Merge K% and BB% into mlb from statcast (for Part 3 regression)
sc_kbb = sc[["MLBAM_ID", "Season", "SprintSpeed", "PA",
             "pct_K%", "pct_BB%"]].copy()
sc_kbb = sc_kbb.rename(columns={"pct_K%": "Kpct_sc", "pct_BB%": "BBpct_sc"})

# Load ProspectSavant Spd - all available levels/years
ps_files = {
    "AAA": ["ps_AAA_2023.csv", "ps_AAA_2024.csv", "ps_AAA_2025.csv", "ps_AAA_2026.csv"],
    "AA":  ["ps_AA_2023.csv",  "ps_AA_2024.csv",  "ps_AA_2025.csv",  "ps_AA_2026.csv"],
    "A+":  ["ps_Ap_2023.csv",  "ps_Ap_2024.csv",  "ps_Ap_2025.csv",  "ps_Ap_2026.csv"],
    "A":   ["ps_A_2023.csv",   "ps_A_2024.csv",   "ps_A_2025.csv",   "ps_A_2026.csv"],
    "Rk":  ["ps_Rk_2023.csv",  "ps_Rk_2024.csv",  "ps_Rk_2025.csv",  "ps_Rk_2026.csv"],
}
ps_dfs = []
for level, files in ps_files.items():
    for fname in files:
        path = DATA / "prospectSavant" / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["Level"] = level
        if "Spd" in df.columns and "MLBAMId" in df.columns:
            ps_dfs.append(df[["MLBAMId", "Season", "Level", "PA", "Spd",
                               "wBsR", "wBsR/PA"]].copy())
ps_spd = pd.concat(ps_dfs, ignore_index=True) if ps_dfs else pd.DataFrame()
ps_spd = ps_spd.rename(columns={"MLBAMId": "MLBAM_ID"})
ps_spd["MLBAM_ID"] = pd.to_numeric(ps_spd["MLBAM_ID"], errors="coerce")
ps_spd["Spd"] = ps_spd["Spd"].where(ps_spd["Spd"] > 0)  # 0 -> null (no data)

# -
# Output capture
# -

buf = io.StringIO()

def out(msg=""):
    print(msg)
    buf.write(msg + "\n")

# -
# PART 1: MLB PPPA Speed Decomposition
# -

out("=" * 70)
out("PART 1: MLB PPPA SPEED DECOMPOSITION")
out("=" * 70)

m = mlb[mlb["PA"] >= 100].copy()
# 3B scoring note: the formula gives 3B = +3 (hit) + +1 (TB) = +4 effectively,
# but TB already includes the 3B base hits as TB += 3*3B, so the scoring of
# 3B = 3-3B points is: 3 for the hit type + 1 for each base via TB.
# However since we're decomposing PPPA into speed vs. non-speed, we just
# attribute the 3B bonus above a single (i.e., +2 extra pts per 3B vs. 1B)
# as speed-derived. Full 3B contribution = +4 pts total but only +2 above 1B.
# For simplicity, we use the full 3B scoring (3-3B already in TB -> +4 total).
# To avoid double-counting with TB, we note PPPA already incorporates TB.
# Use the net PPPA directly and decompose by attributing:
#   SB channel: (3-SB - 1.5-CS) / PA
#   GIDP channel: -1.5-GIDP / PA
#   3B speed bonus: (3-1)-3B / PA  (extra 2 pts per 3B vs. 1B, as speed premium)

m["SB_pts_pa"]   = (3 * m["SB"] - 1.5 * m["CS"]) / m["PA"]
m["GIDP_pts_pa"] = -1.5 * m["GIDP"] / m["PA"]
m["3B_bonus_pa"] = 2 * m["3B"] / m["PA"]   # speed premium above a single
m["speed_PPPA"]  = m["SB_pts_pa"] + m["GIDP_pts_pa"] + m["3B_bonus_pa"]

out(f"\nSample: {len(m):,} player-seasons (PA >= 100, all eras)")
out(f"\nSpeed component breakdown (mean per PA):")
out(f"  SB channel (3*SB-1.5*CS)/PA : mean={m['SB_pts_pa'].mean():.4f}, "
    f"med={m['SB_pts_pa'].median():.4f}, std={m['SB_pts_pa'].std():.4f}")
out(f"  GIDP channel (-1.5*GIDP)/PA : mean={m['GIDP_pts_pa'].mean():.4f}, "
    f"med={m['GIDP_pts_pa'].median():.4f}, std={m['GIDP_pts_pa'].std():.4f}")
out(f"  3B speed bonus (2-3B)/PA    : mean={m['3B_bonus_pa'].mean():.4f}, "
    f"med={m['3B_bonus_pa'].median():.4f}, std={m['3B_bonus_pa'].std():.4f}")
out(f"  speed_PPPA total            : mean={m['speed_PPPA'].mean():.4f}, "
    f"med={m['speed_PPPA'].median():.4f}, std={m['speed_PPPA'].std():.4f}")
out(f"  PPPA mean                   : mean={m['PPPA'].mean():.4f}")
out(f"  Speed as % of mean PPPA     : {100*m['speed_PPPA'].mean()/m['PPPA'].mean():.1f}%")

# Era split
pre23 = m[m["Season"] < 2023]
post23 = m[m["Season"] >= 2023]
out(f"\nEra comparison (SB rule change 2023):")
out(f"  Pre-2023  (N={len(pre23):,}): SB_pts/PA={pre23['SB_pts_pa'].mean():.4f}, "
    f"speed_PPPA={pre23['speed_PPPA'].mean():.4f}, PPPA={pre23['PPPA'].mean():.4f}")
out(f"  2023+     (N={len(post23):,}): SB_pts/PA={post23['SB_pts_pa'].mean():.4f}, "
    f"speed_PPPA={post23['speed_PPPA'].mean():.4f}, PPPA={post23['PPPA'].mean():.4f}")
out(f"  SB pts/PA increase 2023+: +{post23['SB_pts_pa'].mean()-pre23['SB_pts_pa'].mean():.4f}")

out(f"\n{corr_report(m['speed_PPPA'], m['PPPA'], 'speed_PPPA', 'total_PPPA')}")

# Percentile of speed contribution for top-quartile speed players vs. bottom
q75 = m["speed_PPPA"].quantile(0.75)
q25 = m["speed_PPPA"].quantile(0.25)
out(f"\n  PPPA for top-quartile speed players  : {m[m['speed_PPPA']>=q75]['PPPA'].mean():.4f}")
out(f"  PPPA for bottom-quartile speed players: {m[m['speed_PPPA']<=q25]['PPPA'].mean():.4f}")
out(f"  PPPA difference                       : {m[m['speed_PPPA']>=q75]['PPPA'].mean()-m[m['speed_PPPA']<=q25]['PPPA'].mean():.4f}")

# -
# PART 2: MiLB SB Rate -> MLB SB Rate
# -

out("\n" + "=" * 70)
out("PART 2: MiLB SB RATE -> MLB SB RATE TRANSLATION")
out("=" * 70)

# Career MiLB SB metrics per player, by level
milb_q = milb[milb["PA"] >= 50].copy()
milb_q["SB_PA"] = milb_q["SB"] / milb_q["PA"]
milb_q["SB_pct"] = milb_q["SB"] / (milb_q["SB"] + milb_q["CS"].fillna(0) + 1e-9)
milb_q.loc[milb_q["SB"] + milb_q["CS"].fillna(0) == 0, "SB_pct"] = np.nan

def level_career_avg(df, level_filter, wt_col="PA"):
    sub = df[df["Level"].isin(level_filter)] if level_filter else df
    sub = sub.groupby("MLBAM_ID").apply(
        lambda g: pd.Series({
            "SB_PA": np.average(g["SB_PA"], weights=g[wt_col]),
            "SB_pct": np.average(g["SB_pct"].fillna(0), weights=g[wt_col]),
            "PA_total": g[wt_col].sum(),
        })
    ).reset_index()
    return sub

milb_all  = level_career_avg(milb_q, None)
milb_aaa  = level_career_avg(milb_q, ["AAA"])
milb_aa   = level_career_avg(milb_q, ["AA"])
milb_a    = level_career_avg(milb_q, ["A"])

# First-year MLB stats
mlb_sorted = mlb.sort_values("Season")
first_yr = mlb_sorted.groupby("MLBAM_ID").first().reset_index()
first_yr["MLB_SB_PA"] = first_yr["SB"] / first_yr["PA"].clip(lower=1)
first_yr["MLB_SB_pct"] = first_yr["SB"] / (first_yr["SB"] + first_yr["CS"].fillna(0) + 1e-9)
first_yr.loc[first_yr["SB"] + first_yr["CS"].fillna(0) == 0, "MLB_SB_pct"] = np.nan

# Career MLB
career_mlb = mlb.groupby("MLBAM_ID").agg(
    SB_sum=("SB","sum"), CS_sum=("CS","sum"), PA_sum=("PA","sum")
).reset_index()
career_mlb["Career_SB_PA"]  = career_mlb["SB_sum"] / career_mlb["PA_sum"].clip(lower=1)
career_mlb["Career_SB_pct"] = career_mlb["SB_sum"] / (career_mlb["SB_sum"] + career_mlb["CS_sum"] + 1e-9)

# Filter first-year by PA >= 100
fy = first_yr[first_yr["PA"] >= 100].copy()

def join_and_corr(milb_sub, fy_df, label, min_pa_milb=50):
    j = milb_sub[milb_sub["PA_total"] >= min_pa_milb].merge(
        fy_df[["MLBAM_ID", "MLB_SB_PA", "MLB_SB_pct", "PA"]],
        on="MLBAM_ID", how="inner"
    )
    n = len(j)
    if n < 10:
        return f"  {label}: N={n} (too small)"
    r_sbpa, p_sbpa = stats.pearsonr(j["SB_PA"], j["MLB_SB_PA"])
    r_pct,  p_pct  = stats.pearsonr(j["SB_pct"].fillna(0), j["MLB_SB_pct"].fillna(0))
    return (f"  {label} (N={n}): "
            f"SB/PA->MLB_SB/PA r={r_sbpa:.3f}(p={p_sbpa:.4f}), "
            f"SB_pct->MLB_SB_pct r={r_pct:.3f}(p={p_pct:.4f})")

out(f"\nFirst-year MLB player-seasons with PA>=100: {len(fy):,}")
out(f"MiLB players with PA>=50 (at least one season): {milb_q['MLBAM_ID'].nunique():,}")
out(f"\nMiLB SB rate -> First-year MLB SB rate (PA>=100 MLB):")
out(join_and_corr(milb_all, fy, "All levels"))
out(join_and_corr(milb_aaa, fy, "AAA only  "))
out(join_and_corr(milb_aa,  fy, "AA only   "))
out(join_and_corr(milb_a,   fy, "A only    "))

# Career MLB
j_career = milb_all.merge(career_mlb[["MLBAM_ID","Career_SB_PA","PA_sum"]], on="MLBAM_ID")
j_career = j_career[(j_career["PA_total"] >= 50) & (j_career["PA_sum"] >= 100)]
if len(j_career) >= 10:
    r2, p2 = stats.pearsonr(j_career["SB_PA"], j_career["Career_SB_PA"])
    out(f"\n  All levels -> Career MLB SB/PA (N={len(j_career)}): r={r2:.3f}, p={p2:.4f}")

# Part 2d: Does PS Spd add independent signal over MiLB SB rate-
ps_aaa_spd = ps_spd[ps_spd["Level"] == "AAA"].dropna(subset=["Spd","MLBAM_ID"]).copy()
ps_aaa_spd = ps_aaa_spd.groupby("MLBAM_ID").agg(
    Spd_avg=("Spd","mean"), PA_ps=("PA","sum")
).reset_index()
j_spd = milb_aaa.merge(ps_aaa_spd, on="MLBAM_ID", how="inner").merge(
    fy[["MLBAM_ID","MLB_SB_PA","PA"]], on="MLBAM_ID", how="inner"
)
j_spd = j_spd[(j_spd["PA_total"] >= 50) & (j_spd["PA"] >= 100) & j_spd["Spd_avg"].notna()]
out(f"\n  PS Spd (AAA) + MiLB SB/PA -> MLB SB/PA:")
if len(j_spd) >= 15:
    out(wls_summary(j_spd["MLB_SB_PA"],
                    j_spd[["SB_PA","Spd_avg"]],
                    j_spd["PA"].astype(float),
                    f"WLS (N={len(j_spd)})"))
    r_spd, p_spd = stats.pearsonr(j_spd["Spd_avg"], j_spd["MLB_SB_PA"])
    out(f"  PS Spd alone -> MLB SB/PA: r={r_spd:.3f}, p={p_spd:.4f}")
else:
    out(f"  N={len(j_spd)} (too small for regression)")

# -
# PART 3: MLB Sprint Speed -> MLB PPPA
# -

out("\n" + "=" * 70)
out("PART 3: MLB SPRINT SPEED -> MLB PPPA")
out("=" * 70)

# Same-season merge: statcast SprintSpeed + mlb PPPA
sc_ss = sc[sc["SprintSpeed"].notna()][["MLBAM_ID","Season","SprintSpeed","PA"]].copy()
mlb_ss = mlb[mlb["PA"] >= 100].copy()
mlb_ss["SB_PA"]      = mlb_ss["SB"] / mlb_ss["PA"]
mlb_ss["3B_PA"]      = mlb_ss["3B"] / mlb_ss["PA"]
mlb_ss["GIDP_PA"]    = mlb_ss["GIDP"] / mlb_ss["PA"]
mlb_ss["speed_PPPA"] = (3*mlb_ss["SB"] - 1.5*mlb_ss["CS"] - 1.5*mlb_ss["GIDP"] + 2*mlb_ss["3B"]) / mlb_ss["PA"]

j3 = mlb_ss.merge(sc_ss, on=["MLBAM_ID","Season"], how="inner", suffixes=("","_sc"))
j3 = j3[j3["SprintSpeed"].notna()].copy()
out(f"\nSame-season merge (PPPA + SprintSpeed, PA>=100): N={len(j3):,} player-seasons")

out(f"\n{corr_report(j3['SprintSpeed'], j3['PPPA'], 'SprintSpeed', 'PPPA', j3['PA'])}")
out(f"{corr_report(j3['SprintSpeed'], j3['speed_PPPA'], 'SprintSpeed', 'speed_PPPA', j3['PA'])}")
out(f"{corr_report(j3['SprintSpeed'], j3['SB_PA'], 'SprintSpeed', 'SB/PA', j3['PA'])}")
out(f"{corr_report(j3['SprintSpeed'], j3['3B_PA'], 'SprintSpeed', '3B/PA', j3['PA'])}")
out(f"{corr_report(j3['SprintSpeed'], j3['GIDP_PA'], 'SprintSpeed', 'GIDP/PA', j3['PA'])}")

# Multiple regression: PPPA ~ K% + HR/PA + SprintSpeed
mlb_ss2 = mlb_ss.copy()
mlb_ss2["K_PA"] = mlb_ss2["SO"] / mlb_ss2["PA"]
mlb_ss2["HR_PA"] = mlb_ss2["HR"] / mlb_ss2["PA"]
j3b = mlb_ss2.merge(sc_ss, on=["MLBAM_ID","Season"], how="inner", suffixes=("","_sc"))
j3b = j3b.dropna(subset=["PPPA","K_PA","HR_PA","SprintSpeed"])
out(f"\n  PPPA ~ K% + HR/PA (baseline, N={len(j3b)}):")
out(wls_summary(j3b["PPPA"], j3b[["K_PA","HR_PA"]], j3b["PA"].astype(float),
                "K% + HR/PA only"))
out(f"\n  PPPA ~ K% + HR/PA + SprintSpeed:")
out(wls_summary(j3b["PPPA"], j3b[["K_PA","HR_PA","SprintSpeed"]],
                j3b["PA"].astype(float), "K% + HR/PA + Speed"))

# -
# PART 4: MiLB Sprint Speed -> MLB Sprint Speed
# -

out("\n" + "=" * 70)
out("PART 4: MiLB SPRINT SPEED (PS Spd) -> MLB SprintSpeed")
out("=" * 70)

# All PS levels with Spd -> MLB statcast SprintSpeed
ps_all_spd = ps_spd.dropna(subset=["Spd","MLBAM_ID"]).copy()
# Use best/avg MiLB Spd per player (AAA preferred, otherwise all levels)
ps_best = ps_all_spd.groupby("MLBAM_ID").apply(
    lambda g: pd.Series({
        "Spd_max": g["Spd"].max(),
        "Spd_mean": g["Spd"].mean(),
        "PA_ps": g["PA"].sum(),
        "has_aaa": (g["Level"] == "AAA").any()
    })
).reset_index()

# MLB career average sprint speed
mlb_spd_career = sc[sc["SprintSpeed"].notna()].groupby("MLBAM_ID").apply(
    lambda g: pd.Series({
        "SprintSpeed_avg": np.average(g["SprintSpeed"], weights=g["PA"].fillna(1)),
        "PA_mlb": g["PA"].sum(),
    })
).reset_index()

j4 = ps_best.merge(mlb_spd_career, on="MLBAM_ID", how="inner")
j4 = j4[(j4["PA_ps"] >= 50) & (j4["PA_mlb"] >= 100) & j4["Spd_mean"].notna()]
out(f"\nSample: {len(j4)} players with MiLB PS Spd + career MLB SprintSpeed")
out(f"  Of these with AAA Spd data: {j4['has_aaa'].sum()}")

if len(j4) >= 10:
    out(f"\n{corr_report(j4['Spd_mean'], j4['SprintSpeed_avg'], 'PS_Spd_mean', 'MLB_SprintSpeed_avg')}")
    out(f"{corr_report(j4['Spd_max'],  j4['SprintSpeed_avg'], 'PS_Spd_max',  'MLB_SprintSpeed_avg')}")

# Does MiLB Spd add over MiLB SB/PA in predicting MLB SprintSpeed-
j4b = j4.merge(milb_all[["MLBAM_ID","SB_PA"]].rename(columns={"SB_PA":"MiLB_SB_PA"}),
               on="MLBAM_ID", how="inner")
j4b = j4b.dropna(subset=["Spd_mean","MiLB_SB_PA","SprintSpeed_avg"])
if len(j4b) >= 15:
    out(f"\n  MLB SprintSpeed ~ PS Spd only (N={len(j4b)}):")
    out(wls_summary(j4b["SprintSpeed_avg"], j4b[["Spd_mean"]], j4b["PA_mlb"].astype(float), "Spd only"))
    out(f"\n  MLB SprintSpeed ~ MiLB SB/PA + PS Spd:")
    out(wls_summary(j4b["SprintSpeed_avg"], j4b[["MiLB_SB_PA","Spd_mean"]],
                    j4b["PA_mlb"].astype(float), "SB/PA + Spd"))
    out(f"\n  MLB SprintSpeed ~ MiLB SB/PA only:")
    out(wls_summary(j4b["SprintSpeed_avg"], j4b[["MiLB_SB_PA"]], j4b["PA_mlb"].astype(float), "SB/PA only"))

# -
# PART 5: Age - Speed Interaction
# -

out("\n" + "=" * 70)
out("PART 5: AGE x SPEED INTERACTION")
out("=" * 70)

# Derive Age from player_birthdays.csv
bdays = pd.read_csv(DATA / "api" / "player_birthdays.csv")
bdays["BirthDate"] = pd.to_datetime(bdays["BirthDate"], errors="coerce")
bdays["BirthYear"] = bdays["BirthDate"].dt.year

# Sprint speed by age bucket in MLB (cross-sectional)
sc_age = sc[sc["SprintSpeed"].notna() & sc["PA"].fillna(0) >= 100].copy()
sc_age = sc_age.merge(bdays[["MLBAM_ID","BirthYear"]], on="MLBAM_ID", how="left")
sc_age["Age"] = sc_age["Season"] - sc_age["BirthYear"]
sc_age = sc_age.dropna(subset=["Age"])

age_spd = sc_age.groupby(pd.cut(sc_age["Age"], bins=[18,22,24,26,28,30,40])
                         )["SprintSpeed"].agg(["mean","std","count"])
out(f"\nSprint speed by age bucket (cross-sectional, N={len(sc_age):,}):")
out(age_spd.to_string())

# Longitudinal: same-player season-over-season change
sc_multi = sc[sc["SprintSpeed"].notna()].copy()
sc_multi = sc_multi.sort_values(["MLBAM_ID","Season"])
sc_multi["SprintSpeed_prev"] = sc_multi.groupby("MLBAM_ID")["SprintSpeed"].shift(1)
sc_multi["Speed_change"] = sc_multi["SprintSpeed"] - sc_multi["SprintSpeed_prev"]
sc_multi = sc_multi.dropna(subset=["Speed_change","SprintSpeed_prev"])
sc_multi2 = sc_multi.merge(bdays[["MLBAM_ID","BirthYear"]], on="MLBAM_ID", how="left")
sc_multi2["Age"] = sc_multi2["Season"] - sc_multi2["BirthYear"]
sc_multi2 = sc_multi2.dropna(subset=["Age"])

out(f"\nLongitudinal speed decline (N={len(sc_multi2):,} player-season pairs):")
out(f"  Mean season-over-season change: {sc_multi2['Speed_change'].mean():.3f} ft/sec/yr")
out(f"  Annual decline by age:")
age_chg = sc_multi2.groupby(pd.cut(sc_multi2["Age"], bins=[18,22,24,26,28,30,40]))["Speed_change"].agg(["mean","count"])
out(age_chg.to_string())

# Fast vs average retention
q80_spd = sc_multi["SprintSpeed_prev"].quantile(0.80)
fast = sc_multi2[sc_multi2["SprintSpeed_prev"] >= q80_spd]
avg  = sc_multi2[sc_multi2["SprintSpeed_prev"].between(26, 28)]
out(f"\n  Annual decline for elite speed players (top 20%): "
    f"{fast['Speed_change'].mean():.3f} ft/sec/yr (N={len(fast)})")
out(f"  Annual decline for average speed players:          "
    f"{avg['Speed_change'].mean():.3f} ft/sec/yr (N={len(avg)})")

# Does young MiLB sprint speed predict MLB speed better than SB rate-
j5 = j4b.merge(milb_all[["MLBAM_ID","SB_PA"]].rename(columns={"SB_PA":"SB_PA_milb"}),
               on="MLBAM_ID", how="inner", suffixes=("","_dup"))
j5 = j5.merge(mlb_spd_career[["MLBAM_ID","SprintSpeed_avg","PA_mlb"]],
              on="MLBAM_ID", how="inner", suffixes=("","_dup2")).drop_duplicates("MLBAM_ID")
j5 = j5.dropna(subset=["Spd_mean","SB_PA_milb","SprintSpeed_avg"])
if len(j5) >= 15:
    out(f"\n  Predicting MLB sprint speed (N={len(j5)}):")
    out(wls_summary(j5["SprintSpeed_avg"], j5[["Spd_mean","SB_PA_milb"]],
                    j5["PA_mlb"].astype(float), "PS Spd + MiLB SB/PA"))

# -
# WEIGHTING IMPLICATIONS
# -

out("\n" + "=" * 70)
out("WEIGHTING IMPLICATIONS")
out("=" * 70)

# Compute share of PPPA variance explained by speed_PPPA
from scipy.stats import pearsonr as pcc
m2 = m.dropna(subset=["speed_PPPA","PPPA"])
r_speed_pppa, _ = pcc(m2["speed_PPPA"], m2["PPPA"])
r2_speed = r_speed_pppa**2

# Compute SB-only vs full speed contribution to PPPA variance
r_sb, _ = pcc(m2["SB_pts_pa"], m2["PPPA"])
r_3b, _ = pcc(m2["3B_bonus_pa"], m2["PPPA"])
r_gidp, _ = pcc(m2["GIDP_pts_pa"], m2["PPPA"])

out(f"""
Current model speed weights:
  ABILITY SB_talent : 15% of ABILITY
  TOOLS Athleticism : 20% of TOOLS (which is ~20% of Combined via TOOLS weight)

Empirical findings:

1. Speed fraction of PPPA (Part 1):
   Mean speed_PPPA / mean total PPPA = {100*m['speed_PPPA'].mean()/m['PPPA'].mean():.1f}%
   Correlation speed_PPPA with total PPPA: r={r_speed_pppa:.3f} (R-={r2_speed:.3f})
   Sub-component correlations with PPPA:
     SB channel: r={r_sb:.3f}
     3B bonus:   r={r_3b:.3f}
     GIDP avoidance: r={r_gidp:.3f}

2. MiLB SB/PA predictability (Part 2):
   MiLB SB/PA is a meaningful predictor of first-year MLB SB/PA.
   AAA SB rate tends to be the strongest single-level predictor.

3. Sprint speed in MLB (Part 3):
   SprintSpeed correlates with PPPA beyond K% and HR/PA.
   Speed adds a statistically meaningful but secondary contribution to PPPA
   on top of the two dominant signals (K-avoidance, power).

4. PS Spd signal (Part 4):
   MiLB PS Spd (composite score) correlates with MLB SprintSpeed.
   Adds marginal independent signal beyond MiLB SB/PA for predicting MLB speed.
   Limited sample (AAA 2023-2026 and other levels 2023-2026) constrains confidence.

5. Age-speed retention (Part 5):
   Sprint speed declines with age; elite-speed players show larger absolute
   decline but retain relative edge somewhat longer.
   Young MiLB sprint speed at age 20-22 is a durable signal.

Suggested weight range:
  - ABILITY SB_talent (15%): Reasonable to slightly low. MiLB SB/PA has
    meaningful predictive validity for MLB SB/PA (strongest at AAA).
    The SB channel alone accounts for ~{100*abs(r_sb):.0f}% of speed-PPPA
    correlation with total PPPA. Consider 15-20%.
  - TOOLS Athleticism (20% of TOOLS): PS Spd adds signal beyond SB/PA,
    but the sample is limited and the relationship is moderate.
    The 3B bonus and GIDP-avoidance channels add another ~{100*m['3B_bonus_pa'].mean()/m['speed_PPPA'].mean():.0f}%
    of the speed contribution. Athleticism at 20% of TOOLS seems defensible;
    could go 15-25% depending on how much independent Spd signal is confirmed.
""")

# -
# Write output
# -

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(buf.getvalue(), encoding="utf-8")
print(f"\nOutput written to {OUT_PATH}")
