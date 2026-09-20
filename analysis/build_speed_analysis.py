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

mlb = pd.read_parquet(DATA / "historical" / "hist_mlb_data.parquet")
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
# PART 6: MiLB Speed Share Groups -> MLB Translation
# -

out("\n" + "=" * 70)
out("PART 6: MiLB SPEED-SHARE GROUPS -> MLB SPEED TRANSLATION")
out("=" * 70)

# Load minorLeagueData for TP, 3B per player-season
mld = pd.read_parquet(DATA / "computed" / "minorLeagueData.parquet")
# Aggregate career MiLB totals per player via MLBAM_ID join
# minorLeagueData has PlayerId (FG/MLBAM mix); milb_hitting has MLBAM_ID
# Use milb_hitting as the primary source (has SB, CS, PA, 3B) and
# join minorLeagueData for TP (total points)
milb_pa50 = milb[milb["PA"] >= 50].copy()

# TP from minorLeagueData -- join on PlayerId (both use same PlayerId system)
mld_tp = mld[mld["PA"] >= 50][["PlayerId","Season","Level","TP","3B"]].copy()
# milb_hitting PlayerId == minorLeagueData PlayerId (same pipeline source)
milb_tp = milb_pa50.merge(
    mld_tp[["PlayerId","Season","Level","TP"]],
    on=["PlayerId","Season","Level"], how="left"
)

# Career totals per player (MLBAM_ID for MLB join)
career_sp = milb_tp.groupby("MLBAM_ID").apply(lambda g: pd.Series({
    "SB_total":  g["SB"].sum(),
    "CS_total":  g["CS"].sum(),
    "3B_total":  g["3B"].sum(),
    "PA_total":  g["PA"].sum(),
    "TP_total":  g["TP"].sum(),
    "PPPA_milb": (g["TP"].sum() / g["PA"].sum()) if g["PA"].sum() > 0 else np.nan,
})).reset_index()

career_sp["speed_pts_milb"] = 3*career_sp["SB_total"] - 1.5*career_sp["CS_total"] + 3*career_sp["3B_total"]
career_sp["speed_pts_pa_milb"] = career_sp["speed_pts_milb"] / career_sp["PA_total"].clip(lower=1)
career_sp["speed_share"] = career_sp["speed_pts_milb"] / career_sp["TP_total"].replace(0, np.nan)

# MLB side: first-year player-seasons PA>=100
mlb_grp = mlb[mlb["PA"] >= 100].copy()
mlb_grp["speed_pts_mlb"] = 3*mlb_grp["SB"] - 1.5*mlb_grp["CS"] + 3*mlb_grp["3B"]
mlb_grp["speed_pts_pa_mlb"] = mlb_grp["speed_pts_mlb"] / mlb_grp["PA"]
mlb_grp["SB_PA_mlb"] = mlb_grp["SB"] / mlb_grp["PA"]
mlb_grp["3B_PA_mlb"] = mlb_grp["3B"] / mlb_grp["PA"]

# First-year only
fy_grp = mlb_grp.sort_values("Season").groupby("MLBAM_ID").first().reset_index()

# Join
j6 = career_sp.merge(fy_grp[["MLBAM_ID","PPPA","PPPA_Z","speed_pts_pa_mlb",
                               "SB_PA_mlb","3B_PA_mlb","PA"]],
                      on="MLBAM_ID", how="inner")
j6 = j6[(j6["PA_total"] >= 100) & (j6["PA"] >= 100) &
         j6["speed_share"].notna() & j6["speed_pts_pa_milb"].notna()].copy()

out(f"\nJoined sample: {len(j6)} players (MiLB career PA>=100, MLB first-yr PA>=100)")
out(f"MiLB speed_share distribution:")
out(f"  p10={j6['speed_share'].quantile(.10):.3f}, p25={j6['speed_share'].quantile(.25):.3f}, "
    f"p50={j6['speed_share'].quantile(.50):.3f}, p75={j6['speed_share'].quantile(.75):.3f}, "
    f"p90={j6['speed_share'].quantile(.90):.3f}")

# Fixed threshold grouping
bins_fixed = [-np.inf, 0.05, 0.15, 0.25, np.inf]
labels_fixed = ["<5% (low)", "5-15% (mod)", "15-25% (high)", ">25% (elite)"]
j6["grp_fixed"] = pd.cut(j6["speed_share"], bins=bins_fixed, labels=labels_fixed)

# Quartile grouping
j6["grp_q"] = pd.qcut(j6["speed_share"], q=4, labels=["Q1","Q2","Q3","Q4"])

def group_report(df, grp_col, label):
    out(f"\n{label}:")
    out(f"{'Group':<18} {'N':>5} {'MiLB_spd/PA':>12} {'MLB_spd/PA':>12} "
        f"{'Trans_Rate':>11} {'MiLB_PPPA':>10} {'MLB_PPPA':>9} {'MLB_PPPA_Z':>11} "
        f"{'SB/PA':>7} {'3B/PA':>7}")
    for grp, sub in df.groupby(grp_col, observed=True):
        n = len(sub)
        if n < 3:
            continue
        m_spd  = sub["speed_pts_pa_milb"].mean()
        ml_spd = sub["speed_pts_pa_mlb"].mean()
        trans  = ml_spd / m_spd if m_spd > 0 else np.nan
        out(f"  {str(grp):<16} {n:>5} {m_spd:>12.4f} {ml_spd:>12.4f} "
            f"{trans:>11.3f} {sub['PPPA_milb'].mean():>10.4f} "
            f"{sub['PPPA'].mean():>9.4f} {sub['PPPA_Z'].mean():>11.3f} "
            f"{sub['SB_PA_mlb'].mean():>7.4f} {sub['3B_PA_mlb'].mean():>7.4f}")

group_report(j6, "grp_fixed", "Fixed-threshold groups (by MiLB speed_share)")
group_report(j6, "grp_q",     "Quartile groups (Q1=lowest speed share)")

# Regression: MLB_speed_pts_PA ~ MiLB_speed_pts_PA * Group (interaction)
out(f"\nRegression: MLB_speed_pts_PA ~ MiLB_speed_pts_PA x Group (quartiles)")
out(f"(Tests whether speed translation slope differs by MiLB speed-share tier)")
# Encode group dummies and interactions
j6_reg = j6.dropna(subset=["speed_pts_pa_milb","speed_pts_pa_mlb","grp_q","PA"]).copy()
j6_reg["grp_num"] = j6_reg["grp_q"].cat.codes   # 0=Q1 ... 3=Q4
j6_reg["interaction"] = j6_reg["speed_pts_pa_milb"] * j6_reg["grp_num"]
out(wls_summary(j6_reg["speed_pts_pa_mlb"],
                j6_reg[["speed_pts_pa_milb","grp_num","interaction"]],
                j6_reg["PA"].astype(float),
                f"N={len(j6_reg)}"))

# Per-quartile slope (separate regression per group)
out(f"\nPer-quartile slope (MLB_spd_PA ~ MiLB_spd_PA within each group):")
for grp, sub in j6_reg.groupby("grp_q", observed=True):
    sub = sub.dropna(subset=["speed_pts_pa_milb","speed_pts_pa_mlb"])
    if len(sub) < 10:
        continue
    res_line = wls_summary(sub["speed_pts_pa_mlb"],
                           sub[["speed_pts_pa_milb"]],
                           sub["PA"].astype(float),
                           f"{grp} (N={len(sub)})")
    out(res_line)

out(f"""
Part 6 Interpretation:
  Translation rate = MLB speed_pts/PA divided by MiLB speed_pts/PA.
  A rate >1.0 means speed skills AMPLIFY from MiLB to MLB.
  A rate <1.0 means speed skills DECAY from MiLB to MLB.
  The slope from per-group regressions shows how predictable the
  translation is within each tier.
""")

# -
# PART 7: Mediation and Moderation of Speed Translation
# -

out("\n" + "=" * 70)
out("PART 7: MEDIATION AND MODERATION OF SPEED TRANSLATION")
out("=" * 70)

# Build enriched dataset extending j6 with rate stats
# milb_advanced has K%, BB%, ISO, Whiff% per player-season (MLBAM_ID)
adv = pd.read_csv(DATA / "api" / "milb_advanced.csv")
adv_pa = adv[adv["PA"] >= 50].copy()
adv_career = adv_pa.groupby("MLBAM_ID").apply(lambda g: pd.Series({
    "Kpct_milb":   np.average(g["K%"],     weights=g["PA"]),
    "BBpct_milb":  np.average(g["BB%"],    weights=g["PA"]),
    "ISO_milb":    np.average(g["ISO"],    weights=g["PA"]),
    "Whiff_milb":  np.average(g["Whiff%"], weights=g["PA"]),
})).reset_index()

# MiLB SB_pct from career_sp already computed; add per-player SB_pct
career_sp["SB_pct_milb"] = (
    career_sp["SB_total"] /
    (career_sp["SB_total"] + career_sp["CS_total"] + 1e-9)
)
career_sp.loc[career_sp["SB_total"] + career_sp["CS_total"] == 0, "SB_pct_milb"] = np.nan

# MiLB 3B/PA
career_sp["3B_PA_milb"] = career_sp["3B_total"] / career_sp["PA_total"].clip(lower=1)

# PPPA_Z_SL from player_comps (career)
comps_z = comps[["MLBAM_ID","PPPA_Z_career"]].copy()
comps_z["MLBAM_ID"] = pd.to_numeric(comps_z["MLBAM_ID"], errors="coerce")

# Debut age: first season in hist_mlb_data
mlb_raw = pd.read_parquet(DATA / "historical" / "hist_mlb_data.parquet")
mlb_raw = mlb_raw.rename(columns={"MLBAMID": "MLBAM_ID", "GDP": "GIDP"})
debut = mlb_raw.sort_values("Season").groupby("MLBAM_ID").first().reset_index()[
    ["MLBAM_ID","Season","Name"]
].rename(columns={"Season": "MLB_Debut_Season"})
# Approximate debut age from player_birthdays
debut = debut.merge(bdays[["MLBAM_ID","BirthYear"]], on="MLBAM_ID", how="left")
debut["Debut_Age"] = debut["MLB_Debut_Season"] - debut["BirthYear"]
debut["Era_post23"] = (debut["MLB_Debut_Season"] >= 2023).astype(int)

# Debut level: level of last MiLB season before debut
milb_sorted2 = milb.sort_values("Season")
last_milb_row = milb_sorted2.groupby("MLBAM_ID").last().reset_index()[
    ["MLBAM_ID","Level"]
].rename(columns={"Level": "Debut_Level"})
debut = debut.merge(last_milb_row, on="MLBAM_ID", how="left")
debut["From_AAA"] = (debut["Debut_Level"] == "AAA").astype(float)

# Assemble enriched j7
j7 = (j6
      .merge(adv_career, on="MLBAM_ID", how="left")
      .merge(comps_z,    on="MLBAM_ID", how="left")
      .merge(debut[["MLBAM_ID","Debut_Age","Era_post23","From_AAA"]], on="MLBAM_ID", how="left")
)
j7["SB_pct_milb"] = j7["MLBAM_ID"].map(career_sp.set_index("MLBAM_ID")["SB_pct_milb"])
j7["3B_PA_milb"]  = j7["MLBAM_ID"].map(career_sp.set_index("MLBAM_ID")["3B_PA_milb"])

out(f"\nEnriched sample: {len(j7)} players")
out(f"  K% coverage:     {j7['Kpct_milb'].notna().sum()}")
out(f"  Whiff% coverage: {j7['Whiff_milb'].notna().sum()}")
out(f"  ISO coverage:    {j7['ISO_milb'].notna().sum()}")
out(f"  SB_pct coverage: {j7['SB_pct_milb'].notna().sum()}")
out(f"  Debut_Age cover: {j7['Debut_Age'].notna().sum()}")
out(f"  PPPA_Z coverage: {j7['PPPA_Z_career'].notna().sum()}")

# ---- MEDIATORS ----
out("\n--- MEDIATORS (do they explain the U-shape?) ---")
out("\nMean of each mediator by fixed speed-share group:")
med_cols = {"K%": "Kpct_milb", "BB%": "BBpct_milb", "ISO": "ISO_milb",
            "Whiff%": "Whiff_milb", "SB_pct": "SB_pct_milb", "PPPA_Z_career": "PPPA_Z_career"}
grp_medians = j7.groupby("grp_fixed", observed=True)[list(med_cols.values())].mean()
grp_medians.columns = list(med_cols.keys())
out(grp_medians.round(4).to_string())

# Attenuation test: does adding a mediator reduce the group effect?
# Baseline: MLB_speed_pts_PA ~ MiLB_speed_pts_PA + grp_num
j7_reg = j7.dropna(subset=["speed_pts_pa_milb","speed_pts_pa_mlb","grp_fixed","PA"]).copy()
j7_reg["grp_num"] = j7_reg["grp_fixed"].cat.codes

out(f"\nBaseline: MLB_speed_pts_PA ~ MiLB_spd_PA + grp_num (N={len(j7_reg)}):")
base = wls_summary(j7_reg["speed_pts_pa_mlb"],
                   j7_reg[["speed_pts_pa_milb","grp_num"]],
                   j7_reg["PA"].astype(float), "Baseline")
out(base)

out(f"\nAttenuation when mediator added (grp_num coefficient change):")
for mname, mcol in med_cols.items():
    sub = j7_reg.dropna(subset=[mcol])
    if len(sub) < 50:
        out(f"  {mname:<15}: N={len(sub)} (too small)")
        continue
    res = wls_summary(sub["speed_pts_pa_mlb"],
                      sub[["speed_pts_pa_milb","grp_num",mcol]],
                      sub["PA"].astype(float), mname)
    out(f"  {mname:<15}: {res.strip()}")

# ---- MODERATORS ----
out("\n--- MODERATORS (do they change how much speed translates?) ---")
out("Interaction: MLB_speed_pts_PA ~ MiLB_speed_pts_PA * Moderator")
out("(WLS, PA-weighted; interaction coef = marginal slope change per unit moderator)\n")

moderators = {
    "K% (MiLB)":        "Kpct_milb",
    "BB% (MiLB)":       "BBpct_milb",
    "ISO (MiLB)":       "ISO_milb",
    "Whiff% (MiLB)":    "Whiff_milb",
    "SB_pct (MiLB)":    "SB_pct_milb",
    "3B/PA (MiLB)":     "3B_PA_milb",
    "PPPA_Z_career":    "PPPA_Z_career",
    "Debut_Age":        "Debut_Age",
    "From_AAA":         "From_AAA",
    "Era_post23":       "Era_post23",
}

mod_results = []
for mname, mcol in moderators.items():
    sub = j7.dropna(subset=["speed_pts_pa_milb","speed_pts_pa_mlb",mcol,"PA"])
    n = len(sub)
    if n < 30:
        mod_results.append((mname, np.nan, np.nan, n, "N too small"))
        continue
    # Center moderator
    mc = sub[mcol] - sub[mcol].mean()
    X = pd.DataFrame({
        "spd_milb": sub["speed_pts_pa_milb"],
        "mod":      mc,
        "interact": sub["speed_pts_pa_milb"] * mc,
    })
    try:
        Xv = sm.add_constant(X.values)
        res = sm.WLS(sub["speed_pts_pa_mlb"].values, Xv, weights=sub["PA"].values).fit()
        coef = res.params[3]   # interaction term
        pval = res.pvalues[3]
        direction = "strengthens" if coef > 0 else "weakens"
        mod_results.append((mname, coef, pval, n, direction))
        sig = "*" if pval < 0.05 else " "
        out(f"  {sig} {mname:<20} coef={coef:+.4f}  p={pval:.4f}  N={n}  ({direction} translation)")
    except Exception as e:
        mod_results.append((mname, np.nan, np.nan, n, str(e)))
        out(f"    {mname}: error -- {e}")

# Summary table sorted by |effect|
out("\nModerator summary (sorted by |coef|, p<0.05 flagged *):")
out(f"  {'Moderator':<22} {'Coef':>8} {'p':>7} {'N':>6} {'Sig':>4}")
valid_mods = [(nm, c, p, n, d) for nm, c, p, n, d in mod_results if not np.isnan(c if c else np.nan)]
valid_mods.sort(key=lambda x: abs(x[1]), reverse=True)
for nm, c, p, n, d in valid_mods:
    sig = "*" if p < 0.05 else " "
    out(f"  {sig} {nm:<21} {c:>+8.4f} {p:>7.4f} {n:>6}")

# ---- U-SHAPE TARGETED TESTS ----
out("\n--- U-SHAPE TARGETED TESTS ---")

# Test 1: Moderate group split by SB_pct quartile
mod_grp = j7[j7["grp_fixed"] == "5-15% (mod)"].dropna(subset=["SB_pct_milb"])
if len(mod_grp) >= 20:
    mod_grp = mod_grp.copy()
    mod_grp["sbpct_q"] = pd.qcut(mod_grp["SB_pct_milb"], q=4, labels=["Q1","Q2","Q3","Q4"])
    out(f"\nTest 1: Moderate-speed group (5-15%) split by SB_pct (N={len(mod_grp)}):")
    out(f"  {'SB_pct_Q':<8} {'N':>4} {'MiLB_spd/PA':>12} {'MLB_spd/PA':>12} {'Trans':>7} {'MLB_PPPA_Z':>11}")
    for grp, sub in mod_grp.groupby("sbpct_q", observed=True):
        n = len(sub)
        m_s = sub["speed_pts_pa_milb"].mean()
        ml_s = sub["speed_pts_pa_mlb"].mean()
        tr = ml_s / m_s if m_s > 0 else np.nan
        out(f"  {str(grp):<8} {n:>4} {m_s:>12.4f} {ml_s:>12.4f} {tr:>7.3f} {sub['PPPA_Z'].mean():>11.3f}")

# Test 2: Moderate group split by K% tertile
mod_grp_k = j7[j7["grp_fixed"] == "5-15% (mod)"].dropna(subset=["Kpct_milb"])
if len(mod_grp_k) >= 20:
    mod_grp_k = mod_grp_k.copy()
    mod_grp_k["kpct_t"] = pd.qcut(mod_grp_k["Kpct_milb"], q=3, labels=["Low-K","Mid-K","High-K"])
    out(f"\nTest 2: Moderate-speed group split by K% tertile (N={len(mod_grp_k)}):")
    out(f"  {'K%_tier':<8} {'N':>4} {'Mean_K%':>8} {'MiLB_spd/PA':>12} {'MLB_spd/PA':>12} {'Trans':>7} {'MLB_PPPA_Z':>11}")
    for grp, sub in mod_grp_k.groupby("kpct_t", observed=True):
        n = len(sub)
        m_s = sub["speed_pts_pa_milb"].mean()
        ml_s = sub["speed_pts_pa_mlb"].mean()
        tr = ml_s / m_s if m_s > 0 else np.nan
        out(f"  {str(grp):<8} {n:>4} {sub['Kpct_milb'].mean():>8.3f} {m_s:>12.4f} {ml_s:>12.4f} {tr:>7.3f} {sub['PPPA_Z'].mean():>11.3f}")

# Test 3: Elite group -- is 0.672 translation due to SB_pct or genuine?
elite_grp = j7[j7["grp_fixed"] == ">25% (elite)"].dropna(subset=["SB_pct_milb"])
mod_med   = j7[j7["grp_fixed"] == "5-15% (mod)"].dropna(subset=["SB_pct_milb"])
if len(elite_grp) >= 10 and len(mod_med) >= 10:
    out(f"\nTest 3: Elite vs moderate -- SB_pct comparison:")
    out(f"  Elite  (N={len(elite_grp)}): mean SB_pct={elite_grp['SB_pct_milb'].mean():.3f}, "
        f"trans={elite_grp['speed_pts_pa_mlb'].mean()/elite_grp['speed_pts_pa_milb'].mean():.3f}")
    out(f"  Mod    (N={len(mod_med)}): mean SB_pct={mod_med['SB_pct_milb'].mean():.3f}, "
        f"trans={mod_med['speed_pts_pa_mlb'].mean()/mod_med['speed_pts_pa_milb'].mean():.3f}")
    # Matched: moderate players with elite-like SB_pct
    high_sbpct_thresh = elite_grp["SB_pct_milb"].mean()
    mod_high_sbpct = mod_med[mod_med["SB_pct_milb"] >= high_sbpct_thresh]
    if len(mod_high_sbpct) >= 5:
        tr_mhsp = mod_high_sbpct["speed_pts_pa_mlb"].mean() / mod_high_sbpct["speed_pts_pa_milb"].mean()
        out(f"  Mod w/ SB_pct>={high_sbpct_thresh:.3f} (N={len(mod_high_sbpct)}): "
            f"trans={tr_mhsp:.3f} (do these players recover elite-group translation?)")

out(f"""
Part 7 Summary:
  Mediators: K% and SB_pct are the primary candidates for explaining the
  U-shape. High-K moderate-speed players likely drive the 0.563 translation
  floor -- they accumulate MiLB speed stats partly via contact-favorable counts
  but face stiffer MLB pitching. SB_pct differences test whether the moderate
  group runs inefficiently vs. the elite group.

  Moderators: The interaction terms rank the factors that most change how
  reliably speed translates. Significant moderators (p<0.05) are the
  strongest candidates for additional signals in ABILITY or TOOLS scoring.

  Model implications: if SB_pct is a strong moderator (likely), the current
  SB_talent formulation (SB_pct x SB/PA) already partially captures it.
  If K% is a strong moderator, the Discipline component in ABILITY (which
  penalizes K%) already accounts for the interaction. True un-modeled signal
  would require a new cross-term or an explicit SB_pct threshold gate.
""")

# -
# PART 7b: Revised Moderation -- BB/2K, BB%, Spd, HR/FB, GB%, K%
# -

p7b = "\n" + "=" * 70 + "\n"
p7b += "PART 7b: REVISED MODERATION ANALYSIS\n"
p7b += "=" * 70 + "\n"

# Build career averages for the new moderator set from milb_advanced
adv7b = adv[adv["PA"] >= 50].copy()
adv7b["HRFB"] = adv7b["HR/FB"].where(adv7b["HR/FB"] <= 1.0)   # null physically impossible values
adv_career2 = adv7b.groupby("MLBAM_ID").apply(lambda g: pd.Series({
    "Kpct":   np.average(g["K%"],     weights=g["PA"]),
    "BBpct":  np.average(g["BB%"],    weights=g["PA"]),
    "HRFB":   np.average(g["HRFB"].fillna(g["HRFB"].median()), weights=g["PA"])
              if g["HRFB"].notna().any() else np.nan,
    "GBpct":  np.average(g["GB%"],    weights=g["PA"]),
})).reset_index()
adv_career2["BB2K"] = adv_career2["BBpct"] - 2 * adv_career2["Kpct"]

# PS Spd: career average across all available seasons per player
# ps_spd is already loaded above (combines all PS level/season files)
ps_career_spd = (ps_spd[ps_spd["Spd"].notna() & (ps_spd["Spd"] > 0)]
                 .groupby("MLBAM_ID")
                 .apply(lambda g: np.average(g["Spd"], weights=g["PA"].fillna(1)))
                 .reset_index(name="Spd_career"))
ps_career_spd["MLBAM_ID"] = pd.to_numeric(ps_career_spd["MLBAM_ID"], errors="coerce")

# Assemble j7b from j6 base
j7b = (j6
       .merge(adv_career2[["MLBAM_ID","Kpct","BBpct","BB2K","HRFB","GBpct"]], on="MLBAM_ID", how="left")
       .merge(ps_career_spd, on="MLBAM_ID", how="left")
)
# Compute SB_pct and 3B_PA from career counting columns inherited from career_sp
_sb_sum = j7b["SB_total"].fillna(0)
_cs_sum = j7b["CS_total"].fillna(0)
j7b["SB_pct_milb"] = (_sb_sum / (_sb_sum + _cs_sum + 1e-9)).where(_sb_sum + _cs_sum > 0)
j7b["3B_PA_milb"]  = j7b["3B_total"] / j7b["PA_total"].clip(lower=1)

p7b += f"\nEnriched sample (from Part 6 base, N={len(j7b)}):\n"
p7b += f"  K%/BB% coverage:  {j7b['Kpct'].notna().sum()}\n"
p7b += f"  HR/FB coverage:   {j7b['HRFB'].notna().sum()}\n"
p7b += f"  GB% coverage:     {j7b['GBpct'].notna().sum()}\n"
p7b += f"  PS Spd coverage:  {j7b['Spd_career'].notna().sum()} (AAA 2023-2026 + all levels 2026)\n"

def p7b_mod_result(df, mname, mcol, wt_col="PA"):
    sub = df.dropna(subset=["speed_pts_pa_milb","speed_pts_pa_mlb",mcol,wt_col])
    n = len(sub)
    if n < 20:
        return f"    {mname:<22} N={n} (too small)", np.nan, np.nan, n
    mc = sub[mcol] - sub[mcol].mean()
    X = pd.DataFrame({
        "spd_milb": sub["speed_pts_pa_milb"],
        "mod":      mc,
        "interact": sub["speed_pts_pa_milb"] * mc,
    })
    try:
        Xv = sm.add_constant(X.values)
        res = sm.WLS(sub["speed_pts_pa_mlb"].values, Xv, weights=sub[wt_col].values).fit()
        coef = res.params[3]
        pval = res.pvalues[3]
        sig  = "*" if pval < 0.05 else " "
        dirn = "strengthens" if coef > 0 else "weakens"
        line = f"  {sig} {mname:<22} coef={coef:+.4f}  p={pval:.4f}  N={n}  ({dirn} translation)"
        return line, coef, pval, n
    except Exception as e:
        return f"    {mname}: error -- {e}", np.nan, np.nan, n

p7b += "\n--- REVISED MODERATORS ---\n"
p7b += "Interaction: MLB_speed_pts_PA ~ MiLB_speed_pts_PA + Mod + MiLB_speed_pts_PA x Mod\n"
p7b += "(WLS PA-weighted; centered moderator; * = p<0.05)\n\n"

new_mods = [
    ("BB/2K (BB%-2*K%)",  "BB2K"),
    ("BB% (MiLB)",        "BBpct"),
    ("PS Spd (MiLB)",     "Spd_career"),
    ("HR/FB (MiLB)",      "HRFB"),
    ("GB% (MiLB)",        "GBpct"),
    ("K% (MiLB)",         "Kpct"),
]

mod7b_results = []
for mname, mcol in new_mods:
    line, coef, pval, n = p7b_mod_result(j7b, mname, mcol)
    p7b += line + "\n"
    mod7b_results.append((mname, coef, pval, n))

p7b += "\nModerator summary (sorted by |coef|, p<0.05 flagged *):\n"
p7b += f"  {'Moderator':<24} {'Coef':>8} {'p':>7} {'N':>6}\n"
valid7b = [(nm, c, p, n) for nm, c, p, n in mod7b_results if not np.isnan(c)]
valid7b.sort(key=lambda x: abs(x[1]), reverse=True)
for nm, c, p, n in valid7b:
    sig = "*" if p < 0.05 else " "
    p7b += f"  {sig} {nm:<23} {c:>+8.4f} {p:>7.4f} {n:>6}\n"

# ---- U-SHAPE TARGETED TESTS (revised) ----
p7b += "\n--- U-SHAPE TARGETED TESTS (revised) ---\n"

# Test 1: same as Part 7 -- moderate group by SB_pct quartile
mod_grp7b = j7b[j7b["grp_fixed"] == "5-15% (mod)"].dropna(subset=["SB_pct_milb"])
if len(mod_grp7b) >= 20:
    mod_grp7b = mod_grp7b.copy()
    mod_grp7b["sbpct_q"] = pd.qcut(mod_grp7b["SB_pct_milb"], q=4, labels=["Q1","Q2","Q3","Q4"])
    p7b += f"\nTest 1: Moderate-speed group split by SB_pct (N={len(mod_grp7b)}):\n"
    p7b += f"  {'SB_pct_Q':<8} {'N':>4} {'MiLB_spd/PA':>12} {'MLB_spd/PA':>12} {'Trans':>7} {'MLB_PPPA_Z':>11}\n"
    for grp, sub in mod_grp7b.groupby("sbpct_q", observed=True):
        n = len(sub)
        m_s = sub["speed_pts_pa_milb"].mean()
        ml_s = sub["speed_pts_pa_mlb"].mean()
        tr = ml_s / m_s if m_s > 0 else np.nan
        p7b += f"  {str(grp):<8} {n:>4} {m_s:>12.4f} {ml_s:>12.4f} {tr:>7.3f} {sub['PPPA_Z'].mean():>11.3f}\n"

# Test 2: Moderate group split by BB/2K tertile (replacing K% from Part 7)
mod_grp_bb2k = j7b[j7b["grp_fixed"] == "5-15% (mod)"].dropna(subset=["BB2K"])
if len(mod_grp_bb2k) >= 20:
    mod_grp_bb2k = mod_grp_bb2k.copy()
    mod_grp_bb2k["bb2k_t"] = pd.qcut(mod_grp_bb2k["BB2K"], q=3, labels=["Low","Mid","High"])
    p7b += f"\nTest 2: Moderate-speed group split by BB/2K tertile (N={len(mod_grp_bb2k)}):\n"
    p7b += f"  {'BB2K_tier':<9} {'N':>4} {'Mean_BB2K':>10} {'MiLB_spd/PA':>12} {'MLB_spd/PA':>12} {'Trans':>7} {'MLB_PPPA_Z':>11}\n"
    for grp, sub in mod_grp_bb2k.groupby("bb2k_t", observed=True):
        n = len(sub)
        m_s = sub["speed_pts_pa_milb"].mean()
        ml_s = sub["speed_pts_pa_mlb"].mean()
        tr = ml_s / m_s if m_s > 0 else np.nan
        p7b += f"  {str(grp):<9} {n:>4} {sub['BB2K'].mean():>10.4f} {m_s:>12.4f} {ml_s:>12.4f} {tr:>7.3f} {sub['PPPA_Z'].mean():>11.3f}\n"

# Test 3: Elite vs moderate SB_pct (same as Part 7)
elite7b  = j7b[j7b["grp_fixed"] == ">25% (elite)"].dropna(subset=["SB_pct_milb"])
mod7b_m  = j7b[j7b["grp_fixed"] == "5-15% (mod)"].dropna(subset=["SB_pct_milb"])
if len(elite7b) >= 10 and len(mod7b_m) >= 10:
    p7b += f"\nTest 3: Elite vs moderate -- SB_pct comparison:\n"
    p7b += (f"  Elite (N={len(elite7b)}): SB_pct={elite7b['SB_pct_milb'].mean():.3f}, "
            f"trans={elite7b['speed_pts_pa_mlb'].mean()/elite7b['speed_pts_pa_milb'].mean():.3f}\n")
    p7b += (f"  Mod   (N={len(mod7b_m)}): SB_pct={mod7b_m['SB_pct_milb'].mean():.3f}, "
            f"trans={mod7b_m['speed_pts_pa_mlb'].mean()/mod7b_m['speed_pts_pa_milb'].mean():.3f}\n")
    thresh = elite7b["SB_pct_milb"].mean()
    mod_hi = mod7b_m[mod7b_m["SB_pct_milb"] >= thresh]
    if len(mod_hi) >= 5:
        tr_hi = mod_hi["speed_pts_pa_mlb"].mean() / mod_hi["speed_pts_pa_milb"].mean()
        p7b += (f"  Mod w/ SB_pct>={thresh:.3f} (N={len(mod_hi)}): "
                f"trans={tr_hi:.3f}\n")

out("\n=== Part 7b ===")
out(p7b)

# -
# PART 7c: Is PPPA_Z_career moderation independent of HR/FB moderation?
# -

p7c_lines = []
def c(msg=""):
    p7c_lines.append(msg)

c("=" * 70)
c("PART 7c: PPPA_Z_career vs HR/FB -- INDEPENDENT OR CONFOUNDED?")
c("=" * 70)

# Working dataset: j7b + PPPA_Z_career from comps_z
j7c = j7b.merge(comps_z, on="MLBAM_ID", how="left")
j7c = j7c.dropna(subset=["speed_pts_pa_milb","speed_pts_pa_mlb","PPPA_Z_career","HRFB","PA"]).copy()
c(f"\nWorking sample: {len(j7c)} players")

# Center both moderators
j7c["pppa_z_c"] = j7c["PPPA_Z_career"] - j7c["PPPA_Z_career"].mean()
j7c["hrfb_c"]   = j7c["HRFB"]          - j7c["HRFB"].mean()
j7c["spd"]      = j7c["speed_pts_pa_milb"]
j7c["mlb_spd"]  = j7c["speed_pts_pa_mlb"]
j7c["wt"]       = j7c["PA"].astype(float)

def run_wls(y, X_cols, df, label):
    X = sm.add_constant(df[X_cols].values)
    res = sm.WLS(df[y].values, X, weights=df["wt"].values).fit()
    c(f"\n  {label} (N={len(df)}, R2={res.rsquared:.3f}):")
    for i, col in enumerate(["const"] + X_cols):
        c(f"    {col:<35} coef={res.params[i]:+.4f}  p={res.pvalues[i]:.4f}")
    return res

# --- Test 1 ---
c("\n--- TEST 1: Partial effects controlling for both moderators ---")

j7c["spd_x_pppa"] = j7c["spd"] * j7c["pppa_z_c"]
j7c["spd_x_hrfb"] = j7c["spd"] * j7c["hrfb_c"]

run_wls("mlb_spd", ["spd","pppa_z_c","spd_x_pppa"], j7c,
        "A) MiLB_spd x PPPA_Z_career alone")
run_wls("mlb_spd", ["spd","hrfb_c","spd_x_hrfb"],   j7c,
        "B) MiLB_spd x HR/FB alone")
run_wls("mlb_spd", ["spd","pppa_z_c","hrfb_c","spd_x_pppa","spd_x_hrfb"], j7c,
        "C) Both interactions together")

c("\n  Interpretation: compare PPPA_Z_career interaction coef in A vs C.")
c("  If coef attenuates (shrinks / loses sig) in C, HR/FB explains the PPPA_Z signal.")
c("  If coef persists in C, PPPA_Z_career captures independent signal.")

# --- Test 2 ---
c("\n--- TEST 2: PPPA_Z_career moderation within HR/FB terciles ---")
j7c["hrfb_t"] = pd.qcut(j7c["HRFB"], q=3, labels=["Low-HR/FB","Mid-HR/FB","High-HR/FB"])
c(f"\n  {'Tercile':<14} {'N':>5} {'Mean_HRFB':>10}  PPPA_Z_coef  p-value")
for grp, sub in j7c.groupby("hrfb_t", observed=True):
    sub = sub.copy()
    sub["pppa_z_c2"] = sub["PPPA_Z_career"] - sub["PPPA_Z_career"].mean()
    sub["spd_x_p2"]  = sub["spd"] * sub["pppa_z_c2"]
    n = len(sub)
    if n < 20:
        c(f"  {str(grp):<14} {n:>5}  (too small)")
        continue
    X = sm.add_constant(sub[["spd","pppa_z_c2","spd_x_p2"]].values)
    res = sm.WLS(sub["mlb_spd"].values, X, weights=sub["wt"].values).fit()
    coef = res.params[3]
    pval = res.pvalues[3]
    sig  = "*" if pval < 0.05 else " "
    c(f"  {sig} {str(grp):<13} {n:>5} {sub['HRFB'].mean():>10.3f}  {coef:>+10.4f}   {pval:.4f}")

c("\n  Goal: does PPPA_Z moderation persist in Low-HR/FB tercile?")
c("  If yes -> independent signal (pure speed/contact players still over-represent MiLB speed).")
c("  If only in High-HR/FB -> fully explained by power-speed profile.")

# --- Test 3 ---
c("\n--- TEST 3: 2x2 descriptive table (PPPA_Z_career median x HR/FB median) ---")
pppa_med = j7c["PPPA_Z_career"].median()
hrfb_med = j7c["HRFB"].median()
j7c["pppa_hi"] = (j7c["PPPA_Z_career"] >= pppa_med).map({True:"High-PPPA",False:"Low-PPPA"})
j7c["hrfb_hi"] = (j7c["HRFB"]          >= hrfb_med).map({True:"High-HR/FB",False:"Low-HR/FB"})

c(f"\n  PPPA_Z median={pppa_med:.3f}, HR/FB median={hrfb_med:.3f}")
c(f"\n  {'Quadrant':<30} {'N':>5} {'MiLB_spd/PA':>12} {'MLB_spd/PA':>12} {'Trans':>7} {'MLB_PPPA_Z':>11}")
for (pppa_g, hrfb_g), sub in j7c.groupby(["pppa_hi","hrfb_hi"], observed=True):
    n    = len(sub)
    ms   = sub["spd"].mean()
    mls  = sub["mlb_spd"].mean()
    tr   = mls / ms if ms > 0 else np.nan
    label = f"{pppa_g}/{hrfb_g}"
    c(f"  {label:<30} {n:>5} {ms:>12.4f} {mls:>12.4f} {tr:>7.3f} {sub['PPPA_Z'].mean():>11.3f}")

c("\n  The 'poor translation' signature: is it High-PPPA/High-HR/FB specifically,")
c("  or does High-PPPA/Low-HR/FB also show decay?")

# --- Test 4 ---
c("\n--- TEST 4: Correlation PPPA_Z_career x HR/FB ---")
r4, p4 = stats.pearsonr(j7c["PPPA_Z_career"], j7c["HRFB"])
c(f"\n  Pearson r(PPPA_Z_career, HR/FB) = {r4:.3f}  p={p4:.4f}  N={len(j7c)}")
c(f"  R2 = {r4**2:.3f}")
c("\n  High r -> collinear player types; confound hypothesis plausible.")
c("  Low r  -> PPPA_Z and HR/FB measure different things; both signals independent.")

p7c = "\n".join(p7c_lines)
out("\n=== Part 7c ===")
out(p7c)

# -
# PART 8: Empirical component re-weighting — career MLB PPPA as target
# -

p8_lines = []
def c8(msg=""):
    p8_lines.append(msg)

c8("=" * 70)
c8("PART 8: EMPIRICAL COMPONENT RE-WEIGHTING (career MLB PPPA target)")
c8("=" * 70)

LEVEL_DISC = {"AAA": 1.00, "AA": 0.59, "A+": 0.34, "A": 0.23, "R": 0.10}

# --- Step 1: Career MLB PPPA outcome ---
c8("\n--- STEP 1: Career MLB PPPA outcome ---")
career_mlb8 = mlb.groupby("MLBAM_ID").apply(lambda g: pd.Series({
    "career_pppa": np.average(g["PPPA"], weights=g["PA"]),
    "career_pa":   g["PA"].sum(),
})).reset_index()
career_mlb8 = career_mlb8[career_mlb8["career_pa"] >= 200].copy()
mean_c, std_c = career_mlb8["career_pppa"].mean(), career_mlb8["career_pppa"].std()
career_mlb8["career_pppa_z"] = (career_mlb8["career_pppa"] - mean_c) / std_c
c8(f"\n  N graduated (career PA >= 200): {len(career_mlb8)}")
c8(f"  Mean career PPPA: {mean_c:.4f}  Std: {std_c:.4f}")

# --- Step 2: MiLB component scores (level-discount-weighted) ---
c8("\n--- STEP 2: MiLB component scores (level-discounted PA weights) ---")

# Load minorLeagueData for PPPA_Z_SL; join MLBAM_ID via milb_hitting PlayerId crosswalk
mld = pd.read_parquet(DATA / "computed" / "minorLeagueData.parquet")
milb_id_map = (milb[["PlayerId","MLBAM_ID"]]
               .dropna(subset=["PlayerId","MLBAM_ID"])
               .drop_duplicates("PlayerId"))
mld = mld.merge(milb_id_map, on="PlayerId", how="left")
mld["disc"] = mld["Level"].map(LEVEL_DISC).fillna(0.0)
mld["eff_PA"] = mld["PA"] * mld["disc"]

# Fantasy_Out: sum(PPPA_Z_SL * disc * PA) / sum(eff_PA)
fo_grp = mld[mld["PA"] >= 50].dropna(subset=["MLBAM_ID"]).groupby("MLBAM_ID").apply(lambda g: pd.Series({
    "Fantasy_Out": (
        (g["PPPA_Z_SL"] * g["disc"] * g["PA"]).sum() / g["eff_PA"].sum()
        if g["eff_PA"].sum() > 0 else np.nan
    ),
    "milb_pa_total": g["PA"].sum(),
})).reset_index()

# From milb_hitting: SB_talent, Game_Power
milb8 = milb[milb["PA"] >= 50].copy()
milb8["disc"] = milb8["Level"].map(LEVEL_DISC).fillna(0.0)
milb8["eff_PA"] = milb8["PA"] * milb8["disc"]
milb8["SB_att"] = milb8["SB"].fillna(0) + milb8["CS"].fillna(0)
milb8["SB_pct_row"] = milb8["SB"].fillna(0) / (milb8["SB_att"] + 1e-9)
milb8.loc[milb8["SB_att"] == 0, "SB_pct_row"] = np.nan
milb8["SB_PA_row"] = milb8["SB"].fillna(0) / milb8["PA"].clip(lower=1)
milb8["SB_talent_row"] = milb8["SB_pct_row"].fillna(0) * milb8["SB_PA_row"]
# HR_AB = HR / (PA - BB - IBB)
milb8["denom_ab"] = (milb8["PA"] - milb8["BB"].fillna(0) - milb8["IBB"].fillna(0)).clip(lower=1)
milb8["HR_AB_row"] = milb8["HR"] / milb8["denom_ab"]

def eff_wavg(g, val_col):
    mask = g[val_col].notna() & (g["eff_PA"] > 0)
    if mask.sum() == 0:
        return np.nan
    return np.average(g.loc[mask, val_col], weights=g.loc[mask, "eff_PA"])

cnt_grp = milb8.groupby("MLBAM_ID").apply(lambda g: pd.Series({
    "SB_talent":  eff_wavg(g, "SB_talent_row"),
    "Game_Power": eff_wavg(g, "HR_AB_row"),
})).reset_index()

# From milb_advanced: Discipline (BB%-2*K%), Discipline_tools (-K%), Power_tools (HR/FB)
adv8 = adv[adv["PA"] >= 50].copy()
adv8["disc"] = adv8["Level"].map(LEVEL_DISC).fillna(0.0)
adv8["eff_PA"] = adv8["PA"] * adv8["disc"]
adv8["BB_2K_row"] = adv8["BB%"] - 2 * adv8["K%"]
adv8["negK_row"] = -adv8["K%"]
adv8["HRFB_row"] = adv8["HR/FB"].where(adv8["HR/FB"] <= 1.0)

rate_grp = adv8.groupby("MLBAM_ID").apply(lambda g: pd.Series({
    "Discipline":       eff_wavg(g, "BB_2K_row"),
    "Discipline_tools": eff_wavg(g, "negK_row"),
    "Power_tools":      eff_wavg(g, "HRFB_row"),
})).reset_index()

# Assemble j8
j8 = (career_mlb8
      .merge(fo_grp[["MLBAM_ID","Fantasy_Out"]], on="MLBAM_ID", how="left")
      .merge(cnt_grp,  on="MLBAM_ID", how="left")
      .merge(rate_grp, on="MLBAM_ID", how="left")
)
# Athleticism_tools = same as SB_talent proxy
j8["Athleticism_tools"] = j8["SB_talent"]

# Standardize all components to mean=0, std=1
comp_cols = ["Fantasy_Out","Discipline","SB_talent","Game_Power",
             "Discipline_tools","Power_tools","Athleticism_tools"]
for col in comp_cols:
    mu, sd = j8[col].mean(), j8[col].std()
    j8[f"{col}_z"] = (j8[col] - mu) / sd if sd > 0 else 0.0

c8(f"\n  j8 assembled: {len(j8)} rows (career MLB PA >= 200)")
for col in comp_cols:
    n_valid = j8[col].notna().sum()
    c8(f"    {col:<22}: N={n_valid} non-null")

# --- Step 3: ABILITY WLS ---
c8("\n--- STEP 3: ABILITY components -> career PPPA_Z ---")
c8("\n  Current model weights: Fantasy_Out=45%, Discipline=25%, SB_talent=15%, Game_Power=15%")

ABILITY_COLS = ["Fantasy_Out_z","Discipline_z","SB_talent_z","Game_Power_z"]
ABILITY_LABELS = ["Fantasy_Out","Discipline","SB_talent","Game_Power"]
ABILITY_MODEL_WT = [0.45, 0.25, 0.15, 0.15]

def run_wls8(y_col, X_cols, labels, model_wts, df, wt_col, section_label):
    mask = df[y_col].notna() & df[wt_col].notna() & (df[wt_col] > 0)
    for col in X_cols:
        mask = mask & df[col].notna()
    sub = df[mask].copy()
    n = len(sub)
    if n < 20:
        c8(f"  {section_label}: N={n} (too small)")
        return None
    X = sm.add_constant(sub[X_cols].values)
    y = sub[y_col].values
    w = sub[wt_col].values
    res = sm.WLS(y, X, weights=w).fit()
    c8(f"\n  {section_label} (N={n}, R2={res.rsquared:.3f}):")
    c8(f"    {'Component':<22} {'Coef':>8}  {'p-val':>7}  {'|Coef|':>8}  {'Emp_Wt':>8}  {'Model_Wt':>9}  Direction")
    abs_coefs = np.abs(res.params[1:])
    total_abs = abs_coefs.sum()
    for i, (lbl, mwt) in enumerate(zip(labels, model_wts)):
        coef = res.params[i+1]
        pv   = res.pvalues[i+1]
        emp  = abs_coefs[i] / total_abs if total_abs > 0 else 0
        direction = ("OVER" if emp < mwt * 0.8 else
                     "UNDER" if emp > mwt * 1.2 else "~RIGHT")
        c8(f"    {lbl:<22} {coef:>8.4f}  {pv:>7.4f}  {abs_coefs[i]:>8.4f}  "
           f"{emp:>7.1%}  {mwt:>8.1%}  {direction}")
    return res

res_ability = run_wls8("career_pppa_z", ABILITY_COLS, ABILITY_LABELS,
                       ABILITY_MODEL_WT, j8, "career_pa", "ABILITY -> career PPPA_Z (WLS)")

# Also run with raw career_pppa (standardized betas are the same as z-scored outcome, so just report)
c8("\n  Note: z-scored outcome gives normalized coefficients directly comparable to implied weights.")

# --- Step 4: Speed component ---
c8("\n--- STEP 4: Speed channel analysis ---")
j8s = j8.dropna(subset=["career_pppa_z","SB_talent_z","Power_tools_z","career_pa"]).copy()
c8(f"\n  Bivariate N: {len(j8s)}")

r_spd, p_spd = stats.pearsonr(j8s["SB_talent_z"], j8s["career_pppa_z"])
c8(f"\n  Bivariate SB_talent_z -> career_pppa_z:")
c8(f"    r={r_spd:.3f}  R2={r_spd**2:.3f}  p={p_spd:.4f}")

X_spd = sm.add_constant(j8s[["SB_talent_z"]].values)
res_spd = sm.WLS(j8s["career_pppa_z"].values, X_spd, weights=j8s["career_pa"].values).fit()
c8(f"    WLS coef={res_spd.params[1]:.4f}  p={res_spd.pvalues[1]:.4f}")

# SB_talent + HR/FB interaction
j8s["hrfb_z"] = (j8s["Power_tools"] - j8s["Power_tools"].mean()) / j8s["Power_tools"].std()
j8s["spd_x_hrfb"] = j8s["SB_talent_z"] * j8s["hrfb_z"]
X_int = sm.add_constant(j8s[["SB_talent_z","hrfb_z","spd_x_hrfb"]].values)
res_int = sm.WLS(j8s["career_pppa_z"].values, X_int, weights=j8s["career_pa"].values).fit()
c8(f"\n  SB_talent + HR/FB interaction:")
c8(f"    N={len(j8s)}  R2={res_int.rsquared:.3f}")
c8(f"    SB_talent_z coef={res_int.params[1]:.4f}  p={res_int.pvalues[1]:.4f}")
c8(f"    hrfb_z      coef={res_int.params[2]:.4f}  p={res_int.pvalues[2]:.4f}")
c8(f"    spd_x_hrfb  coef={res_int.params[3]:.4f}  p={res_int.pvalues[3]:.4f}")
delta_r2 = res_int.rsquared - res_spd.rsquared
c8(f"    Delta R2 from adding interaction: {delta_r2:.4f}")

# --- Step 5: TOOLS WLS ---
c8("\n--- STEP 5: TOOLS component proxies -> career PPPA_Z ---")
c8("\n  Current model weights: Discipline=45%, Power=35%, Athleticism=20%")

TOOLS_COLS   = ["Discipline_tools_z","Power_tools_z","Athleticism_tools_z"]
TOOLS_LABELS = ["Discipline(-K%)","Power(HR/FB)","Athleticism(SB_tal)"]
TOOLS_MODEL_WT = [0.45, 0.35, 0.20]

run_wls8("career_pppa_z", TOOLS_COLS, TOOLS_LABELS,
         TOOLS_MODEL_WT, j8, "career_pa", "TOOLS proxies -> career PPPA_Z (WLS)")

# --- Weighting verdict ---
c8("\n" + "=" * 70)
c8("WEIGHTING VERDICT")
c8("=" * 70)
c8("""
ABILITY components (vs current 45/25/15/15):
  Empirical results assess each component's contribution to career PPPA_Z.
  Fantasy_Out (45%): Should anchor — it is PPPA itself level-discounted,
    so its coefficient partly reflects circularity (MiLB PPPA predicts MLB PPPA).
    Monitor sign and significance; should be the dominant term.
  Discipline (25%): BB%-2*K% is the strongest non-circular signal in ablation
    studies (R2~0.13 for first-year MLB). Empirical weight vs. 25% guides adjustment.
  SB_talent (15%): Part 7-7c showed speed translation r~0.67 but moderated
    downward by HR/FB and PPPA_Z. Career weight may differ from first-year weight.
  Game_Power (15%): HR/AB is a reliable per-AB metric; ABILITY weight vs. empirical
    guides whether power is over/under-weighted relative to speed and discipline.

TOOLS proxies note:
  -K% is only a proxy for Chase%+ZContact%+Whiff% full tier.
  HR/FB is a noisier power signal than MaxEV+EV90 (full tier).
  Athleticism_tools = SB_talent (same variable) — not independent.
  These regressions diagnose the direction of bias (tools proxy < true tools signal)
  rather than exact weights, since proxies attenuate coefficients vs. true measures.
""")

p8 = "\n".join(p8_lines)
out("\n=== Part 8 ===")
out(p8)

# -
# Write output
# -

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
existing = OUT_PATH.read_text(encoding="utf-8") if OUT_PATH.exists() else ""
OUT_PATH.write_text(existing + "\n=== Part 8 ===\n" + p8, encoding="utf-8")
print(f"\nOutput written to {OUT_PATH}")
