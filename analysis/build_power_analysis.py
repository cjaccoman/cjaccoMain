"""Power metric stickiness and MLB PPPA predictiveness analysis.

Metrics tested: HR/FB, HR_AB, PullAir%, MaxEV, EV90 (PS), xSLG (PS), Barrel%BBE (PS).
ISO is explicitly excluded.

Output: data/historical/power_analysis_output.txt
"""

import sys
import numpy as np
import pandas as pd
import glob
import os
from pathlib import Path
from itertools import combinations
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = DATA_DIR / "historical" / "power_analysis_output.txt"

# ---------------------------------------------------------------------------
# Tee: write to stdout and file simultaneously
# ---------------------------------------------------------------------------
class Tee:
    def __init__(self, path):
        self._file = open(path, "w", encoding="utf-8")
        self._stdout = sys.stdout
    def write(self, msg):
        self._stdout.write(msg)
        self._file.write(msg)
    def flush(self):
        self._stdout.flush()
        self._file.flush()
    def close(self):
        self._file.close()

tee = Tee(OUT_PATH)
sys.stdout = tee

SEP = "=" * 72


def sep(title):
    print(f"\n{SEP}\n{title}\n{SEP}")


# ---------------------------------------------------------------------------
# Load base data
# ---------------------------------------------------------------------------
sep("LOADING DATA")

pf = pd.read_parquet(DATA_DIR / "rankings" / "prospect_features.parquet")
print(f"prospect_features: {len(pf):,} rows")

# Null impossible PullAir% values (>1 are bad rows)
pf.loc[pf["PullAir%"] > 1.0, "PullAir%"] = np.nan
# Null MaxEV/EV90 zeros (PS outputs 0.0 for missing, not NaN)
pf.loc[pf["MaxEV"] <= 0, "MaxEV"] = np.nan
pf.loc[pf["EV90"] <= 0, "EV90"] = np.nan
# Null HR/FB > 1.0
pf.loc[pf["HR/FB"] > 1.0, "HR/FB"] = np.nan

pa50 = pf[pf["PA"] >= 50].copy()
print(f"Rows with PA >= 50: {len(pa50):,}")

pc = pd.read_csv(DATA_DIR / "computed" / "player_comps.csv")
# Null impossible values in player_comps career columns
pc.loc[pc["HRFB_career"] > 1.0, "HRFB_career"] = np.nan
print(f"player_comps: {len(pc):,} rows")

mlb = pd.read_parquet(DATA_DIR / "historical" / "hist_mlb_data.parquet")
mlb["MLB_HR_AB"] = mlb["HR"] / (mlb["PA"] - mlb["BB"] - mlb["IBB"]).replace(0, np.nan)
print(f"hist_mlb_data: {len(mlb):,} rows")

mlb_sc = pd.read_csv(DATA_DIR / "api" / "mlb_statcast.csv",
                     usecols=["Season", "MLBAM_ID", "MaxEV", "Brl%", "xSLG", "EV95%"])
print(f"mlb_statcast: {len(mlb_sc):,} rows")

chad = pd.read_csv(DATA_DIR / "api" / "chadwick.csv")
chad = chad.dropna(subset=["key_fangraphs"])
chad["key_fangraphs"] = chad["key_fangraphs"].astype(int).astype(str)
chad["key_mlbam"] = chad["key_mlbam"].astype(int)
fg_to_mlbam = dict(zip(chad["key_fangraphs"], chad["key_mlbam"]))
mlbam_to_fg  = dict(zip(chad["key_mlbam"], chad["key_fangraphs"]))

# ---------------------------------------------------------------------------
# Load ProspectSavant data (xSLG + Barrel%BBE)
# ---------------------------------------------------------------------------
ps_files = sorted(glob.glob(str(DATA_DIR / "prospectSavant" / "ps_*.csv")))
ps_frames = []
level_map = {"AAA": "AAA", "AA": "AA", "Ap": "A+", "A": "A", "Rk": "R"}

for f in ps_files:
    fname = os.path.basename(f)
    if "savant" in fname:
        continue
    # parse level from filename: ps_AAA_2023.csv, ps_Ap_2026.csv etc
    parts = fname.replace("ps_", "").replace(".csv", "").split("_")
    lvl_raw = parts[0]
    yr = int(parts[1])
    lvl = level_map.get(lvl_raw, lvl_raw)
    df = pd.read_csv(f, usecols=["MLBAMId", "PA", "xSLG", "Barrel%BBE", "Barrel%PA"])
    # Barrel%BBE = 0 for AA and A+ (no real data) -- treat as null
    if df["Barrel%BBE"].max() == 0:
        df["Barrel%BBE"] = np.nan
        df["Barrel%PA"] = np.nan
    df = df.rename(columns={"MLBAMId": "MLBAM_ID_ps"})
    df["Season"] = yr
    df["Level"] = lvl
    ps_frames.append(df)

ps_all = pd.concat(ps_frames, ignore_index=True)
ps_all = ps_all[ps_all["PA"] >= 50].copy()
print(f"ProspectSavant rows (PA>=50): {len(ps_all):,}")
print(f"  xSLG non-null: {ps_all['xSLG'].notna().sum():,}")
print(f"  Barrel%BBE non-null: {ps_all['Barrel%BBE'].notna().sum():,}")

# Career averages from PS (weighted by PA)
def pa_weighted_career(df, col, id_col="MLBAM_ID_ps"):
    valid = df.dropna(subset=[col])
    num = (valid[col] * valid["PA"]).groupby(valid[id_col]).sum()
    den = valid.groupby(id_col)["PA"].sum()
    return (num / den).rename(col + "_career_ps")

xslg_career = pa_weighted_career(ps_all, "xSLG")
brl_career   = pa_weighted_career(ps_all, "Barrel%BBE")

ps_career = pd.concat([xslg_career, brl_career], axis=1).reset_index()
ps_career = ps_career.rename(columns={"MLBAM_ID_ps": "MLBAM_ID"})
print(f"PS career rows (xSLG): {ps_career['xSLG_career_ps'].notna().sum():,}")
print(f"PS career rows (Barrel%BBE): {ps_career['Barrel%BBE_career_ps'].notna().sum():,}")

# Merge PS career into player_comps (pc has MLBAM_ID)
pc2 = pc.merge(ps_career, on="MLBAM_ID", how="left")

# ---------------------------------------------------------------------------
# Graduated players for translation/multivariate analysis
# ---------------------------------------------------------------------------
grad = pc2[pc2["graduated"] == True].copy()
# Get first-year MLB outcomes
mlb_debut = (
    mlb.sort_values("Season")
       .groupby("PlayerId")
       .first()
       .reset_index()
       [["PlayerId", "Season", "PA", "PPPA_Z", "MLB_HR_AB"]]
       .rename(columns={"Season": "Debut_Season", "PA": "Debut_PA",
                        "PPPA_Z": "FirstYr_PPPA_Z_mlb", "MLB_HR_AB": "FirstYr_HR_AB_mlb"})
)
# Career MLB outcomes
mlb_career = (
    mlb.groupby("PlayerId")
       .apply(lambda g: pd.Series({
           "Career_MLB_PA_raw": g["PA"].sum(),
           "Career_PPPA_Z_raw": (g["PPPA_Z"] * g["PA"]).sum() / g["PA"].sum(),
           "Career_HR_AB_raw": g["HR"].sum() / max(1, (g["PA"] - g["BB"] - g["IBB"]).sum()),
       }), include_groups=False)
       .reset_index()
)

grad = grad.merge(mlb_debut, left_on="PlayerId", right_on="PlayerId", how="left")
grad = grad.merge(mlb_career, on="PlayerId", how="left")

grad100 = grad[grad["Debut_PA"] >= 100].copy()
grad200 = grad[grad["Career_MLB_PA"] >= 200].copy()
print(f"\nGraduated players: {len(grad):,}")
print(f"  with Debut_PA >= 100: {len(grad100):,}")
print(f"  with Career_MLB_PA >= 200: {len(grad200):,}")


# ---------------------------------------------------------------------------
# Part 1 -- Coverage Audit
# ---------------------------------------------------------------------------
sep("PART 1 -- COVERAGE AUDIT")

METRICS = {
    "HR/FB":      pa50["HR/FB"],
    "HR_AB":      pa50["HR_AB"],
    "PullAir%":   pa50["PullAir%"],
    "MaxEV":      pa50["MaxEV"],
    "EV90":       pa50["EV90"],
}

print(f"\n{'Metric':<14} {'Non-null':>9} {'Coverage%':>10} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("-" * 75)
for name, s in METRICS.items():
    nn = s.notna().sum()
    pct = nn / len(pa50) * 100
    print(f"{name:<14} {nn:>9,} {pct:>9.1f}% {s.mean():>8.3f} {s.std():>8.3f} {s.min():>8.3f} {s.max():>8.3f}")

# PS metrics coverage
print(f"\nProspectSavant metrics (from ps_*.csv files, PA >= 50):")
print(f"{'Metric':<18} {'Non-null':>9} {'Levels/Seasons'}")
print("-" * 60)
for col in ["xSLG", "Barrel%BBE"]:
    sub = ps_all.dropna(subset=[col])
    coverage = sub.groupby(["Level","Season"]).size().reset_index()
    lvl_yr_list = ", ".join(f"{r.Level}{r.Season}" for _, r in coverage.iterrows())
    print(f"{col:<18} {len(sub):>9,}  {lvl_yr_list}")

print(f"\nEV metrics by level (MaxEV non-null, PA>=50):")
ev_cov = pa50[pa50["MaxEV"].notna()].groupby(["Level","Season"]).size().reset_index(name="N")
print(ev_cov.to_string(index=False))


# ---------------------------------------------------------------------------
# Part 2 -- YoY Stickiness
# ---------------------------------------------------------------------------
sep("PART 2 -- YEAR-OVER-YEAR STICKINESS")

STICK_COLS = ["HR/FB", "HR_AB", "PullAir%"]  # MaxEV/EV90 too sparse for YoY

print("\nSame level, consecutive seasons (N1 and N2 both PA>=50):")
print(f"\n{'Metric':<14} {'Same-Level r':>13} {'N pairs':>8} {'Across-Level r':>15} {'N pairs':>8}")
print("-" * 65)

for col in STICK_COLS:
    sub = pa50[["PlayerId", "Season", "Level", "PA", col]].dropna(subset=[col]).copy()

    # Same level YoY
    sub2 = sub.merge(
        sub[["PlayerId", "Season", "Level", col]].rename(
            columns={"Season": "Season_next", col: col + "_next"}),
        on=["PlayerId", "Level"],
    )
    sub2 = sub2[sub2["Season_next"] == sub2["Season"] + 1]
    r_same = sub2[col].corr(sub2[col + "_next"])
    n_same = len(sub2)

    # Across level: year N at level L -> year N+1 at L+1
    level_order = {"R": 0, "A": 1, "A+": 2, "AA": 3, "AAA": 4}
    sub["Level_num"] = sub["Level"].map(level_order)
    sub3 = sub.merge(
        sub[["PlayerId", "Season", "Level_num", col]].rename(
            columns={"Season": "Season_next", "Level_num": "Level_num_next", col: col + "_next"}),
        on="PlayerId",
    )
    sub3 = sub3[
        (sub3["Season_next"] == sub3["Season"] + 1) &
        (sub3["Level_num_next"] == sub3["Level_num"] + 1)
    ]
    r_cross = sub3[col].corr(sub3[col + "_next"]) if len(sub3) >= 30 else np.nan
    n_cross = len(sub3)

    print(f"{col:<14} {r_same:>13.3f} {n_same:>8,} {r_cross:>15.3f} {n_cross:>8,}")

# Also test xSLG YoY from PS
ps_yoy = ps_all[["MLBAM_ID_ps", "Season", "Level", "xSLG", "Barrel%BBE"]].dropna(subset=["xSLG"])
ps_yoy2 = ps_yoy.merge(
    ps_yoy[["MLBAM_ID_ps", "Season", "Level", "xSLG", "Barrel%BBE"]].rename(
        columns={"Season": "Season_next", "xSLG": "xSLG_next", "Barrel%BBE": "Brl_next"}),
    on=["MLBAM_ID_ps", "Level"],
)
ps_yoy2 = ps_yoy2[ps_yoy2["Season_next"] == ps_yoy2["Season"] + 1]
for col, col_next, n_col in [("xSLG", "xSLG_next", "xSLG"), ("Barrel%BBE", "Brl_next", "Barrel%BBE")]:
    sub_yoy = ps_yoy2.dropna(subset=[col, col_next])
    r_yoy = sub_yoy[col].corr(sub_yoy[col_next])
    print(f"{'PS '+col:<14} {r_yoy:>13.3f} {len(sub_yoy):>8,}  (PS data, AAA 2023-2026)")


# ---------------------------------------------------------------------------
# Part 3 -- MiLB Career -> MLB Translation
# ---------------------------------------------------------------------------
sep("PART 3 -- MiLB CAREER AVG -> MLB TRANSLATION")

CAREER_METRICS = {
    "HRFB_career":    "HR/FB",
    "PullAir_career": "PullAir%",
    "MaxEV_career":   "MaxEV (sparse)",
    "EV90_career":    "EV90 (sparse)",
}

print(f"\nOutcome 1: First-year MLB HR_AB (Debut_PA >= 100)")
print(f"{'MiLB Metric':<20} {'r raw':>8} {'N':>6} {'r partial':>10}  (controlling K%+BB2K)")
print("-" * 60)
base_g = grad100.dropna(subset=["FirstYr_HR_AB_mlb", "Kpct_career", "BB2K_career"])
for col, label in CAREER_METRICS.items():
    sub = base_g.dropna(subset=[col])
    if len(sub) < 30:
        print(f"{label:<20} {'n/a':>8} {len(sub):>6}  (too few)")
        continue
    r_raw = sub[col].corr(sub["FirstYr_HR_AB_mlb"])
    # partial correlation: regress out K% and BB2K from both
    from sklearn.linear_model import LinearRegression
    controls = sub[["Kpct_career", "BB2K_career"]].values
    X = controls
    resid_milb = sub[col].values - LinearRegression().fit(X, sub[col].values).predict(X)
    resid_mlb  = sub["FirstYr_HR_AB_mlb"].values - LinearRegression().fit(X, sub["FirstYr_HR_AB_mlb"].values).predict(X)
    r_partial = pd.Series(resid_milb).corr(pd.Series(resid_mlb))
    print(f"{label:<20} {r_raw:>8.3f} {len(sub):>6,} {r_partial:>10.3f}")

print(f"\nOutcome 2: Career MLB PPPA_Z (Career_MLB_PA >= 200)")
print(f"{'MiLB Metric':<20} {'r raw':>8} {'N':>6} {'r partial':>10}  (controlling K%+BB2K)")
print("-" * 60)
base_c = grad200.dropna(subset=["Career_PPPA_Z_raw", "Kpct_career", "BB2K_career"])
for col, label in CAREER_METRICS.items():
    sub = base_c.dropna(subset=[col])
    if len(sub) < 30:
        print(f"{label:<20} {'n/a':>8} {len(sub):>6}  (too few)")
        continue
    r_raw = sub[col].corr(sub["Career_PPPA_Z_raw"])
    controls = sub[["Kpct_career", "BB2K_career"]].values
    resid_milb = sub[col].values - LinearRegression().fit(controls, sub[col].values).predict(controls)
    resid_mlb  = sub["Career_PPPA_Z_raw"].values - LinearRegression().fit(controls, sub["Career_PPPA_Z_raw"].values).predict(controls)
    r_partial  = pd.Series(resid_milb).corr(pd.Series(resid_mlb))
    print(f"{label:<20} {r_raw:>8.3f} {len(sub):>6,} {r_partial:>10.3f}")

# Also test HR_AB career -- compute from hist_mlb
pf_agg = pa50[["PlayerId", "Season", "Level", "PA", "HR_AB", "HR/FB", "PullAir%"]].copy()
hr_ab_career = (
    pf_agg.dropna(subset=["HR_AB"])
          .assign(wtd=lambda d: d["HR_AB"] * d["PA"])
          .groupby("PlayerId")
          .apply(lambda g: g["wtd"].sum() / g["PA"].sum(), include_groups=False)
          .rename("HR_AB_career")
          .reset_index()
)
# join to grad200
grad200_ext = grad200.merge(hr_ab_career, on="PlayerId", how="left")
base_ext = grad200_ext.dropna(subset=["Career_PPPA_Z_raw", "Kpct_career", "BB2K_career", "HR_AB_career"])
r_hrab = base_ext["HR_AB_career"].corr(base_ext["Career_PPPA_Z_raw"])
controls = base_ext[["Kpct_career", "BB2K_career"]].values
r_hrab_milb = base_ext["HR_AB_career"].values - LinearRegression().fit(controls, base_ext["HR_AB_career"].values).predict(controls)
r_hrab_mlb  = base_ext["Career_PPPA_Z_raw"].values - LinearRegression().fit(controls, base_ext["Career_PPPA_Z_raw"].values).predict(controls)
r_hrab_partial = pd.Series(r_hrab_milb).corr(pd.Series(r_hrab_mlb))
print(f"{'HR_AB (career)':<20} {r_hrab:>8.3f} {len(base_ext):>6,} {r_hrab_partial:>10.3f}  (HR_AB from pf agg)")


# ---------------------------------------------------------------------------
# Part 4 -- Multivariate: which combination best predicts Career PPPA_Z?
# ---------------------------------------------------------------------------
sep("PART 4 -- MULTIVARIATE: BEST POWER COMBINATION -> CAREER PPPA_Z")

grad200_mv = grad200_ext.dropna(subset=["Career_PPPA_Z_raw"]).copy()
mv_features = {
    "HRFB":    "HRFB_career",
    "HR_AB":   "HR_AB_career",
    "PullAir": "PullAir_career",
}
# Add discipline as baseline
mv_features["BB2K"]  = "BB2K_career"
mv_features["Kpct"]  = "Kpct_career"

# All subsets up to 3 power variables (exclude discipline for the power-only test)
power_vars = list(mv_features.keys())[:3]  # HRFB, HR_AB, PullAir
disc_vars   = list(mv_features.keys())[3:]  # BB2K, Kpct

from itertools import combinations
import warnings

def adj_r2(X, y):
    n, p = X.shape
    model = LinearRegression().fit(X, y)
    ss_res = ((y - model.predict(X))**2).sum()
    ss_tot = ((y - y.mean())**2).sum()
    r2 = 1 - ss_res / ss_tot
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

print("\nBest single power metric:")
results = []
for k in power_vars:
    col = mv_features[k]
    sub = grad200_mv.dropna(subset=[col])
    if len(sub) < 50:
        continue
    X = sub[[col]].values
    y = sub["Career_PPPA_Z_raw"].values
    ar2 = adj_r2(X, y)
    r   = np.corrcoef(sub[col].values, y)[0, 1]
    results.append((k, [col], ar2, r, len(sub)))

results.sort(key=lambda x: -x[2])
for name, cols, ar2, r, n in results:
    print(f"  {name:<12} adj_R?={ar2:.3f}  r={r:.3f}  N={n:,}")

print("\nBest 2-metric power combo:")
combos2 = []
for c1, c2 in combinations(power_vars, 2):
    cols = [mv_features[c1], mv_features[c2]]
    sub = grad200_mv.dropna(subset=cols)
    if len(sub) < 50:
        continue
    X = sub[cols].values
    y = sub["Career_PPPA_Z_raw"].values
    ar2 = adj_r2(X, y)
    combos2.append((f"{c1}+{c2}", cols, ar2, len(sub)))
combos2.sort(key=lambda x: -x[2])
for name, cols, ar2, n in combos2[:3]:
    print(f"  {name:<20} adj_R?={ar2:.3f}  N={n:,}")

print("\nBest 3-metric combo:")
all3_cols = [mv_features[k] for k in power_vars]
sub = grad200_mv.dropna(subset=all3_cols)
if len(sub) >= 50:
    ar2_all3 = adj_r2(sub[all3_cols].values, sub["Career_PPPA_Z_raw"].values)
    print(f"  HRFB+HR_AB+PullAir  adj_R?={ar2_all3:.3f}  N={len(sub):,}")

print("\nWith discipline added (best power + K% + BB2K):")
for k in power_vars:
    col = mv_features[k]
    disc_c = [mv_features["BB2K"], mv_features["Kpct"]]
    all_c  = [col] + disc_c
    sub = grad200_mv.dropna(subset=all_c)
    if len(sub) < 50:
        continue
    ar2_disc = adj_r2(sub[all_c].values, sub["Career_PPPA_Z_raw"].values)
    # base: discipline only
    sub_base = grad200_mv.dropna(subset=disc_c)
    ar2_base = adj_r2(sub_base[disc_c].values, sub_base["Career_PPPA_Z_raw"].values)
    print(f"  {k}+disc  adj_R?={ar2_disc:.3f}  (discipline alone={ar2_base:.3f})  N={len(sub):,}  +?R?={ar2_disc-ar2_base:.3f}")


# ---------------------------------------------------------------------------
# Part 5 -- Exit Velocity Specifically
# ---------------------------------------------------------------------------
sep("PART 5 -- EXIT VELOCITY: GENUINE PS DATA ONLY")

# Genuine MaxEV = not imputed = from player_comps MaxEV_career where we know it's PS
# player_comps MaxEV_career was computed from prospect_features which zeroed nulls
# Use players where MaxEV_career is non-null AND non-zero
ev_grad = grad200[grad200["MaxEV_career"].notna() & (grad200["MaxEV_career"] > 0)].copy()
print(f"\nGraduated players with genuine MaxEV_career: {len(ev_grad):,} / {len(grad200):,} ({len(ev_grad)/len(grad200)*100:.1f}%)")

if len(ev_grad) >= 20:
    r_maxev = ev_grad["MaxEV_career"].corr(ev_grad["Career_PPPA_Z_raw"])
    r_hrfb_ev = ev_grad["HRFB_career"].corr(ev_grad["Career_PPPA_Z_raw"])
    print(f"\nIn MaxEV subset (N={len(ev_grad):,}):")
    print(f"  r(MaxEV_career, Career_PPPA_Z):    {r_maxev:.3f}")
    print(f"  r(HRFB_career, Career_PPPA_Z):     {r_hrfb_ev:.3f}  (same subset)")

    # Does MaxEV add beyond HRFB?
    ev_sub = ev_grad.dropna(subset=["MaxEV_career", "HRFB_career", "Career_PPPA_Z_raw"])
    if len(ev_sub) >= 20:
        X_hrfb = ev_sub[["HRFB_career"]].values
        X_both = ev_sub[["HRFB_career", "MaxEV_career"]].values
        y = ev_sub["Career_PPPA_Z_raw"].values
        ar2_hrfb = adj_r2(X_hrfb, y)
        ar2_both = adj_r2(X_both, y)
        print(f"  adj_R? HRFB alone:       {ar2_hrfb:.3f}")
        print(f"  adj_R? HRFB + MaxEV:     {ar2_both:.3f}  (?R?={ar2_both-ar2_hrfb:+.3f})")

        # EV90 separately
        ev_sub2 = ev_grad.dropna(subset=["EV90_career", "HRFB_career", "Career_PPPA_Z_raw"])
        if len(ev_sub2) >= 20:
            r_ev90 = ev_sub2["EV90_career"].corr(ev_sub2["Career_PPPA_Z_raw"])
            X_both2 = ev_sub2[["HRFB_career", "EV90_career"]].values
            ar2_both2 = adj_r2(X_both2, ev_sub2["Career_PPPA_Z_raw"].values)
            ar2_hrfb2 = adj_r2(ev_sub2[["HRFB_career"]].values, ev_sub2["Career_PPPA_Z_raw"].values)
            print(f"  r(EV90_career, Career_PPPA_Z):     {r_ev90:.3f}")
            print(f"  adj_R? HRFB + EV90:      {ar2_both2:.3f}  (?R?={ar2_both2-ar2_hrfb2:+.3f})")

    # Compare MaxEV to MLB Brl% (Statcast) -- do MiLB MaxEV players have higher MLB Brl%?
    ev_mlbam = ev_grad.dropna(subset=["MLBAM_ID", "MaxEV_career"])
    mlb_brl = mlb_sc.groupby("MLBAM_ID").agg({"Brl%": "mean", "MaxEV": "mean"}).reset_index()
    ev_sc_join = ev_mlbam.merge(mlb_brl, left_on="MLBAM_ID", right_on="MLBAM_ID", how="inner")
    if len(ev_sc_join) >= 20:
        r_milb_mlb_ev = ev_sc_join["MaxEV_career"].corr(ev_sc_join["MaxEV"])
        r_milb_mlb_brl = ev_sc_join["MaxEV_career"].corr(ev_sc_join["Brl%"])
        print(f"\n  MiLB MaxEV -> MLB MaxEV: r={r_milb_mlb_ev:.3f}  N={len(ev_sc_join):,}")
        print(f"  MiLB MaxEV -> MLB Brl%:  r={r_milb_mlb_brl:.3f}  N={len(ev_sc_join):,}")
else:
    print(f"  Too few EV players ({len(ev_grad)}) for meaningful analysis")


# ---------------------------------------------------------------------------
# Part 6 -- xSLG and Barrel% from ProspectSavant
# ---------------------------------------------------------------------------
sep("PART 6 -- xSLG AND BARREL%BBE (ProspectSavant)")

# grad200 already has xSLG_career_ps and Barrel%BBE_career_ps from the pc2 merge above
# (pc2 = pc.merge(ps_career, ...) and grad200 = pc2[graduated==True])
grad200_ps2 = grad200

xslg_valid = grad200_ps2.dropna(subset=["xSLG_career_ps", "Career_PPPA_Z_raw", "HRFB_career"])
brl_valid   = grad200_ps2.dropna(subset=["Barrel%BBE_career_ps", "Career_PPPA_Z_raw", "HRFB_career"])

print(f"\nxSLG_career_ps: {grad200_ps2['xSLG_career_ps'].notna().sum():,} / {len(grad200_ps2):,} grad players")
print(f"Barrel%BBE_career_ps: {grad200_ps2['Barrel%BBE_career_ps'].notna().sum():,} / {len(grad200_ps2):,} grad players")

if len(xslg_valid) >= 30:
    r_xslg_pppa = xslg_valid["xSLG_career_ps"].corr(xslg_valid["Career_PPPA_Z_raw"])
    r_xslg_hrab = xslg_valid["xSLG_career_ps"].corr(xslg_valid["Career_HR_AB_raw"])
    r_hrfb_pppa = xslg_valid["HRFB_career"].corr(xslg_valid["Career_PPPA_Z_raw"])
    print(f"\nIn xSLG subset (N={len(xslg_valid):,}):")
    print(f"  r(xSLG, Career_PPPA_Z):  {r_xslg_pppa:.3f}")
    print(f"  r(xSLG, Career_HR_AB):   {r_xslg_hrab:.3f}")
    print(f"  r(HRFB, Career_PPPA_Z):  {r_hrfb_pppa:.3f}  (same subset, for comparison)")

    # partial r: xSLG controlling for HRFB
    ctrl = xslg_valid[["HRFB_career"]].values
    res_xslg = xslg_valid["xSLG_career_ps"].values - LinearRegression().fit(ctrl, xslg_valid["xSLG_career_ps"].values).predict(ctrl)
    res_pppa  = xslg_valid["Career_PPPA_Z_raw"].values - LinearRegression().fit(ctrl, xslg_valid["Career_PPPA_Z_raw"].values).predict(ctrl)
    r_xslg_partial = pd.Series(res_xslg).corr(pd.Series(res_pppa))
    print(f"  r(xSLG partial, PPPA_Z):  {r_xslg_partial:.3f}  (controlling HR/FB)")

    # xSLG + HRFB combined
    X_comb = xslg_valid[["xSLG_career_ps", "HRFB_career"]].values
    y = xslg_valid["Career_PPPA_Z_raw"].values
    ar2_comb = adj_r2(X_comb, y)
    ar2_hrfb_only = adj_r2(xslg_valid[["HRFB_career"]].values, y)
    ar2_xslg_only = adj_r2(xslg_valid[["xSLG_career_ps"]].values, y)
    print(f"  adj_R? xSLG alone:   {ar2_xslg_only:.3f}")
    print(f"  adj_R? HRFB alone:   {ar2_hrfb_only:.3f}")
    print(f"  adj_R? xSLG+HRFB:   {ar2_comb:.3f}")
else:
    print(f"\n  Too few xSLG players ({len(xslg_valid)}) for analysis")

if len(brl_valid) >= 30:
    r_brl = brl_valid["Barrel%BBE_career_ps"].corr(brl_valid["Career_PPPA_Z_raw"])
    r_brl_hrab = brl_valid["Barrel%BBE_career_ps"].corr(brl_valid["Career_HR_AB_raw"])
    r_hrfb_brl = brl_valid["HRFB_career"].corr(brl_valid["Career_PPPA_Z_raw"])
    print(f"\nIn Barrel%BBE subset (N={len(brl_valid):,}):")
    print(f"  r(Barrel%BBE, Career_PPPA_Z):  {r_brl:.3f}")
    print(f"  r(Barrel%BBE, Career_HR_AB):   {r_brl_hrab:.3f}")
    print(f"  r(HRFB, Career_PPPA_Z):        {r_hrfb_brl:.3f}  (same subset)")

    ctrl_b = brl_valid[["HRFB_career"]].values
    res_brl  = brl_valid["Barrel%BBE_career_ps"].values - LinearRegression().fit(ctrl_b, brl_valid["Barrel%BBE_career_ps"].values).predict(ctrl_b)
    res_pppa_b = brl_valid["Career_PPPA_Z_raw"].values - LinearRegression().fit(ctrl_b, brl_valid["Career_PPPA_Z_raw"].values).predict(ctrl_b)
    r_brl_partial = pd.Series(res_brl).corr(pd.Series(res_pppa_b))
    print(f"  r(Barrel%BBE partial, PPPA_Z): {r_brl_partial:.3f}  (controlling HR/FB)")
else:
    print(f"\n  Too few Barrel%BBE players ({len(brl_valid)}) for meaningful analysis")


# ---------------------------------------------------------------------------
# Part 7 -- Summary Ranking
# ---------------------------------------------------------------------------
sep("PART 7 -- SUMMARY RANKING")

# Collect stickiness, translation, coverage for all metrics
summary = []

# HR/FB
yoy_sub = pa50[["PlayerId","Season","Level","PA","HR/FB"]].dropna(subset=["HR/FB"])
yoy_sub2 = yoy_sub.merge(yoy_sub[["PlayerId","Season","Level","HR/FB"]].rename(
    columns={"Season":"S2","HR/FB":"HRFB2"}), on=["PlayerId","Level"])
yoy_sub2 = yoy_sub2[yoy_sub2["S2"] == yoy_sub2["Season"]+1]
r_stick_hrfb = yoy_sub2["HR/FB"].corr(yoy_sub2["HRFB2"])

base_t = grad200.dropna(subset=["HRFB_career","Career_PPPA_Z_raw"])
r_trans_hrfb = base_t["HRFB_career"].corr(base_t["Career_PPPA_Z_raw"])
cov_hrfb = pa50["HR/FB"].notna().mean()
summary.append(("HR/FB",      r_stick_hrfb, r_trans_hrfb, cov_hrfb))

# HR_AB
yoy_hrab = pa50[["PlayerId","Season","Level","PA","HR_AB"]].dropna(subset=["HR_AB"])
yoy_hrab2 = yoy_hrab.merge(yoy_hrab[["PlayerId","Season","Level","HR_AB"]].rename(
    columns={"Season":"S2","HR_AB":"HRAB2"}), on=["PlayerId","Level"])
yoy_hrab2 = yoy_hrab2[yoy_hrab2["S2"] == yoy_hrab2["Season"]+1]
r_stick_hrab = yoy_hrab2["HR_AB"].corr(yoy_hrab2["HRAB2"])
r_trans_hrab = grad200_ext.dropna(subset=["HR_AB_career","Career_PPPA_Z_raw"])["HR_AB_career"].corr(
    grad200_ext.dropna(subset=["HR_AB_career","Career_PPPA_Z_raw"])["Career_PPPA_Z_raw"])
cov_hrab = pa50["HR_AB"].notna().mean()
summary.append(("HR_AB",      r_stick_hrab, r_trans_hrab, cov_hrab))

# PullAir%
yoy_pa = pa50[["PlayerId","Season","Level","PA","PullAir%"]].dropna(subset=["PullAir%"])
yoy_pa2 = yoy_pa.merge(yoy_pa[["PlayerId","Season","Level","PullAir%"]].rename(
    columns={"Season":"S2","PullAir%":"PA2"}), on=["PlayerId","Level"])
yoy_pa2 = yoy_pa2[yoy_pa2["S2"] == yoy_pa2["Season"]+1]
r_stick_pull = yoy_pa2["PullAir%"].corr(yoy_pa2["PA2"])
base_pull = grad200.dropna(subset=["PullAir_career","Career_PPPA_Z_raw"])
r_trans_pull = base_pull["PullAir_career"].corr(base_pull["Career_PPPA_Z_raw"])
cov_pull = pa50["PullAir%"].notna().mean()
summary.append(("PullAir%",   r_stick_pull, r_trans_pull, cov_pull))

# MaxEV (EV data)
r_stick_ev = np.nan  # sparse, can't reliably compute
r_trans_ev = ev_grad.dropna(subset=["MaxEV_career","Career_PPPA_Z_raw"])["MaxEV_career"].corr(
    ev_grad.dropna(subset=["MaxEV_career","Career_PPPA_Z_raw"])["Career_PPPA_Z_raw"]) if len(ev_grad) > 20 else np.nan
cov_ev = pa50["MaxEV"].notna().mean()
summary.append(("MaxEV",      r_stick_ev,   r_trans_ev,   cov_ev))

# EV90
r_trans_ev90 = ev_grad.dropna(subset=["EV90_career","Career_PPPA_Z_raw"])["EV90_career"].corr(
    ev_grad.dropna(subset=["EV90_career","Career_PPPA_Z_raw"])["Career_PPPA_Z_raw"]) if len(ev_grad) > 20 else np.nan
summary.append(("EV90",       r_stick_ev,   r_trans_ev90, cov_ev))

# xSLG
r_stick_xslg = ps_yoy2.dropna(subset=["xSLG","xSLG_next"])["xSLG"].corr(
    ps_yoy2.dropna(subset=["xSLG","xSLG_next"])["xSLG_next"]) if "xSLG_next" in ps_yoy2.columns else np.nan
r_trans_xslg = xslg_valid["xSLG_career_ps"].corr(xslg_valid["Career_PPPA_Z_raw"]) if len(xslg_valid) >= 30 else np.nan
cov_xslg = len(ps_all.dropna(subset=["xSLG"])) / len(pa50)
summary.append(("xSLG (PS)",  r_stick_xslg, r_trans_xslg, cov_xslg))

# Barrel%BBE
r_stick_brl_val = ps_yoy2.dropna(subset=["Barrel%BBE","Brl_next"])["Barrel%BBE"].corr(
    ps_yoy2.dropna(subset=["Barrel%BBE","Brl_next"])["Brl_next"]) if "Brl_next" in ps_yoy2.columns else np.nan
r_trans_brl = brl_valid["Barrel%BBE_career_ps"].corr(brl_valid["Career_PPPA_Z_raw"]) if len(brl_valid) >= 30 else np.nan
cov_brl = len(ps_all.dropna(subset=["Barrel%BBE"])) / len(pa50)
summary.append(("Barrel%BBE", r_stick_brl_val, r_trans_brl, cov_brl))

print(f"\n{'Metric':<14} {'Stickiness':>11} {'Translation':>12} {'Coverage':>10} {'Usefulness':>11}")
print(f"{'':14} {'(YoY r)':>11} {'(r, CareerZ)':>12} {'(PA>=50)':>10} {'s?t??c':>11}")
print("-" * 65)

for name, stick, trans, cov in sorted(summary, key=lambda x: -(x[1]*x[2]*np.sqrt(x[3])) if not np.isnan(x[1]*x[2]) else -99):
    if np.isnan(stick):
        stick_s = "    n/a"
    else:
        stick_s = f"{stick:>11.3f}"
    if np.isnan(trans):
        trans_s = "         n/a"
    else:
        trans_s = f"{trans:>12.3f}"
    cov_pct = f"{cov*100:>9.1f}%"
    if not np.isnan(stick) and not np.isnan(trans):
        use = stick * trans * np.sqrt(cov)
    else:
        use = np.nan
    use_s = f"{use:>11.3f}" if not np.isnan(use) else "        n/a"
    print(f"{name:<14} {stick_s} {trans_s} {cov_pct} {use_s}")


# ---------------------------------------------------------------------------
# Part 8 -- Implications for Model
# ---------------------------------------------------------------------------
sep("PART 8 -- IMPLICATIONS FOR CURRENT MODEL")

print("""
Current ABILITY Game_Power: 0.5 ? PullAir% + 0.5 ? HR_AB
Current TOOLS Power fallback: HRFB_adj (100%)
Current TOOLS Power full tier: MaxEV (35%) + EV90 (35%) + HRFB (30%)
""")

# Direct comparison: which single metric best predicts career PPPA_Z?
best = max([(name, trans) for name, stick, trans, cov in summary
            if not np.isnan(trans) and cov > 0.03], key=lambda x: x[1])
print(f"Best single predictor of Career_PPPA_Z (among adequately-covered metrics): {best[0]} (r={best[1]:.3f})")

# Compare current formula to HRFB alone
formula_sub = grad200_ext.dropna(subset=["HR_AB_career", "PullAir_career", "Career_PPPA_Z_raw", "HRFB_career"])
if len(formula_sub) >= 50:
    formula_sub["current_formula"] = 0.5 * formula_sub["HR_AB_career"] + 0.5 * formula_sub["PullAir_career"]
    r_formula = formula_sub["current_formula"].corr(formula_sub["Career_PPPA_Z_raw"])
    r_hrfb_only = formula_sub["HRFB_career"].corr(formula_sub["Career_PPPA_Z_raw"])
    r_hrab_only = formula_sub["HR_AB_career"].corr(formula_sub["Career_PPPA_Z_raw"])
    r_pull_only = formula_sub["PullAir_career"].corr(formula_sub["Career_PPPA_Z_raw"])
    X_formula = formula_sub[["current_formula"]].values
    X_hrfb    = formula_sub[["HRFB_career"]].values
    X_both    = formula_sub[["HRFB_career","HR_AB_career"]].values
    X_all3    = formula_sub[["HRFB_career","HR_AB_career","PullAir_career"]].values
    y = formula_sub["Career_PPPA_Z_raw"].values
    print(f"\nComparison (N={len(formula_sub):,}):")
    print(f"  Current formula (0.5?HR_AB + 0.5?PullAir%):  r={r_formula:.3f}  adj_R?={adj_r2(X_formula, y):.3f}")
    print(f"  HR_AB alone:                                   r={r_hrab_only:.3f}  adj_R?={adj_r2(formula_sub[['HR_AB_career']].values, y):.3f}")
    print(f"  PullAir% alone:                                r={r_pull_only:.3f}")
    print(f"  HR/FB alone:                                   r={r_hrfb_only:.3f}  adj_R?={adj_r2(X_hrfb, y):.3f}")
    print(f"  HR/FB + HR_AB:                                 adj_R?={adj_r2(X_both, y):.3f}")
    print(f"  HR/FB + HR_AB + PullAir%:                      adj_R?={adj_r2(X_all3, y):.3f}")

    # Recommendation
    ar2_formula = adj_r2(X_formula, y)
    ar2_hrfb = adj_r2(X_hrfb, y)
    ar2_both = adj_r2(X_both, y)

    print("\n--- RECOMMENDATION ---")
    if ar2_hrfb > ar2_formula + 0.01:
        print(f"HRFB alone outperforms the current formula by {ar2_hrfb - ar2_formula:.3f} adj_R?.")
        print("Suggest: replace 0.5?HR_AB + 0.5?PullAir% with HRFB_adj in ABILITY Game_Power.")
    elif ar2_both > ar2_formula + 0.01:
        print(f"HRFB + HR_AB (50/50) outperforms current formula by {ar2_both - ar2_formula:.3f} adj_R?.")
        print("Suggest: replace PullAir% share with HRFB in ABILITY Game_Power: 0.5?HRFB + 0.5?HR_AB.")
    else:
        print("Current formula (0.5?HR_AB + 0.5?PullAir%) is competitive with alternatives.")
        print("No strong statistical case for replacement -- PullAir% may add non-power dimension worth keeping.")

    if r_hrfb_only > r_pull_only + 0.05:
        print(f"\nNote: PullAir% (r={r_pull_only:.3f}) adds less signal than HR/FB (r={r_hrfb_only:.3f}).")
        print("TOOLS Power full tier using MaxEV + EV90 + HRFB is well-ordered.")

    if len(xslg_valid) >= 30 and not np.isnan(r_trans_xslg):
        print(f"\nxSLG (PS, limited coverage): r={r_trans_xslg:.3f} vs HRFB r={r_hrfb_pppa:.3f} in same subset.")
        if r_trans_xslg > r_hrfb_pppa + 0.05:
            print("xSLG outperforms HRFB within its coverage window -- warrants monitoring as PS expands.")
        else:
            print("xSLG does not meaningfully outperform HRFB; coverage < 10% makes adoption premature.")

print(f"\nOutput written to {OUT_PATH}")
sys.stdout = tee._stdout
tee.close()
print(f"Done. Output at {OUT_PATH}")
