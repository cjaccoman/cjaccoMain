"""Build AFL prospect scores mirroring the FT TOOLS / ABILITY formulas.

Standalone — not wired into the prospect pipeline. Scores are relative to AFL
peers within each season (no era adjustment, no level discount needed).

TOOLS_Score — physical skills (from afl_pitches_agg.csv):
  Discipline 45%:  −Chase%_z (40%) + ZContact%_z (35%) + −Whiff%_z (25%)
  Power      35%:  MaxEV_z (50%) + HardHit%_z (50%)
  Athleticism 20%: SB_PA_z (100%)

ABILITY_Score — demonstrated production (from afl_hitting.csv):
  Fantasy Output 47%: PPPA_z
  Discipline     28%: BB_2K_z  = (BB% − 2×K%)_z
  SB Talent       8%: (SB_pct × SB/PA)_z
  Game Power     17%: HR_AB_z  = (HR / AB)_z

Age adjustment on every TOOLS and ABILITY component: × (1 + 0.20 × −Age_Z),
clipped ±2 SD. Age_Z is z-scored within the AFL pool per season.

AFL_Score = 0.30 × TOOLS_Score + 0.50 × ABILITY_Score + 0.20 × Age_Score
All three standardized to 50±10 within the season pool, clipped at 0.

Outputs: data/api/afl_scores.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "api"
OUT_PATH = DATA_DIR / "afl_scores.csv"

MIN_PA     = 20    # minimum PA (from afl_hitting) to score
AFL_START  = (10, 1)  # (month, day) — approximate AFL season start for age calc
AGE_CLIP   = 2.0   # SD clip for age adjustment

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def z_score(s: pd.Series) -> pd.Series:
    """Z-score within the non-null values of s."""
    mu, sd = s.mean(), s.std()
    if sd == 0 or pd.isna(sd):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def age_adj(component_z: pd.Series, age_z: pd.Series) -> pd.Series:
    """Multiply each component z-score by (1 + 0.20 × −Age_Z), clipped ±AGE_CLIP SD."""
    adj_factor = 1.0 + 0.20 * (-age_z).clip(-AGE_CLIP, AGE_CLIP)
    return component_z * adj_factor


def standardize(s: pd.Series) -> pd.Series:
    """Map to 50 ± 10 scale and clip at 0."""
    mu, sd = s.mean(), s.std()
    if sd == 0 or pd.isna(sd):
        return pd.Series(50.0, index=s.index)
    return (50.0 + 10.0 * (s - mu) / sd).clip(lower=0)


def blend(*pairs) -> pd.Series:
    """Weighted blend of (series, weight) pairs; missing values get 0 (neutral)."""
    total_w = sum(w for _, w in pairs)
    result  = sum(s.fillna(0) * w for s, w in pairs)
    return result / total_w


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

print("Loading data...")

hit = pd.read_csv(
    DATA_DIR / "afl_hitting.csv",
    dtype={"MLBAM_ID": "Int64"},
    low_memory=False,
)

pit = pd.read_csv(
    DATA_DIR / "afl_pitches_agg.csv",
    dtype={"MLBAM_ID": "Int64"},
    low_memory=False,
)

bdays = pd.read_csv(
    DATA_DIR / "player_birthdays.csv",
    dtype={"MLBAM_ID": "Int64"},
)

print(f"  hitting:  {len(hit):,} rows")
print(f"  pitches:  {len(pit):,} rows")
print(f"  bdays:    {len(bdays):,} rows")

# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

# Pitch metrics needed for TOOLS
pit_cols = ["MLBAM_ID", "Season",
            "Chase%", "Z-Contact%", "Whiff%",
            "EV_Max", "HardHit%", "SB", "CS", "PA"]
pit_sub = pit[pit_cols].rename(columns={
    "Chase%":     "Chase_pct",
    "Z-Contact%": "ZContact_pct",
    "Whiff%":     "Whiff_pct_pit",
    "EV_Max":     "MaxEV",
    "SB":         "SB_pit",
    "CS":         "CS_pit",
    "PA":         "PA_pit",
})

df = hit.merge(pit_sub, on=["MLBAM_ID", "Season"], how="left")

# Apply PA filter from afl_hitting (official record)
df = df[df["PA"] >= MIN_PA].copy()
print(f"\nAfter PA ≥ {MIN_PA} filter: {len(df):,} player-seasons")

# ---------------------------------------------------------------------------
# Age (as of AFL start date for each season)
# ---------------------------------------------------------------------------

bdays["BirthDate"] = pd.to_datetime(bdays["BirthDate"], errors="coerce")
df = df.merge(bdays[["MLBAM_ID", "BirthDate"]], on="MLBAM_ID", how="left")

def calc_age(row) -> float:
    bd = row["BirthDate"]
    if pd.isna(bd):
        return np.nan
    ref = date(int(row["Season"]), AFL_START[0], AFL_START[1])
    return (ref - bd.date()).days / 365.25

df["Age"] = df.apply(calc_age, axis=1)

# Z-score Age within each season pool
df["Age_Z"] = df.groupby("Season")["Age"].transform(z_score)

# ---------------------------------------------------------------------------
# TOOLS components
# ---------------------------------------------------------------------------

# SB/PA from pitch data (more granular than hitting SB/PA)
df["SB_PA"] = (df["SB_pit"] / df["PA_pit"]).where(df["PA_pit"] > 0)

# Z-score everything within season
def season_z(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("Season")[col].transform(z_score)

df["Chase_z"]    = season_z(df, "Chase_pct")
df["ZCont_z"]    = season_z(df, "ZContact_pct")
df["Whiff_z"]    = season_z(df, "Whiff_pct_pit")
df["MaxEV_z"]    = season_z(df, "MaxEV")
df["HardHit_z"]  = season_z(df, "HardHit%")
df["SB_PA_z"]    = season_z(df, "SB_PA")

# Compute discipline, power, athleticism sub-scores (then age-adjust)
def tools_subcomp(df: pd.DataFrame) -> pd.DataFrame:
    age_z = df["Age_Z"].fillna(0)

    disc = blend(
        (-df["Chase_z"],   0.40),
        ( df["ZCont_z"],   0.35),
        (-df["Whiff_z"],   0.25),
    )
    power = blend(
        (df["MaxEV_z"],   0.50),
        (df["HardHit_z"], 0.50),
    )
    ath = df["SB_PA_z"].fillna(0)

    df["T_Disc"]  = age_adj(disc,  age_z)
    df["T_Power"] = age_adj(power, age_z)
    df["T_Ath"]   = ath   # athleticism: no age adj (sprint speed equivalent)

    df["TOOLS_raw"] = (
        0.45 * df["T_Disc"]
      + 0.35 * df["T_Power"]
      + 0.20 * df["T_Ath"]
    )
    return df

df = tools_subcomp(df)

# ---------------------------------------------------------------------------
# ABILITY components
# ---------------------------------------------------------------------------

# BB_2K = BB% − 2×K%
df["BB_2K"] = df["BB_pct"] - 2 * df["K_pct"]

# SBTalent = SB_pct × SB/PA
df["SBTalent"] = df["SB_pct"] * (df["SB"] / df["PA"].clip(lower=1))

# HR_AB = HR / AB (AB already excludes walks)
df["HR_AB"] = (df["HR"] / df["AB"]).where(df["AB"] > 0)

df["PPPA_z"]    = season_z(df, "PPPA")
df["BB2K_z"]    = season_z(df, "BB_2K")
df["SBT_z"]     = season_z(df, "SBTalent")
df["HRAB_z"]    = season_z(df, "HR_AB")

def ability_subcomp(df: pd.DataFrame) -> pd.DataFrame:
    age_z = df["Age_Z"].fillna(0)

    fantasy_out = df["PPPA_z"].fillna(0)
    disc        = df["BB2K_z"].fillna(0)
    sbt         = df["SBT_z"].fillna(0)
    power       = df["HRAB_z"].fillna(0)

    df["A_Output"] = age_adj(fantasy_out, age_z)
    df["A_Disc"]   = age_adj(disc,        age_z)
    df["A_SBT"]    = age_adj(sbt,         age_z)
    df["A_Power"]  = age_adj(power,       age_z)

    # Dynamic power/discipline scaling (mirrors FT ABILITY model)
    gp = df["A_Power"].clip(-3, 3)
    power_w = (0.17 + np.maximum(0, gp) * 0.05).clip(0.17, 0.32)
    disc_w  = (0.28 - np.maximum(0, gp) * 0.05).clip(0.13, 0.28)

    df["ABILITY_raw"] = (
        0.47 * df["A_Output"]
      + disc_w  * df["A_Disc"]
      + 0.08 * df["A_SBT"]
      + power_w * df["A_Power"]
    )
    return df

df = ability_subcomp(df)

# ---------------------------------------------------------------------------
# Final scores — standardize within season then blend
# ---------------------------------------------------------------------------

df["TOOLS_Score"]   = df.groupby("Season")["TOOLS_raw"].transform(standardize)
df["ABILITY_Score"] = df.groupby("Season")["ABILITY_raw"].transform(standardize)
df["Age_Score"]     = df.groupby("Season")["Age_Z"].transform(
    lambda s: standardize(-s)   # younger = higher score
)

df["AFL_Score"] = (
    0.30 * df["TOOLS_Score"]
  + 0.50 * df["ABILITY_Score"]
  + 0.20 * df["Age_Score"]
)

# Re-standardize AFL_Score within each season so ranking scale is consistent
df["AFL_Score"] = df.groupby("Season")["AFL_Score"].transform(standardize)
df["AFL_Rank"]  = df.groupby("Season")["AFL_Score"].rank(ascending=False).astype("Int64")

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

keep = [
    "MLBAM_ID", "Name", "Team", "Season", "PA", "Age",
    "AFL_Score", "AFL_Rank",
    "TOOLS_Score", "ABILITY_Score", "Age_Score",
    # sub-components for inspection
    "T_Disc", "T_Power", "T_Ath",
    "A_Output", "A_Disc", "A_SBT", "A_Power",
    # key raw metrics
    "PPPA", "BB_2K", "SBTalent", "HR_AB",
    "Chase_pct", "ZContact_pct", "Whiff_pct_pit", "MaxEV", "HardHit%",
]
keep = [c for c in keep if c in df.columns]
out = df[keep].sort_values(["Season", "AFL_Rank"])

for col in ["AFL_Score", "TOOLS_Score", "ABILITY_Score", "Age_Score",
            "T_Disc", "T_Power", "T_Ath",
            "A_Output", "A_Disc", "A_SBT", "A_Power"]:
    if col in out.columns:
        out[col] = out[col].round(2)

out.to_csv(OUT_PATH, index=False)
print(f"\nWrote {len(out)} rows → {OUT_PATH.name}")

# ---------------------------------------------------------------------------
# Summary — current season top 20
# ---------------------------------------------------------------------------

current = out[out["Season"] == out["Season"].max()].head(20)
print(f"\n=== AFL {out['Season'].max()} — Top 20 ===")
print(f"{'Rk':>3}  {'Name':<25}  {'PA':>3}  {'Age':>4}  "
      f"{'AFL':>5}  {'TOOLS':>5}  {'ABIL':>5}  {'PPPA':>6}  {'MaxEV':>6}")
print("─" * 80)
for _, r in current.iterrows():
    ev_str   = f"{r['MaxEV']:6.1f}" if pd.notna(r.get("MaxEV")) else "   —  "
    pppa_str = f"{r['PPPA']:6.3f}" if pd.notna(r.get("PPPA")) else "   —  "
    print(f"  {int(r['AFL_Rank']):>2}  {r['Name']:<25}  {int(r['PA']):>3}  "
          f"{r['Age']:>4.1f}  "
          f"{r['AFL_Score']:>5.1f}  {r['TOOLS_Score']:>5.1f}  "
          f"{r['ABILITY_Score']:>5.1f}  {pppa_str}  {ev_str}")
