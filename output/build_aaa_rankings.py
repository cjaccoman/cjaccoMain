"""Build data/rankings/aaa_{year}.csv — AAA player data table for power rankings.

Sources:
  data/prospectSavant/ps_AAA_{year}.csv — fetched via fetch_prospectsavant.py
  minorLeagueData.csv — pipeline PPPA and season-accurate Age

Age resolution (priority order):
  1. minorLeagueData Age — API-reported age for that season (most accurate)
  2. floor(ExactAge)    — derived from birth date vs. season end date
  3. NaN               — players excluded from Age_R; Overall renormalized to 0.95 weight total

Usage:
  python build_aaa_rankings.py          # builds all available seasons
  python build_aaa_rankings.py 2026     # builds a single season
"""

import sys
import json
import time
import urllib.request
from pathlib import Path
import re
import unicodedata
import pandas as pd
from datetime import date

DATA_DIR  = Path(__file__).resolve().parent.parent / "data"
PS_DIR    = DATA_DIR / "prospectSavant"
BDAY_PATH = DATA_DIR / "api" / "player_birthdays.csv"

AVAILABLE_SEASONS = [2023, 2024, 2025, 2026]
BIRTH_BATCH   = 50
CALL_DELAY    = 0.3
MLB_STATS_URL = "https://statsapi.mlb.com/api/v1"


def normalize_name(name: str) -> str:
    name = str(name).lower().strip()
    name = name.replace(".", "")
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"\s+jr$", "", name)
    name = re.sub(r"\s+", " ", name)
    return name


def fetch_missing_birth_dates(missing_ids: pd.DataFrame) -> dict:
    """Fetch birth dates for players not yet in cache.

    Args:
        missing_ids: DataFrame with [_norm, MLBAMId] for players missing birth dates.
    Returns:
        Full updated bday_dict {mlbam_id (int): birth_date_str}.
    """
    cached = (
        pd.read_csv(BDAY_PATH, dtype={"MLBAM_ID": "Int64", "BirthDate": str})
        if BDAY_PATH.exists()
        else pd.DataFrame(columns=["MLBAM_ID", "BirthDate"])
    )
    cached_dict = dict(zip(cached["MLBAM_ID"].astype(int), cached["BirthDate"]))

    to_fetch = [int(i) for i in missing_ids["MLBAMId"].dropna().unique()
                if int(i) not in cached_dict]

    if not to_fetch:
        return cached_dict

    print(f"  Fetching {len(to_fetch)} birth dates from API...")
    results = dict(cached_dict)
    batches = [to_fetch[i:i + BIRTH_BATCH] for i in range(0, len(to_fetch), BIRTH_BATCH)]
    for batch in batches:
        ids_str = ",".join(str(i) for i in batch)
        url = f"{MLB_STATS_URL}/people?personIds={ids_str}&fields=people,id,birthDate"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "prospectsMain/1.0"})
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.loads(r.read())
            for person in data.get("people", []):
                pid = person.get("id")
                bd  = person.get("birthDate")
                if pid and bd:
                    results[int(pid)] = bd
        except Exception as e:
            print(f"  Warning: birth date fetch failed: {e}")
        time.sleep(CALL_DELAY)

    rows = [{"MLBAM_ID": k, "BirthDate": v} for k, v in sorted(results.items())]
    pd.DataFrame(rows).to_csv(BDAY_PATH, index=False)
    fetched = len(results) - len(cached_dict)
    print(f"  Birth date cache updated: +{fetched} new ({len(results)} total)")
    return results


PS_COL_MAP = {
    "Max EV":      "MaxEV",
    "90th% EV":    "EV90",
    "Hard Hit%":   "HardHit%",
    "Barrel%":     "Barrel%PA",
    "Pull Air":    "PullAir%",
    "Z-Contact%":  "ZContact%",
    "Speed":       "Spd",
}


def build_season(year: int, ml: pd.DataFrame, mlb_veterans: set, mlb_vet_ids: set) -> None:
    ps = pd.read_csv(PS_DIR / f"ps_AAA_{year}.csv")
    ps = ps.rename(columns=PS_COL_MAP)
    ps["_norm"] = ps["Name"].apply(normalize_name)

    # Exclude MLB veterans using MLBAM ID first (avoids Jr/Sr name collisions),
    # falling back to name for rows with no ID match in hist_mlb_data.
    if year == date.today().year and "MLBAMId" in ps.columns:
        has_id = ps["MLBAMId"].notna()
        id_is_vet   = has_id  & ps["MLBAMId"].astype("Int64").isin(mlb_vet_ids)
        name_is_vet = ~has_id & ps["_norm"].isin(mlb_veterans)
        ps = ps[~(id_is_vet | name_is_vet)].copy()
    ml["_norm"] = ml["Name"].apply(normalize_name)

    aaa = (
        ml[(ml["Season"] == year) & (ml["Level"] == "AAA")]
        .sort_values("PA", ascending=False)
        .drop_duplicates(subset="_norm", keep="first")
    )

    # ── Age resolution ────────────────────────────────────────────────────────
    if "MLBAMId" in ps.columns:
        bday = (
            pd.read_csv(BDAY_PATH, dtype={"MLBAM_ID": "Int64", "BirthDate": str})
            if BDAY_PATH.exists()
            else pd.DataFrame(columns=["MLBAM_ID", "BirthDate"])
        )
        bday_dict = dict(zip(bday["MLBAM_ID"].astype(int), bday["BirthDate"]))
        ps["BirthDate"] = ps["MLBAMId"].apply(
            lambda x: bday_dict.get(int(x)) if pd.notna(x) else None
        )
        still_missing = ps[ps["BirthDate"].isna() & ps["MLBAMId"].notna()][["_norm", "MLBAMId"]]
        if not still_missing.empty:
            bday_dict = fetch_missing_birth_dates(still_missing)
            ps["BirthDate"] = ps["MLBAMId"].apply(
                lambda x: bday_dict.get(int(x)) if pd.notna(x) else None
            )
        ref_date = date.today() if year == date.today().year else date(year, 9, 30)

        def exact_age(bd_str):
            if pd.isna(bd_str):
                return None
            bd = date.fromisoformat(str(bd_str))
            return round((ref_date - bd).days / 365.25, 3)

        ps["ExactAge"] = ps["BirthDate"].apply(exact_age)
    else:
        ps["ExactAge"] = ps["Age"].astype(float)
    ps["BB%-2K%"] = (ps["BB%"] - 2 * ps["K%"]).round(1)

    # ── Position lookup ───────────────────────────────────────────────────────
    pos_path = DATA_DIR / "api" / "player_positions.csv"
    if pos_path.exists() and "MLBAMId" in ps.columns:
        pos_df = pd.read_csv(pos_path, dtype={"MLBAM_ID": "Int64"})
        ps["_mlbam"] = pd.to_numeric(ps["MLBAMId"], errors="coerce").astype("Int64")
        ps = ps.merge(pos_df.rename(columns={"MLBAM_ID": "_mlbam", "Position": "Pos"}),
                      on="_mlbam", how="left").drop(columns="_mlbam")
    elif "Pos" not in ps.columns:
        ps["Pos"] = None

    # ── Build output table ────────────────────────────────────────────────────
    ps = ps[ps["AB"] >= 40].copy()

    out = ps[["Name", "Org", "Pos", "ExactAge", "PA", "BB%-2K%",
              "Chase%", "ZContact%", "Whiff%",
              "MaxEV", "EV90", "HardHit%", "Barrel%PA", "PullAir%",
              "xBA", "xSLG", "xwOBA", "Spd", "_norm"]].copy()
    out = out.rename(columns={
        "Org":       "Team",
        "Barrel%PA": "Barrel%",
    })

    # PPPA + SB stats from pipeline; also pull season-accurate Age
    pppa_lookup = aaa[["_norm", "PPPA", "SB", "CS", "PA", "Age"]].copy()
    pppa_lookup["SB_rate"] = (pppa_lookup["SB"] / pppa_lookup["PA"]).round(3)
    pppa_lookup["SB_succ"] = (
        pppa_lookup["SB"] / (pppa_lookup["SB"] + pppa_lookup["CS"])
    ).fillna(0).round(3)
    out = out.merge(
        pppa_lookup[["_norm", "PPPA", "SB_rate", "SB_succ", "Age"]],
        on="_norm", how="left"
    ).drop(columns="_norm")

    # Age priority: minorLeagueData → floor(ExactAge) → NaN
    age_from_exact = out["ExactAge"].apply(lambda x: int(x) if pd.notna(x) else pd.NA)
    out["Age"] = out["Age"].where(out["Age"].notna(), age_from_exact)

    col_order = ["Name", "Team", "Pos", "Age", "ExactAge", "PA", "PPPA", "BB%-2K%", "Chase%",
                 "ZContact%", "Whiff%", "MaxEV", "EV90", "HardHit%", "Barrel%", "PullAir%",
                 "xBA", "xSLG", "xwOBA", "Spd", "SB_rate", "SB_succ"]
    out = out[col_order]

    num_cols = out.select_dtypes(include="float").columns
    out[num_cols] = out[num_cols].round(3)

    # ── Percentile ranks ──────────────────────────────────────────────────────
    STAT_COLS = ["PPPA", "BB%-2K%", "Chase%", "ZContact%", "Whiff%",
                 "MaxEV", "EV90", "HardHit%", "Barrel%", "PullAir%", "xBA", "xSLG",
                 "Spd", "SB_rate", "SB_succ"]
    LOWER_IS_BETTER = {"Chase%", "Whiff%"}

    out["Age_R"] = (
        out["ExactAge"].rank(pct=True, ascending=False, na_option="keep") * 100
    ).round(1)

    final_cols = ["Name", "Team", "Pos", "Age", "ExactAge", "Age_R", "PA"]
    for col in STAT_COLS:
        asc = col not in LOWER_IS_BETTER
        out[col + "_R"] = (
            out[col].rank(pct=True, ascending=asc, na_option="keep") * 100
        ).round(1)
        final_cols += [col, col + "_R"]

    # ── Composites ───────────────────────────────────────────────────────────
    POWER_WEIGHTS = {
        "Barrel%_R":  0.35,
        "xSLG_R":     0.20,
        "MaxEV_R":    0.17,
        "EV90_R":     0.18,
        "PullAir%_R": 0.10,
    }
    out["Power"] = sum(out[col] * w for col, w in POWER_WEIGHTS.items()).round(1)
    out["Power_R"] = (
        out["Power"].rank(pct=True, ascending=True, na_option="keep") * 100
    ).round(1)

    DISCIPLINE_WEIGHTS = {
        "BB%-2K%_R": 0.50,
        "Chase%_R":  0.30,
        "Whiff%_R":  0.20,
    }
    out["Discipline"] = sum(out[col] * w for col, w in DISCIPLINE_WEIGHTS.items()).round(1)
    out["Discipline_R"] = (
        out["Discipline"].rank(pct=True, ascending=True, na_option="keep") * 100
    ).round(1)

    CONTACT_WEIGHTS = {
        "xBA_R":       0.50,
        "ZContact%_R": 0.50,
    }
    out["Contact"] = sum(out[col] * w for col, w in CONTACT_WEIGHTS.items()).round(1)
    out["Contact_R"] = (
        out["Contact"].rank(pct=True, ascending=True, na_option="keep") * 100
    ).round(1)

    SPEED_WEIGHTS = {
        "Spd_R":     0.40,
        "SB_rate_R": 0.35,
        "SB_succ_R": 0.25,
    }
    out["Speed"] = sum(out[col] * w for col, w in SPEED_WEIGHTS.items()).round(1)
    out["Speed_R"] = (
        out["Speed"].rank(pct=True, ascending=True, na_option="keep") * 100
    ).round(1)

    out["ConPow"] = ((out["Contact"] + out["Power"]) / 2).round(1)
    out["ConPow_R"] = (
        out["ConPow"].rank(pct=True, ascending=True, na_option="keep") * 100
    ).round(1)

    out["ConDis"] = ((out["Contact"] + out["Discipline"]) / 2).round(1)
    out["ConDis_R"] = (
        out["ConDis"].rank(pct=True, ascending=True, na_option="keep") * 100
    ).round(1)

    out["ConDisSpd"] = ((out["Contact"] + out["Discipline"] + out["Speed"]) / 3).round(1)
    out["ConDisSpd_R"] = (
        out["ConDisSpd"].rank(pct=True, ascending=True, na_option="keep") * 100
    ).round(1)

    # Overall: renormalize to 0.95 weight total when Age_R is NaN
    OVERALL_WEIGHTS = {
        "PPPA_R":     0.42,
        "Discipline": 0.22,
        "Power":      0.13,
        "Speed":      0.12,
        "Contact":    0.06,
        "Age_R":      0.05,
    }
    non_age_w     = {k: v for k, v in OVERALL_WEIGHTS.items() if k != "Age_R"}
    non_age_sum   = sum(out[col] * w for col, w in non_age_w.items())
    non_age_total = sum(non_age_w.values())  # 0.95

    has_age = out["Age_R"].notna()
    out["Overall"] = (
        (non_age_sum + out["Age_R"].fillna(0) * OVERALL_WEIGHTS["Age_R"])
        .where(has_age, non_age_sum / non_age_total)
    ).round(1)
    out["Overall_R"] = (
        out["Overall"].rank(pct=True, ascending=True, na_option="keep") * 100
    ).round(1)

    pppa_r_idx = final_cols.index("PPPA_R")
    final_cols = (
        final_cols[:pppa_r_idx + 1]
        + ["Overall", "Overall_R", "Discipline", "Discipline_R", "Power", "Power_R",
           "Contact", "Contact_R", "ConPow", "ConPow_R", "ConDis", "ConDis_R",
           "ConDisSpd", "ConDisSpd_R", "Speed", "Speed_R"]
        + final_cols[pppa_r_idx + 1:]
    )

    out = out[final_cols]
    # Name-based fallback for current year when MLBAMId unavailable in ps source
    if year == date.today().year and "MLBAMId" not in ps.columns:
        out = out[~out["Name"].apply(normalize_name).isin(mlb_veterans)]

    out = out.sort_values("Overall", ascending=False).reset_index(drop=True)
    out.insert(0, "Rank", range(1, len(out) + 1))

    out_path = DATA_DIR / "rankings" / f"aaa_{year}.csv"
    out.to_csv(out_path, index=False)

    age_null   = out["Age"].isna().sum()
    exact_null = out["ExactAge"].isna().sum()
    print(f"[{year}] Wrote {len(out)} players to {out_path.name}  "
          f"(PPPA: {out['PPPA'].notna().sum()} populated  "
          f"Age: {len(out)-age_null} resolved, {age_null} null  "
          f"ExactAge: {len(out)-exact_null} resolved, {exact_null} null)")


COMBINED_COLS = [
    "Season", "Name", "Team", "Pos", "Age", "ExactAge", "Age_R", "PA",
    "Overall", "Overall_R", "PPPA", "PPPA_R",
    "Discipline", "Discipline_R", "Power", "Power_R",
    "Contact", "Contact_R", "ConPow", "ConPow_R",
    "ConDis", "ConDis_R", "ConDisSpd", "ConDisSpd_R", "Speed", "Speed_R",
]


def build_combined() -> None:
    frames = []
    for year in AVAILABLE_SEASONS:
        path = DATA_DIR / "rankings" / f"aaa_{year}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df.insert(0, "Season", year)
        # Pos column added in 2026; back-fill for older seasons
        if "Pos" not in df.columns:
            df["Pos"] = None
        frames.append(df[COMBINED_COLS])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("Overall", ascending=False).reset_index(drop=True)
    out_path = DATA_DIR / "rankings" / "aaa_combined.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nCombined: {len(combined)} rows across {len(frames)} seasons -> {out_path.name}")


def main() -> None:
    if len(sys.argv) > 1:
        seasons = [int(sys.argv[1])]
    else:
        seasons = AVAILABLE_SEASONS

    ml  = pd.read_csv(DATA_DIR / "computed" / "minorLeagueData.csv")
    mlb = pd.read_csv(DATA_DIR / "historical" / "hist_mlb_data.csv",
                      usecols=["Name", "PA", "MLBAMID"])
    mlb["_norm"]      = mlb["Name"].apply(normalize_name)
    mlb_career_pa     = mlb.groupby("_norm")["PA"].sum()
    mlb_veterans      = set(mlb_career_pa[mlb_career_pa >= 50].index)
    mlb_id_pa         = mlb.groupby("MLBAMID")["PA"].sum()
    mlb_vet_ids       = set(mlb_id_pa[mlb_id_pa >= 50].index)

    for year in seasons:
        build_season(year, ml.copy(), mlb_veterans, mlb_vet_ids)

    build_combined()


if __name__ == "__main__":
    main()
