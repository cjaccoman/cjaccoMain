"""Build data/rankings/rk_2026.csv — Rk-ball power rankings using PS Savant data.

Source: data/prospectSavant/ps_Rk_2026_savant.csv (players with Statcast EV data only)

Adapted from build_aaa_rankings.py. Two composites differ from AAA due to
xBA and xSLG being unavailable at Rk level:
  Contact: ZContact%(50%) + HardHit%(50%)
  Power:   MaxEV(22%) + EV90(22%) + Barrel%PA(41%) + PullAir%(15%)

Usage:
  python build_rk_rankings.py
"""

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

BIRTH_BATCH   = 50
CALL_DELAY    = 0.3
MLB_STATS_URL = "https://statsapi.mlb.com/api/v1"
MIN_AB        = 40


def normalize_name(name: str) -> str:
    name = str(name).lower().strip()
    name = name.replace(".", "")
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"\s+jr$", "", name)
    name = re.sub(r"\s+", " ", name)
    return name


def fetch_missing_birth_dates(missing_ids: pd.DataFrame) -> dict:
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


def main() -> None:
    year = 2026
    ps = pd.read_csv(PS_DIR / "ps_Rk_2026_savant.csv")
    ps = ps[ps["AB"] >= MIN_AB].copy()
    ps["_norm"] = ps["Name"].apply(normalize_name)

    ml = pd.read_csv(DATA_DIR / "computed" / "minorLeagueData.csv")
    ml["_norm"] = ml["Name"].apply(normalize_name)

    rk = (
        ml[(ml["Season"] == year) & (ml["Level"] == "R")]
        .sort_values("PA", ascending=False)
        .drop_duplicates(subset="_norm", keep="first")
    )

    # ── Age resolution ────────────────────────────────────────────────────────
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

    ref_date = date.today()

    def exact_age(bd_str):
        if pd.isna(bd_str):
            return None
        bd = date.fromisoformat(str(bd_str))
        return round((ref_date - bd).days / 365.25, 3)

    ps["ExactAge"] = ps["BirthDate"].apply(exact_age)
    ps["BB%-2K%"]  = (ps["BB%"] - 2 * ps["K%"]).round(1)

    # ── Build output table ────────────────────────────────────────────────────
    out = ps[["Name", "Org", "ExactAge", "PA", "BB%-2K%",
              "Chase%", "ZContact%", "Whiff%",
              "MaxEV", "EV90", "HardHit%", "Barrel%PA",
              "Spd", "_norm"]].copy()
    out = out.rename(columns={
        "Org":       "Team",
        "Barrel%PA": "Barrel%",
    })

    # PPPA + SB stats from pipeline
    pppa_lookup = rk[["_norm", "PPPA", "SB", "CS", "PA", "Age"]].copy()
    pppa_lookup["SB_rate"] = (pppa_lookup["SB"] / pppa_lookup["PA"]).round(3)
    pppa_lookup["SB_succ"] = (
        pppa_lookup["SB"] / (pppa_lookup["SB"] + pppa_lookup["CS"])
    ).fillna(0).round(3)
    out = out.merge(
        pppa_lookup[["_norm", "PPPA", "SB_rate", "SB_succ", "Age"]],
        on="_norm", how="left"
    ).drop(columns="_norm")

    age_from_exact = out["ExactAge"].apply(lambda x: int(x) if pd.notna(x) else pd.NA)
    out["Age"] = out["Age"].where(out["Age"].notna(), age_from_exact)

    col_order = ["Name", "Team", "Age", "ExactAge", "PA", "PPPA", "BB%-2K%", "Chase%",
                 "ZContact%", "Whiff%", "MaxEV", "EV90", "HardHit%", "Barrel%",
                 "Spd", "SB_rate", "SB_succ"]
    out = out[col_order]

    num_cols = out.select_dtypes(include="float").columns
    out[num_cols] = out[num_cols].round(3)

    # ── Percentile ranks ──────────────────────────────────────────────────────
    STAT_COLS = ["PPPA", "BB%-2K%", "Chase%", "ZContact%", "Whiff%",
                 "MaxEV", "EV90", "HardHit%", "Barrel%",
                 "Spd", "SB_rate", "SB_succ"]
    LOWER_IS_BETTER = {"Chase%", "Whiff%"}

    out["Age_R"] = (
        out["ExactAge"].rank(pct=True, ascending=False, na_option="keep") * 100
    ).round(1)

    final_cols = ["Name", "Team", "Age", "ExactAge", "Age_R", "PA"]
    for col in STAT_COLS:
        asc = col not in LOWER_IS_BETTER
        out[col + "_R"] = (
            out[col].rank(pct=True, ascending=asc, na_option="keep") * 100
        ).round(1)
        final_cols += [col, col + "_R"]

    # ── Composites ───────────────────────────────────────────────────────────
    # Power: xSLG and PullAir% unavailable at Rk — split between EV metrics and Barrel%
    POWER_WEIGHTS = {
        "MaxEV_R":   0.40,
        "EV90_R":    0.40,
        "Barrel%_R": 0.20,
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

    # Contact: xBA unavailable — ZContact% + HardHit% equally weighted
    CONTACT_WEIGHTS = {
        "ZContact%_R": 0.50,
        "HardHit%_R":  0.50,
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

    out["RawPower"] = (out["MaxEV_R"] * 0.60 + out["EV90_R"] * 0.40).round(1)
    out["PowAge"]   = ((out["RawPower"] + out["Age_R"]) / 2).round(1)

    OVERALL_WEIGHTS = {
        "Contact":    0.15,
        "Discipline": 0.25,
        "Power":      0.25,
        "Speed":      0.10,
        "Age_R":      0.25,
    }
    non_age_w     = {k: v for k, v in OVERALL_WEIGHTS.items() if k != "Age_R"}
    non_age_sum   = sum(out[col] * w for col, w in non_age_w.items())
    non_age_total = sum(non_age_w.values())

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
           "ConDisSpd", "ConDisSpd_R", "Speed", "Speed_R",
           "RawPower", "PowAge"]
        + final_cols[pppa_r_idx + 1:]
    )

    out = out[final_cols]
    out = out.sort_values("Overall", ascending=False).reset_index(drop=True)
    out.insert(0, "Rank", range(1, len(out) + 1))

    out_path = DATA_DIR / "rankings" / f"rk_{year}.csv"
    out.to_csv(out_path, index=False)

    age_null   = out["Age"].isna().sum()
    exact_null = out["ExactAge"].isna().sum()
    print(f"[Rk {year}] Wrote {len(out)} players to {out_path.name}  "
          f"(PPPA: {out['PPPA'].notna().sum()} populated  "
          f"Age: {len(out)-age_null} resolved, {age_null} null  "
          f"ExactAge: {len(out)-exact_null} resolved, {exact_null} null)")


if __name__ == "__main__":
    main()
