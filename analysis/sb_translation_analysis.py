"""
SB translation analysis — chained level-to-level transitions.

For each player, finds every one-step promotion:
  R->A, A->A+, A+->AA, AA->AAA, AAA->MLB

For each promotion, compares:
  - SB/PA in the most recent qualifying season at the lower level
  - SB/PA in the first qualifying season at the higher level (must follow lower season)

Chain: A+->MLB = (A+->AA) x (AA->AAA) x (AAA->MLB)
Validates chain against direct jump estimates where sample allows.

Outputs:
  data/historical/sb_translation_log.csv      -- one row per player-promotion
  data/historical/sb_translation_rates.csv    -- level-pair x season
  data/historical/sb_translation_eras.csv     -- level-pair x era
  data/historical/sb_translation_chain.csv    -- chained discount factors level->MLB by era
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HIST_DIR = DATA_DIR / "historical"

MILB_PA_MIN   = 80
MLB_PA_MIN    = 100
MILB_SB_MIN   = 5       # min SB in "from" season for rate aggregations
MAX_GAP_YRS   = 3       # max calendar years between from-season and to-season

LEVEL_ORDER = {"R": 1, "A": 2, "A+": 3, "AA": 4, "AAA": 5, "MLB": 6}
TRANSITIONS = [("R","A"), ("A","A+"), ("A+","AA"), ("AA","AAA"), ("AAA","MLB")]

ERA_BREAKS = [
    (2006, 2014, "2006-2014"),
    (2015, 2019, "2015-2019"),
    (2020, 2022, "2020-2022"),
    (2023, 2026, "2023-2026"),
]


def assign_era(season):
    for start, end, label in ERA_BREAKS:
        if start <= season <= end:
            return label
    return "Unknown"


def main():
    mil = pd.read_csv(DATA_DIR / "historical" / "ovr_hist_data.csv",
                      usecols=["PlayerId","Season","Name","Level","PA","SB"])
    mlb = pd.read_csv(HIST_DIR / "hist_mlb_data.csv",
                      usecols=["PlayerId","Season","Name","PA","SB"])

    # Qualifying MiLB seasons
    mil_q = mil[mil["PA"] >= MILB_PA_MIN].copy()
    mil_q["SB_rate"] = mil_q["SB"] / mil_q["PA"]

    # Qualifying MLB seasons — tag as "MLB" level
    mlb_q = mlb[mlb["PA"] >= MLB_PA_MIN].copy()
    mlb_q["Level"]   = "MLB"
    mlb_q["SB_rate"] = mlb_q["SB"] / mlb_q["PA"]

    # Combine into one pool
    all_q = pd.concat([
        mil_q[["PlayerId","Season","Name","Level","PA","SB","SB_rate"]],
        mlb_q[["PlayerId","Season","Name","Level","PA","SB","SB_rate"]],
    ], ignore_index=True)

    all_q["level_num"] = all_q["Level"].map(LEVEL_ORDER)

    # Build transition records
    records = []
    for pid, grp in all_q.groupby("PlayerId"):
        grp = grp.sort_values("Season")
        name = grp["Name"].iloc[0]

        for (from_lvl, to_lvl) in TRANSITIONS:
            from_rows = grp[grp["Level"] == from_lvl]
            to_rows   = grp[grp["Level"] == to_lvl]
            if from_rows.empty or to_rows.empty:
                continue

            # First qualifying season at "to" level
            first_to = to_rows.sort_values("Season").iloc[0]

            # Most recent qualifying season at "from" level that precedes first_to
            valid_from = from_rows[from_rows["Season"] < first_to["Season"]]
            if valid_from.empty:
                continue

            last_from = valid_from.sort_values("Season").iloc[-1]

            # Gap check
            if first_to["Season"] - last_from["Season"] > MAX_GAP_YRS:
                continue

            records.append({
                "PlayerId":      pid,
                "Name":          name,
                "Transition":    f"{from_lvl}->{to_lvl}",
                "From_Level":    from_lvl,
                "To_Level":      to_lvl,
                "From_Season":   int(last_from["Season"]),
                "To_Season":     int(first_to["Season"]),
                "From_PA":       int(last_from["PA"]),
                "From_SB":       int(last_from["SB"]),
                "From_SB_rate":  last_from["SB_rate"],
                "To_PA":         int(first_to["PA"]),
                "To_SB":         int(first_to["SB"]),
                "To_SB_rate":    first_to["SB_rate"],
                "Translation":   (first_to["SB_rate"] / last_from["SB_rate"]
                                  if last_from["SB_rate"] > 0 else np.nan),
            })

    log = pd.DataFrame(records)
    log["Era"] = log["From_Season"].apply(assign_era)
    log.to_csv(HIST_DIR / "sb_translation_log.csv", index=False)

    print(f"Total transitions logged: {len(log)}")
    for t in [f"{a}->{b}" for a,b in TRANSITIONS]:
        sub = log[log["Transition"] == t]
        print(f"  {t:10s}  N={len(sub):4d}  (From_SB>={MILB_SB_MIN}: {(sub['From_SB']>=MILB_SB_MIN).sum():4d})")
    print()

    # --- Aggregation helper ---
    def agg(g):
        return pd.Series({
            "N":                 len(g),
            "From_SB_rate_mean": g["From_SB_rate"].mean(),
            "To_SB_rate_mean":   g["To_SB_rate"].mean(),
            "Translation_mean":  g["Translation"].mean(),
            "Translation_median":g["Translation"].median(),
            "Translation_p25":   g["Translation"].quantile(0.25),
            "Translation_p75":   g["Translation"].quantile(0.75),
        })

    # Rates filtered to meaningful base stealers
    log_clean = log[log["From_SB"] >= MILB_SB_MIN].copy()

    rates = (
        log_clean.groupby(["Transition", "From_Season"])
        .apply(agg, include_groups=False)
        .reset_index()
        .round(4)
    )
    rates.to_csv(HIST_DIR / "sb_translation_rates.csv", index=False)

    eras = (
        log_clean.groupby(["Transition", "Era"])
        .apply(agg, include_groups=False)
        .reset_index()
        .round(4)
    )
    eras.to_csv(HIST_DIR / "sb_translation_eras.csv", index=False)

    # --- Print era table ---
    print("=== SB Translation Median by Transition x Era (From_SB >= 5) ===\n")
    pivot = eras.pivot(index="Transition", columns="Era", values=["N","Translation_median"])
    pivot.index = pd.CategoricalIndex(pivot.index,
        categories=[f"{a}->{b}" for a,b in TRANSITIONS], ordered=True)
    pivot = pivot.sort_index()
    print(pivot.to_string())
    print()

    # --- Chained factors by era ---
    era_labels = [e for _,_,e in ERA_BREAKS]
    chain_rows = []
    for era in era_labels:
        era_data = eras[eras["Era"] == era].set_index("Transition")["Translation_median"]
        # Step medians (fill missing with all-era median)
        all_era  = eras.groupby("Transition")["Translation_median"].median()

        def step(t):
            return era_data.get(t, all_era.get(t, np.nan))

        steps = {f"{a}->{b}": step(f"{a}->{b}") for a,b in TRANSITIONS}

        # Chained from each level to MLB
        aaa_mlb = steps.get("AAA->MLB", np.nan)
        aa_mlb  = steps.get("AA->AAA",  np.nan) * aaa_mlb
        ap_mlb  = steps.get("A+->AA",   np.nan) * aa_mlb
        a_mlb   = steps.get("A->A+",    np.nan) * ap_mlb
        r_mlb   = steps.get("R->A",     np.nan) * a_mlb

        chain_rows.append({
            "Era":      era,
            "AAA->MLB": round(aaa_mlb, 4) if not np.isnan(aaa_mlb) else None,
            "AA->MLB":  round(aa_mlb,  4) if not np.isnan(aa_mlb)  else None,
            "A+->MLB":  round(ap_mlb,  4) if not np.isnan(ap_mlb)  else None,
            "A->MLB":   round(a_mlb,   4) if not np.isnan(a_mlb)   else None,
            "R->MLB":   round(r_mlb,   4) if not np.isnan(r_mlb)   else None,
        })

    chain = pd.DataFrame(chain_rows)
    chain.to_csv(HIST_DIR / "sb_translation_chain.csv", index=False)

    print("=== Chained SB discount factors (level -> MLB) by era ===")
    print("(e.g. 0.60 = multiply MiLB SB/PA by 0.60 to project MLB SB/PA)")
    print()
    print(chain.to_string(index=False))
    print()

    # Validate against direct-jump estimates from prior analysis
    direct = {
        "AAA": {"2006-2014": 0.507, "2015-2019": 0.502, "2020-2022": 0.615, "2023-2026": 0.689},
        "AA":  {"2006-2014": 0.600, "2015-2019": 0.455, "2020-2022": 0.419, "2023-2026": 0.446},
        "A+":  {"2006-2014": 0.406, "2015-2019": 0.351, "2020-2022": 0.566, "2023-2026": 0.439},
    }
    print("=== Validation: chained vs direct-jump medians ===")
    for era in era_labels:
        crow = chain[chain["Era"] == era].iloc[0]
        print(f"  {era}")
        for lvl, col in [("AAA","AAA->MLB"),("AA","AA->MLB"),("A+","A+->MLB")]:
            chained = crow[col]
            direct_val = direct.get(lvl, {}).get(era, None)
            diff = f"  delta={chained - direct_val:+.3f}" if direct_val else ""
            print(f"    {lvl:4s}  chained={chained:.3f}  direct={direct_val or 'n/a'}{diff}")
        print()


if __name__ == "__main__":
    main()
