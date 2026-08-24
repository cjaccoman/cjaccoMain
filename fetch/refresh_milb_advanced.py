"""Re-fetch seasonAdvanced for all years to populate Whiff% (and any other
fields added to parse_advanced_splits after the initial build).

Does NOT touch milb_hitting.csv. Rewrites milb_advanced.csv in full.
Run: python refresh_milb_advanced.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fetch_milb_data as fm

SEASONS = [s for s in fm.SEASONS if s != 2020]


def main() -> None:
    print("=== refresh_milb_advanced.py ===\n")

    chadwick = fm.load_chadwick()
    hit_df   = pd.read_csv(fm.HITTING_OUT)  # needed for HR/FB join

    mlb_org     = fm.build_mlb_org_lookup()
    all_advanced: list[dict] = []

    total = len(SEASONS) * len(fm.SPORT_IDS)
    done  = 0
    for season in SEASONS:
        for level, sport_id in fm.SPORT_IDS.items():
            done += 1
            tag = f"[{done:3}/{total}] {season} {level:3}"
            team_lkp   = fm.build_team_lookup(sport_id, season, mlb_org)
            league_lkp = fm.build_league_lookup(sport_id, season)
            adv_splits = fm.fetch_splits("seasonAdvanced", sport_id, season)
            if adv_splits:
                rows = fm.parse_advanced_splits(adv_splits, team_lkp, league_lkp)
                rows = [r for r in rows
                        if (r.get("PA") or 0) >= fm.MIN_PA
                        and (r.get("Age") or 99) <= fm.MAX_AGE]
                all_advanced.extend(rows)
                print(f"{tag}  {len(rows):4} rows")
            else:
                print(f"{tag}  no data")

    print(f"\nBuilding milb_advanced.csv ({len(all_advanced):,} rows)...")
    adv_df = pd.DataFrame(all_advanced)
    adv_df = fm.apply_crosswalk(adv_df, chadwick)

    # HR/FB: join HR from milb_hitting on (MLBAM_ID, Season, Team, Level)
    hr_lookup = hit_df[["MLBAM_ID", "Season", "Team", "Level", "HR"]].copy()
    hr_lookup["MLBAM_ID"] = pd.to_numeric(hr_lookup["MLBAM_ID"], errors="coerce").astype("Int64")
    adv_df = adv_df.merge(hr_lookup, on=["MLBAM_ID", "Season", "Team", "Level"], how="left")
    fly_pop = adv_df.pop("_fly_pop")
    hr      = adv_df.pop("HR")
    adv_df["HR/FB"] = (hr / fly_pop).where(fly_pop > 0).round(3)

    out_cols = [
        "PlayerId", "MLBAM_ID", "Season", "Name", "Team", "Level", "League", "Age", "PA",
        "AVG", "OBP", "SLG", "OPS", "ISO", "BABIP",
        "BB%", "K%", "GB%", "LD%", "FB%", "GB/FB", "IFFB%", "HR/FB",
        "SwStr%", "Whiff%", "Pitches",
    ]
    adv_df = adv_df[[c for c in out_cols if c in adv_df.columns]]
    adv_df.to_csv(fm.ADVANCED_OUT, index=False)
    print(f"Wrote {len(adv_df):,} rows -> {fm.ADVANCED_OUT.name}")

    whiff_pop = adv_df["Whiff%"].notna().sum()
    print(f"Whiff% populated: {whiff_pop:,} / {len(adv_df):,} "
          f"({whiff_pop/len(adv_df)*100:.1f}%)")


if __name__ == "__main__":
    main()

