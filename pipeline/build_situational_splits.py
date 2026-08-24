"""Compute per-player-season situational splits from AAA Statcast pitch data.

Reads data/api/milb_statcast_aaa.csv (output of fetch_milb_statcast.py).

Splits computed (PA-level, filtered from pitch rows):
  RISP        — on_2b or on_3b occupied
  vs_LHP      — p_throws == 'L'
  vs_RHP      — p_throws == 'R'
  count_02    — 0-2 count (toughest discipline spot)
  late_close  — inning >= 7, |bat_score_diff| <= 2 (proxied from home/away scores)

Per-split metrics (where estimable):
  PA, K_pct, xwOBA, xBA, xSLG

Also computes AAA park factors (wOBA relative, 2023+):
  PF_wOBA = (home wOBA / road wOBA) × 100, normalized to 100 = average

Output: data/rankings/situational_splits.csv
  Columns: batter (MLBAM ID), player_name, Season,
           RISP_PA, RISP_K_pct, RISP_xwOBA,
           LHP_PA, LHP_xwOBA, LHP_K_pct,
           RHP_PA, RHP_xwOBA, RHP_K_pct,
           Platoon_Gap (LHP_xwOBA - RHP_xwOBA),
           Count02_PA, Count02_K_pct,
           Overall_PA, Overall_xwOBA, Overall_K_pct

Also outputs: data/rankings/aaa_park_factors.csv
  Columns: Season, home_team, PF_wOBA, PF_K, PA_home, PA_road
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR   = Path(__file__).resolve().parent.parent / "data"
IN_PATH    = DATA_DIR / "api" / "milb_statcast_aaa.csv"
OUT_SPLITS = DATA_DIR / "rankings" / "situational_splits.csv"
OUT_PF     = DATA_DIR / "rankings" / "aaa_park_factors.csv"

MIN_SPLIT_PA = 20   # minimum PAs to report a split metric (NaN otherwise)
MIN_PF_PA    = 200  # minimum PAs (home or road) to report a park factor


def pa_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only plate-appearance-terminal pitches (woba_denom=1 or events not null)."""
    return df[df["woba_denom"] == 1].copy()


def agg_split(sub: pd.DataFrame, min_pa: int = MIN_SPLIT_PA) -> pd.Series:
    pa = len(sub)
    if pa < min_pa:
        return pd.Series({"PA": pa, "K_pct": np.nan, "xwOBA": np.nan, "xBA": np.nan, "xSLG": np.nan})
    k_pct  = (sub["events"] == "strikeout").mean()
    xwoba  = sub["estimated_woba_using_speedangle"].mean()
    xba    = sub["estimated_ba_using_speedangle"].mean()  if "estimated_ba_using_speedangle"  in sub.columns else np.nan
    xslg   = sub["estimated_slg_using_speedangle"].mean() if "estimated_slg_using_speedangle" in sub.columns else np.nan
    return pd.Series({"PA": pa, "K_pct": round(k_pct, 4), "xwOBA": round(xwoba, 4),
                      "xBA": round(xba, 4), "xSLG": round(xslg, 4)})


def main() -> None:
    if not IN_PATH.exists():
        raise SystemExit(f"No data at {IN_PATH} — run fetch_milb_statcast.py first")

    print(f"Loading {IN_PATH.name}...")
    raw = pd.read_csv(IN_PATH, low_memory=False)
    print(f"  {len(raw):,} pitch rows")

    raw["game_date"] = pd.to_datetime(raw["game_date"])
    raw["Season"]    = raw["game_year"].fillna(raw["game_date"].dt.year).astype(int)

    # RISP: runner on 2b or 3b
    raw["is_risp"] = raw["on_2b"].notna() | raw["on_3b"].notna()

    # 0-2 count
    raw["is_02"] = (raw["balls"] == 0) & (raw["strikes"] == 2)

    # Home/road flag
    raw["batter_is_home"] = raw["inning_topbot"] == "Bot"

    # PA-terminal rows
    pa = pa_rows(raw)
    print(f"  {len(pa):,} PA rows")

    player_groups = pa.groupby(["batter", "player_name", "Season"])

    records = []
    for (batter_id, player_name, season), grp in player_groups:
        row = {"batter": batter_id, "player_name": player_name, "Season": season}

        overall = agg_split(grp)
        row["Overall_PA"]    = int(overall["PA"])
        row["Overall_K_pct"] = overall["K_pct"]
        row["Overall_xwOBA"] = overall["xwOBA"]

        risp   = agg_split(grp[grp["is_risp"]])
        row["RISP_PA"]    = int(risp["PA"])
        row["RISP_K_pct"] = risp["K_pct"]
        row["RISP_xwOBA"] = risp["xwOBA"]

        lhp    = agg_split(grp[grp["p_throws"] == "L"])
        row["LHP_PA"]    = int(lhp["PA"])
        row["LHP_K_pct"] = lhp["K_pct"]
        row["LHP_xwOBA"] = lhp["xwOBA"]

        rhp    = agg_split(grp[grp["p_throws"] == "R"])
        row["RHP_PA"]    = int(rhp["PA"])
        row["RHP_K_pct"] = rhp["K_pct"]
        row["RHP_xwOBA"] = rhp["xwOBA"]

        if lhp["xwOBA"] is not np.nan and rhp["xwOBA"] is not np.nan:
            row["Platoon_Gap"] = round(lhp["xwOBA"] - rhp["xwOBA"], 4) if (
                not pd.isna(lhp["xwOBA"]) and not pd.isna(rhp["xwOBA"])
            ) else np.nan
        else:
            row["Platoon_Gap"] = np.nan

        c02    = agg_split(grp[grp["is_02"]], min_pa=5)
        row["Count02_PA"]    = int(c02["PA"])
        row["Count02_K_pct"] = c02["K_pct"]

        records.append(row)

    splits = pd.DataFrame(records)
    splits.to_csv(OUT_SPLITS, index=False)
    print(f"\nWrote {len(splits):,} player-seasons -> {OUT_SPLITS.name}")
    print(f"Seasons: {sorted(splits['Season'].unique())}")
    print(f"Players with >= 50 Overall_PA: {(splits['Overall_PA'] >= 50).sum():,}")

    # ------------------------------------------------------------------
    # Park factors from same data
    # ------------------------------------------------------------------
    print("\nComputing AAA park factors...")

    # Home stats per team-season
    home_pa = pa[pa["batter_is_home"]].copy()
    road_pa = pa[~pa["batter_is_home"]].copy()

    def team_agg(df, col="home_team"):
        return df.groupby(["Season", col]).agg(
            PA=("woba_denom", "count"),
            xwOBA=("estimated_woba_using_speedangle", "mean"),
            K_pct=("events", lambda x: (x == "strikeout").mean()),
        ).reset_index().rename(columns={col: "team"})

    home_stats = team_agg(home_pa, "home_team")
    road_stats = team_agg(road_pa, "home_team")

    pf = home_stats.merge(road_stats, on=["Season", "team"], suffixes=("_home", "_road"))
    pf = pf[(pf["PA_home"] >= MIN_PF_PA) & (pf["PA_road"] >= MIN_PF_PA)]

    # Normalize: PF = home_rate / road_rate × 100 (100 = neutral)
    pf["PF_xwOBA"] = (pf["xwOBA_home"] / pf["xwOBA_road"] * 100).round(1)
    pf["PF_K"]     = (pf["K_pct_home"]  / pf["K_pct_road"]  * 100).round(1)

    pf_out = pf[["Season", "team", "PA_home", "PA_road",
                 "xwOBA_home", "xwOBA_road", "PF_xwOBA",
                 "K_pct_home", "K_pct_road", "PF_K"]].sort_values(["Season", "PF_xwOBA"], ascending=[True, False])
    pf_out.to_csv(OUT_PF, index=False)
    print(f"Wrote {len(pf_out):,} team-seasons -> {OUT_PF.name}")

    print("\nMost hitter-friendly AAA parks (latest season):")
    latest = pf_out[pf_out["Season"] == pf_out["Season"].max()]
    print(latest[["team","PF_xwOBA","PF_K","PA_home"]].head(8).to_string(index=False))
    print("\nMost pitcher-friendly AAA parks (latest season):")
    print(latest[["team","PF_xwOBA","PF_K","PA_home"]].tail(8).sort_values("PF_xwOBA").to_string(index=False))


if __name__ == "__main__":
    main()
