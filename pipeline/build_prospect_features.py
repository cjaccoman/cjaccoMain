"""Build prospect_features.csv — one row per player-season-level, 2006-2026.

All columns relevant to the TOOLS_Score + ABILITY_Score overhaul model.
Run standalone: python build_prospect_features.py

Sources joined onto ovr_hist_data spine:
  milb_hitting     -> Team, 3B, HR, BB, IBB, CS, MLBAM_ID
  milb_advanced    -> BB%, HR/FB, Whiff%
  milb_pitches_agg -> PullAir%, Chase%, Z-Contact%   (MLBAM_ID join)
  data/prospectSavant/  -> Spd, MaxEV, EV90               (MLBAM_ID join)

Derived:
  3B_PA        = 3B / PA
  HR_AB        = HR / (PA - BB - IBB)   [approx AB; excludes HBP/SF not in data]
  SB_pct       = SB / (SB + CS)
  BB_2K        = BB% - 2*K%
  career_HR_FB = FB-weighted career HR/FB across all seasons with FB_est >= 15
                 (replaces single-season HR/FB in TOOLS Power; r=0.60 vs 0.42 YOY)
"""

import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PS_DIR   = DATA_DIR / "prospectSavant"
OUT_PATH = DATA_DIR / "rankings" / "prospect_features.csv"

PS_FILES = [
    ("AAA", 2023), ("AAA", 2024), ("AAA", 2025), ("AAA", 2026),
    ("AA",  2026),
    ("A+",  2026),
    ("A",   2023), ("A",   2024), ("A",   2025), ("A",   2026),
    ("Rk",  2026),
]
PS_LEVEL_MAP = {"Rk": "R"}   # normalise PS level names to pipeline names


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s))
    return "".join(c for c in s if unicodedata.category(c) != "Mn").lower().strip()


def load_ps() -> pd.DataFrame:
    frames = []
    for level, year in PS_FILES:
        safe = level.replace("+", "p")
        path = PS_DIR / f"ps_{safe}_{year}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=lambda c: c in {
            "Name", "MLBAMId", "Season", "Level",
            "Spd", "MaxEV", "EV90",
            "Chase%", "ZContact%", "Whiff%", "PullAir%",
        })
        df["Season"] = year
        df["Level"]  = PS_LEVEL_MAP.get(level, level)
        # Align ZContact% to pipeline naming convention
        if "ZContact%" in df.columns:
            df = df.rename(columns={"ZContact%": "Z-Contact%"})
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    ps = pd.concat(frames, ignore_index=True)
    ps = ps.rename(columns={"MLBAMId": "MLBAM_ID"})
    ps["MLBAM_ID"] = pd.to_numeric(ps["MLBAM_ID"], errors="coerce").astype("Int64")
    ps["_norm"] = ps["Name"].apply(_norm)

    # 0.0 is physically impossible for exit velocity (mph) and sprint speed (ft/sec).
    # ProspectSavant outputs 0.0 when it has no tracked data for that player-season,
    # not a true measurement.  Treat as null so these rows fall back to HRFB/3B_PA_adj
    # rather than dragging down Level z-score pools and producing extreme negative z-scores
    # for every real player in the same Level group.
    for col in ["MaxEV", "EV90", "Spd"]:
        if col in ps.columns:
            ps[col] = ps[col].where(ps[col] > 0)

    return ps


def main() -> None:
    print("Loading spine (ovr_hist_data)...")
    spine = pd.read_csv(DATA_DIR / "historical" / "ovr_hist_data.csv")
    spine["_norm"] = spine["Name"].apply(_norm)
    print(f"  {len(spine):,} rows")

    # ------------------------------------------------------------------
    # milb_hitting — Team, 3B, HR, BB, IBB, CS, MLBAM_ID
    # ------------------------------------------------------------------
    print("Joining milb_hitting...")
    hit = pd.read_csv(
        DATA_DIR / "api" / "milb_hitting.csv",
        usecols=["PlayerId", "MLBAM_ID", "Season", "Name", "Team", "Level",
                 "3B", "HR", "BB", "IBB", "CS"],
    )
    hit["MLBAM_ID"] = pd.to_numeric(hit["MLBAM_ID"], errors="coerce").astype("Int64")
    hit["_norm_hit"] = hit["Name"].apply(_norm)

    # Primary join: PlayerId + Season + Level
    hit_key = hit.drop_duplicates(subset=["PlayerId", "Season", "Level"])
    merged = spine.merge(
        hit_key[["PlayerId", "Season", "Level", "Team", "3B", "HR",
                 "BB", "IBB", "CS", "MLBAM_ID", "_norm_hit"]],
        on=["PlayerId", "Season", "Level"], how="left",
    )

    # Name fallback for unmatched rows
    unmatched = merged["MLBAM_ID"].isna()
    if unmatched.any():
        hit_name = hit.drop_duplicates(subset=["_norm_hit", "Season", "Level"])
        fill = merged[unmatched][["_norm", "Season", "Level"]].merge(
            hit_name[["_norm_hit", "Season", "Level", "Team", "3B", "HR",
                      "BB", "IBB", "CS", "MLBAM_ID"]],
            left_on=["_norm", "Season", "Level"],
            right_on=["_norm_hit", "Season", "Level"],
            how="left",
        )
        for col in ["Team", "3B", "HR", "BB", "IBB", "CS", "MLBAM_ID"]:
            merged.loc[unmatched, col] = fill[col].values
    merged = merged.drop(columns=["_norm_hit"], errors="ignore")
    print(f"  MLBAM_ID resolved: {merged['MLBAM_ID'].notna().sum():,} / {len(merged):,}")

    # ------------------------------------------------------------------
    # milb_advanced — BB%, HR/FB, Whiff%
    # ------------------------------------------------------------------
    print("Joining milb_advanced...")
    adv = pd.read_csv(
        DATA_DIR / "api" / "milb_advanced.csv",
        usecols=["PlayerId", "Season", "Name", "Level", "BB%", "HR/FB", "SwStr%", "Pitches", "Whiff%"],
    )
    adv = adv.rename(columns={"Pitches": "Pitches_adv", "SwStr%": "SwStr%_adv", "Whiff%": "Whiff%_adv"})
    adv["_norm_adv"] = adv["Name"].apply(_norm)
    adv_key = adv.drop_duplicates(subset=["PlayerId", "Season", "Level"])
    merged = merged.merge(
        adv_key[["PlayerId", "Season", "Level", "BB%", "HR/FB", "SwStr%_adv", "Pitches_adv", "Whiff%_adv"]],
        on=["PlayerId", "Season", "Level"], how="left",
    )
    unmatched = merged["BB%"].isna()
    if unmatched.any():
        adv_name = adv.drop_duplicates(subset=["_norm_adv", "Season", "Level"])
        fill = merged[unmatched][["_norm", "Season", "Level"]].merge(
            adv_name[["_norm_adv", "Season", "Level", "BB%", "HR/FB", "SwStr%_adv", "Pitches_adv", "Whiff%_adv"]],
            left_on=["_norm", "Season", "Level"],
            right_on=["_norm_adv", "Season", "Level"], how="left",
        )
        for col in ["BB%", "HR/FB", "SwStr%_adv", "Pitches_adv", "Whiff%_adv"]:
            merged.loc[unmatched, col] = fill[col].values
    print(f"  BB% populated: {merged['BB%'].notna().sum():,} / {len(merged):,}")

    # HR/FB > 1.0 is physically impossible (can't hit more HRs than fly balls).
    # FanGraphs returns values like 2.0 or 3.0 for tiny samples where the fly-ball
    # denominator is mis-counted.  Null these out before any downstream computation.
    impossible = merged["HR/FB"] > 1.0
    if impossible.any():
        print(f"  HR/FB > 1.0 nulled: {impossible.sum()} rows "
              f"({merged.loc[impossible, 'Name'].tolist()})")
        merged.loc[impossible, "HR/FB"] = np.nan

    # ------------------------------------------------------------------
    # milb_pitches_agg — PullAir%, Chase%, Z-Contact%, Whiff% (MLBAM_ID join)
    # ------------------------------------------------------------------
    print("Joining milb_pitches_agg...")
    agg_cols = ["MLBAM_ID", "Season", "Level", "PullAir%", "Chase%", "Z-Contact%",
                "InZoneSwings", "OutsideSwings"]
    # Whiff% natively tracked only for games fetched after the SwStr update
    agg_file_cols = pd.read_csv(DATA_DIR / "api" / "milb_pitches_agg.csv", nrows=0).columns
    if "Whiff%" in agg_file_cols:
        agg_cols.append("Whiff%")
    agg = pd.read_csv(DATA_DIR / "api" / "milb_pitches_agg.csv", usecols=agg_cols)
    agg["MLBAM_ID"] = pd.to_numeric(agg["MLBAM_ID"], errors="coerce").astype("Int64")
    agg = agg.drop_duplicates(subset=["MLBAM_ID", "Season", "Level"])
    merged = merged.merge(agg, on=["MLBAM_ID", "Season", "Level"], how="left")
    print(f"  PullAir% populated: {merged['PullAir%'].notna().sum():,} / {len(merged):,}")
    print(f"  Chase% populated:   {merged['Chase%'].notna().sum():,} / {len(merged):,}")

    # ------------------------------------------------------------------
    # ProspectSavant — Spd, MaxEV, EV90 + Chase%/Z-Contact%/Whiff%/PullAir%
    # fills (MLBAM_ID primary, name fallback)
    # ------------------------------------------------------------------
    print("Joining ProspectSavant (Spd, MaxEV, EV90, Chase%, Z-Contact%, Whiff%, PullAir%)...")
    ps = load_ps()
    if not ps.empty:
        ps_key = ps.drop_duplicates(subset=["MLBAM_ID", "Season", "Level"])
        ps_cols_avail = [c for c in ["Spd", "MaxEV", "EV90", "Chase%", "Z-Contact%", "Whiff%", "PullAir%"]
                         if c in ps_key.columns]
        merged = merged.merge(
            ps_key[["MLBAM_ID", "Season", "Level"] + ps_cols_avail].add_suffix("_ps").rename(
                columns={"MLBAM_ID_ps": "MLBAM_ID", "Season_ps": "Season", "Level_ps": "Level"}
            ),
            on=["MLBAM_ID", "Season", "Level"], how="left",
        )
        # Name fallback for rows where MLBAM_ID join missed
        ps_name = ps.drop_duplicates(subset=["_norm", "Season", "Level"])
        ps_name_cols = [c for c in ps_cols_avail if c in ps_name.columns]
        spd_mask = merged["Spd_ps"].isna() if "Spd_ps" in merged.columns else pd.Series(False, index=merged.index)
        if spd_mask.any():
            fill = merged[spd_mask][["_norm", "Season", "Level"]].merge(
                ps_name[["_norm", "Season", "Level"] + ps_name_cols],
                on=["_norm", "Season", "Level"], how="left",
            )
            for col in ps_name_cols:
                merged.loc[spd_mask, f"{col}_ps"] = fill[col].values

        # Apply PS values: primary for Spd/MaxEV/EV90; fill-nulls for Chase%/Z-Contact%/Whiff%/PullAir%
        for col in ["Spd", "MaxEV", "EV90"]:
            ps_col = f"{col}_ps"
            if ps_col in merged.columns:
                merged[col] = merged[ps_col]
        for col in ["Chase%", "Z-Contact%", "Whiff%", "PullAir%"]:
            ps_col = f"{col}_ps"
            if ps_col in merged.columns:
                if col not in merged.columns:
                    merged[col] = merged[ps_col]
                else:
                    null_mask = merged[col].isna() & merged[ps_col].notna()
                    merged.loc[null_mask, col] = merged.loc[null_mask, ps_col]

        # Drop _ps working columns
        merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_ps")], errors="ignore")

        print(f"  Spd populated:        {merged['Spd'].notna().sum():,} / {len(merged):,}")
        print(f"  Chase% populated:     {merged['Chase%'].notna().sum():,} / {len(merged):,}")
        print(f"  Z-Contact% populated: {merged['Z-Contact%'].notna().sum():,} / {len(merged):,}")
    else:
        for col in ["Spd", "MaxEV", "EV90"]:
            merged[col] = None

    # ------------------------------------------------------------------
    # Derived columns
    # ------------------------------------------------------------------
    print("Computing derived columns...")
    merged["3B_PA"]  = (merged["3B"] / merged["PA"]).round(4)
    merged["HR_AB"]  = (merged["HR"] / (merged["PA"] - merged["BB"] - merged["IBB"])).round(4)
    merged["SB_pct"] = (merged["SB"] / (merged["SB"] + merged["CS"])).round(4)
    merged["BB_2K"]  = (merged["BB%"] - 2 * merged["K%"]).round(4)

    # Career HR/FB — FB-weighted average and total career FBs across all qualifying
    # player-seasons.  Single-season HR/FB tops out at YOY r≈0.52; career average
    # reaches r≈0.60 but introduces a z-score artifact for young players (compressed
    # std from long-career peers inflates their z-scores).
    #
    # career_HR_FB  is stored as a reference column (not used directly in z-scoring).
    # career_FBs_est is used in build_tools_score.py to shrink HRFB_adj in the
    # fallback tier: players with more career evidence get higher weight.
    # FB_est = HR / (HR/FB); only seasons with FB_est >= 15 contribute.
    _MIN_CAREER_FB = 15
    _hrfb_src = merged[
        merged["HR"].gt(0) & merged["HR/FB"].gt(0)
    ].copy()
    _hrfb_src["_fb_est"] = _hrfb_src["HR"] / _hrfb_src["HR/FB"]
    _hrfb_src = _hrfb_src[_hrfb_src["_fb_est"] >= _MIN_CAREER_FB]
    _career = (
        _hrfb_src.groupby("PlayerId")
        .apply(lambda g: pd.Series({
            "career_HR_FB":   float(np.average(g["HR/FB"], weights=g["_fb_est"])),
            "career_FBs_est": float(g["_fb_est"].sum()),
        }))
        .reset_index()
    )
    merged = merged.merge(_career, on="PlayerId", how="left")
    merged["career_HR_FB"]   = merged["career_HR_FB"].round(4)
    merged["career_FBs_est"] = merged["career_FBs_est"].round(1)
    print(f"  career_HR_FB populated:   {merged['career_HR_FB'].notna().sum():,} / {len(merged):,}")
    print(f"  career_FBs_est populated: {merged['career_FBs_est'].notna().sum():,} / {len(merged):,}")

    # prior_FBs_est: career FBs EXCLUDING this row's own season.  Used in
    # build_tools_score.py so that current-season HRFB_adj is weighted only by
    # evidence from prior seasons — prevents a single breakout year from both
    # generating the extreme HRFB_adj value AND supplying most of its own shrinkage weight.
    _row_fb = pd.Series(np.nan, index=merged.index)
    _qual_mask = merged["HR"].gt(0) & merged["HR/FB"].gt(0)
    _row_fb[_qual_mask] = merged.loc[_qual_mask, "HR"] / merged.loc[_qual_mask, "HR/FB"]
    _row_contrib = _row_fb.where(_row_fb >= _MIN_CAREER_FB, 0.0).fillna(0.0)
    merged["prior_FBs_est"] = (merged["career_FBs_est"] - _row_contrib).clip(lower=0).round(1)
    print(f"  prior_FBs_est populated:  {merged['prior_FBs_est'].notna().sum():,} / {len(merged):,}")

    # Whiff% — milb_advanced is the authoritative source (official API seasonAdvanced
    # endpoint, covers all levels and seasons).  pitches_agg Whiff% (game-feed derived)
    # is unreliable: all AAA 2026 rows are 0 due to an unimplemented computation, and
    # treating that 0 as real data produces wildly incorrect Whiff%_adj values.
    # Apply milb_advanced unconditionally; pitches_agg fills only residual nulls.
    if "Whiff%" not in merged.columns:
        merged["Whiff%"] = None
    # Primary: milb_advanced overrides anything pitches_agg set (including zeros)
    adv_mask = merged["Whiff%_adv"].notna()
    merged.loc[adv_mask, "Whiff%"] = merged.loc[adv_mask, "Whiff%_adv"]
    # Remaining gaps: formula fallback
    total_swings = merged["InZoneSwings"] + merged["OutsideSwings"]
    formula_whiff = (merged["SwStr%_adv"] * merged["Pitches_adv"] / total_swings).round(4)
    missing_whiff = merged["Whiff%"].isna() & total_swings.gt(0) & merged["SwStr%"].notna()
    merged.loc[missing_whiff, "Whiff%"] = formula_whiff[missing_whiff]
    print(f"  Whiff% populated: {merged['Whiff%'].notna().sum():,} / {len(merged):,}")

    # ------------------------------------------------------------------
    # Situational splits (AAA 2023-2025, from Baseball Savant Statcast)
    # ------------------------------------------------------------------
    splits_path = DATA_DIR / "rankings" / "situational_splits.csv"
    if splits_path.exists():
        print("Joining situational splits (AAA 2023-2025)...")
        splits = pd.read_csv(splits_path, usecols=[
            "batter", "Season",
            "RISP_PA", "RISP_K_pct", "RISP_xwOBA",
            "LHP_PA", "LHP_xwOBA", "LHP_K_pct",
            "RHP_PA", "RHP_xwOBA", "RHP_K_pct",
            "Platoon_Gap",
            "Count02_PA", "Count02_K_pct",
            "Overall_xwOBA",
        ])
        splits = splits.rename(columns={"batter": "MLBAM_ID"})
        splits["MLBAM_ID"] = pd.to_numeric(splits["MLBAM_ID"], errors="coerce").astype("Int64")
        # Join only to AAA rows (splits are AAA-only by construction)
        aaa_mask = merged["Level"] == "AAA"
        merged = merged.merge(splits, on=["MLBAM_ID", "Season"], how="left")
        # Zero out any splits that landed on non-AAA rows (shouldn't happen but guard)
        split_cols = ["RISP_PA", "RISP_K_pct", "RISP_xwOBA",
                      "LHP_PA", "LHP_xwOBA", "LHP_K_pct",
                      "RHP_PA", "RHP_xwOBA", "RHP_K_pct",
                      "Platoon_Gap", "Count02_PA", "Count02_K_pct", "Overall_xwOBA"]
        for col in split_cols:
            if col in merged.columns:
                merged.loc[~aaa_mask, col] = None
        n_joined = merged.loc[aaa_mask, "RISP_K_pct"].notna().sum()
        print(f"  RISP_K_pct populated: {n_joined:,} AAA rows")
    else:
        print("  situational_splits.csv not found — skipping (run fetch_milb_statcast.py + build_situational_splits.py)")
        split_cols = []

    # ------------------------------------------------------------------
    # Final column order and output
    # ------------------------------------------------------------------
    out_cols = [
        # Identity
        "PlayerId", "MLBAM_ID", "Season", "Name", "Team", "Level", "League", "Age", "PA",
        # ABILITY — fantasy output
        "PPPA", "PPPA_Z_SL",
        # ABILITY — contact/discipline
        "BB%", "K%", "BB_2K", "BB/K",
        # ABILITY — game power
        "PullAir%", "HR_AB",
        # ABILITY — SB talent
        "SB", "CS", "SB_pct",
        # TOOLS — discipline
        "Chase%", "Z-Contact%", "Whiff%", "SwStr%",
        # TOOLS — raw power
        "HR/FB", "career_HR_FB", "career_FBs_est", "prior_FBs_est", "MaxEV", "EV90",
        # TOOLS — athleticism
        "Spd", "3B_PA",
        # TOOLS — dev runway
        "Age_Z_SL",
        # Supporting
        "ISO", "wRC+", "GB/FB",
        # Raw counting
        "3B", "HR", "BB", "IBB",
        # Situational splits (AAA 2023-2025)
        "RISP_PA", "RISP_K_pct", "RISP_xwOBA",
        "LHP_PA", "LHP_xwOBA", "LHP_K_pct",
        "RHP_PA", "RHP_xwOBA", "RHP_K_pct",
        "Platoon_Gap", "Count02_PA", "Count02_K_pct", "Overall_xwOBA",
    ]
    # Drop working columns not in final output
    for drop_col in ["InZoneSwings", "OutsideSwings", "Pitches_adv", "SwStr%_adv", "Whiff%_adv"]:
        merged = merged.drop(columns=[drop_col], errors="ignore")
    out_cols = [c for c in out_cols if c in merged.columns]
    out = merged[out_cols].copy()

    # Round floats
    float_cols = out.select_dtypes(include="float").columns
    out[float_cols] = out[float_cols].round(4)

    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(out):,} rows -> {OUT_PATH.name}")
    print(f"Columns ({len(out_cols)}): {out_cols}")

    # Coverage summary
    print("\nNull rates for key columns:")
    key_cols = ["PPPA_Z_SL", "BB%", "K%", "BB_2K", "PullAir%", "Chase%",
                "Whiff%", "HR/FB", "Spd", "MaxEV", "SB_pct", "Age_Z_SL"]
    for c in key_cols:
        if c in out.columns:
            pct = out[c].isna().mean() * 100
            print(f"  {c:<15} {pct:5.1f}% null")


if __name__ == "__main__":
    main()
