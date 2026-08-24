"""Build data/rankings/2026ProspectPool.csv — Phase 3 unified prospect ranking.

Combined_Score = 0.40 × Skill_Score + 0.25 × MLB_Proj_Score + 0.22 × PPPA_Score + 0.13 × Age_Score
All signals are re-standardized within the pool to 50±10 before blending.
Players missing a signal default to 50 (pool neutral) for that component.
BW_Rank is retained as a reference column but excluded from the blend.
Age_Score is derived from Age_Z_SL (inverted: younger-for-level = higher score).
PPPA_Score is PA-weighted PPPA_Z_SL with level discounts derived from chained SB
  translation factors (2023+ era, normalized to AAA=1.0):
  AAA=1.00, AA=0.84, A+=0.71, A=0.57, R=0.39
Pool includes all players active in 2025 or 2026 (Season >= 2025).
"""

from pathlib import Path
import re
import unicodedata
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HIST_DIR = DATA_DIR / "historical"

SKILL_W    = 0.40
MLB_PROJ_W = 0.25
PPPA_W     = 0.22
AGE_W      = 0.13

MIN_SEASON = 2025
MAX_PROSPECT_AGE = 24   # age 25-26 in pipeline only for AAA rankings, not prospect pool


def normalize_name(name: str) -> str:
    name = name.lower().strip()
    name = name.replace(".", "")
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"\s+jr$", "", name)
    name = re.sub(r"\s+", " ", name)
    return name


def _pool_scale(df: pd.DataFrame, col: str, out_col: str) -> None:
    """Standardize col within non-null pool rows to 50±10; fill nulls with 50."""
    mask = df[col].notna()
    if mask.sum() > 1:
        s = df.loc[mask, col]
        m, sd = s.mean(), s.std()
        df[out_col] = 50.0
        df.loc[mask, out_col] = ((50 + 10 * (s - m) / sd).clip(lower=0, upper=100).round(2))
    else:
        df[out_col] = 50.0


def _born_name_lookup() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (hit_bday, hit_name) lookup frames for resolving TBC players → PlayerId."""
    bday_path = DATA_DIR / "api" / "player_birthdays.csv"
    hit_path  = DATA_DIR / "api" / "milb_hitting.csv"
    if bday_path.exists() and hit_path.exists():
        bdays   = pd.read_csv(bday_path, dtype={"MLBAM_ID": "Int64", "BirthDate": str})
        hitting = pd.read_csv(hit_path, usecols=["PlayerId", "MLBAM_ID", "Name"],
                              dtype={"PlayerId": str, "MLBAM_ID": "Int64"})
        hitting["_norm"] = hitting["Name"].apply(normalize_name)
        hit_bday = (hitting
                    .merge(bdays.rename(columns={"BirthDate": "born"}), on="MLBAM_ID", how="inner")
                    .drop_duplicates(subset=["_norm", "born"])[["_norm", "born", "PlayerId"]])
        name_counts = hitting.groupby("_norm")["PlayerId"].nunique()
        uniq_names  = name_counts[name_counts == 1].index
        hit_name    = (hitting[hitting["_norm"].isin(uniq_names)]
                       .drop_duplicates("_norm")[["_norm", "PlayerId"]])
    else:
        hit_bday = pd.DataFrame(columns=["_norm", "born", "PlayerId"])
        hit_name = pd.DataFrame(columns=["_norm", "PlayerId"])
    return hit_bday, hit_name


def _tbc_ranks_for_year(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (ba_df, pipe_df) with columns [PlayerId, _norm, BA_Rank / PIPE_Rank].

    Join strategy:
      Primary  — born date + normalized name → MLBAM_ID → PlayerId (via player_birthdays + milb_hitting)
      Fallback — normalized name only (for players whose born date is missing from either side)
    """
    tbc_path = DATA_DIR / "rankings" / "tbc_rankings.csv"
    if not tbc_path.exists():
        return pd.DataFrame(columns=["PlayerId", "_norm", "BA_Rank"]), \
               pd.DataFrame(columns=["PlayerId", "_norm", "PIPE_Rank"])

    tbc = pd.read_csv(tbc_path, dtype={"born": str})
    tbc26 = tbc[(tbc["year"] == year) & (tbc["source"].isin(["BA", "PIPE"]))].copy()
    tbc26["_norm"] = tbc26["player"].apply(normalize_name)

    hit_bday, hit_name = _born_name_lookup()

    def _resolve(df: pd.DataFrame, rank_col: str) -> pd.DataFrame:
        # Primary: born + name
        merged = df.merge(hit_bday, on=["_norm", "born"], how="left")
        # Fallback: name only for unmatched rows
        no_id = merged["PlayerId"].isna()
        if no_id.any():
            fill = merged.loc[no_id, ["_norm"]].merge(hit_name, on="_norm", how="left")
            merged.loc[no_id, "PlayerId"] = fill["PlayerId"].values
        merged = merged.rename(columns={"rank": rank_col})
        return merged[["PlayerId", "_norm", rank_col]].dropna(subset=[rank_col])

    results = {}
    for source, rank_col in [("BA", "BA_Rank"), ("PIPE", "PIPE_Rank")]:
        sub = tbc26[tbc26["source"] == source].copy()
        results[rank_col] = _resolve(sub, rank_col)

    return results["BA_Rank"], results["PIPE_Rank"]


def _tbc_org_ranks_for_year(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (ba_org_df, pipe_org_df) with org-level ranks from franchise prospect lists.

    Source: data/rankings/tbc_team_rankings_{year}.csv (scraped from TBC franchise pages).
    Columns returned: [PlayerId, _norm, born, player, pos, BA_Org_Rank / PIPE_Org_Rank]
    """
    team_path = DATA_DIR / "rankings" / f"tbc_team_rankings_{year}.csv"
    empty_ba   = pd.DataFrame(columns=["PlayerId", "_norm", "born", "player", "pos", "BA_Org_Rank"])
    empty_pipe = pd.DataFrame(columns=["PlayerId", "_norm", "born", "player", "pos", "PIPE_Org_Rank"])
    if not team_path.exists():
        return empty_ba, empty_pipe

    team = pd.read_csv(team_path, dtype={"born": str, "rank": "Int64"})
    team = team[team["status"] != "Active"].copy()  # exclude MLB-active players
    team["_norm"] = team["player"].apply(normalize_name)
    team["born"]  = team["born"].fillna("")

    hit_bday, hit_name = _born_name_lookup()

    def _resolve_org(df: pd.DataFrame, rank_col: str) -> pd.DataFrame:
        merged = df.merge(hit_bday, on=["_norm", "born"], how="left")
        no_id = merged["PlayerId"].isna()
        if no_id.any():
            fill = merged.loc[no_id, ["_norm"]].merge(hit_name, on="_norm", how="left")
            merged.loc[no_id, "PlayerId"] = fill["PlayerId"].values
        merged = merged.rename(columns={"rank": rank_col})
        # Keep best (lowest) org rank per player-source
        merged = (merged.sort_values(rank_col)
                        .drop_duplicates(subset="_norm", keep="first"))
        return merged[["PlayerId", "_norm", "born", "player", "pos", rank_col]]

    results = {}
    for source, rank_col in [("BA", "BA_Org_Rank"), ("PIPE", "PIPE_Org_Rank")]:
        sub = team[team["source"] == source].copy()
        results[rank_col] = _resolve_org(sub, rank_col)

    return results["BA_Org_Rank"], results["PIPE_Org_Rank"]


def main() -> None:
    bw = pd.read_csv(DATA_DIR / "rankings" / "bwRankings.csv", encoding="latin-1")
    ml = pd.read_csv(DATA_DIR / "computed" / "minorLeagueData.csv")

    bw["_norm"] = bw["Name"].apply(normalize_name)
    ml["_norm"] = ml["Name"].apply(normalize_name)

    # Most recent season per player: highest Season, then most PA if tied
    pool = (
        ml
        .sort_values(["PlayerId", "Season", "PA"], ascending=[True, False, False])
        .drop_duplicates(subset="PlayerId", keep="first")
    )

    # Deduplicate by normalized name (keep one PlayerId per name)
    pool = (
        pool
        .sort_values(["_norm", "Season", "PA"], ascending=[True, False, False])
        .drop_duplicates(subset="_norm", keep="first")
    )

    # Override Level with highest level reached across all career seasons
    LEVEL_ORDER = {"R": 1, "A": 2, "A+": 3, "AA": 4, "AAA": 5}
    ml["_level_ord"] = ml["Level"].map(LEVEL_ORDER)
    highest = (
        ml.dropna(subset=["_level_ord"])
        .sort_values("_level_ord", ascending=False)
        .drop_duplicates(subset="PlayerId", keep="first")
        [["PlayerId", "Level"]]
        .rename(columns={"Level": "Level_Top"})
    )
    pool = pool.merge(highest, on="PlayerId", how="left")
    pool["Level"] = pool["Level_Top"].fillna(pool["Level"])
    pool = pool.drop(columns="Level_Top")

    # Only players active in 2025 or 2026
    pool = pool[pool["Season"] >= MIN_SEASON]

    # Exclude age 25-26 players — pipeline now fetches up to Age 26 for AAA rankings,
    # but they are not prospects and must not appear in the prospect pool.
    pool = pool[pool["Age"] <= MAX_PROSPECT_AGE]

    # Exclude players with ≥50 career MLB PA.
    # ID systems may not align (hist_mlb uses FanGraphs IDs; pool may have MLBAM IDs for recent
    # players not yet in the Chadwick crosswalk), so filter on both PlayerId and normalized Name.
    mlb = pd.read_csv(DATA_DIR / "historical" / "hist_mlb_data.csv", usecols=["PlayerId", "Name", "PA"])
    mlb["_norm"]    = mlb["Name"].apply(normalize_name)
    mlb_id_pa       = mlb.groupby("PlayerId")["PA"].sum()
    mlb_name_pa     = mlb.groupby("_norm")["PA"].sum()
    mlb_ids         = set(mlb_id_pa[mlb_id_pa >= 50].index.astype(str))
    mlb_names       = set(mlb_name_pa[mlb_name_pa >= 50].index)
    pool["_norm"]   = pool["Name"].apply(normalize_name)
    pool = pool[
        ~pool["PlayerId"].astype(str).isin(mlb_ids) &
        ~pool["_norm"].isin(mlb_names)
    ]
    pool = pool.drop(columns="_norm")

    out = pool[["PlayerId", "Name", "Team", "Level", "Age", "Age_Z_SL"]].reset_index(drop=True)

    # BW rank — left join so non-BW players get null BW_Rank
    bw_rank = bw[["_norm", "Rank"]].rename(columns={"Rank": "BW_Rank"})
    out["_norm"] = out["Name"].apply(normalize_name)
    out = out.merge(bw_rank, on="_norm", how="left")

    # TBC 2026 rankings (BA + Pipeline) — reference only, like BW_Rank
    current_year = int(ml["Season"].max())
    ba_df, pipe_df = _tbc_ranks_for_year(current_year)

    # Primary join by PlayerId; fallback by normalized name for unmatched rows
    def _join_tbc(frame: pd.DataFrame, tbc_df: pd.DataFrame, rank_col: str) -> pd.DataFrame:
        tbc_ids = tbc_df[["PlayerId", rank_col]].dropna(subset=["PlayerId"]).copy()
        # Cast tbc PlayerId to match frame's dtype to avoid merge type error
        try:
            tbc_ids["PlayerId"] = tbc_ids["PlayerId"].astype(frame["PlayerId"].dtype)
        except (ValueError, TypeError):
            tbc_ids["PlayerId"] = tbc_ids["PlayerId"].astype(str)
            frame = frame.copy()
            frame["PlayerId"] = frame["PlayerId"].astype(str)
        frame = frame.merge(tbc_ids, on="PlayerId", how="left")
        no_match = frame[rank_col].isna() & frame["_norm"].isin(tbc_df["_norm"])
        if no_match.any():
            name_fill = frame.loc[no_match, ["_norm"]].merge(
                tbc_df[["_norm", rank_col]], on="_norm", how="left"
            )
            frame.loc[no_match, rank_col] = name_fill[rank_col].values
        return frame

    out = _join_tbc(out, ba_df,   "BA_Rank")
    out = _join_tbc(out, pipe_df, "PIPE_Rank")

    # TBC 2026 org (franchise) rankings — broader coverage than national top-100
    ba_org_df, pipe_org_df = _tbc_org_ranks_for_year(current_year)
    out = _join_tbc(out, ba_org_df,   "BA_Org_Rank")
    out = _join_tbc(out, pipe_org_df, "PIPE_Org_Rank")
    out = out.drop(columns="_norm")

    # Historical fantasy score (reference only — no longer in Combined_Score)
    scores = pd.read_csv(DATA_DIR / "rankings" / "player_scores.csv")
    pool_scores = scores[scores["PlayerId"].isin(out["PlayerId"])].copy()
    pool_scores["Score_Rank"] = (
        pool_scores["Score"].rank(ascending=False, method="min").astype("Int64")
    )
    out = out.merge(pool_scores[["PlayerId", "Total_PA", "Score", "Score_Rank"]], on="PlayerId", how="left")
    out = out.rename(columns={"Total_PA": "PA"})

    # Phase 1: Skill_Score (BB%, K%, GB%, SwStr%, ISO, Age_Z_SL — era-weighted)
    skill = pd.read_csv(HIST_DIR / "skill_scores.csv")[["PlayerId", "Skill_Score"]]
    out = out.merge(skill, on="PlayerId", how="left")

    # Phase 2: MLB_Proj_Score (Phase 2 WLS coefficients applied to recent MiLB stats)
    proj = pd.read_csv(HIST_DIR / "prospect_mlb_proj.csv")[["PlayerId", "MLB_Proj_Score"]]
    out = out.merge(proj, on="PlayerId", how="left")

    # Re-standardize all signals within pool to 50±10
    _pool_scale(out, "Skill_Score",    "_skill_s")
    _pool_scale(out, "MLB_Proj_Score", "_proj_s")

    # Age_Score: two-component penalty.
    # (1) Level-discount Age_Z_SL — lower levels get less credit for peer-relative youth.
    # (2) Absolute age-for-level penalty — years above the elite-prospect target age at each
    #     level subtract from the signal, catching old players whose peers are also old (e.g. 22 in A+).
    LEVEL_AGE_WEIGHT = {"AAA": 1.0, "AA": 1.0, "A+": 0.75, "A": 0.60, "R": 0.40}
    LEVEL_AGE_TARGET = {"R": 18, "A": 19, "A+": 20, "AA": 21, "AAA": 22}
    age_target  = out["Level"].map(LEVEL_AGE_TARGET).fillna(20)
    age_penalty = (out["Age"] - age_target).clip(lower=0)
    out["_age_inv"] = (
        -(out["Age_Z_SL"].fillna(0) * out["Level"].map(LEVEL_AGE_WEIGHT).fillna(0.60))
        - 0.5 * age_penalty
    )
    _pool_scale(out, "_age_inv", "_age_s")
    out["Age_Score"] = out["_age_s"].round(2)

    # PPPA_Score: PA-weighted PPPA_Z_SL across all career seasons, level-discounted.
    # Level weights from chained SB translation factors (2023+ era, normalized AAA=1.0).
    LEVEL_PPPA_WEIGHT = {"AAA": 1.00, "AA": 0.84, "A+": 0.71, "A": 0.57, "R": 0.39}
    hist = pd.read_csv(HIST_DIR / "ovr_hist_data.csv",
                       usecols=["PlayerId", "Level", "PA", "PPPA_Z_SL"])
    hist = hist[hist["PPPA_Z_SL"].notna()].copy()
    hist["_lvl_wt"]  = hist["Level"].map(LEVEL_PPPA_WEIGHT).fillna(0.57)
    hist["_wt"]      = hist["PA"] * hist["_lvl_wt"]
    hist["_wt_pppa"] = hist["PPPA_Z_SL"] * hist["_wt"]
    pppa_agg = (
        hist.groupby("PlayerId")[["_wt_pppa", "_wt"]]
        .sum()
        .reset_index()
    )
    pppa_agg["_pppa_raw"] = pppa_agg["_wt_pppa"] / pppa_agg["_wt"]
    out = out.merge(pppa_agg[["PlayerId", "_pppa_raw"]], on="PlayerId", how="left")
    _pool_scale(out, "_pppa_raw", "_pppa_s")
    out["PPPA_Score"] = out["_pppa_s"].round(2)

    # Phase 3 blend (BW_Rank retained as reference only, not in blend)
    out["Combined_Score"] = (
        SKILL_W    * out["_skill_s"] +
        MLB_PROJ_W * out["_proj_s"] +
        PPPA_W     * out["_pppa_s"] +
        AGE_W      * out["_age_s"]
    ).round(2)
    out["Combined_Rank"] = (
        out["Combined_Score"].rank(ascending=False, method="min").astype("Int64")
    )

    col_order = [
        "Combined_Rank", "Name", "Team", "Level", "Age", "PA",
        "BW_Rank", "BA_Rank", "PIPE_Rank", "BA_Org_Rank", "PIPE_Org_Rank",
        "Score", "Score_Rank",
        "Skill_Score", "MLB_Proj_Score", "PPPA_Score", "Age_Score", "Combined_Score", "PlayerId",
    ]
    out = (
        out.drop(columns=["Age_Z_SL", "_age_inv", "_skill_s", "_proj_s", "_age_s", "_pppa_raw", "_pppa_s"])
        [col_order]
        .sort_values("Combined_Rank")
        .reset_index(drop=True)
    )

    out_path = DATA_DIR / "rankings" / "2026ProspectPool.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} players to {out_path.name}")

    # Report BW-ranked players not matched in pool
    pool_norms = set(out["Name"].apply(normalize_name))
    unmatched = bw[~bw["_norm"].isin(pool_norms)]["Name"].tolist()
    print(f"\n{len(unmatched)} BW-ranked players not found in 2025/2026 data:")
    for name in sorted(unmatched):
        print(f"  {name}")

    # tbc_vs_model.csv — all pool players with any TBC ranking (national or org-level)
    out["_norm"] = out["Name"].apply(normalize_name)
    ranked_mask = (
        out["BA_Rank"].notna() | out["PIPE_Rank"].notna() |
        out["BA_Org_Rank"].notna() | out["PIPE_Org_Rank"].notna()
    )
    vs = out[ranked_mask].copy()

    # TBC metadata (name on TBC, pos, born) — pull from org data frame (broader coverage)
    tbc_meta = pd.concat([
        ba_org_df[["PlayerId", "player", "pos", "born"]].assign(_src="BA"),
        pipe_org_df[["PlayerId", "player", "pos", "born"]].assign(_src="PIPE"),
    ]).dropna(subset=["PlayerId"])
    try:
        tbc_meta["PlayerId"] = tbc_meta["PlayerId"].astype(vs["PlayerId"].dtype)
    except (ValueError, TypeError):
        tbc_meta["PlayerId"] = tbc_meta["PlayerId"].astype(str)
        vs = vs.copy(); vs["PlayerId"] = vs["PlayerId"].astype(str)
    tbc_meta = tbc_meta.drop_duplicates(subset="PlayerId").drop(columns="_src")
    vs = vs.merge(tbc_meta, on="PlayerId", how="left")

    # Age at ranking year (Jan 1 convention)
    vs["Age_At_Rank"] = vs["born"].apply(
        lambda b: (current_year - int(str(b)[:4]))
        if pd.notna(b) and str(b).strip() else None
    )

    vs["BA_Delta"]   = (vs["BA_Rank"]   - vs["Combined_Rank"]).round(0)
    vs["PIPE_Delta"] = (vs["PIPE_Rank"] - vs["Combined_Rank"]).round(0)

    vs_out = (vs[[
        "Combined_Rank", "Name", "Team", "Level",
        "BA_Rank", "BA_Delta", "PIPE_Rank", "PIPE_Delta",
        "BA_Org_Rank", "PIPE_Org_Rank",
        "player", "pos", "born", "Age_At_Rank",
        "Skill_Score", "MLB_Proj_Score", "PPPA_Score", "Age_Score", "Combined_Score",
    ]].rename(columns={"player": "TBC_Name", "pos": "Pos", "born": "Born"})
      .sort_values("Combined_Rank").reset_index(drop=True))

    vs_path = DATA_DIR / "rankings" / "tbc_vs_model.csv"
    vs_out.to_csv(vs_path, index=False)
    n_national = (vs_out["BA_Rank"].notna() | vs_out["PIPE_Rank"].notna()).sum()
    print(f"\ntbc_vs_model.csv: {len(vs_out)} players ranked by TBC "
          f"({n_national} on national lists, {len(vs_out)-n_national} org-only)")
    out = out.drop(columns="_norm")

    # bw_vs_model.csv — all 309 BW-ranked players vs. Combined_Rank
    pool_lookup = out.copy()
    pool_lookup["_norm"] = pool_lookup["Name"].apply(normalize_name)
    pool_idx = pool_lookup.set_index("_norm")

    bw_rows = []
    for _, bw_row in bw.iterrows():
        norm = bw_row["_norm"]
        in_pool = norm in pool_idx.index
        if in_pool:
            p = pool_idx.loc[norm]
            bw_rows.append({
                "BW_Rank":        int(bw_row["Rank"]),
                "BW_Name":        bw_row["Name"],
                "BW_Team":        bw_row["Team"],
                "Pos":            bw_row["Pos"],
                "BW_Age":         bw_row["Age"],
                "In_Pool":        True,
                "Combined_Rank":  int(p["Combined_Rank"]),
                "Delta":          int(bw_row["Rank"]) - int(p["Combined_Rank"]),
                "Pool_Name":      p["Name"],
                "Pool_Team":      p["Team"],
                "Level":          p["Level"],
                "Skill_Score":    p["Skill_Score"],
                "MLB_Proj_Score": p["MLB_Proj_Score"],
                "PPPA_Score":     p["PPPA_Score"],
                "Age_Score":      p["Age_Score"],
                "Combined_Score": p["Combined_Score"],
            })
        else:
            bw_rows.append({
                "BW_Rank":  int(bw_row["Rank"]),
                "BW_Name":  bw_row["Name"],
                "BW_Team":  bw_row["Team"],
                "Pos":      bw_row["Pos"],
                "BW_Age":   bw_row["Age"],
                "In_Pool":  False,
                "Combined_Rank": None, "Delta": None,
                "Pool_Name": None, "Pool_Team": None, "Level": None,
                "Skill_Score": None, "MLB_Proj_Score": None,
                "PPPA_Score": None, "Age_Score": None, "Combined_Score": None,
            })

    bw_vs = pd.DataFrame(bw_rows).sort_values("BW_Rank").reset_index(drop=True)
    bw_vs_path = DATA_DIR / "rankings" / "bw_vs_model.csv"
    bw_vs.to_csv(bw_vs_path, index=False)
    n_matched = bw_vs["In_Pool"].sum()
    print(f"\nbw_vs_model.csv: {len(bw_vs)} BW players ({n_matched} matched, {len(bw_vs)-n_matched} not in pool)")


if __name__ == "__main__":
    main()
