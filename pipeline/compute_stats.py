"""Recompute derived columns on minorLeagueData.csv and refresh the averages_*.csv baselines.

Reads `data/minorLeagueData.csv` (which is expected to contain only the raw counting-stat
columns PlayerId..CS), computes TB, TP, PPG, PPPA and z-scores against season+league+age
and season+league baselines, and writes the augmented file back in place. Also refreshes
the averages_*.csv files used as baselines.

Scoring weights come from images/scoringSystem.png.
"""

from pathlib import Path

import pandas as pd

DATA_DIR     = Path(__file__).resolve().parent.parent / "data"
COMPUTED_DIR = DATA_DIR / "computed"
API_CSV  = DATA_DIR / "api" / "milb_hitting.csv"
OUT_CSV  = COMPUTED_DIR / "minorLeagueData.csv"
LEAGUES_DIR = DATA_DIR / "Leagues"
AVG_SEASON_LEAGUE_CSV     = COMPUTED_DIR / "averages_season_league.csv"
AVG_SEASON_LEAGUE_AGE_CSV = COMPUTED_DIR / "averages_season_league_age.csv"
PLAYER_SCORES_CSV = DATA_DIR / "rankings" / "player_scores.csv"

BASE_COLS = [
    "PlayerId", "Season", "Name", "Team", "Level", "Age",
    "G", "PA", "1B", "2B", "3B", "HR",
    "R", "RBI", "BB", "IBB", "SO", "GIDP", "SB", "CS",
]
DERIVED_COLS = [
    "TB", "TP", "PPG", "PPPA",
    "PPG_Z", "PPPA_Z", "PPG_Z_SL", "PPPA_Z_SL", "Age_Z_SL",
]
OUTPUT_COLS = [
    "PlayerId", "Season", "Name", "Team", "Level", "League", "Age",
    "G", "PA", "1B", "2B", "3B", "HR",
    "R", "RBI", "BB", "IBB", "SO", "GIDP", "SB", "CS",
    *DERIVED_COLS,
]

SCORING_WEIGHTS = {
    "1B": 1, "2B": 2, "3B": 3, "HR": 4,
    "R": 1, "RBI": 2, "BB": 1, "IBB": 1.5,
    "SO": -2, "GIDP": -1.5, "SB": 3, "CS": -1.5,
    "TB": 1,
}


def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df[BASE_COLS].copy()
    df["TB"] = df["1B"] + 2 * df["2B"] + 3 * df["3B"] + 4 * df["HR"]
    df["TP"] = sum(weight * df[col] for col, weight in SCORING_WEIGHTS.items())
    df["PPG"] = (df["TP"] / df["G"]).round(2)
    df["PPPA"] = (df["TP"] / df["PA"]).round(2)
    return df


def compute_averages(df: pd.DataFrame):
    season_league = (
        df.groupby(["Season", "League"])
        .agg(
            Avg_Age=("Age", "mean"),
            Std_Age=("Age", "std"),
            Avg_PPG=("PPG", "mean"),
            Avg_PPPA=("PPPA", "mean"),
            Std_PPG=("PPG", "std"),
            Std_PPPA=("PPPA", "std"),
        )
        .round(2)
        .reset_index()
    )
    season_league_age = (
        df.groupby(["Season", "League", "Age"])
        .agg(
            Avg_PPG=("PPG", "mean"),
            Avg_PPPA=("PPPA", "mean"),
            Std_PPG=("PPG", "std"),
            Std_PPPA=("PPPA", "std"),
        )
        .round(2)
        .reset_index()
    )
    return season_league, season_league_age


def attach_league(df: pd.DataFrame) -> pd.DataFrame:
    """Look up each row's league by matching (PlayerId, Season, Level) against
    every CSV in data/Leagues/. League column is set to the source filename
    (minus .csv); rows with no match get 'DiscLeague'."""
    parts = []
    for league_csv in sorted(LEAGUES_DIR.glob("*.csv")):
        part = pd.read_csv(league_csv, usecols=["PlayerId", "Season", "Level"])
        part["League"] = league_csv.stem
        parts.append(part)
    lookup = (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates(subset=["PlayerId", "Season", "Level"])
    )
    df = df.merge(lookup, on=["PlayerId", "Season", "Level"], how="left")
    df["League"] = df["League"].fillna("DiscLeague")
    return df


def attach_z_scores(df: pd.DataFrame, season_league: pd.DataFrame, season_league_age: pd.DataFrame) -> pd.DataFrame:
    sla = season_league_age.rename(columns={
        "Avg_PPG": "_avg_ppg_sla", "Avg_PPPA": "_avg_pppa_sla",
        "Std_PPG": "_std_ppg_sla", "Std_PPPA": "_std_pppa_sla",
    })
    df = df.merge(sla, on=["Season", "League", "Age"], how="left")

    sl = season_league[["Season", "League", "Avg_Age", "Std_Age", "Avg_PPG", "Avg_PPPA", "Std_PPG", "Std_PPPA"]].rename(columns={
        "Avg_Age": "_avg_age_sl", "Std_Age": "_std_age_sl",
        "Avg_PPG": "_avg_ppg_sl", "Avg_PPPA": "_avg_pppa_sl",
        "Std_PPG": "_std_ppg_sl", "Std_PPPA": "_std_pppa_sl",
    })
    df = df.merge(sl, on=["Season", "League"], how="left")

    df["PPG_Z"] = ((df["PPG"] - df["_avg_ppg_sla"]) / df["_std_ppg_sla"]).round(2)
    df["PPPA_Z"] = ((df["PPPA"] - df["_avg_pppa_sla"]) / df["_std_pppa_sla"]).round(2)
    df["PPG_Z_SL"] = ((df["PPG"] - df["_avg_ppg_sl"]) / df["_std_ppg_sl"]).round(2)
    df["PPPA_Z_SL"] = ((df["PPPA"] - df["_avg_pppa_sl"]) / df["_std_pppa_sl"]).round(2)
    df["Age_Z_SL"] = ((df["Age"] - df["_avg_age_sl"]) / df["_std_age_sl"]).round(2)

    helper_cols = [c for c in df.columns if c.startswith("_")]
    return df.drop(columns=helper_cols)


def compute_player_scores(df: pd.DataFrame) -> pd.DataFrame:
    """PA-weighted mean of (PPPA_Z - Age_Z_SL) across each player's seasons.

    Age_Z_SL is subtracted because negative means younger-than-peers, which is
    a positive prospect signal. Rows missing either z-score are dropped from
    the weighted average (typically single-sample baseline groups).
    """
    rows = df.dropna(subset=["PPPA_Z", "Age_Z_SL", "PA"]).copy()
    rows["_row_score"] = 0.6 * rows["PPPA_Z"] - 0.4 * rows["Age_Z_SL"]
    rows["_weighted"] = rows["_row_score"] * rows["PA"]

    grouped = rows.groupby("PlayerId").agg(
        _weighted_sum=("_weighted", "sum"),
        Total_PA=("PA", "sum"),
        Seasons=("Season", "nunique"),
    )
    # Score = 50 + 10 * raw_score: 50 = exactly-average prospect, each ±10 = ±1 std
    # dev on the (PPPA_Z - Age_Z_SL) composite. No hard cap, stable across refreshes.
    raw_score = grouped["_weighted_sum"] / grouped["Total_PA"]
    grouped["Score"] = (50 + 10 * raw_score).clip(lower=0).round(2)

    season_range = (
        df.groupby("PlayerId")["Season"]
        .agg(lambda s: f"{s.min()}" if s.min() == s.max() else f"{s.min()}-{s.max()}")
        .rename("Season_Range")
    )

    latest_name = (
        df.sort_values("Season")
        .groupby("PlayerId")["Name"]
        .last()
        .rename("Name")
    )
    out = grouped.join(latest_name).join(season_range).reset_index()
    out = out[["PlayerId", "Name", "Seasons", "Season_Range", "Total_PA", "Score"]]
    return out.sort_values("Score", ascending=False).reset_index(drop=True)


def main() -> None:
    raw = pd.read_csv(API_CSV)
    missing = [c for c in BASE_COLS if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns in {API_CSV.name}: {missing}")

    # League is already supplied by the API fetch — preserve it before compute_derived
    # strips the DataFrame down to BASE_COLS.
    league_map = (
        raw[["PlayerId", "Season", "Team", "Level", "League"]]
        .drop_duplicates(subset=["PlayerId", "Season", "Team", "Level"])
    )

    df = compute_derived(raw)

    # Re-attach League from the API data; fall back to DiscLeague for any gaps.
    df = df.merge(league_map, on=["PlayerId", "Season", "Team", "Level"], how="left")
    df["League"] = df["League"].fillna("DiscLeague")

    season_league, season_league_age = compute_averages(df)
    season_league.to_csv(AVG_SEASON_LEAGUE_CSV, index=False)
    season_league_age.to_csv(AVG_SEASON_LEAGUE_AGE_CSV, index=False)

    df = attach_z_scores(df, season_league, season_league_age)
    df = df[OUTPUT_COLS]
    df.to_csv(OUT_CSV, index=False)

    scores = compute_player_scores(df)
    scores.to_csv(PLAYER_SCORES_CSV, index=False)


if __name__ == "__main__":
    main()

