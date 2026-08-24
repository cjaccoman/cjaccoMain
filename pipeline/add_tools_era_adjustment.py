"""Add era-adjusted TOOLS columns to prospect_features.csv.

For each raw TOOLS stat, compute a PA-weighted mean and std within the player's
Level x era combination, then z-score: (raw - cell_mean) / cell_std.

Using Level x era rather than era-only: HR/FB at R-ball is structurally different
from AAA, and era means pooled across levels would be dominated by lower-level
distributions (which have far more rows). Cells with fewer than MIN_ROWS qualifying
rows fall back to era-only means.

Era labels must be pre-populated by add_era_columns.py.

  Whiff%, Chase%, Z-Contact%  ->  EraK%    (strikeout environment)
  HR/FB                        ->  EraHR/FB (power environment)
  3B_PA                        ->  EraSB    (speed / baserunning environment)

Not era-adjusted:
  MaxEV, EV90  -- exit velocity is physics-based, era drift is minimal
  Spd          -- only available 2023-2026, all within the same era
  Age_Z_SL     -- already peer-relative within season+level

Direction for TOOLS scorer:
  Whiff%_adj    lower is better -> invert sign when scoring
  Chase%_adj    lower is better -> invert sign when scoring
  ZContact%_adj higher is better
  HRFB_adj      higher is better
  3B_PA_adj     higher is better
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR    = Path(__file__).resolve().parent.parent / "data"
FEATURES_IN = DATA_DIR / "rankings" / "prospect_features.csv"
MIN_PA      = 50    # minimum PA for a row to count toward cell mean/std
MIN_ROWS    = 10    # minimum qualifying rows for a Level x era cell before falling back

ADJUSTMENTS = [
    # (raw_col,     era_col,    out_col)
    ("Whiff%",      "EraK%",    "Whiff%_adj"),
    ("Chase%",      "EraK%",    "Chase%_adj"),
    ("Z-Contact%",  "EraK%",    "ZContact%_adj"),
    ("HR/FB",       "EraHR/FB", "HRFB_adj"),
    ("3B_PA",       "EraSB",    "3B_PA_adj"),
]


# ---------------------------------------------------------------------------
# Weighted stats helpers
# ---------------------------------------------------------------------------

def _wstats(g: pd.DataFrame, col: str) -> pd.Series:
    w   = g["PA"].values.astype(float)
    x   = g[col].values.astype(float)
    mu  = np.dot(w, x) / w.sum()
    var = np.dot(w, (x - mu) ** 2) / w.sum()
    return pd.Series({"mean": mu, "std": max(np.sqrt(var), 1e-9), "n": float(len(g))})


def build_lookup(
    df: pd.DataFrame, stat_col: str, era_col: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (level_era_stats, era_stats) DataFrames indexed by (era, level) / era."""
    valid = df[(df["PA"] >= MIN_PA)].dropna(subset=[stat_col])

    le_stats = valid.groupby([era_col, "Level"], observed=True).apply(
        lambda g: _wstats(g, stat_col), include_groups=False
    )
    e_stats  = valid.groupby(era_col, observed=True).apply(
        lambda g: _wstats(g, stat_col), include_groups=False
    )
    return le_stats, e_stats


def era_level_adjust(df: pd.DataFrame, stat_col: str, era_col: str) -> pd.Series:
    """Vectorised Level x era z-score with era-only fallback for sparse cells."""
    le_stats, e_stats = build_lookup(df, stat_col, era_col)

    mean_s = pd.Series(np.nan, index=df.index, dtype=float)
    std_s  = pd.Series(np.nan, index=df.index, dtype=float)

    for (era, level), idx in df.groupby([era_col, "Level"], observed=True).groups.items():
        try:
            cell = le_stats.loc[(era, level)]
            if cell["n"] >= MIN_ROWS:
                mean_s.iloc[df.index.get_indexer(idx)] = cell["mean"]
                std_s.iloc[df.index.get_indexer(idx)]  = cell["std"]
                continue
        except KeyError:
            pass
        # Fallback to era-only
        try:
            cell = e_stats.loc[era]
            mean_s.iloc[df.index.get_indexer(idx)] = cell["mean"]
            std_s.iloc[df.index.get_indexer(idx)]  = cell["std"]
        except KeyError:
            pass

    return ((df[stat_col] - mean_s) / std_s).round(4)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, stat_col: str, era_col: str, out_col: str) -> None:
    le_stats, _ = build_lookup(df, stat_col, era_col)
    print(f"{out_col}  ({stat_col} by {era_col} x Level):")
    for (era, level), row in le_stats.iterrows():
        flag = " [fallback]" if row["n"] < MIN_ROWS else ""
        print(f"  {era:<28} {level:<5}  "
              f"mean={row['mean']:.4f}  std={row['std']:.4f}  n={int(row['n']):,}{flag}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = pd.read_csv(FEATURES_IN)

    # Drop existing _adj columns to rebuild clean
    stale = [o for _, _, o in ADJUSTMENTS if o in df.columns]
    if stale:
        df = df.drop(columns=stale)

    for stat_col, era_col, out_col in ADJUSTMENTS:
        if stat_col not in df.columns or era_col not in df.columns:
            print(f"SKIP {out_col} — {stat_col} or {era_col} missing\n")
            continue

        print_summary(df, stat_col, era_col, out_col)

        adj = era_level_adjust(df, stat_col, era_col)
        adj.name = out_col

        n_total = df[stat_col].notna().sum()
        n_adj   = adj.notna().sum()
        print(f"  -> {n_adj:,} / {n_total:,} rows adjusted\n")

        pos = df.columns.get_loc(stat_col) + 1
        df.insert(pos, out_col, adj)

    df.to_csv(FEATURES_IN, index=False)
    adj_cols = [o for _, _, o in ADJUSTMENTS if o in df.columns]
    print(f"Wrote {len(df):,} rows -> {FEATURES_IN.name}")
    print(f"Added columns: {adj_cols}")


if __name__ == "__main__":
    main()
