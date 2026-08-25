"""Assign player profile archetypes to player_comps.csv using k-means clustering.

For each player, computes a single career-weighted score per skill dimension
(PA-weighted across levels, with level-discount factors), then clusters on
6 independent skill axes:
  - Kpct      : strikeout rate (negative — higher = worse discipline)
  - BBpct     : walk rate
  - ISO       : isolated power (SLG - AVG)
  - HRFB      : HR/FB rate (power quality)
  - PullAir   : pull-air batted ball profile (power shape)
  - SBTalent  : SB_pct × SB/PA (speed/baserunning)

Whiff% and BB2K are intentionally excluded: Whiff% is highly collinear with
K%, and BB2K = BB% − 2×K% is derived from axes already included.

Output: adds 'Profile' and 'Profile_K', 'Profile_BB', 'Profile_ISO',
'Profile_Speed', 'Profile_Power' subscores to player_comps.csv.

Usage:
  python analysis/build_comp_profiles.py            # cluster + print summary
  python analysis/build_comp_profiles.py --k 7      # override cluster count
  python analysis/build_comp_profiles.py --show     # print per-cluster centroid table
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA    = Path(__file__).resolve().parent.parent / "data"
OUT     = DATA / "computed" / "player_comps.csv"

LEVELS         = ["AAA", "AA", "A+", "A", "R"]
LEVEL_DISCOUNT = {"AAA": 1.00, "AA": 0.59, "A+": 0.34, "A": 0.23, "R": 0.10}
MIN_PA         = 80
N_CLUSTERS     = 6
RANDOM_STATE   = 42

# Clustering features: (csv_prefix, direction)
# direction: +1 = higher is better/more of that trait, -1 = higher is worse
PROFILE_FEATURES = [
    ("Kpct",    -1),   # strikeout rate — high K% = free swinger
    ("BBpct",   +1),   # walk rate — patience/discipline
    ("ISO",     +1),   # isolated power
    ("HRFB",    +1),   # HR/FB — power quality
    ("PullAir", +1),   # pull-air profile
    ("SBTalent",+1),   # speed/baserunning
]

# Human-readable archetype names keyed by cluster label after inspection.
# These are assigned after running the first time and inspecting centroids.
# Keys match the cluster integer; values are updated via --relabel below.
ARCHETYPE_MAP: dict[int, str] = {}  # populated at runtime after fit


def career_composite(df: pd.DataFrame) -> pd.DataFrame:
    """Compute PA+level-discount weighted career average for each profile feature."""
    rows = []
    lk_list = [l.replace("+", "plus") for l in LEVELS]
    discounts = {l.replace("+", "plus"): LEVEL_DISCOUNT[l] for l in LEVELS}

    for _, row in df.iterrows():
        rec: dict = {"PlayerId": row["PlayerId"], "Name": row["Name"]}
        for feat, _ in PROFILE_FEATURES:
            num = 0.0
            denom = 0.0
            for lk in lk_list:
                pa  = row.get(f"PA_{lk}")
                val = row.get(f"{feat}_{lk}")
                if pd.isna(pa) or pa < MIN_PA or pd.isna(val):
                    continue
                wt   = pa * discounts[lk]
                num  += wt * val
                denom += wt
            rec[feat] = num / denom if denom > 0 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def label_cluster(centroid: pd.Series, feat_names: list[str]) -> str:
    """Derive a descriptive label from a cluster centroid (z-scored values)."""
    k_z   = centroid["Kpct"]
    bb_z  = centroid["BBpct"]
    iso_z = centroid["ISO"]
    hr_z  = centroid["HRFB"]
    sb_z  = centroid["SBTalent"]

    # Speed is the clearest separator — SBTalent distribution is right-skewed
    if sb_z > 1.5:
        return "Speed / Athleticism"

    # Pure power: both ISO and HRFB strongly elevated
    if iso_z > 1.0 and hr_z > 1.0:
        return "Power / Free Swinger" if k_z > 0.5 else "Power"

    # Moderate power with decent contact (ISO elevated, K below avg)
    if iso_z > 0.3 and k_z < 0.3:
        return "Gap Power"

    # High BB, low-to-moderate power → patient hitters
    if bb_z > 1.0:
        return "Disciplined"

    # High K, low power, low BB → swing and miss without the upside
    if k_z > 0.8 and iso_z < -0.3 and hr_z < -0.2:
        return "Swing & Miss"

    # Low K, low power → contact/slap profile
    if k_z < -0.5 and iso_z < -0.5:
        return "Contact / Slap"

    return "All-Around"


def run(k: int = N_CLUSTERS, show: bool = False) -> pd.DataFrame:
    df = pd.read_csv(OUT)
    print(f"Loaded {len(df):,} rows from {OUT}")

    comp = career_composite(df)
    feat_names = [f for f, _ in PROFILE_FEATURES]

    # Drop rows missing any feature
    valid = comp.dropna(subset=feat_names).copy()
    print(f"Players with full career composite: {len(valid):,} / {len(comp):,}")

    X = valid[feat_names].values
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
    valid["cluster"] = km.fit_predict(Xz)

    # Build centroid DataFrame (in z-score space)
    centers_z = pd.DataFrame(km.cluster_centers_, columns=feat_names)

    # Label each cluster
    cluster_labels = {}
    for ci, row in centers_z.iterrows():
        cluster_labels[ci] = label_cluster(row, feat_names)

    valid["Profile"] = valid["cluster"].map(cluster_labels)

    # Print centroid summary
    print(f"\n{'='*70}")
    print(f"K-means clusters (k={k})  —  centroids in z-score units")
    print(f"{'='*70}")
    centers_raw = pd.DataFrame(
        scaler.inverse_transform(km.cluster_centers_), columns=feat_names
    )
    for ci in range(k):
        n = (valid["cluster"] == ci).sum()
        grad = valid.loc[valid["cluster"] == ci, "PlayerId"].isin(
            df.loc[df["graduated"] == True, "PlayerId"]
        ).sum() if "graduated" in df.columns else "?"
        lbl = cluster_labels[ci]
        r = centers_raw.iloc[ci]
        print(f"\n  [{ci}] {lbl}  (n={n}, {grad} graduates)")
        print(f"       K%={r['Kpct']:.1%}  BB%={r['BBpct']:.1%}  "
              f"ISO={r['ISO']:.3f}  HR/FB={r['HRFB']:.1%}  "
              f"PullAir={r['PullAir']:.1%}  SBTalent={r['SBTalent']:.4f}")
    print()

    if show:
        print("\nCluster size breakdown:")
        print(valid.groupby("Profile").size().sort_values(ascending=False).to_string())

    # Merge Profile back into main df (drop stale column if present)
    if "Profile" in df.columns:
        df = df.drop(columns=["Profile"])
    df = df.merge(valid[["PlayerId", "Profile"]], on="PlayerId", how="left")

    # Move Profile column after Name
    cols = [c for c in df.columns if c != "Profile"]
    name_idx = cols.index("Name")
    cols.insert(name_idx + 1, "Profile")
    df = df[cols]

    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df):,} rows → {OUT}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",    type=int, default=N_CLUSTERS)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    run(k=args.k, show=args.show)
