"""AA player-season scatterplot: BB%-2K% vs Age_R, colored by MLB top-100 PPPA outcomes."""

from pathlib import Path
import re
import unicodedata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = Path.home() / "Documents" / "prospectFiles" / "prospectGraphs" / "aa_scatter_discipline_age.png"


def normalize_name(name: str) -> str:
    name = str(name).lower().strip().replace(".", "")
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"\s+jr$", "", name)
    return re.sub(r"\s+", " ", name)


# ── MiLB: AA player-seasons, PA ≥ 50 ─────────────────────────────────────────
ovr = pd.read_csv(DATA_DIR / "historical" / "ovr_hist_data.csv")
aa = ovr[(ovr["Level"] == "AA") & (ovr["PA"] >= 50)].copy()
aa = aa.dropna(subset=["BB/K", "K%"])

# Detect K% scale (decimal vs percentage points)
k_median = aa["K%"].median()
if k_median < 1:
    scale = 100.0   # stored as decimal → convert to pp
else:
    scale = 1.0     # already in pp

aa["BB_pct"]  = aa["BB/K"] * aa["K%"] * scale   # BB% in pp
aa["K_pct"]   = aa["K%"] * scale                 # K% in pp
aa["BB_m_2K"] = (aa["BB_pct"] - 2 * aa["K_pct"]).round(2)

# Age_R: within each Season+League group at AA (younger = higher percentile)
aa["Age_R"] = (
    aa.groupby(["Season", "League"])["Age"]
      .rank(pct=True, ascending=False) * 100
).round(1)

print(f"AA rows (PA≥50, BB/K+K% present): {len(aa):,}")
print(f"K% scale: {'decimal → *100' if scale == 100 else 'already pp'}")
print(f"BB%-2K% range: {aa['BB_m_2K'].min():.1f} to {aa['BB_m_2K'].max():.1f}")

# ── MLB: rank PPPA within each season, count top-100 finishes ─────────────────
mlb = pd.read_csv(DATA_DIR / "historical" / "hist_mlb_data.csv")
mlb = mlb[mlb["PA"] >= 100].copy()
mlb["season_rank"] = mlb.groupby("Season")["PPPA"].rank(ascending=False, method="min")
top100 = mlb[mlb["season_rank"] <= 100].copy()

# By MLBAMID
top100_by_id = top100.groupby("MLBAMID").size().reset_index(name="top100_n")

# By normalized name (fallback)
top100["_norm"] = top100["Name"].apply(normalize_name)
top100_by_name = top100.groupby("_norm").size().reset_index(name="top100_n_name")

# ── Join MiLB → MLB outcomes ──────────────────────────────────────────────────
aa = aa.merge(top100_by_id, left_on="PlayerId", right_on="MLBAMID", how="left")

# Name fallback for unmatched rows
aa["_norm"] = aa["Name"].apply(normalize_name)
unmatched = aa["top100_n"].isna()
if unmatched.any():
    fill = aa.loc[unmatched, ["_norm"]].merge(top100_by_name, on="_norm", how="left")
    aa.loc[unmatched, "top100_n"] = fill["top100_n_name"].values

aa["top100_n"] = aa["top100_n"].fillna(0).astype(int)

# Clip to Age_R ≥ 40
aa = aa[aa["Age_R"] >= 40].copy()

n_blue  = (aa["top100_n"] == 0).sum()
n_red   = (aa["top100_n"] == 1).sum()
n_green = (aa["top100_n"] >= 2).sum()
print(f"Outcome split — Blue: {n_blue:,} / Red: {n_red:,} / Green: {n_green:,}")

# ── Plot ──────────────────────────────────────────────────────────────────────
BG    = "#12141c"
PANEL = "#1a1d2e"
GRID  = "#2a2d42"

fig, ax = plt.subplots(figsize=(13, 10), facecolor=BG)
ax.set_facecolor(PANEL)

LAYERS = [
    ("#3a86ff", 0, "0 top-100 MLB PPPA seasons",  2, 0.35, 14),
    ("#ff5e57", 1, "1 top-100 MLB PPPA season",   3, 0.55, 18),
    ("#2ec46a", 2, "2+ top-100 MLB PPPA seasons", 4, 0.80, 22),
]

for color, outcome, label, zorder, alpha, size in LAYERS:
    if outcome == 2:
        mask = aa["top100_n"] >= 2
    else:
        mask = aa["top100_n"] == outcome
    ax.scatter(
        aa.loc[mask, "BB_m_2K"],
        aa.loc[mask, "Age_R"],
        c=color, alpha=alpha, s=size, linewidths=0,
        zorder=zorder, label=label,
    )

# ── Trend line (OLS across all points) ───────────────────────────────────────
valid = aa.dropna(subset=["BB_m_2K", "Age_R"])
m, b = np.polyfit(valid["BB_m_2K"], valid["Age_R"], 1)
x_line = np.linspace(valid["BB_m_2K"].min(), valid["BB_m_2K"].max(), 200)
ax.plot(x_line, m * x_line + b,
        color="#ffffff", linewidth=1.6, alpha=0.45, linestyle="-", zorder=5,
        label=f"Trend  (slope={m:.2f})")

# Reference lines
ax.axvline(0, color="#ffffff", linewidth=0.6, alpha=0.15, linestyle="--", zorder=1)
ax.axhline(50, color="#ffffff", linewidth=0.6, alpha=0.15, linestyle="--", zorder=1)

# Labels & styling
ax.set_xlabel("BB% − 2×K%  (percentage points, higher = better discipline)",
              color="#aaaacc", fontsize=11, labelpad=10)
ax.set_ylabel("Age Percentile Rank  (higher = younger for level)",
              color="#aaaacc", fontsize=11, labelpad=10)
ax.set_title(
    "AA Player-Seasons: Discipline vs. Age Relative to League  (2006–2026)\n"
    "PA ≥ 50  ·  Colored by MLB Top-100 PPPA Outcomes (PA ≥ 100)",
    color="white", fontsize=13, fontweight="bold", pad=14,
)

ax.tick_params(colors="#888899", labelsize=9)
for spine in ax.spines.values():
    spine.set_color("#2a2d42")
ax.grid(color=GRID, linewidth=0.5, linestyle="--", alpha=0.8)

# Quadrant corner labels
xl, xr = ax.get_xlim()
yb, yt = ax.get_ylim()
pad_x = (xr - xl) * 0.02
pad_y = (yt - yb) * 0.02
kw = dict(color="#ffffff", alpha=0.18, fontsize=8, fontstyle="italic")
ax.text(xr - pad_x, yt - pad_y, "Young + Elite Discipline", ha="right", va="top", **kw)
ax.text(xl + pad_x, yb + pad_y, "Old + Poor Discipline",    ha="left",  va="bottom", **kw)
ax.text(xr - pad_x, yb + pad_y, "Old + Elite Discipline",  ha="right", va="bottom", **kw)
ax.text(xl + pad_x, yt - pad_y, "Young + Poor Discipline",  ha="left",  va="top",    **kw)

# Legend
legend = ax.legend(
    loc="center right", framealpha=0.30, edgecolor="#3a3d52",
    labelcolor="#ccccdd", fontsize=10, markerscale=2.0,
    title="MLB Outcome", title_fontsize=9,
)
legend.get_title().set_color("#aaaacc")

# ── Player labels ─────────────────────────────────────────────────────────────
# Elite greens: best combined score of Age_R + BB_m_2K (normalized)
green_df = aa[aa["top100_n"] >= 2].copy()
green_df["elite_score"] = (
    (green_df["Age_R"] - green_df["Age_R"].min()) / (green_df["Age_R"].max() - green_df["Age_R"].min()) +
    (green_df["BB_m_2K"] - green_df["BB_m_2K"].min()) / (green_df["BB_m_2K"].max() - green_df["BB_m_2K"].min())
)
# Pick top elite candidates, deduplicate by name (keep best season per player)
top_elite = (green_df.sort_values("elite_score", ascending=False)
                      .drop_duplicates(subset="Name", keep="first")
                      .head(5))

# Outlier: highest BB_m_2K of any single season (regardless of color), exclude Onil Perez (small-sample ghost)
top_disc = (aa[aa["Name"] != "Onil Perez"]
            .sort_values("BB_m_2K", ascending=False)
            .drop_duplicates("Name")
            .head(2))

# Young + poor discipline: top Age_R but very low BB_m_2K (bottom 5th percentile of discipline)
disc_floor = aa["BB_m_2K"].quantile(0.05)
young_poor = (aa[aa["BB_m_2K"] <= disc_floor]
              .sort_values("Age_R", ascending=False)
              .drop_duplicates("Name")
              .head(2))

label_rows = pd.concat([top_elite, top_disc, young_poor]).drop_duplicates(subset=["Name", "Season"])

def add_label(row, dot_color):
    x, y = row["BB_m_2K"], row["Age_R"]
    name = row["Name"].split()[-1]   # last name only
    season = int(row["Season"])
    txt = f"{name} '{str(season)[2:]}"
    # offset: push right if left half, push left if right half
    xl2, xr2 = ax.get_xlim()
    xoff = 4 if x < (xl2 + xr2) / 2 else -4
    yoff = 3
    ax.annotate(
        txt,
        xy=(x, y), xytext=(x + xoff, y + yoff),
        fontsize=7.5, color="white", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="#12141c", ec=dot_color, alpha=0.75, lw=0.8),
        arrowprops=dict(arrowstyle="-", color=dot_color, lw=0.8, alpha=0.7),
        zorder=10,
    )

for _, row in top_elite.iterrows():
    add_label(row, "#2ec46a")
for _, row in top_disc.iterrows():
    add_label(row, "#ffffff")
for _, row in young_poor.iterrows():
    add_label(row, "#3a86ff")

# Footnote
ax.text(0.01, 0.005,
        f"n = {len(aa):,} player-seasons  ·  "
        f"Blue {n_blue:,}  ·  Red {n_red:,}  ·  Green {n_green:,}",
        transform=ax.transAxes, color="#555566", fontsize=7.5, va="bottom")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=160, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"\nSaved: {OUT_PATH}")
