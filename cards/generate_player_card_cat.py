"""Generate a category-league radar profile card.

Differences from generate_player_card.py (points league):
  - wRC+_R replaces PPPA_R on the radar and Overall formula
  - Discipline uses BB%-K% (not BB%-2K%) — no double K% penalty
  - Footer shows wRC+, BB%-K%, K%, Chase%, xSLG

Supports AAA and Rk levels (detects automatically).

Usage:
    python generate_player_card_cat.py "A.J. Ewing"
    python generate_player_card_cat.py "jorwin pulido"
"""

import sys
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PS_DIR   = Path(__file__).resolve().parent.parent / "data" / "prospectSavant"
OUT_DIR  = Path.home() / "Documents" / "PlayerCards"

# ── Team colors: (primary, secondary) ─────────────────────────────────────────
# Primary keys match abbreviations used in aaa/rk CSV files (FanGraphs API style)
TEAM_COLORS = {
    "ARI": ("#A71930", "#E3D4AD"),   # Diamondbacks: red/sand
    "ATH": ("#003831", "#EFB21E"),   # Athletics: green/gold
    "ATL": ("#CE1141", "#13274F"),   # Braves: red/navy
    "BAL": ("#DF4601", "#27251F"),   # Orioles: orange/black
    "BOS": ("#BD3039", "#0C2340"),   # Red Sox: red/navy
    "CHC": ("#0E3386", "#CC3433"),   # Cubs: blue/red
    "CHW": ("#27251F", "#C4CED4"),   # White Sox: black/silver
    "CIN": ("#C6011F", "#000000"),   # Reds: red/black
    "CLE": ("#00385D", "#E31937"),   # Guardians: navy/red
    "COL": ("#33006F", "#C4CED4"),   # Rockies: purple/silver
    "DET": ("#0C2340", "#FA4616"),   # Tigers: navy/orange
    "HOU": ("#002D62", "#EB6E1F"),   # Astros: navy/orange
    "KCR": ("#004687", "#C09A5B"),   # Royals: blue/gold
    "LAA": ("#BA0021", "#003263"),   # Angels: red/navy
    "LAD": ("#005A9C", "#EF3E42"),   # Dodgers: blue/red
    "MIA": ("#00A3E0", "#EF3340"),   # Marlins: blue/red
    "MIL": ("#12284B", "#FFC52F"),   # Brewers: navy/gold
    "MIN": ("#002B5C", "#D31145"),   # Twins: navy/red
    "NYM": ("#002D72", "#FF5910"),   # Mets: blue/orange
    "NYY": ("#003087", "#C4A44C"),   # Yankees: navy/gold
    "PHI": ("#E81828", "#002D72"),   # Phillies: red/navy
    "PIT": ("#27251F", "#FDB827"),   # Pirates: black/gold
    "SDP": ("#2F241D", "#FFC425"),   # Padres: brown/gold
    "SEA": ("#0C2C56", "#005C5C"),   # Mariners: navy/teal
    "SFG": ("#27251F", "#FD5A1E"),   # Giants: black/orange
    "STL": ("#C41E3A", "#FEDB00"),   # Cardinals: red/gold
    "TBR": ("#092C5C", "#8FBCE6"),   # Rays: navy/light blue
    "TEX": ("#003278", "#C0111F"),   # Rangers: blue/red
    "TOR": ("#134A8E", "#E8291C"),   # Blue Jays: blue/red
    "WSN": ("#AB0003", "#14225A"),   # Nationals: red/navy
    # Common aliases
    "CWS": ("#27251F", "#C4CED4"),
    "KC":  ("#004687", "#C09A5B"),
    "OAK": ("#003831", "#EFB21E"),
    "SD":  ("#2F241D", "#FFC425"),
    "SF":  ("#27251F", "#FD5A1E"),
    "TB":  ("#092C5C", "#8FBCE6"),
    "WSH": ("#AB0003", "#14225A"),
    "WAS": ("#AB0003", "#14225A"),
}
DEFAULT_COLORS = ("#1a2744", "#4a6fa5")

GRID_COLOR = "#d5d8dc"

MIN_AB = 40

OVERALL_WEIGHTS_AAA = {
    "wRC+_R":     0.20,
    "Discipline": 0.30,
    "Power":      0.20,
    "Speed":      0.10,
    "Contact":    0.10,
    "Age_R":      0.10,
}
OVERALL_WEIGHTS_RK = {
    "wRC+_R":     0.40,
    "Discipline": 0.25,
    "Power":      0.15,
    "Speed":      0.10,
    "Age_R":      0.10,
}

POWER_WEIGHTS_AAA = {
    "Barrel%_R":  0.35,
    "xSLG_R":     0.20,
    "MaxEV_R":    0.17,
    "EV90_R":     0.18,
    "PullAir%_R": 0.10,
}
POWER_WEIGHTS_RK = {
    "MaxEV_R":    0.22,
    "EV90_R":     0.22,
    "Barrel%_R":  0.41,
    "PullAir%_R": 0.15,
}
DISCIPLINE_WEIGHTS = {
    "BB%-K%_R":  0.50,
    "Chase%_R":  0.30,
    "Whiff%_R":  0.20,
}
CONTACT_WEIGHTS_AAA = {"xBA_R": 0.50, "ZContact%_R": 0.50}
CONTACT_WEIGHTS_RK  = {"ZContact%_R": 0.50, "HardHit%_R": 0.50}
SPEED_WEIGHTS = {"Spd_R": 0.40, "SB_rate_R": 0.35, "SB_succ_R": 0.25}
LOWER_IS_BETTER = {"Chase%", "Whiff%", "K%"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def hex_to_rgb(h: str):
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def darken(hex_color: str, f: float = 0.55) -> str:
    r, g, b = hex_to_rgb(hex_color)
    return f"#{int(r * f):02x}{int(g * f):02x}{int(b * f):02x}"


def get_team_colors(team: str):
    return TEAM_COLORS.get(str(team).upper(), DEFAULT_COLORS)


def normalize_name(name: str) -> str:
    name = str(name).lower().strip().replace(".", "")
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"\s+jr$", "", name)
    return re.sub(r"\s+", " ", name)


def pct_color(pct: float) -> str:
    if pct >= 70:   return "#27ae60"
    elif pct >= 50: return "#f39c12"
    elif pct >= 35: return "#e67e22"
    else:           return "#e74c3c"


def safe_float(val, default: float = 50.0) -> float:
    try:
        v = float(val)
        return v if not np.isnan(v) else default
    except (TypeError, ValueError):
        return default


def find_player(df: pd.DataFrame, query: str):
    df = df.copy()
    df["_norm"] = df["Name"].apply(normalize_name)
    norm = normalize_name(query)
    exact = df[df["_norm"] == norm]
    if not exact.empty:
        return exact.iloc[0]
    partial = df[df["_norm"].str.contains(norm.split()[-1], regex=False)]
    if not partial.empty:
        if len(partial) > 1:
            print(f"Multiple matches: {', '.join(partial['Name'].tolist())} — using first.")
        return partial.iloc[0]
    return None


def pct_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    return (series.rank(pct=True, ascending=ascending, na_option="keep") * 100).round(1)


# ── Pool builder (unchanged logic) ────────────────────────────────────────────

def build_pool(ps_raw: pd.DataFrame, ml: pd.DataFrame, hist: pd.DataFrame,
               level: str, year: int, is_aaa: bool) -> pd.DataFrame:
    ps = ps_raw[ps_raw["AB"] >= MIN_AB].copy()
    ps["_norm"] = ps["Name"].apply(normalize_name)
    ml["_norm"]  = ml["Name"].apply(normalize_name)

    lvl_key = "AAA" if is_aaa else "R"
    sub = (
        ml[(ml["Season"] == year) & (ml["Level"] == lvl_key)]
        .sort_values("PA", ascending=False)
        .drop_duplicates(subset="_norm", keep="first")
    )

    pppa_lkp = sub[["_norm", "SB", "CS", "PA"]].copy()
    pppa_lkp["SB_rate"] = (pppa_lkp["SB"] / pppa_lkp["PA"]).round(3)
    pppa_lkp["SB_succ"] = (pppa_lkp["SB"] / (pppa_lkp["SB"] + pppa_lkp["CS"])).fillna(0).round(3)
    ps = ps.merge(pppa_lkp[["_norm", "SB_rate", "SB_succ"]], on="_norm", how="left")

    # wRC+ — primary: minorLeaguewrcp.csv; fallback: ovr_hist_data; last resort: xwOBA_R
    wrcp_path = DATA_DIR / "fangraphs" / "minorLeaguewrcp.csv"
    if wrcp_path.exists():
        wrcp = pd.read_csv(wrcp_path)
        wrcp["_norm"] = wrcp["NameASCII"].apply(normalize_name)
        wrcp_lkp = (
            wrcp[(wrcp["Season"] == year) & (wrcp["Level"] == lvl_key)]
            .sort_values("PA", ascending=False)
            .drop_duplicates(subset="_norm", keep="first")
        )[["_norm", "wRC+"]]
        ps = ps.merge(wrcp_lkp, on="_norm", how="left", suffixes=("_ps", ""))
        if ps["wRC+"].isna().any():
            hist["_norm"] = hist["Name"].apply(normalize_name)
            hist_lkp = (
                hist[(hist["Season"] == year) & (hist["Level"] == lvl_key)]
                .sort_values("PA", ascending=False)
                .drop_duplicates(subset="_norm", keep="first")
            )[["_norm", "wRC+"]].rename(columns={"wRC+": "_hist_wrc"})
            ps = ps.merge(hist_lkp, on="_norm", how="left")
            ps["wRC+"] = ps["wRC+"].fillna(ps["_hist_wrc"])
            ps = ps.drop(columns="_hist_wrc")
    else:
        hist["_norm"] = hist["Name"].apply(normalize_name)
        wrc_lkp = (
            hist[(hist["Season"] == year) & (hist["Level"] == lvl_key)]
            .sort_values("PA", ascending=False)
            .drop_duplicates(subset="_norm", keep="first")
        )[["_norm", "wRC+"]]
        ps = ps.merge(wrc_lkp, on="_norm", how="left", suffixes=("_ps", ""))

    ps["BB%-K%"] = (ps["BB%"] - ps["K%"]).round(1)

    # Percentile ranks
    stat_cols = {
        "wRC+": True, "BB%-K%": True, "Chase%": False, "ZContact%": True,
        "Whiff%": False, "MaxEV": True, "EV90": True, "HardHit%": True,
        "Barrel%PA": True, "PullAir%": True, "xBA": True, "xSLG": True,
        "Spd": True, "SB_rate": True, "SB_succ": True, "K%": False,
    }
    for col, asc in stat_cols.items():
        if col in ps.columns:
            ps[col + "_R"] = pct_rank(ps[col], ascending=asc)

    # xwOBA fallback for wRC+_R
    if "xwOBA" in ps.columns:
        xwoba_valid = ps["xwOBA"].notna() & (ps["xwOBA"] > 0)
        if xwoba_valid.any():
            ps["xwOBA_R"] = pct_rank(ps["xwOBA"].where(xwoba_valid))
            missing_wrc = ps["wRC+_R"].isna() & ps["xwOBA_R"].notna()
            ps.loc[missing_wrc, "wRC+_R"] = ps.loc[missing_wrc, "xwOBA_R"]

    ps["Age_R"] = pct_rank(ps.get("Age", pd.Series(dtype=float)), ascending=False)

    if "Barrel%PA_R" in ps.columns:
        ps["Barrel%_R"] = ps["Barrel%PA_R"]
    if "Barrel%PA" in ps.columns:
        ps["Barrel%"] = ps["Barrel%PA"]

    power_w   = POWER_WEIGHTS_AAA if is_aaa else POWER_WEIGHTS_RK
    contact_w = CONTACT_WEIGHTS_AAA if is_aaa else CONTACT_WEIGHTS_RK
    overall_w = OVERALL_WEIGHTS_AAA if is_aaa else OVERALL_WEIGHTS_RK

    ps["Power"]      = sum(ps[c] * w for c, w in power_w.items()   if c in ps.columns).round(1)
    ps["Contact"]    = sum(ps[c] * w for c, w in contact_w.items() if c in ps.columns).round(1)
    ps["Speed"]      = sum(ps[c] * w for c, w in SPEED_WEIGHTS.items() if c in ps.columns).round(1)
    ps["Discipline"] = sum(ps[c] * w for c, w in DISCIPLINE_WEIGHTS.items() if c in ps.columns).round(1)

    non_age_w   = {k: v for k, v in overall_w.items() if k != "Age_R"}
    non_age_sum = sum(ps[c] * w for c, w in non_age_w.items() if c in ps.columns)
    non_age_tot = sum(non_age_w.values())
    has_age     = ps["Age_R"].notna()
    ps["Overall"] = (
        (non_age_sum + ps["Age_R"].fillna(0) * overall_w["Age_R"])
        .where(has_age, non_age_sum / non_age_tot)
    ).round(1)

    for comp in ["Power", "Contact", "Speed", "Discipline", "Overall"]:
        ps[comp + "_R"] = pct_rank(ps[comp]).round(1)

    ps["wRC+_R"] = pct_rank(ps["wRC+"])

    return ps


# ── Radar ─────────────────────────────────────────────────────────────────────

def draw_radar(ax, values, categories, fill_color, border_color):
    N = len(categories)
    angles = [np.pi / 2 - i * 2 * np.pi / N for i in range(N)]
    angles_closed = angles + [angles[0]]
    vals_norm  = [v / 100.0 for v in values]
    vals_closed = vals_norm + [vals_norm[0]]

    # Faint polygon background
    bg_x = [np.cos(a) for a in angles] + [np.cos(angles[0])]
    bg_y = [np.sin(a) for a in angles] + [np.sin(angles[0])]
    ax.fill(bg_x, bg_y, color="#f4f6f8", alpha=0.55, zorder=0)

    # Grid rings
    theta = np.linspace(0, 2 * np.pi, 300)
    for r, lw, ls in [(0.25, 0.5, "--"), (0.50, 0.7, "--"), (0.75, 0.7, "--"), (1.0, 1.3, "-")]:
        ax.plot(r * np.cos(theta), r * np.sin(theta),
                color=GRID_COLOR, linewidth=lw, linestyle=ls, zorder=1)
    for a in angles:
        ax.plot([0, np.cos(a)], [0, np.sin(a)], color=GRID_COLOR, linewidth=0.7, zorder=1)

    # Data polygon
    x_vals = [v * np.cos(a) for v, a in zip(vals_closed, angles_closed)]
    y_vals = [v * np.sin(a) for v, a in zip(vals_closed, angles_closed)]
    ax.fill(x_vals, y_vals, color=fill_color, alpha=0.22, zorder=2)
    ax.plot(x_vals, y_vals, color=border_color, linewidth=2.5, zorder=3)
    ax.scatter(
        [v * np.cos(a) for v, a in zip(vals_norm, angles)],
        [v * np.sin(a) for v, a in zip(vals_norm, angles)],
        color=border_color, s=55, zorder=4, edgecolors="white", linewidths=1.0,
    )

    # Axis labels + percentile values
    label_r = 1.22
    for i, (angle, cat, val) in enumerate(zip(angles, categories, values)):
        lx = label_r * np.cos(angle)
        ly = label_r * np.sin(angle)
        ha = "center"
        if lx > 0.15:   ha = "left"
        elif lx < -0.15: ha = "right"
        va = "center"
        if ly > 0.15:   va = "bottom"
        elif ly < -0.15: va = "top"
        ax.text(lx, ly, cat, fontsize=9.5, ha=ha, va=va,
                color="#2c2c2c", fontweight="bold", zorder=5)
        tip = vals_norm[i]
        ax.text((tip + 0.11) * np.cos(angle), (tip + 0.11) * np.sin(angle), f"{val:.0f}",
                fontsize=7.5, ha="center", va="center",
                color=pct_color(val), fontweight="bold", zorder=6)

    # Ring tick labels
    tick_angle = angles[0] - 0.18
    for r, label in [(0.25, "25"), (0.50, "50"), (0.75, "75")]:
        ax.text(r * np.cos(tick_angle), r * np.sin(tick_angle), label,
                fontsize=6, color="#cccccc", ha="center", va="center", zorder=5)

    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")


# ── Card generator ────────────────────────────────────────────────────────────

def generate_card(player_name: str, year: int = 2026) -> None:
    ml   = pd.read_csv(DATA_DIR / "computed" / "minorLeagueData.csv")
    hist = pd.read_csv(DATA_DIR / "historical" / "ovr_hist_data.csv",
                       usecols=["Season", "Name", "Level", "PA", "wRC+"])

    aaa_rank = pd.read_csv(DATA_DIR / "rankings" / f"aaa_{year}.csv")
    rk_rank  = pd.read_csv(DATA_DIR / "rankings" / f"rk_{year}.csv")

    p_aaa = find_player(aaa_rank, player_name)
    p_rk  = find_player(rk_rank,  player_name)

    if p_aaa is not None:
        is_aaa    = True
        ps_raw    = pd.read_csv(PS_DIR / f"ps_AAA_{year}.csv")
        level_lbl = f"AAA {year}"
    elif p_rk is not None:
        is_aaa    = False
        ps_raw    = pd.read_csv(PS_DIR / f"ps_Rk_{year}_savant.csv")
        level_lbl = f"Rk {year}"
    else:
        print(f"Player '{player_name}' not found in aaa_{year} or rk_{year}.")
        return

    pool = build_pool(ps_raw, ml.copy(), hist, "AAA" if is_aaa else "R", year, is_aaa)
    p = find_player(pool, player_name)
    if p is None:
        print(f"Player '{player_name}' not found after pool build.")
        return

    team = str(p.get("Org", p.get("Team", "UNK")))
    primary, secondary = get_team_colors(team)
    dark_pri = darken(primary, 0.60)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(7, 7.5), facecolor="#d8dce4")

    # Drop shadow
    shadow = fig.add_axes([0.053, 0.033, 0.92, 0.92])
    shadow.set_facecolor("#b0b4bc")
    shadow.axis("off")

    # White card body
    card = fig.add_axes([0.04, 0.04, 0.92, 0.92])
    card.set_facecolor("white")
    card.set_xlim(0, 1)
    card.set_ylim(0, 1)
    card.axis("off")

    # ── Header gradient (left → right: primary → darker) ─────────────────────
    HB = 0.84
    n  = 256
    xs = np.linspace(0, 1, n + 1)
    ys = np.array([HB, 1.0])
    X, Y = np.meshgrid(xs, ys)
    Z = np.linspace(0, 1, n).reshape(1, -1)
    cmap_hdr = LinearSegmentedColormap.from_list("hdr", [primary, dark_pri])
    card.pcolormesh(X, Y, Z, cmap=cmap_hdr, shading="flat", zorder=1)
    card.set_xlim(0, 1)
    card.set_ylim(0, 1)

    # ── Accent stripe ─────────────────────────────────────────────────────────
    STRIPE_H = 0.016
    card.add_patch(mpatches.Rectangle(
        (0, HB - STRIPE_H), 1.0, STRIPE_H,
        facecolor=secondary, edgecolor="none", zorder=3,
    ))

    # ── Name & meta ───────────────────────────────────────────────────────────
    hh = 1.0 - HB
    card.text(0.06, HB + hh * 0.80, p["Name"],
              fontsize=18, fontweight="bold", color="white",
              va="center", ha="left", zorder=4)

    age_val = p.get("Age", None)
    age_str = str(int(age_val)) if pd.notna(age_val) else "?"
    pa_val  = p.get("PA", "?")
    meta = f"{team}  ·  Age {age_str}  ·  {int(pa_val)} PA  ·  {level_lbl}  ·  Category"
    card.text(0.06, HB + hh * 0.28, meta,
              fontsize=8.5, color="#b8cfe0", va="center", ha="left", zorder=4)

    # ── Overall score badge ───────────────────────────────────────────────────
    bx, by = 0.865, HB + hh * 0.52
    bw, bh = 0.148, 0.112
    card.add_patch(mpatches.FancyBboxPatch(
        (bx - bw / 2, by - bh / 2), bw, bh,
        boxstyle="round,pad=0.006",
        facecolor="white", alpha=0.13,
        edgecolor="white", linewidth=0.8, zorder=3,
    ))
    card.text(bx, by + bh * 0.25, "OVERALL",
              fontsize=6.5, color="white", fontweight="bold",
              ha="center", va="center", zorder=4)
    card.text(bx, by - bh * 0.12, f"{p['Overall']:.1f}",
              fontsize=22, fontweight="bold",
              color=pct_color(safe_float(p["Overall"])),
              ha="center", va="center", zorder=4)

    # ── Radar ─────────────────────────────────────────────────────────────────
    radar_ax = fig.add_axes([0.08, 0.13, 0.84, 0.66])
    radar_ax.set_facecolor("none")
    draw_radar(
        radar_ax,
        [safe_float(p["wRC+_R"]),      safe_float(p["Discipline_R"]),
         safe_float(p["Power_R"]),     safe_float(p["Contact_R"]),
         safe_float(p["Speed_R"])],
        ["wRC+", "Discipline", "Power", "Contact", "Speed"],
        primary, primary,
    )

    # ── Footer divider ────────────────────────────────────────────────────────
    card.axhline(0.118, xmin=0.03, xmax=0.97, color="#e5e8ec", linewidth=0.9, zorder=2)

    # ── Footer stats ──────────────────────────────────────────────────────────
    wrc_val   = p.get("wRC+", float("nan"))
    xwoba_val = p.get("xwOBA", None)
    if pd.notna(wrc_val):
        wrc_lbl, wrc_disp = "wRC+", f"{wrc_val:.0f}"
    elif pd.notna(xwoba_val) and xwoba_val > 0:
        wrc_lbl, wrc_disp = "xwOBA", f"{xwoba_val:.3f}"
    else:
        wrc_lbl, wrc_disp = "wRC+", "—"

    xslg_disp = f"{p['xSLG']:.3f}" if "xSLG" in p.index and pd.notna(p.get("xSLG")) else "—"

    raw_stats = [
        (wrc_lbl,  wrc_disp),
        ("BB%-K%",  f"{p['BB%-K%']:.1f}"),
        ("K%",      f"{p['K%']:.1f}"),
        ("Chase%",  f"{p['Chase%']:.1f}"),
        ("xSLG",    xslg_disp),
    ]
    gap = 0.90 / len(raw_stats)
    for j, (label, val) in enumerate(raw_stats):
        x = 0.05 + j * gap
        card.add_patch(mpatches.Rectangle(
            (x, 0.086), 0.034, 0.005,
            facecolor=secondary, edgecolor="none", zorder=2, alpha=0.85,
        ))
        card.text(x, 0.076, label, fontsize=7, va="top", ha="left",
                  color="#9aa5b0", zorder=3)
        card.text(x, 0.024, val, fontsize=10, va="bottom", ha="left",
                  color="#1a2744", fontweight="bold", zorder=3)

    # ── Watermark ─────────────────────────────────────────────────────────────
    card.text(0.97, 0.008, "CATEGORY LEAGUE",
              fontsize=5.5, color="#cccccc", ha="right", va="bottom",
              fontstyle="italic", zorder=2)

    # ── Save ──────────────────────────────────────────────────────────────────
    team_dir = OUT_DIR / str(year) / team
    team_dir.mkdir(parents=True, exist_ok=True)
    slug = normalize_name(p["Name"]).replace(" ", "_")
    out_path = team_dir / f"{slug}_cat.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#d8dce4")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0].isdigit() and len(args[0]) == 4:
        year = int(args.pop(0))
    else:
        year = 2026
    query = " ".join(args) if args else input("Player name: ")
    generate_card(query, year)
