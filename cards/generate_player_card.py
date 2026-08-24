"""Generate a radar profile card for a 2026 AAA player (points league).

Usage:
    python generate_player_card.py "A.J. Ewing"
    python generate_player_card.py sean keys
"""

import sys
import re
import unicodedata
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR  = Path.home() / "Documents" / "prospectFiles" / "PlayerCards"

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


def ordinal(n: float) -> str:
    i = int(round(n))
    suffix = "th" if 11 <= (i % 100) <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(i % 10, "th")
    return f"{i}{suffix}%"


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
    tokens = norm.split()
    for token in tokens:
        partial = df[df["_norm"].str.contains(token, regex=False)]
        if len(partial) == 1:
            return partial.iloc[0]
    partial = df[df["_norm"].str.contains(tokens[-1], regex=False)]
    if not partial.empty:
        if len(partial) > 1:
            print(f"Multiple matches: {', '.join(partial['Name'].tolist())} — using first.")
        return partial.iloc[0]
    return None


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


def generate_card(player_name: str, year: int = 2026, level: str = "aaa") -> None:
    fname = f"rk_{year}.csv" if level == "rk" else f"aaa_{year}.csv"
    df = pd.read_csv(DATA_DIR / "rankings" / fname)
    p = find_player(df, player_name)
    if p is None:
        print(f"Player '{player_name}' not found in {fname}.")
        return

    primary, secondary = get_team_colors(p["Team"])
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
    HB = 0.84  # header bottom in card data coords
    n  = 256
    xs = np.linspace(0, 1, n + 1)
    ys = np.array([HB, 1.0])
    X, Y = np.meshgrid(xs, ys)
    Z = np.linspace(0, 1, n).reshape(1, -1)
    cmap = LinearSegmentedColormap.from_list("hdr", [primary, dark_pri])
    card.pcolormesh(X, Y, Z, cmap=cmap, shading="flat", zorder=1)
    card.set_xlim(0, 1)
    card.set_ylim(0, 1)

    # ── Accent stripe ─────────────────────────────────────────────────────────
    STRIPE_H = 0.016
    card.add_patch(mpatches.Rectangle(
        (0, HB - STRIPE_H), 1.0, STRIPE_H,
        facecolor=secondary, edgecolor="none", zorder=3,
    ))

    # ── Name & meta ───────────────────────────────────────────────────────────
    hh = 1.0 - HB  # header height in data units
    card.text(0.06, HB + hh * 0.80, p["Name"],
              fontsize=18, fontweight="bold", color="white",
              va="center", ha="left", zorder=4)

    exact_age = p.get("ExactAge")
    age_str = f"{exact_age:.1f}" if pd.notna(exact_age) else str(int(p["Age"]))
    rank_num = p.get("Rank")
    rank_part = f"  ·  #{int(rank_num)}" if pd.notna(rank_num) else ""
    level_label = "Rk" if level == "rk" else "AAA"
    meta = f"{p['Team']}  ·  Age {age_str}  ·  {int(p['PA'])} PA  ·  {level_label} {year}{rank_part}"
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
        [safe_float(p["PPPA_R"]),     safe_float(p["Discipline_R"]),
         safe_float(p["Power_R"]),    safe_float(p["Contact_R"]),
         safe_float(p["Speed_R"])],
        ["P/PA", "Discipline", "Power", "Contact", "Speed"],
        primary, primary,
    )

    # ── Footer divider ────────────────────────────────────────────────────────
    card.axhline(0.118, xmin=0.03, xmax=0.97, color="#e5e8ec", linewidth=0.9, zorder=2)

    # ── Footer stats ──────────────────────────────────────────────────────────
    if level == "rk":
        raw_stats = [
            ("P/PA",      f"{p['PPPA']:.3f}"),
            ("BB%-2K%",   ordinal(safe_float(p["BB%-2K%_R"]))),
            ("Chase%",    f"{p['Chase%']:.1f}"),
            ("Barrel%",   f"{p['Barrel%']:.1f}"),
            ("SB/PA",     f"{p['SB_rate']:.3f}"),
        ]
    else:
        raw_stats = [
            ("P/PA",      f"{p['PPPA']:.3f}"),
            ("BB%-2K%",   ordinal(safe_float(p["BB%-2K%_R"]))),
            ("Chase%",    f"{p['Chase%']:.1f}"),
            ("Barrel%",   f"{p['Barrel%']:.1f}"),
            ("xSLG",      f"{p['xSLG']:.3f}"),
        ]
    gap = 0.90 / len(raw_stats)
    for j, (label, val) in enumerate(raw_stats):
        x = 0.05 + j * gap
        # Colored accent bar above label
        card.add_patch(mpatches.Rectangle(
            (x, 0.086), 0.034, 0.005,
            facecolor=secondary, edgecolor="none", zorder=2, alpha=0.85,
        ))
        card.text(x, 0.076, label, fontsize=7, va="top", ha="left",
                  color="#9aa5b0", zorder=3)
        card.text(x, 0.024, val, fontsize=10, va="bottom", ha="left",
                  color="#1a2744", fontweight="bold", zorder=3)

    # ── Watermark ─────────────────────────────────────────────────────────────
    csv_path = DATA_DIR / "rankings" / fname
    updated_str = datetime.fromtimestamp(os.path.getmtime(csv_path)).strftime("%#m/%#d/%y")
    card.text(0.97, 0.016, "POINTS LEAGUE",
              fontsize=5.5, color="#cccccc", ha="right", va="bottom",
              fontstyle="italic", zorder=2)
    card.text(0.97, 0.004, f"Last Updated: {updated_str}",
              fontsize=5.0, color="#cccccc", ha="right", va="bottom",
              fontstyle="italic", zorder=2)

    # ── Save ──────────────────────────────────────────────────────────────────
    team_dir = OUT_DIR / str(year) / str(p["Team"])
    team_dir.mkdir(parents=True, exist_ok=True)
    slug = normalize_name(p["Name"]).replace(" ", "_")
    out_path = team_dir / f"{slug}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#d8dce4")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    args = sys.argv[1:]
    level = "aaa"
    if "--rk" in args:
        level = "rk"
        args.remove("--rk")
    if args and args[0].isdigit() and len(args[0]) == 4:
        year = int(args.pop(0))
    else:
        year = 2026
    query = " ".join(args) if args else input("Player name: ")
    generate_card(query, year, level)
