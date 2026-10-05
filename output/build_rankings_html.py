"""Generate interactive prospect rankings HTML artifact."""
import json
import math
import os
import pandas as pd
from pathlib import Path

DATA_DIR   = Path(__file__).resolve().parent.parent / "data"

# RANKINGS_OUT_DIR env var overrides output location (used by CI/GitHub Actions).
# When set, output filename is index.html; otherwise uses local scratchpad path.
_out_env = os.environ.get("RANKINGS_OUT_DIR")
if _out_env:
    SCRATCHPAD = Path(_out_env)
    OUT_PATH   = SCRATCHPAD / "index.html"
else:
    SCRATCHPAD = Path(r"C:\Users\cjacc\AppData\Local\Temp\claude\C--Users-cjacc-PycharmProjects-prospectsMain\5dc88b54-dc8f-45dc-934f-30c09d0bfd05\scratchpad")
    OUT_PATH   = SCRATCHPAD / "prospect_rankings.html"

# ── Prospects data ──────────────────────────────────────────────────────────
ps = pd.read_csv(DATA_DIR / "rankings" / "prospect_scores.csv")
ps["Discipline_Flag"] = ps["Discipline_Flag"].fillna("")
ps["Career_Disc_Flag"] = ps["Career_Disc_Flag"].fillna("")
ps["Hard_Floor_Flag"] = ps["Hard_Floor_Flag"].fillna("")
ps["Below_OVR_Floor"] = ps["Below_OVR_Floor"].fillna(False)
ps["AgeAA_Combo_Flag_Label"] = ps["AgeAA_Combo_Flag_Label"].fillna("")
ps["Pos"] = ps["Pos"].fillna("")
ps["Age"] = ps["Age"].fillna("").astype(str).str.replace(".0", "", regex=False)
for c in ["Total_Weighted_PA", "TOOLS_Score", "TOOLS_Disc", "TOOLS_Power", "TOOLS_Ath",
          "ABILITY_Score", "Fantasy_Out", "ABILITY_Disc", "SB_Talent", "Game_Power",
          "Current_Score", "OVR_Score", "Combined_Score"]:
    if c in ps.columns:
        ps[c] = ps[c].round(1)

ps["Pos_Adj_Score"] = ps["Pos_Adj_Score"].round(1)
_sub_cols = [c for c in ["TOOLS_Disc", "TOOLS_Power", "TOOLS_Ath",
                          "Fantasy_Out", "ABILITY_Disc", "SB_Talent", "Game_Power"]
             if c in ps.columns]
cols = ["Combined_Rank", "Pos_Adj_Rank", "Name", "Pos", "Team", "Level", "Age", "Last_Season",
        "Career_PA", "TOOLS_Score"] + _sub_cols[:3] + ["ABILITY_Score"] + _sub_cols[3:] + [
        "Current_Score", "OVR_Score", "Combined_Score", "Pos_Bonus", "Pos_Adj_Score",
        "Discipline_Flag", "Career_Disc_Flag", "Hard_Floor_Flag", "Below_OVR_Floor",
        "AgeAA_Combo_Flag_Label"]
raw = json.dumps(ps[cols].to_dict(orient="records"), separators=(",", ":"))
covered = (ps["Pos"] != "").sum()
print(f"Position coverage: {covered:,} / {len(ps):,} ({covered/len(ps)*100:.1f}%)")

# ── AAA 2026 data ───────────────────────────────────────────────────────────
aaa = pd.read_csv(DATA_DIR / "rankings" / "aaa_2026.csv")
aaa_num_cols = ["PPPA", "Overall", "Overall_R", "Discipline", "Discipline_R",
                "Power", "Power_R", "Contact", "Contact_R", "Speed", "Speed_R",
                "Chase%", "ZContact%", "Whiff%", "BB%-2K%",
                "MaxEV", "Barrel%", "xSLG", "SB_rate", "SB_succ",
                "PPPA_R", "Age_R"]
for c in aaa_num_cols:
    if c in aaa.columns:
        aaa[c] = pd.to_numeric(aaa[c], errors="coerce").round(2)

aaa_cols = ["Rank", "Name", "Team", "Pos", "Age", "PA",
            "PPPA", "PPPA_R",
            "Overall", "Overall_R",
            "Discipline", "Discipline_R",
            "Power", "Power_R",
            "Contact", "Contact_R",
            "Speed", "Speed_R",
            "Chase%", "ZContact%", "Whiff%", "BB%-2K%",
            "MaxEV", "Barrel%", "xSLG", "SB_rate", "SB_succ",
            "Age_R"]

def _to_json_val(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v

aaa_records = [
    {k: _to_json_val(row[k]) for k in aaa_cols if k in aaa.columns}
    for _, row in aaa.iterrows()
]
raw_aaa = json.dumps(aaa_records, separators=(",", ":"))
print(f"AAA 2026: {len(aaa_records)} players")

# ── Pitcher data ─────────────────────────────────────────────────────────────
_pitcher_path = DATA_DIR / "rankings" / "pitcher_scores.csv"
if _pitcher_path.exists():
    pit = pd.read_csv(_pitcher_path, dtype={"PlayerId": str})
    for c in ["STUFF_Score", "PERFORMANCE_Score",
              "Current_Score", "OVR_Score", "Combined_Score"]:
        if c in pit.columns:
            pit[c] = pit[c].round(1)
    pit["Age"] = pit["Age"].fillna("").astype(str).str.replace(".0", "", regex=False)
    pit["Career_IP"] = pit["Career_IP"].round(1)
    pit_cols = ["Combined_Rank", "SP_Rank", "RP_Rank", "Name", "Team", "Level", "Age", "Role",
                "Career_IP", "Career_G", "Career_GS",
                "STUFF_Score", "PERFORMANCE_Score",
                "Current_Score", "OVR_Score", "Combined_Score"]
    pit_cols = [c for c in pit_cols if c in pit.columns]
    sp_df = pit[pit["Role"] == "SP"].copy()
    rp_df = pit[pit["Role"] == "RP"].copy()
    raw_sp = json.dumps(
        [{k: _to_json_val(row[k]) for k in pit_cols if k in sp_df.columns}
         for _, row in sp_df.iterrows()], separators=(",", ":"))
    raw_rp = json.dumps(
        [{k: _to_json_val(row[k]) for k in pit_cols if k in rp_df.columns}
         for _, row in rp_df.iterrows()], separators=(",", ":"))
    print(f"SP prospects: {len(sp_df)} | RP prospects: {len(rp_df)}")
else:
    raw_sp = "[]"
    raw_rp = "[]"
    print("pitcher_scores.csv not found — SP/RP tabs will be empty")

# ── Luck tracker data ────────────────────────────────────────────────────────
luck_df = pd.read_csv(DATA_DIR / "computed" / "babip_luck.csv")
luck_df["PlayerId"] = luck_df["PlayerId"].astype(str)
ps_luck = ps.copy()
ps_luck["PlayerId"] = ps_luck["PlayerId"].astype(str)

# Most recent qualifying season (2025/2026) per player
luck_recent = (
    luck_df[luck_df["Season"].isin([2025, 2026])]
    .sort_values(["PlayerId", "Season"], ascending=[True, False])
    .drop_duplicates(subset=["PlayerId"], keep="first")
)
# Prior season PPPA_Z_SL for jump calculation
luck_prior = (
    luck_df[luck_df["Season"].isin([2024, 2025])]
    .sort_values(["PlayerId", "Season"], ascending=[True, False])
    .drop_duplicates(subset=["PlayerId"], keep="first")[["PlayerId", "Season", "PPPA_Z_SL"]]
    .rename(columns={"PPPA_Z_SL": "PPPA_prior", "Season": "Prior_Season"})
)
luck_recent = luck_recent.merge(luck_prior, on="PlayerId", how="left")
luck_recent["PPPA_Jump"] = (luck_recent["PPPA_Z_SL"] - luck_recent["PPPA_prior"]).where(
    luck_recent["Season"] != luck_recent["Prior_Season"]
)

# Merge with prospect pool for Combined_Rank / FantasyPos
luck_merged = luck_recent.merge(
    ps_luck[["PlayerId", "Combined_Rank", "FantasyPos"]], on="PlayerId", how="inner"
)

luck_out_cols = [
    "Combined_Rank", "Name", "FantasyPos", "Level", "Season", "PA",
    "Prior_Career_PA",
    "BABIP", "BABIP_career", "BABIP_delta_z", "BABIP_Delta_Slope",
    "HR/FB", "HRFB_career", "HRFB_delta_z",
    "Luck_PPPA", "Luck_PPPA_pct",
    "PPPA_Z_SL", "PPPA_Jump",
]
luck_merged = luck_merged.sort_values("Luck_PPPA", ascending=False, na_position="last")
for c in ["BABIP", "BABIP_career", "BABIP_delta_z", "BABIP_Delta_Slope", "HR/FB", "HRFB_career",
          "HRFB_delta_z", "Luck_PPPA", "Luck_PPPA_pct", "PPPA_Z_SL", "PPPA_Jump"]:
    if c in luck_merged.columns:
        luck_merged[c] = pd.to_numeric(luck_merged[c], errors="coerce").round(2)

luck_records = [
    {k: _to_json_val(row[k]) for k in luck_out_cols if k in luck_merged.columns}
    for _, row in luck_merged.iterrows()
]
raw_luck = json.dumps(luck_records, separators=(",", ":"))
print(f"Luck tracker: {len(luck_records)} prospects")

# ── Comps data ────────────────────────────────────────────────────────────────
_comps_path = DATA_DIR / "rankings" / "prospect_comps.json"
if _comps_path.exists():
    with open(_comps_path, "r", encoding="utf-8") as _f:
        raw_comps = _f.read()
    _n_comps = raw_comps.count('"rank"')
    print(f"Comps: {_n_comps} prospects loaded")
else:
    raw_comps = "{}"
    print("prospect_comps.json not found — Comps tab will be empty")

HTML = """\
<title>2026 Prospect Rankings</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {
  --bg:#ffffff;--surface:#f6f8fa;--surface2:#eaeef2;--border:#d0d7de;
  --text:#1f2328;--muted:#656d76;--accent:#0969da;--row-hover:#f0f6ff;
  --row-alt:#f6f8fa;--input-bg:#ffffff;
  --flag-soft-bg:#fff3cd;--flag-soft-text:#7d4e00;
  --flag-hard-bg:#ffeaea;--flag-hard-text:#a10000;
  --flag-whiff-bg:#fff0e0;--flag-whiff-text:#7a3900;
  --flag-good-bg:#e6f4ea;--flag-good-text:#1a7a3d;
  --tab-active-bg:#ffffff;--tab-active-border:var(--accent);
}
@media(prefers-color-scheme:dark){:root{
  --bg:#0d1117;--surface:#161b22;--surface2:#21262d;--border:#30363d;
  --text:#e6edf3;--muted:#8b949e;--accent:#58a6ff;--row-hover:#1c2433;
  --row-alt:#161b22;--input-bg:#0d1117;
  --flag-soft-bg:#2d2000;--flag-soft-text:#e3a008;
  --flag-hard-bg:#2d0000;--flag-hard-text:#ff6b6b;
  --flag-whiff-bg:#2d1500;--flag-whiff-text:#f97316;
  --flag-good-bg:#0d2e1a;--flag-good-text:#4ade80;
  --tab-active-bg:#0d1117;
}}
:root[data-theme="light"]{
  --bg:#ffffff;--surface:#f6f8fa;--surface2:#eaeef2;--border:#d0d7de;
  --text:#1f2328;--muted:#656d76;--accent:#0969da;--row-hover:#f0f6ff;
  --row-alt:#f6f8fa;--input-bg:#ffffff;
  --flag-soft-bg:#fff3cd;--flag-soft-text:#7d4e00;
  --flag-hard-bg:#ffeaea;--flag-hard-text:#a10000;
  --flag-whiff-bg:#fff0e0;--flag-whiff-text:#7a3900;
  --flag-good-bg:#e6f4ea;--flag-good-text:#1a7a3d;
  --tab-active-bg:#ffffff;
}
:root[data-theme="dark"]{
  --bg:#0d1117;--surface:#161b22;--surface2:#21262d;--border:#30363d;
  --text:#e6edf3;--muted:#8b949e;--accent:#58a6ff;--row-hover:#1c2433;
  --row-alt:#161b22;--input-bg:#0d1117;
  --flag-soft-bg:#2d2000;--flag-soft-text:#e3a008;
  --flag-hard-bg:#2d0000;--flag-hard-text:#ff6b6b;
  --flag-whiff-bg:#2d1500;--flag-whiff-text:#f97316;
  --flag-good-bg:#0d2e1a;--flag-good-text:#4ade80;
  --tab-active-bg:#0d1117;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  background:var(--bg);color:var(--text);height:100vh;
  display:flex;flex-direction:column;overflow:hidden;font-size:13px}

/* ── Tab bar ── */
.tab-bar{background:var(--surface);border-bottom:1px solid var(--border);
  display:flex;align-items:flex-end;padding:0 14px;gap:2px;flex-shrink:0}
.tab-btn{background:none;border:1px solid transparent;border-bottom:none;
  border-radius:6px 6px 0 0;color:var(--muted);cursor:pointer;
  font-size:12px;font-weight:600;letter-spacing:.02em;
  padding:7px 16px;transition:color .15s,background .15s;white-space:nowrap;
  margin-bottom:-1px}
.tab-btn:hover{color:var(--text)}
.tab-btn.active{background:var(--tab-active-bg);border-color:var(--border);
  color:var(--accent);border-bottom-color:var(--tab-active-bg)}

/* ── Panels ── */
.panel{display:none;flex-direction:column;flex:1;min-height:0}
.panel.active{display:flex}

/* ── Controls ── */
.controls{background:var(--surface);border-bottom:1px solid var(--border);
  padding:10px 14px;display:flex;flex-wrap:wrap;gap:8px;align-items:center;flex-shrink:0}
.controls-title{font-weight:700;font-size:14px;color:var(--text);margin-right:4px;white-space:nowrap}
.controls-count{font-size:12px;color:var(--muted);white-space:nowrap;min-width:80px}
input[type="search"],select{background:var(--input-bg);border:1px solid var(--border);
  border-radius:6px;color:var(--text);font-size:12px;padding:5px 9px;outline:none;
  transition:border-color .15s}
input[type="search"]{width:190px}
input[type="search"]:focus,select:focus{border-color:var(--accent)}
select option{background:var(--surface)}
.spacer{flex:1}
.clear-btn{background:none;border:1px solid var(--border);border-radius:6px;
  color:var(--muted);cursor:pointer;font-size:11px;padding:5px 10px;
  transition:color .15s,border-color .15s}
.clear-btn:hover{color:var(--text);border-color:var(--text)}

/* ── Tables ── */
.table-wrap{flex:1;overflow:auto;min-height:0}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
thead{position:sticky;top:0;z-index:10;background:var(--surface)}
th{padding:7px 10px;text-align:left;font-size:11px;font-weight:600;
  letter-spacing:.04em;text-transform:uppercase;color:var(--muted);
  white-space:nowrap;cursor:pointer;user-select:none;
  border-bottom:2px solid var(--border)}
th:hover{color:var(--text)}
th.sort-asc::after{content:" ↑";color:var(--accent)}
th.sort-desc::after{content:" ↓";color:var(--accent)}
th.num,td.num{text-align:right}
td{padding:5px 10px;border-bottom:1px solid var(--border);
  white-space:nowrap;color:var(--text);font-size:12px}
td.rank{text-align:right;color:var(--muted);font-weight:500;font-size:12px}
td.name{font-weight:500}
tbody tr:nth-child(even){background:var(--row-alt)}
tbody tr:hover{background:var(--row-hover)}

/* ── Score bars ── */
.flag{display:inline-block;border-radius:4px;font-size:10px;font-weight:700;
  padding:1px 5px;letter-spacing:.02em;text-transform:uppercase}
.flag-soft,.flag-softwhiff{background:var(--flag-soft-bg);color:var(--flag-soft-text)}
.flag-hard,.flag-hardwhiff{background:var(--flag-hard-bg);color:var(--flag-hard-text)}
.pos-bonus{font-size:0.7em;opacity:0.7;margin-left:4px;font-variant-numeric:tabular-nums}
.flag-whiff{background:var(--flag-whiff-bg);color:var(--flag-whiff-text)}
.flag-good{background:var(--flag-good-bg);color:var(--flag-good-text)}
.score-bar-wrap{display:flex;align-items:center;gap:5px;justify-content:flex-end}
.score-val{min-width:34px;text-align:right;font-weight:600}
.bar{width:42px;height:4px;border-radius:2px;background:var(--surface2);overflow:hidden;display:inline-block;vertical-align:middle}
.bar-fill{height:100%;border-radius:2px;background:var(--accent)}

/* ── Level badges ── */
.lvl{display:inline-block;border-radius:3px;font-size:10px;font-weight:700;
  padding:1px 5px;letter-spacing:.03em;background:var(--surface2);color:var(--muted)}
.lvl-AAA{background:#1a3a6c22;color:#4a90d9}
.lvl-AA {background:#1a4a2022;color:#2d8a4a}
.lvl-Ap {background:#3a2a0022;color:#b07d00}
.lvl-A  {background:#3a0a0022;color:#c05050}
.lvl-R  {background:#2a002a22;color:#9060b0}
@media(prefers-color-scheme:dark){
  .lvl-AAA{background:#1a3a6c55;color:#79b8ff}
  .lvl-AA {background:#1a4a2055;color:#56d364}
  .lvl-Ap {background:#3a2a0055;color:#e3a008}
  .lvl-A  {background:#3a0a0055;color:#ff7b72}
  .lvl-R  {background:#2a002a55;color:#d2a8ff}
}
:root[data-theme="dark"] .lvl-AAA{background:#1a3a6c55;color:#79b8ff}
:root[data-theme="dark"] .lvl-AA {background:#1a4a2055;color:#56d364}
:root[data-theme="dark"] .lvl-Ap {background:#3a2a0055;color:#e3a008}
:root[data-theme="dark"] .lvl-A  {background:#3a0a0055;color:#ff7b72}
:root[data-theme="dark"] .lvl-R  {background:#2a002a55;color:#d2a8ff}
:root[data-theme="light"] .lvl-AAA{background:#1a3a6c22;color:#4a90d9}
:root[data-theme="light"] .lvl-AA {background:#1a4a2022;color:#2d8a4a}
:root[data-theme="light"] .lvl-Ap {background:#3a2a0022;color:#b07d00}
:root[data-theme="light"] .lvl-A  {background:#3a0a0022;color:#c05050}
:root[data-theme="light"] .lvl-R  {background:#2a002a22;color:#9060b0}

.no-results{text-align:center;padding:48px;color:var(--muted);font-size:13px}
.null-cell{color:var(--muted)}

/* ── Expandable detail row ── */
tbody tr.data-row{cursor:pointer}
tbody tr.data-row:hover td{background:var(--row-hover)}
tr.detail-row td{padding:0;border-bottom:2px solid var(--accent);background:var(--surface)!important}
tr.detail-row td > .detail-inner{padding:10px 14px;display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start}
.comp-group{display:flex;flex-direction:column;gap:4px;min-width:160px}
.comp-group-label{font-size:10px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;
  color:var(--muted);margin-bottom:2px}
.comp-row{display:flex;align-items:center;gap:7px;font-size:11px}
.comp-name{width:72px;color:var(--muted);white-space:nowrap}
.comp-val{width:30px;text-align:right;font-weight:600;font-variant-numeric:tabular-nums}
.bar-sm{width:60px;height:5px;border-radius:3px;background:var(--surface2);overflow:hidden;display:inline-block;vertical-align:middle}
.bar-sm-fill{height:100%;border-radius:3px;background:var(--accent)}

/* ── Luck score coloring ── */
.luck-hot2{background:#ff4d0022;color:#c0392b;font-weight:700}
.luck-hot1{background:#ff8c0018;color:#e67e00;font-weight:600}
.luck-cold2{background:#0066cc22;color:#1a6fb5;font-weight:700}
.luck-cold1{background:#00998818;color:#007a6a;font-weight:600}
:root[data-theme="dark"] .luck-hot2,:root[data-theme="dark"] .luck-hot2{background:#ff4d0033;color:#ff7f7f}
:root[data-theme="dark"] .luck-hot1{background:#ff8c0028;color:#ffb347}
:root[data-theme="dark"] .luck-cold2{background:#0066cc33;color:#79b8ff}
:root[data-theme="dark"] .luck-cold1{background:#00998828;color:#56d3b5}
@media(prefers-color-scheme:dark){
  .luck-hot2{background:#ff4d0033;color:#ff7f7f}
  .luck-hot1{background:#ff8c0028;color:#ffb347}
  .luck-cold2{background:#0066cc33;color:#79b8ff}
  .luck-cold1{background:#00998828;color:#56d3b5}
}

/* ── Mobile ── */
@media(max-width:640px){
  body{font-size:12px}
  .controls{padding:8px 10px;gap:6px}
  .controls-title{font-size:13px}
  input[type="search"]{width:100%;flex:1 1 100%}
  select{flex:1 1 auto;min-width:80px}
  .spacer{display:none}
  td{padding:5px 7px}
  th{padding:6px 7px}
  .bar{width:28px}
  /* Prospects: show Rank Name Pos Team Level Age TOOLS ABILITY Combined (cols 1-6, 9, 10, 14)
     hide: Season(7) CareerPA(8) AgeScore(11) Current(12) OVR(13) RecentFlag(15) CareerFlag(16) */
  #table th:nth-child(7),#table td:nth-child(7),
  #table th:nth-child(8),#table td:nth-child(8),
  #table th:nth-child(11),#table td:nth-child(11),
  #table th:nth-child(12),#table td:nth-child(12),
  #table th:nth-child(13),#table td:nth-child(13),
  #table th:nth-child(15),#table td:nth-child(15),
  #table th:nth-child(16),#table td:nth-child(16),
  #table th:nth-child(17),#table td:nth-child(17){display:none}
  /* AAA: show Rank Name Team Pos Age PA PPPA Overall (cols 1-8), hide 9+ */
  #aaa-table th:nth-child(n+9),#aaa-table td:nth-child(n+9){display:none}
  /* Luck: show Rank Name Pos Level PA Luck PPPA_Z (cols 1-4,6,12,13), hide others */
  #luck-table th:nth-child(5),#luck-table td:nth-child(5),
  #luck-table th:nth-child(7),#luck-table td:nth-child(7),
  #luck-table th:nth-child(9),#luck-table td:nth-child(9),
  #luck-table th:nth-child(10),#luck-table td:nth-child(10),
  #luck-table th:nth-child(11),#luck-table td:nth-child(11),
  #luck-table th:nth-child(14),#luck-table td:nth-child(14){display:none}
}

/* ── Comps tab ── */
#comps-table{width:100%;border-collapse:collapse;table-layout:fixed}
#comps-table th,#comps-table td{padding:6px 10px;border-bottom:1px solid var(--border);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#comps-table th{background:var(--surface);font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);cursor:pointer;position:sticky;top:0;z-index:1;user-select:none}
#comps-table th:hover{color:var(--text)}
#comps-table tr.comp-row:hover td{background:var(--row-hover);cursor:pointer}
#comps-table tr.comp-row.expanded td{background:var(--surface)}
.proj-bar-wrap{display:flex;align-items:center;gap:6px;min-width:0}
.proj-bar-outer{flex:1;height:10px;background:var(--surface2);border-radius:5px;position:relative;overflow:visible;min-width:60px}
.proj-bar-fill{position:absolute;top:0;height:100%;background:#3fb950;border-radius:5px;opacity:.75}
.proj-bar-median{position:absolute;top:-2px;width:3px;height:14px;background:var(--text);border-radius:2px;transform:translateX(-50%)}
.proj-label{font-size:11px;color:var(--muted);white-space:nowrap;min-width:90px}
.comp-detail{display:none;padding:0}
.comp-detail.open{display:table-row}
.comp-detail-inner{padding:14px 16px;background:var(--surface);border-bottom:2px solid var(--border);display:flex;gap:24px;flex-wrap:wrap}
.comp-detail-section{flex:1;min-width:260px}
.comp-detail-section h4{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin-bottom:8px}
.mini-table{width:100%;border-collapse:collapse;font-size:12px}
.mini-table th,.mini-table td{padding:4px 8px;border-bottom:1px solid var(--border);text-align:right}
.mini-table th:first-child,.mini-table td:first-child{text-align:left}
.mini-table th{background:var(--surface2);font-weight:600;color:var(--muted);font-size:11px}
.mini-table td.match-hi{color:#3fb950;font-weight:700}
.mini-table td.match-med{color:var(--accent)}
.mini-table td.grad-yes{color:#3fb950}
.mini-table td.grad-no{color:var(--muted)}
.mini-table tr:hover td{background:var(--row-hover)}
.proj-summary{display:flex;gap:16px;margin-bottom:10px;flex-wrap:wrap}
.proj-stat{display:flex;flex-direction:column;align-items:center;background:var(--surface2);border-radius:6px;padding:6px 12px;min-width:60px}
.proj-stat-label{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.proj-stat-val{font-size:15px;font-weight:700;margin-top:2px}
.proj-stat-val.ceiling{color:#3fb950}
.proj-stat-val.median{color:var(--accent)}
.proj-stat-val.floor{color:var(--muted)}
.grad-badge{display:inline-block;font-size:10px;font-weight:700;padding:1px 5px;border-radius:3px;vertical-align:middle}
.grad-badge.yes{background:var(--flag-good-bg);color:var(--flag-good-text)}
.grad-badge.no{background:var(--surface2);color:var(--muted)}
</style>

<!-- ══ Tab bar ══ -->
<div class="tab-bar">
  <button class="tab-btn active" data-tab="prospects">Prospects</button>
  <button class="tab-btn" data-tab="sp">SP Prospects</button>
  <button class="tab-btn" data-tab="rp">RP Prospects</button>
  <button class="tab-btn" data-tab="aaa">AAA 2026</button>
  <button class="tab-btn" data-tab="luck">Luck Tracker</button>
  <button class="tab-btn" data-tab="comps">Comps</button>
</div>

<!-- ══ Prospects panel ══ -->
<div class="panel active" id="panel-prospects">
  <div class="controls">
    <span class="controls-title">2026 Prospect Rankings</span>
    <span class="controls-count" id="count"></span>
    <input type="search" id="search" placeholder="Search player…" autocomplete="off" />
    <select id="team-filter"><option value="">All teams</option></select>
    <select id="level-filter">
      <option value="">All levels</option>
      <option>R</option><option>A</option><option>A+</option>
      <option>AA</option><option>AAA</option>
    </select>
    <select id="pos-filter"><option value="">All positions</option></select>
    <select id="flag-filter">
      <option value="">All flags</option>
      <option value="any_recent">Any recent flag</option>
      <option value="any_career">Any career flag</option>
      <option value="hard">Hard floor (either)</option>
      <option value="soft">Soft floor (either)</option>
      <option value="whiff">Whiff (either)</option>
      <option value="hidden">Hidden: clean recent, flagged career</option>
      <option value="hardfloor">Hard Floor (K%/HRFB/SBTalent)</option>
      <option value="youngaaelite">Young AA Elite (Age&le;20 + AA PPPA_Z&ge;1.0)</option>
      <option value="clean">Fully clean (no flags)</option>
    </select>
    <div class="spacer"></div>
    <button class="clear-btn" id="ovr-floor-btn" title="Prospects graded below Harold Castro's all-time OVR score are hidden by default (still findable via search or flags)">Below Floor: HIDDEN</button>
    <button class="clear-btn" id="pos-adj-btn" title="Toggle positional scarcity adjustment">Pos Adj: OFF</button>
    <button class="clear-btn" id="clear-btn">Clear</button>
  </div>
  <div class="table-wrap">
    <table id="table">
      <thead><tr>
        <th class="num" data-col="Combined_Rank" data-type="num">Rank</th>
        <th data-col="Name" data-type="str">Name</th>
        <th data-col="Pos" data-type="str">Pos</th>
        <th data-col="Team" data-type="str">Team</th>
        <th data-col="Level" data-type="level">Level</th>
        <th class="num" data-col="Age" data-type="num">Age</th>
        <th class="num" data-col="Last_Season" data-type="num">Season</th>
        <th class="num" data-col="Career_PA" data-type="num">Career PA</th>
        <th class="num" data-col="TOOLS_Score" data-type="num">TOOLS</th>
        <th class="num" data-col="ABILITY_Score" data-type="num">ABILITY</th>
        <th class="num" data-col="Current_Score" data-type="num">Current</th>
        <th class="num" data-col="OVR_Score" data-type="num">OVR</th>
        <th class="num" data-col="Combined_Score" data-type="num">Combined</th>
        <th data-col="Discipline_Flag" data-type="str">Recent Flag</th>
        <th data-col="Career_Disc_Flag" data-type="str">Career Flag</th>
        <th data-col="Hard_Floor_Flag" data-type="str">Hard Floor</th>
        <th data-col="AgeAA_Combo_Flag_Label" data-type="str">Young AA Elite</th>
      </tr></thead>
      <tbody id="tbody"></tbody>
    </table>
  </div>
</div>

<!-- ══ SP Prospects panel ══ -->
<div class="panel" id="panel-sp">
  <div class="controls">
    <span class="controls-title">SP Prospects 2026</span>
    <span class="controls-count" id="sp-count"></span>
    <input type="search" id="sp-search" placeholder="Search pitcher…" autocomplete="off" />
    <select id="sp-team-filter"><option value="">All teams</option></select>
    <select id="sp-level-filter">
      <option value="">All levels</option>
      <option>R</option><option>A</option><option>A+</option>
      <option>AA</option><option>AAA</option>
    </select>
    <div class="spacer"></div>
    <button class="clear-btn" id="sp-clear-btn">Clear</button>
  </div>
  <div class="table-wrap">
    <table id="sp-table">
      <thead><tr>
        <th class="num" data-sp-col="SP_Rank" data-type="num">Rank</th>
        <th data-sp-col="Name" data-type="str">Name</th>
        <th data-sp-col="Team" data-type="str">Team</th>
        <th data-sp-col="Level" data-type="level">Level</th>
        <th class="num" data-sp-col="Age" data-type="num">Age</th>
        <th class="num" data-sp-col="Career_IP" data-type="num">Career IP</th>
        <th class="num" data-sp-col="STUFF_Score" data-type="num">STUFF</th>
        <th class="num" data-sp-col="PERFORMANCE_Score" data-type="num">PERF</th>
        <th class="num" data-sp-col="Current_Score" data-type="num">Current</th>
        <th class="num" data-sp-col="OVR_Score" data-type="num">OVR</th>
        <th class="num" data-sp-col="Combined_Score" data-type="num">Combined</th>
      </tr></thead>
      <tbody id="sp-tbody"></tbody>
    </table>
  </div>
</div>

<!-- ══ RP Prospects panel ══ -->
<div class="panel" id="panel-rp">
  <div class="controls">
    <span class="controls-title">RP Prospects 2026</span>
    <span class="controls-count" id="rp-count"></span>
    <input type="search" id="rp-search" placeholder="Search pitcher…" autocomplete="off" />
    <select id="rp-team-filter"><option value="">All teams</option></select>
    <select id="rp-level-filter">
      <option value="">All levels</option>
      <option>R</option><option>A</option><option>A+</option>
      <option>AA</option><option>AAA</option>
    </select>
    <div class="spacer"></div>
    <button class="clear-btn" id="rp-clear-btn">Clear</button>
  </div>
  <div class="table-wrap">
    <table id="rp-table">
      <thead><tr>
        <th class="num" data-rp-col="RP_Rank" data-type="num">Rank</th>
        <th data-rp-col="Name" data-type="str">Name</th>
        <th data-rp-col="Team" data-type="str">Team</th>
        <th data-rp-col="Level" data-type="level">Level</th>
        <th class="num" data-rp-col="Age" data-type="num">Age</th>
        <th class="num" data-rp-col="Career_IP" data-type="num">Career IP</th>
        <th class="num" data-rp-col="STUFF_Score" data-type="num">STUFF</th>
        <th class="num" data-rp-col="PERFORMANCE_Score" data-type="num">PERF</th>
        <th class="num" data-rp-col="Current_Score" data-type="num">Current</th>
        <th class="num" data-rp-col="OVR_Score" data-type="num">OVR</th>
        <th class="num" data-rp-col="Combined_Score" data-type="num">Combined</th>
      </tr></thead>
      <tbody id="rp-tbody"></tbody>
    </table>
  </div>
</div>

<!-- ══ AAA 2026 panel ══ -->
<div class="panel" id="panel-aaa">
  <div class="controls">
    <span class="controls-title">AAA 2026 Power Rankings</span>
    <span class="controls-count" id="aaa-count"></span>
    <input type="search" id="aaa-search" placeholder="Search player…" autocomplete="off" />
    <select id="aaa-team-filter"><option value="">All teams</option></select>
    <select id="aaa-pa-filter">
      <option value="">All PA</option>
      <option value="50">≥ 50 PA</option>
      <option value="100">≥ 100 PA</option>
      <option value="150">≥ 150 PA</option>
      <option value="200">≥ 200 PA</option>
    </select>
    <div class="spacer"></div>
    <button class="clear-btn" id="aaa-clear-btn">Clear</button>
  </div>
  <div class="table-wrap">
    <table id="aaa-table">
      <thead><tr>
        <th class="num" data-aaa-col="Rank" data-type="num">Rank</th>
        <th data-aaa-col="Name" data-type="str">Name</th>
        <th data-aaa-col="Team" data-type="str">Team</th>
        <th data-aaa-col="Pos" data-type="str">Pos</th>
        <th class="num" data-aaa-col="Age" data-type="num">Age</th>
        <th class="num" data-aaa-col="PA" data-type="num">PA</th>
        <th class="num" data-aaa-col="PPPA" data-type="num">PPPA</th>
        <th class="num" data-aaa-col="Overall" data-type="num">Overall</th>
        <th class="num" data-aaa-col="Discipline" data-type="num">Disc</th>
        <th class="num" data-aaa-col="Power" data-type="num">Power</th>
        <th class="num" data-aaa-col="Contact" data-type="num">Contact</th>
        <th class="num" data-aaa-col="Speed" data-type="num">Speed</th>
        <th class="num" data-aaa-col="Chase%" data-type="num">Chase%</th>
        <th class="num" data-aaa-col="ZContact%" data-type="num">ZCon%</th>
        <th class="num" data-aaa-col="Whiff%" data-type="num">Whiff%</th>
        <th class="num" data-aaa-col="BB%-2K%" data-type="num">BB-2K%</th>
        <th class="num" data-aaa-col="MaxEV" data-type="num">MaxEV</th>
        <th class="num" data-aaa-col="Barrel%" data-type="num">Brl%</th>
        <th class="num" data-aaa-col="xSLG" data-type="num">xSLG</th>
        <th class="num" data-aaa-col="SB_rate" data-type="num">SB/PA</th>
        <th class="num" data-aaa-col="SB_succ" data-type="num">SB%</th>
      </tr></thead>
      <tbody id="aaa-tbody"></tbody>
    </table>
  </div>
</div>

<!-- ══ Luck Tracker panel ══ -->
<div class="panel" id="panel-luck">
  <div class="controls">
    <span class="controls-title">Luck Tracker</span>
    <span class="controls-count" id="luck-count"></span>
    <input type="search" id="luck-search" placeholder="Search player…" autocomplete="off" />
    <select id="luck-level-filter">
      <option value="">All levels</option>
      <option>R</option><option>A</option><option>A+</option>
      <option>AA</option><option>AAA</option>
    </select>
    <select id="luck-pos-filter">
      <option value="">All positions</option>
      <option value="C">C</option><option value="1B">1B</option>
      <option value="2B">2B</option><option value="3B">3B</option>
      <option value="SS">SS</option><option value="OF">OF</option>
    </select>
    <select id="luck-dir-filter">
      <option value="">All players</option>
      <option value="lucky">Lucky (regression risk)</option>
      <option value="unlucky">Unlucky (buy-low)</option>
    </select>
    <select id="luck-pa-filter">
      <option value="">All PA</option>
      <option value="200">≥ 200 PA</option>
      <option value="300">≥ 300 PA</option>
      <option value="400">≥ 400 PA</option>
    </select>
    <div class="spacer"></div>
    <button class="clear-btn" id="luck-clear-btn">Clear</button>
  </div>
  <div class="table-wrap">
    <table id="luck-table">
      <thead><tr>
        <th class="num" data-luck-col="Combined_Rank" data-type="num">Rank</th>
        <th data-luck-col="Name" data-type="str">Name</th>
        <th data-luck-col="FantasyPos" data-type="str">Pos</th>
        <th data-luck-col="Level" data-type="level">Level</th>
        <th class="num" data-luck-col="Season" data-type="num">Year</th>
        <th class="num" data-luck-col="PA" data-type="num">PA</th>
        <th class="num" data-luck-col="Prior_Career_PA" data-type="num" title="Career PA from prior seasons — drives baseline reliability">Prior PA</th>
        <th class="num" data-luck-col="BABIP" data-type="num">BABIP</th>
        <th class="num" data-luck-col="BABIP_career" data-type="num" title="Career PA-weighted BABIP baseline (current season excluded)">Career BABIP</th>
        <th class="num" data-luck-col="BABIP_delta_z" data-type="num" title="BABIP vs career baseline, z-scored within Season+Level peers">BABIP Δz</th>
        <th class="num" data-luck-col="BABIP_Delta_Slope" data-type="num" title="PA-weighted OLS slope of (BABIP − Career BABIP) on Season. Positive = gap widening over career; negative = regressing toward or below baseline.">BABIP Δ slope</th>
        <th class="num" data-luck-col="HRFB_delta_z" data-type="num" title="HR/FB vs career baseline, z-scored within Season+Level peers">HRFB Δz</th>
        <th class="num" data-luck-col="Luck_PPPA" data-type="num" title="Luck expressed in PPPA units (shown with % of season PPPA). BABIP deviation × BIP rate × 2.8 + HR/FB deviation × FB rate × 8, shrunk by prior PA reliability. Positive = lucky.">Luck PPPA</th>
        <th class="num" data-luck-col="PPPA_Z_SL" data-type="num" title="PPPA z-score vs same Season+Level peers">PPPA Z</th>
        <th class="num" data-luck-col="PPPA_Jump" data-type="num" title="PPPA_Z change vs prior season">PPPA Δ</th>
      </tr></thead>
      <tbody id="luck-tbody"></tbody>
    </table>
  </div>
</div>

<!-- ══ Comps panel ══ -->
<div class="panel" id="panel-comps">
  <div class="controls">
    <span class="controls-title">Prospect Comps</span>
    <span class="controls-count" id="comps-count"></span>
    <input type="search" id="comps-search" placeholder="Search player…" autocomplete="off" />
    <div class="spacer"></div>
    <span style="font-size:11px;color:var(--muted)">Top 250 prospects · click row to expand</span>
  </div>
  <div class="table-wrap">
    <table id="comps-table">
      <thead><tr>
        <th style="width:52px" data-comps-col="rank" data-type="num">Rank</th>
        <th style="width:160px" data-comps-col="name" data-type="str">Name</th>
        <th style="width:50px" data-comps-col="team" data-type="str">Team</th>
        <th style="width:46px" data-comps-col="level" data-type="level">Lvl</th>
        <th style="width:38px" data-comps-col="age" data-type="num">Age</th>
        <th style="min-width:180px" data-comps-col="median" data-type="num">Projection  <span style="font-weight:400;font-size:10px">(floor — median — ceiling)</span></th>
        <th style="width:130px" data-comps-col="archetype" data-type="str">Archetype</th>
        <th style="width:60px" data-comps-col="grad_pct" data-type="num" title="Fraction of top-10 comps that reached MLB — proxy for graduation odds">Grad%</th>
        <th style="width:50px" data-comps-col="n_grads" data-type="num"># Grads</th>
      </tr></thead>
      <tbody id="comps-tbody"></tbody>
    </table>
  </div>
</div>

<script>
/* ════════════════════════════════════════════
   PROSPECTS TAB
   ════════════════════════════════════════════ */
const RAW = PROSPECTS_DATA_PLACEHOLDER;
const LVL = {R:1,A:2,'A+':3,AA:4,AAA:5};
let sortCol='Combined_Rank', sortDir=1, filtered=RAW.slice();
let posAdj=false;
let hideFloor=true;

const tf=document.getElementById('team-filter');
[...new Set(RAW.map(r=>r.Team))].sort().forEach(t=>{
  const o=document.createElement('option');o.value=t;o.textContent=t;tf.appendChild(o);
});

const POS_BUCKETS=[
  {val:'C',label:'C — Catcher'},{val:'1B',label:'1B'},{val:'2B',label:'2B'},
  {val:'3B',label:'3B'},{val:'SS',label:'SS'},{val:'IF',label:'IF — Infielder'},
  {val:'OF',label:'OF (all outfield)'},{val:'DH',label:'DH'},
];
const POS_OF_SET=new Set(['OF','LF','CF','RF']);
const pf=document.getElementById('pos-filter');
POS_BUCKETS.forEach(({val,label})=>{
  const o=document.createElement('option');o.value=val;o.textContent=label;pf.appendChild(o);
});

function fclass(f){
  if(!f)return'';
  if(f==='soft'||f==='soft+whiff')return'flag-soft';
  if(f==='hard'||f==='hard+whiff'||f==='Hard Floor')return'flag-hard';
  if(f==='whiff')return'flag-whiff';
  if(f==='Young AA Elite')return'flag-good';
  return'';
}
function fcell(f){
  if(!f)return'<td></td>';
  return`<td><span class="flag ${fclass(f)}">${f}</span></td>`;
}
function lclass(l){return'lvl lvl-'+(l==='A+'?'Ap':l)}
function bar(v){
  const p=Math.max(0,Math.min(100,((v-30)/70)*100));
  return`<div class="score-bar-wrap"><span class="score-val">${v.toFixed(1)}</span>`+
    `<span class="bar"><span class="bar-fill" style="width:${p.toFixed(1)}%"></span></span></div>`;
}

function compBar(v){
  if(v==null)return'';
  const p=Math.max(0,Math.min(100,((v-30)/70)*100));
  return`<span class="comp-val">${v.toFixed(1)}</span>`+
    `<span class="bar-sm"><span class="bar-sm-fill" style="width:${p.toFixed(1)}%"></span></span>`;
}
function detailHTML(r){
  const tCols=[['Disc',r.TOOLS_Disc],['Power',r.TOOLS_Power],['Ath',r.TOOLS_Ath]];
  const aCols=[['Fantasy',r.Fantasy_Out],['Disc',r.ABILITY_Disc],['SB',r.SB_Talent],['Power',r.Game_Power]];
  const hasT=tCols.some(([,v])=>v!=null), hasA=aCols.some(([,v])=>v!=null);
  if(!hasT&&!hasA)return'';
  let html='<div class="detail-inner">';
  if(hasT){
    html+='<div class="comp-group"><div class="comp-group-label">TOOLS breakdown</div>';
    tCols.forEach(([n,v])=>{
      if(v!=null)html+=`<div class="comp-row"><span class="comp-name">${n}</span>${compBar(v)}</div>`;
    });
    html+='</div>';
  }
  if(hasA){
    html+='<div class="comp-group"><div class="comp-group-label">ABILITY breakdown</div>';
    aCols.forEach(([n,v])=>{
      if(v!=null)html+=`<div class="comp-row"><span class="comp-name">${n}</span>${compBar(v)}</div>`;
    });
    html+='</div>';
  }
  html+='</div>';
  return html;
}

let expandedIdx=null;

function render(){
  const tbody=document.getElementById('tbody');
  document.getElementById('count').textContent=filtered.length.toLocaleString()+' players';
  if(!filtered.length){
    tbody.innerHTML='<tr><td colspan="17" class="no-results">No players match.</td></tr>';
    return;
  }
  expandedIdx=null;
  tbody.innerHTML=filtered.map((r,i)=>{
    const rankVal=posAdj?r.Pos_Adj_Rank:r.Combined_Rank;
    const scoreCell=posAdj
      ?`${bar(r.Pos_Adj_Score)}<span class="pos-bonus" title="Pos bonus">${r.Pos_Bonus>=0?'+':''}${r.Pos_Bonus}</span>`
      :bar(r.Combined_Score);
    return `<tr class="data-row" data-i="${i}">
    <td class="rank">${rankVal}</td>
    <td class="name">${r.Name}</td>
    <td>${r.Pos||''}</td>
    <td>${r.Team}</td>
    <td><span class="${lclass(r.Level)}">${r.Level}</span></td>
    <td class="num">${r.Age}</td>
    <td class="num">${r.Last_Season}</td>
    <td class="num">${r.Career_PA.toLocaleString()}</td>
    <td class="num">${bar(r.TOOLS_Score)}</td>
    <td class="num">${bar(r.ABILITY_Score)}</td>
    <td class="num">${bar(r.Current_Score)}</td>
    <td class="num">${bar(r.OVR_Score)}</td>
    <td class="num">${scoreCell}</td>
    ${fcell(r.Discipline_Flag)}
    ${fcell(r.Career_Disc_Flag)}
    ${fcell(r.Hard_Floor_Flag)}
    ${fcell(r.AgeAA_Combo_Flag_Label)}
  </tr>`;
  }).join('');

}

document.getElementById('tbody').addEventListener('click',function(e){
  const row=e.target.closest('tr.data-row');
  if(!row)return;
  const i=parseInt(row.dataset.i);
  const detail=detailHTML(filtered[i]);
  if(!detail)return;
  const existing=row.nextElementSibling;
  if(existing&&existing.classList.contains('detail-row')){
    existing.remove();
    expandedIdx=null;
    return;
  }
  document.getElementById('tbody').querySelectorAll('tr.detail-row').forEach(d=>d.remove());
  expandedIdx=i;
  const cols=row.querySelectorAll('td').length;
  const dr=document.createElement('tr');
  dr.className='detail-row';
  dr.innerHTML=`<td colspan="${cols}">${detail}</td>`;
  row.after(dr);
});

function applyFilters(){
  const q=document.getElementById('search').value.trim().toLowerCase();
  const team=document.getElementById('team-filter').value;
  const lvl=document.getElementById('level-filter').value;
  const pos=document.getElementById('pos-filter').value;
  const flag=document.getElementById('flag-filter').value;
  const anyFilterActive=!!(q||team||lvl||pos||flag);
  filtered=RAW.filter(r=>{
    if(hideFloor&&!anyFilterActive&&r.Below_OVR_Floor)return false;
    if(q&&!r.Name.toLowerCase().includes(q))return false;
    if(team&&r.Team!==team)return false;
    if(lvl&&r.Level!==lvl)return false;
    if(pos){
      const rp=r.Pos||'';
      if(pos==='OF'){if(!POS_OF_SET.has(rp))return false;}
      else if(rp!==pos)return false;
    }
    const rf=r.Discipline_Flag||'', cf=r.Career_Disc_Flag||'', hf=r.Hard_Floor_Flag||'', yf=r.AgeAA_Combo_Flag_Label||'';
    if(flag==='any_recent'&&!rf)return false;
    if(flag==='any_career'&&!cf)return false;
    if(flag==='hard'&&!rf.includes('hard')&&!cf.includes('hard'))return false;
    if(flag==='soft'&&!rf.startsWith('soft')&&!cf.startsWith('soft'))return false;
    if(flag==='whiff'&&!rf.includes('whiff')&&!cf.includes('whiff'))return false;
    if(flag==='hidden'&&!(rf===''&&cf!==''))return false;
    if(flag==='hardfloor'&&!hf)return false;
    if(flag==='youngaaelite'&&!yf)return false;
    if(flag==='clean'&&(rf!==''||cf!==''||hf!==''||yf!==''))return false;
    return true;
  });
  sort();
}

function sort(){
  const type=document.querySelector(`th[data-col="${sortCol}"]`)?.dataset.type;
  filtered.sort((a,b)=>{
    let av=a[sortCol],bv=b[sortCol];
    if(type==='level'){av=LVL[av]||0;bv=LVL[bv]||0;}
    if(type==='str')return sortDir*String(av||'').localeCompare(String(bv||''));
    return sortDir*((av??-Infinity)-(bv??-Infinity));
  });
  render();
}

document.querySelectorAll('th[data-col]').forEach(th=>{
  th.addEventListener('click',()=>{
    const col=th.dataset.col;
    sortDir=(sortCol===col)?-sortDir:(col==='Combined_Rank'?1:-1);
    sortCol=col;
    document.querySelectorAll('th[data-col]').forEach(h=>h.classList.remove('sort-asc','sort-desc'));
    th.classList.add(sortDir===1?'sort-asc':'sort-desc');
    sort();
  });
});

['search','team-filter','level-filter','pos-filter','flag-filter'].forEach(id=>{
  document.getElementById(id).addEventListener(id==='search'?'input':'change',applyFilters);
});
document.getElementById('ovr-floor-btn').addEventListener('click',()=>{
  hideFloor=!hideFloor;
  document.getElementById('ovr-floor-btn').textContent='Below Floor: '+(hideFloor?'HIDDEN':'SHOWN');
  document.getElementById('ovr-floor-btn').style.opacity=hideFloor?'0.7':'1';
  applyFilters();
});
document.getElementById('pos-adj-btn').addEventListener('click',()=>{
  posAdj=!posAdj;
  document.getElementById('pos-adj-btn').textContent='Pos Adj: '+(posAdj?'ON':'OFF');
  document.getElementById('pos-adj-btn').style.opacity=posAdj?'1':'0.7';
  const rankCol=posAdj?'Pos_Adj_Rank':'Combined_Rank';
  sortCol=rankCol;sortDir=1;
  document.querySelectorAll('th[data-col]').forEach(h=>h.classList.remove('sort-asc','sort-desc'));
  const rankTh=document.querySelector('th[data-col="Combined_Rank"]');
  if(rankTh)rankTh.classList.add('sort-asc');
  sort();
});
document.getElementById('clear-btn').addEventListener('click',()=>{
  document.getElementById('search').value='';
  ['team-filter','level-filter','pos-filter','flag-filter'].forEach(id=>{
    document.getElementById(id).value='';
  });
  applyFilters();
});
document.querySelector('th[data-col="Combined_Rank"]').classList.add('sort-asc');
applyFilters();

/* ════════════════════════════════════════════
   PITCHER TABS (SP + RP)
   ════════════════════════════════════════════ */
const SP_RAW = SP_DATA_PLACEHOLDER;
const RP_RAW = RP_DATA_PLACEHOLDER;

function makePitcherTab(RAW, prefix, rankCol){
  let sortCol=rankCol, sortDir=1, filtered=RAW.slice();

  const tf=document.getElementById(prefix+'-team-filter');
  [...new Set(RAW.map(r=>r.Team))].sort().forEach(t=>{
    const o=document.createElement('option');o.value=t;o.textContent=t;tf.appendChild(o);
  });

  function renderPit(){
    const tbody=document.getElementById(prefix+'-tbody');
    document.getElementById(prefix+'-count').textContent=filtered.length.toLocaleString()+' pitchers';
    if(!filtered.length){
      tbody.innerHTML='<tr><td colspan="12" class="no-results">No pitchers match.</td></tr>';
      return;
    }
    tbody.innerHTML=filtered.map(r=>`<tr>
      <td class="rank">${r[rankCol]??'—'}</td>
      <td class="name">${r.Name}</td>
      <td>${r.Team}</td>
      <td><span class="${lclass(r.Level)}">${r.Level}</span></td>
      <td class="num">${r.Age}</td>
      <td class="num">${r.Career_IP!=null?(+r.Career_IP).toFixed(1):'—'}</td>
      <td class="num">${r.STUFF_Score!=null?bar(r.STUFF_Score):''}</td>
      <td class="num">${r.PERFORMANCE_Score!=null?bar(r.PERFORMANCE_Score):''}</td>
      <td class="num">${r.Current_Score!=null?bar(r.Current_Score):''}</td>
      <td class="num">${r.OVR_Score!=null?bar(r.OVR_Score):''}</td>
      <td class="num">${r.Combined_Score!=null?bar(r.Combined_Score):''}</td>
    </tr>`).join('');
  }

  function applyFilters(){
    const q=document.getElementById(prefix+'-search').value.trim().toLowerCase();
    const team=document.getElementById(prefix+'-team-filter').value;
    const lvl=document.getElementById(prefix+'-level-filter').value;
    filtered=RAW.filter(r=>{
      if(q&&!r.Name.toLowerCase().includes(q))return false;
      if(team&&r.Team!==team)return false;
      if(lvl&&r.Level!==lvl)return false;
      return true;
    });
    doSort();
  }

  function doSort(){
    const type=document.querySelector(`th[data-${prefix}-col="${sortCol}"]`)?.dataset.type;
    filtered.sort((a,b)=>{
      let av=a[sortCol],bv=b[sortCol];
      if(type==='level'){av=LVL[av]||0;bv=LVL[bv]||0;}
      if(type==='str')return sortDir*String(av||'').localeCompare(String(bv||''));
      return sortDir*((av??Infinity)-(bv??Infinity));
    });
    renderPit();
  }

  document.querySelectorAll(`th[data-${prefix}-col]`).forEach(th=>{
    th.addEventListener('click',()=>{
      const col=th.dataset[prefix+'Col']||(prefix==='sp'?th.dataset.spCol:th.dataset.rpCol);
      sortDir=(sortCol===col)?-sortDir:(col===rankCol?1:-1);
      sortCol=col;
      document.querySelectorAll(`th[data-${prefix}-col]`).forEach(h=>h.classList.remove('sort-asc','sort-desc'));
      th.classList.add(sortDir===1?'sort-asc':'sort-desc');
      doSort();
    });
  });

  [prefix+'-search',prefix+'-team-filter',prefix+'-level-filter'].forEach(id=>{
    document.getElementById(id).addEventListener(id.endsWith('-search')?'input':'change',applyFilters);
  });
  document.getElementById(prefix+'-clear-btn').addEventListener('click',()=>{
    document.getElementById(prefix+'-search').value='';
    [prefix+'-team-filter',prefix+'-level-filter'].forEach(id=>{ document.getElementById(id).value=''; });
    applyFilters();
  });
  const rankTh=document.querySelector(`th[data-${prefix}-col="${rankCol}"]`);
  if(rankTh)rankTh.classList.add('sort-asc');
  applyFilters();
}

if(SP_RAW.length) makePitcherTab(SP_RAW,'sp','SP_Rank');
if(RP_RAW.length) makePitcherTab(RP_RAW,'rp','RP_Rank');

/* ════════════════════════════════════════════
   AAA 2026 TAB
   ════════════════════════════════════════════ */
const AAA_RAW = AAA_DATA_PLACEHOLDER;
let aaaSortCol='Rank', aaaSortDir=1, aaaFiltered=AAA_RAW.slice();

const aaaTF=document.getElementById('aaa-team-filter');
[...new Set(AAA_RAW.map(r=>r.Team))].sort().forEach(t=>{
  const o=document.createElement('option');o.value=t;o.textContent=t;aaaTF.appendChild(o);
});

function fmt(v,dec=1){
  if(v==null||isNaN(v))return'<td class="num null-cell">—</td>';
  return`<td class="num">${(+v).toFixed(dec)}</td>`;
}
function barPct(v,pct,dec=1){
  if(v==null||isNaN(v))return'<td class="num null-cell">—</td>';
  const p=Math.max(0,Math.min(100,pct??0));
  return`<td class="num"><div class="score-bar-wrap"><span class="score-val">${(+v).toFixed(dec)}</span>`+
    `<span class="bar"><span class="bar-fill" style="width:${p.toFixed(1)}%"></span></span></div></td>`;
}
function pctFmt(v,dec=1){
  if(v==null||isNaN(v))return'<td class="num null-cell">—</td>';
  return`<td class="num">${(v*100).toFixed(dec)}%</td>`;
}

function renderAAA(){
  const tbody=document.getElementById('aaa-tbody');
  document.getElementById('aaa-count').textContent=aaaFiltered.length.toLocaleString()+' players';
  if(!aaaFiltered.length){
    tbody.innerHTML='<tr><td colspan="21" class="no-results">No players match.</td></tr>';
    return;
  }
  tbody.innerHTML=aaaFiltered.map(r=>`<tr>
    <td class="rank">${r.Rank}</td>
    <td class="name">${r.Name}</td>
    <td>${r.Team}</td>
    <td>${r.Pos||'—'}</td>
    <td class="num">${r.Age!=null?(+r.Age).toFixed(1):'—'}</td>
    <td class="num">${r.PA??'—'}</td>
    ${barPct(r.PPPA, r.PPPA_R, 2)}
    ${barPct(r.Overall, r.Overall_R)}
    ${barPct(r.Discipline, r.Discipline_R)}
    ${barPct(r.Power, r.Power_R)}
    ${barPct(r.Contact, r.Contact_R)}
    ${barPct(r.Speed, r.Speed_R)}
    ${fmt(r['Chase%'])}
    ${fmt(r['ZContact%'])}
    ${fmt(r['Whiff%'])}
    ${fmt(r['BB%-2K%'])}
    ${fmt(r['MaxEV'])}
    ${fmt(r['Barrel%'])}
    ${fmt(r['xSLG'],3)}
    ${fmt(r['SB_rate'],3)}
    ${pctFmt(r['SB_succ'])}
  </tr>`).join('');
}

function applyAAAFilters(){
  const q=document.getElementById('aaa-search').value.trim().toLowerCase();
  const team=document.getElementById('aaa-team-filter').value;
  const minPA=parseInt(document.getElementById('aaa-pa-filter').value)||0;
  aaaFiltered=AAA_RAW.filter(r=>{
    if(q&&!r.Name.toLowerCase().includes(q))return false;
    if(team&&r.Team!==team)return false;
    if(minPA&&(r.PA??0)<minPA)return false;
    return true;
  });
  sortAAA();
}

function sortAAA(){
  const type=document.querySelector(`th[data-aaa-col="${aaaSortCol}"]`)?.dataset.type;
  aaaFiltered.sort((a,b)=>{
    let av=a[aaaSortCol],bv=b[aaaSortCol];
    if(type==='str')return aaaSortDir*String(av||'').localeCompare(String(bv||''));
    return aaaSortDir*((av??-Infinity)-(bv??-Infinity));
  });
  renderAAA();
}

document.querySelectorAll('th[data-aaa-col]').forEach(th=>{
  th.addEventListener('click',()=>{
    const col=th.dataset.aaaCol;
    aaaSortDir=(aaaSortCol===col)?-aaaSortDir:(col==='Rank'?1:-1);
    aaaSortCol=col;
    document.querySelectorAll('th[data-aaa-col]').forEach(h=>h.classList.remove('sort-asc','sort-desc'));
    th.classList.add(aaaSortDir===1?'sort-asc':'sort-desc');
    sortAAA();
  });
});

['aaa-search','aaa-team-filter','aaa-pa-filter'].forEach(id=>{
  document.getElementById(id).addEventListener(id==='aaa-search'?'input':'change',applyAAAFilters);
});
document.getElementById('aaa-clear-btn').addEventListener('click',()=>{
  document.getElementById('aaa-search').value='';
  ['aaa-team-filter','aaa-pa-filter'].forEach(id=>{ document.getElementById(id).value=''; });
  applyAAAFilters();
});
document.querySelector('th[data-aaa-col="Rank"]').classList.add('sort-asc');
applyAAAFilters();

/* ════════════════════════════════════════════
   LUCK TRACKER TAB
   ════════════════════════════════════════════ */
const LUCK_RAW = LUCK_DATA_PLACEHOLDER;
let luckSortCol='Luck_PPPA', luckSortDir=-1, luckFiltered=LUCK_RAW.slice();

function luckClass(v){
  if(v==null||isNaN(v))return'';
  if(v>=2.5)return'luck-hot2';
  if(v>=1.0)return'luck-hot1';
  if(v<=-2.5)return'luck-cold2';
  if(v<=-1.0)return'luck-cold1';
  return'';
}
function luckCell(v){
  if(v==null||isNaN(v))return'<td class="num null-cell">—</td>';
  const cls=luckClass(v);
  return`<td class="num${cls?' '+cls:''}">${(+v).toFixed(2)}</td>`;
}
function luckPPPAClass(v){
  if(v==null||isNaN(v))return'';
  if(v>=0.20)return'luck-hot2';
  if(v>=0.08)return'luck-hot1';
  if(v<=-0.20)return'luck-cold2';
  if(v<=-0.08)return'luck-cold1';
  return'';
}
function luckPPPACell(pppa,pct){
  if(pppa==null||isNaN(pppa))return'<td class="num null-cell">—</td>';
  const cls=luckPPPAClass(pppa);
  const sign=pppa>=0?'+':'';
  const pctStr=(pct!=null&&!isNaN(pct))?` <span style="opacity:.7;font-size:.85em">(${pct>=0?'+':''}${(+pct).toFixed(0)}%)</span>`:'';
  return`<td class="num${cls?' '+cls:''}">${sign}${(+pppa).toFixed(3)}${pctStr}</td>`;
}
function numCell(v,dec=2){
  if(v==null||isNaN(v))return'<td class="num null-cell">—</td>';
  return`<td class="num">${(+v).toFixed(dec)}</td>`;
}
function signCell(v,dec=2){
  if(v==null||isNaN(v))return'<td class="num null-cell">—</td>';
  return`<td class="num">${v>=0?'+':''}${(+v).toFixed(dec)}</td>`;
}

function renderLuck(){
  const tbody=document.getElementById('luck-tbody');
  document.getElementById('luck-count').textContent=luckFiltered.length.toLocaleString()+' players';
  if(!luckFiltered.length){
    tbody.innerHTML='<tr><td colspan="15" class="no-results">No players match.</td></tr>';
    return;
  }
  tbody.innerHTML=luckFiltered.map(r=>`<tr>
    <td class="rank">${r.Combined_Rank}</td>
    <td class="name">${r.Name}</td>
    <td>${r.FantasyPos||'—'}</td>
    <td><span class="${lclass(r.Level)}">${r.Level}</span></td>
    <td class="num">${r.Season}</td>
    <td class="num">${r.PA}</td>
    <td class="num">${r.Prior_Career_PA!=null?Math.round(r.Prior_Career_PA):'—'}</td>
    ${numCell(r.BABIP,3)}
    ${numCell(r.BABIP_career,3)}
    ${luckCell(r.BABIP_delta_z)}
    ${r.BABIP_Delta_Slope!=null?signCell(r.BABIP_Delta_Slope,4):'<td class="num null-cell">—</td>'}
    ${r.HRFB_delta_z!=null?luckCell(r.HRFB_delta_z):'<td class="num null-cell">—</td>'}
    ${luckPPPACell(r.Luck_PPPA,r.Luck_PPPA_pct)}
    ${numCell(r.PPPA_Z_SL,2)}
    ${r.PPPA_Jump!=null?signCell(r.PPPA_Jump,2):'<td class="num null-cell">—</td>'}
  </tr>`).join('');
}

function applyLuckFilters(){
  const q=document.getElementById('luck-search').value.trim().toLowerCase();
  const lvl=document.getElementById('luck-level-filter').value;
  const pos=document.getElementById('luck-pos-filter').value;
  const dir=document.getElementById('luck-dir-filter').value;
  const minPA=parseInt(document.getElementById('luck-pa-filter').value)||0;
  luckFiltered=LUCK_RAW.filter(r=>{
    if(q&&!r.Name.toLowerCase().includes(q))return false;
    if(lvl&&r.Level!==lvl)return false;
    if(pos&&r.FantasyPos!==pos)return false;
    if(minPA&&(r.PA??0)<minPA)return false;
    if(dir==='lucky'&&!(r.Luck_PPPA!=null&&r.Luck_PPPA>0))return false;
    if(dir==='unlucky'&&!(r.Luck_PPPA!=null&&r.Luck_PPPA<0))return false;
    return true;
  });
  sortLuck();
}

function sortLuck(){
  const type=document.querySelector(`th[data-luck-col="${luckSortCol}"]`)?.dataset.type;
  luckFiltered.sort((a,b)=>{
    let av=a[luckSortCol],bv=b[luckSortCol];
    if(type==='level'){av=LVL[av]||0;bv=LVL[bv]||0;}
    if(type==='str')return luckSortDir*String(av||'').localeCompare(String(bv||''));
    return luckSortDir*((av??-Infinity)-(bv??-Infinity));
  });
  renderLuck();
}

document.querySelectorAll('th[data-luck-col]').forEach(th=>{
  th.addEventListener('click',()=>{
    const col=th.dataset.luckCol;
    luckSortDir=(luckSortCol===col)?-luckSortDir:(col==='Combined_Rank'?1:-1);
    luckSortCol=col;
    document.querySelectorAll('th[data-luck-col]').forEach(h=>h.classList.remove('sort-asc','sort-desc'));
    th.classList.add(luckSortDir===1?'sort-asc':'sort-desc');
    sortLuck();
  });
});
['luck-search','luck-level-filter','luck-pos-filter','luck-dir-filter','luck-pa-filter'].forEach(id=>{
  document.getElementById(id).addEventListener(
    id==='luck-search'?'input':'change', applyLuckFilters);
});
document.getElementById('luck-clear-btn').addEventListener('click',()=>{
  document.getElementById('luck-search').value='';
  ['luck-level-filter','luck-pos-filter','luck-dir-filter','luck-pa-filter'].forEach(id=>{
    document.getElementById(id).value='';
  });
  applyLuckFilters();
});
document.querySelector('th[data-luck-col="Luck_PPPA"]').classList.add('sort-desc');
applyLuckFilters();

/* ════════════════════════════════════════════
   COMPS TAB
   ════════════════════════════════════════════ */
const COMPS_RAW = COMPS_DATA_PLACEHOLDER;
// Convert object-by-PlayerId to sorted array
const COMPS_ARR = Object.entries(COMPS_RAW).map(([pid,d])=>({...d,pid}))
  .sort((a,b)=>a.rank-b.rank);

let compsSortCol='rank', compsSortDir=1, compsFiltered=COMPS_ARR.slice();
let compsOpenPid=null;

const PROJ_MIN=-1.5, PROJ_MAX=2.5, PROJ_RANGE=PROJ_MAX-PROJ_MIN;

function projPct(v){return Math.max(0,Math.min(100,((v-PROJ_MIN)/PROJ_RANGE)*100));}

function projBarHtml(proj){
  if(!proj||proj.n_grads<2)return'<span style="color:var(--muted);font-size:11px">— not enough comps</span>';
  const fl=proj.floor, med=proj.median, ceil=proj.ceiling;
  if(fl==null||med==null||ceil==null)return'<span style="color:var(--muted);font-size:11px">—</span>';
  const left=projPct(fl), right=projPct(ceil), medPct=projPct(med);
  const width=Math.max(right-left,2);
  const sign=v=>(v>=0?'+':'')+v.toFixed(2);
  return`<div class="proj-bar-wrap">
    <div class="proj-bar-outer">
      <div class="proj-bar-fill" style="left:${left}%;width:${width}%"></div>
      <div class="proj-bar-median" style="left:${medPct}%"></div>
    </div>
    <span class="proj-label">${sign(fl)} / <strong>${sign(med)}</strong> / ${sign(ceil)}</span>
  </div>`;
}

function renderComps(){
  const tbody=document.getElementById('comps-tbody');
  document.getElementById('comps-count').textContent=compsFiltered.length.toLocaleString()+' prospects';
  if(!compsFiltered.length){
    tbody.innerHTML='<tr><td colspan="8" class="no-results">No prospects match.</td></tr>';
    return;
  }
  const rows=[];
  compsFiltered.forEach(d=>{
    const open=d.pid===compsOpenPid;
    const gradPct=d.projection?Math.round(d.projection.grad_pct*100):null;
    const nGrads=d.projection?d.projection.n_grads:0;
    rows.push(`<tr class="comp-row${open?' expanded':''}" data-pid="${d.pid}">
      <td class="rank num">${d.rank}</td>
      <td class="name">${d.name}</td>
      <td>${d.team||'—'}</td>
      <td><span class="${lclass(d.level)}">${d.level||'—'}</span></td>
      <td class="num">${d.age??'—'}</td>
      <td>${projBarHtml(d.projection)}</td>
      <td style="font-size:11px;color:var(--muted)">${d.archetype||'—'}</td>
      <td class="num">${gradPct!=null?gradPct+'%':'—'}</td>
      <td class="num">${nGrads}</td>
    </tr>`);
    if(open){
      rows.push(`<tr class="comp-detail open"><td colspan="8"><div class="comp-detail-inner">${buildDetailHtml(d)}</div></td></tr>`);
    }
  });
  tbody.innerHTML=rows.join('');
  tbody.querySelectorAll('tr.comp-row').forEach(tr=>{
    tr.addEventListener('click',()=>{
      const pid=tr.dataset.pid;
      compsOpenPid=(compsOpenPid===pid)?null:pid;
      renderComps();
    });
  });
}

function fmtZ(v){if(v==null)return'—';return(v>=0?'+':'')+v.toFixed(2);}
function fmtPct(v){if(v==null)return'—';return(v*100).toFixed(1)+'%';}
function fmtN(v,d=0){if(v==null)return'—';return(+v).toFixed(d);}

function buildDetailHtml(d){
  // ── Projection summary ────────────────────────────────────────────────
  const proj=d.projection||{};
  const gradRate=proj.grad_pct!=null?Math.round(proj.grad_pct*100)+'%':'—';
  const hasDist=proj.n_grads>=2;
  let projHtml=`<div class="proj-summary">
    <div class="proj-stat"><span class="proj-stat-label">Grad Rate</span><span class="proj-stat-val" style="color:var(--text)">${gradRate}</span></div>
    <div class="proj-stat"><span class="proj-stat-label">Grads</span><span class="proj-stat-val" style="color:var(--text)">${proj.n_grads??0}/10</span></div>`;
  if(hasDist){
    projHtml+=`
    <div class="proj-stat"><span class="proj-stat-label">Ceiling p75</span><span class="proj-stat-val ceiling">${fmtZ(proj.ceiling)}</span></div>
    <div class="proj-stat"><span class="proj-stat-label">Median p50</span><span class="proj-stat-val median">${fmtZ(proj.median)}</span></div>
    <div class="proj-stat"><span class="proj-stat-label">Floor p25</span><span class="proj-stat-val floor">${fmtZ(proj.floor)}</span></div>`;
  }
  projHtml+='</div>';

  // ── Player profile ────────────────────────────────────────────────────
  let profileRows='';
  (d.profile||[]).forEach(lv=>{
    profileRows+=`<tr>
      <td><span class="${lclass(lv.level)}">${lv.level}</span></td>
      <td>${fmtZ(lv.pppa_z)}</td>
      <td>${fmtN(lv.age,1)}</td>
      <td>${fmtN(lv.pa,0)}</td>
      <td>${lv.bb2k!=null?(lv.bb2k>=0?'+':'')+fmtPct(lv.bb2k):'—'}</td>
      <td>${lv.kpct!=null?fmtPct(lv.kpct):'—'}</td>
      <td>${lv.hrfb!=null?fmtPct(lv.hrfb):'—'}</td>
      <td>${lv.sbtalent!=null?(+lv.sbtalent*100).toFixed(2)+'%':'—'}</td>
    </tr>`;
  });
  const playerSection=`<div class="comp-detail-section">
    <h4>${d.name} — MiLB Profile</h4>
    ${projHtml}
    <table class="mini-table">
      <thead><tr><th>Lvl</th><th>PPPA Z</th><th>Age</th><th>PA</th><th>BB−2K</th><th>K%</th><th>HR/FB</th><th>SB Tal</th></tr></thead>
      <tbody>${profileRows||'<tr><td colspan="8" style="color:var(--muted)">No level data</td></tr>'}</tbody>
    </table>
  </div>`;

  // ── Comp table ────────────────────────────────────────────────────────
  let compRows='';
  (d.comps||[]).forEach((c,i)=>{
    const matchCls=c.match_pct>=75?'match-hi':c.match_pct>=60?'match-med':'';
    const gradCls=c.graduated?'grad-yes':'grad-no';
    const gradBadge=c.graduated
      ?'<span class="grad-badge yes">MLB</span>'
      :'<span class="grad-badge no">MiLB</span>';
    // Show comp's level stats at shared levels (profile entries)
    let statsCell='';
    if(c.profile&&c.profile.length){
      statsCell=c.profile.map(lv=>`<span style="font-size:11px;margin-right:8px"><span class="${lclass(lv.level)}">${lv.level}</span> ${fmtZ(lv.pppa_z)}</span>`).join('');
    }
    compRows+=`<tr>
      <td class="num">${i+1}</td>
      <td>${c.name}</td>
      <td class="num ${matchCls}">${c.match_pct!=null?(+c.match_pct).toFixed(0)+'%':'—'}</td>
      <td class="num">${c.shared??'—'}</td>
      <td>${gradBadge}</td>
      <td class="num ${gradCls}">${fmtZ(c.career_z)}</td>
      <td class="num">${fmtZ(c.firstyr_z)}</td>
      <td style="font-size:11px">${statsCell||'—'}</td>
    </tr>`;
  });
  const compSection=`<div class="comp-detail-section" style="flex:2;min-width:340px">
    <h4>Top 10 Historical Comps</h4>
    <table class="mini-table">
      <thead><tr><th>#</th><th>Name</th><th>Match</th><th>Shared</th><th>Status</th><th>Career Z</th><th>1st Yr Z</th><th>Level PPPA Z</th></tr></thead>
      <tbody>${compRows||'<tr><td colspan="8" style="color:var(--muted)">No comps found</td></tr>'}</tbody>
    </table>
  </div>`;

  return playerSection+compSection;
}

function applyCompsFilters(){
  const q=(document.getElementById('comps-search').value||'').toLowerCase();
  compsFiltered=COMPS_ARR.filter(d=>{
    if(q&&!d.name.toLowerCase().includes(q)&&!(d.team||'').toLowerCase().includes(q))return false;
    return true;
  });
  // sort
  compsFiltered.sort((a,b)=>{
    let av,bv;
    if(compsSortCol==='rank'){av=a.rank;bv=b.rank;}
    else if(compsSortCol==='name'){av=a.name;bv=b.name;return compsSortDir*(av<bv?-1:av>bv?1:0);}
    else if(compsSortCol==='team'){av=a.team||'';bv=b.team||'';return compsSortDir*(av<bv?-1:av>bv?1:0);}
    else if(compsSortCol==='archetype'){av=a.archetype||'';bv=b.archetype||'';return compsSortDir*(av<bv?-1:av>bv?1:0);}
    else if(compsSortCol==='level'){const LV={R:1,A:2,'A+':3,AA:4,AAA:5};av=LV[a.level]||0;bv=LV[b.level]||0;}
    else if(compsSortCol==='age'){av=a.age??999;bv=b.age??999;}
    else if(compsSortCol==='median'){av=a.projection?.median??-99;bv=b.projection?.median??-99;}
    else if(compsSortCol==='grad_pct'){av=a.projection?.grad_pct??-1;bv=b.projection?.grad_pct??-1;}
    else if(compsSortCol==='n_grads'){av=a.projection?.n_grads??-1;bv=b.projection?.n_grads??-1;}
    else{av=a.rank;bv=b.rank;}
    return compsSortDir*(av-bv);
  });
  renderComps();
}

document.getElementById('comps-search').addEventListener('input',applyCompsFilters);
document.querySelectorAll('#comps-table th[data-comps-col]').forEach(th=>{
  th.addEventListener('click',()=>{
    const col=th.dataset.compsCol;
    if(compsSortCol===col)compsSortDir*=-1;
    else{compsSortCol=col;compsSortDir=col==='name'||col==='team'?1:-1;}
    applyCompsFilters();
  });
});
applyCompsFilters();

/* ════════════════════════════════════════════
   TAB SWITCHING
   ════════════════════════════════════════════ */
document.querySelectorAll('.tab-btn').forEach(btn=>{
  btn.addEventListener('click',()=>{
    document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('panel-'+btn.dataset.tab).classList.add('active');
  });
});
</script>
"""

html = HTML \
    .replace('PROSPECTS_DATA_PLACEHOLDER', raw) \
    .replace('SP_DATA_PLACEHOLDER', raw_sp) \
    .replace('RP_DATA_PLACEHOLDER', raw_rp) \
    .replace('AAA_DATA_PLACEHOLDER', raw_aaa) \
    .replace('LUCK_DATA_PLACEHOLDER', raw_luck) \
    .replace('COMPS_DATA_PLACEHOLDER', raw_comps)

SCRATCHPAD.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"Written: {OUT_PATH}")
print(f"Size: {len(html) // 1024} KB")
