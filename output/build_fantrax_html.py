import json, pandas as pd
from pathlib import Path

ROOT  = Path(__file__).parent.parent
src   = ROOT / "data" / "rankings_fantrax" / "prospect_scores.csv"
tmpl  = ROOT / "output" / "fantrax_template.html"
out   = ROOT / "output" / "fantrax_rankings.html"

df = pd.read_csv(src)
df = df[~df.get("Below_OVR_Floor", pd.Series(False, index=df.index))]

cols = ["Combined_Rank","Pos_Adj_Rank","Name","Team","Level","Age",
        "FantasyPos","Archetype","TOOLS_Score","ABILITY_Score",
        "Combined_Score","Career_PA","Career_Disc_Flag","Bats"]
keep = [c for c in cols if c in df.columns]
top = df.nsmallest(300, "Combined_Rank")[keep].copy()

# 0-100 rating anchored to OVR historical distribution.
# LOW  = OVR 1st percentile (32.95) — floor of the ranked prospect range.
# HIGH = 107.0 — theoretical ceiling no real player reaches; the best in our
#         database (De Paula, 101.78 old scale) maps to ~93, so 100 stays unattainable.
_LOW, _HIGH = 32.95, 107.0
top["Rating"] = ((top["Combined_Score"] - _LOW) / (_HIGH - _LOW) * 100).round(1)

for c in ["TOOLS_Score","ABILITY_Score","Combined_Score","Age"]:
    if c in top.columns:
        top[c] = top[c].round(2)

records = top.where(top.notna(), None).to_dict(orient="records")
js_data = json.dumps(records, ensure_ascii=False)

tmpl_text = tmpl.read_text(encoding="utf-8")
final = tmpl_text.replace("PLACEHOLDER_DATA", js_data)
out.write_text(final, encoding="utf-8")

print(f"Wrote {len(top)} rows -> {out}")
print(top[["Combined_Rank","Name","Team","Level","ABILITY_Score","Combined_Score","Rating"]].head(5).to_string(index=False))
