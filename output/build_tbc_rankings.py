"""Build data/rankings/tbc_rankings.csv — cleaned TBC prospect rankings.

Reads tbc_prospect_rankings.csv (scraped from thebaseballcube.com) and
produces a tidy file with one row per source-year-rank entry.

Columns retained / derived:
  year         – ranking year (pre-season)
  source       – BA / PIPE / FG / ESPN / BP
  rank         – ordinal rank on that list
  player       – full name (as listed by TBC)
  pos          – position
  born         – birth date (YYYY-MM-DD)
  age_at_rank  – age on Jan 1 of ranking year (prospect age convention)
  draft_year   – year drafted (parsed from draft_info)
  draft_round  – round drafted
  draft_pick   – overall pick
  draft_org    – org that drafted them
  hilvl        – highest MiLB/MLB level ever reached
  made_mlb     – bool: did they reach MLB
  mlb_first    – first MLB season (int, nullable)
  mlb_last     – last MLB season (int, nullable)
  cur_org      – current org at time of TBC snapshot
  cur_lev      – current level at time of TBC snapshot
"""

import re
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IN_PATH  = DATA_DIR / "rankings" / "tbc_prospect_rankings.csv"
OUT_PATH = DATA_DIR / "rankings" / "tbc_rankings.csv"


def parse_draft_info(s):
    """Parse '2024-  1-   9-PIT' → (year, round, pick, org). UDFA → (year, None, None, org)."""
    if pd.isna(s) or not str(s).strip():
        return None, None, None, None
    s = str(s).strip()
    # UDFA pattern: '2024-  UDFA  -LAN'
    udfa = re.match(r'(\d{4})-\s*UDFA\s*-([A-Z]+)', s)
    if udfa:
        return int(udfa.group(1)), None, None, udfa.group(2)
    # Normal: '2024-  1-   9-PIT'
    m = re.match(r'(\d{4})-\s*(\d+)-\s*(\d+)-([A-Z]+)', s)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
    return None, None, None, None


def parse_mlb_years(s):
    """Parse '2018-2023' → (2018, 2023). Empty/nan → (None, None)."""
    if pd.isna(s) or not str(s).strip():
        return None, None
    m = re.match(r'(\d{4})-(\d{4})', str(s).strip())
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def age_on_jan1(born_str, year):
    """Age as of Jan 1 of ranking year — standard prospect age convention."""
    if pd.isna(born_str):
        return None
    try:
        birth_year = int(str(born_str)[:4])
        return year - birth_year
    except (ValueError, TypeError):
        return None


def main():
    df = pd.read_csv(IN_PATH)

    # Parse draft info
    parsed = df["draft_info"].apply(parse_draft_info)
    df["draft_year"]  = [x[0] for x in parsed]
    df["draft_round"] = [x[1] for x in parsed]
    df["draft_pick"]  = [x[2] for x in parsed]
    df["draft_org"]   = [x[3] for x in parsed]

    # Parse MLB years
    mlb = df["mlb_years"].apply(parse_mlb_years)
    df["mlb_first"] = [x[0] for x in mlb]
    df["mlb_last"]  = [x[1] for x in mlb]
    df["made_mlb"]  = df["mlb_first"].notna()

    # Age at ranking year
    df["age_at_rank"] = df.apply(lambda r: age_on_jan1(r["born"], r["year"]), axis=1)

    # Select and order final columns
    out = df[[
        "year", "source", "rank",
        "player", "pos", "born", "age_at_rank",
        "draft_year", "draft_round", "draft_pick", "draft_org",
        "hilvl", "made_mlb", "mlb_first", "mlb_last",
        "cur_org", "cur_lev",
    ]].copy()

    out = out.sort_values(["year", "source", "rank"]).reset_index(drop=True)

    out.to_csv(OUT_PATH, index=False)

    # Summary
    total = len(out)
    by_source = out.groupby("source").size().to_dict()
    yr_min, yr_max = out["year"].min(), out["year"].max()
    made_mlb_pct = out.drop_duplicates("player")["made_mlb"].mean() * 100
    print(f"Wrote {total:,} rows to {OUT_PATH.name}")
    print(f"Years: {yr_min}–{yr_max}")
    print(f"By source: {by_source}")
    print(f"Unique players who made MLB: {made_mlb_pct:.1f}% of distinct names")


if __name__ == "__main__":
    main()
