"""Fetch Arizona Fall League stats — current season or full history (2005+).

AFL is sportId=17, leagueId=119. Player IDs are MLBAM IDs — joinable
directly to prospect_scores.csv, pitcher_scores.csv, and player_comps.csv.

Pulls both hitting and pitching. Computes PPPA for hitters and PPI_skill
for pitchers so results can be compared against prospect pool scores.

Output:
  data/api/afl_hitting.csv   -- one row per player × season
  data/api/afl_pitching.csv  -- one row per pitcher × season

Usage:
  python fetch/fetch_afl_data.py            # current season (incremental)
  python fetch/fetch_afl_data.py --season 2024
  python fetch/fetch_afl_data.py --full     # all seasons 2005–current
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_HIT  = DATA_DIR / "api" / "afl_hitting.csv"
OUT_PIT  = DATA_DIR / "api" / "afl_pitching.csv"

SPORT_ID   = 17
LEAGUE_ID  = 119
FIRST_YEAR = 2005   # earliest AFL data in API
NO_SEASON  = {2020} # AFL cancelled (COVID)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (compatible; research)"})


def _get(url: str, **params) -> dict:
    r = SESSION.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def _fetch_splits(season: int, group: str, sitcode: str) -> dict:
    """Return {pid: stat_dict} for a given situational split."""
    base = "https://statsapi.mlb.com/api/v1/stats"
    try:
        data = _get(base, stats="season", group=group, season=season,
                    sportId=SPORT_ID, leagueId=LEAGUE_ID,
                    sitCodes=sitcode, limit=500)
        return {s["player"]["id"]: s["stat"]
                for s in data.get("stats", [{}])[0].get("splits", [])}
    except Exception:
        return {}


def _fetch_adv_splits(season: int, group: str, sitcode: str) -> dict:
    """Return {pid: stat_dict} for advanced stats at a given situational split."""
    base = "https://statsapi.mlb.com/api/v1/stats"
    try:
        data = _get(base, stats="seasonAdvanced", group=group, season=season,
                    sportId=SPORT_ID, leagueId=LEAGUE_ID,
                    sitCodes=sitcode, limit=500)
        return {s["player"]["id"]: s["stat"]
                for s in data.get("stats", [{}])[0].get("splits", [])}
    except Exception:
        return {}


def _bip_rates(a: dict) -> dict:
    """Compute LD%/GB%/FB%/POP% from an advanced stat dict."""
    go  = int(a.get("groundOuts",  0)); gh = int(a.get("groundHits", 0))
    fo  = int(a.get("flyOuts",     0)); fh = int(a.get("flyHits",    0))
    lo  = int(a.get("lineOuts",    0)); lh = int(a.get("lineHits",   0))
    po  = int(a.get("popOuts",     0)); ph = int(a.get("popHits",    0))
    total = go + gh + fo + fh + lo + lh + po + ph
    if total == 0:
        return {"LD_pct": None, "GB_pct": None, "FB_pct": None, "POP_pct": None, "GB_FB": None}
    gb = (go + gh) / total
    fb = (fo + fh) / total
    ld = (lo + lh) / total
    pp = (po + ph) / total
    return {
        "LD_pct":  round(ld, 4),
        "GB_pct":  round(gb, 4),
        "FB_pct":  round(fb, 4),
        "POP_pct": round(pp, 4),
        "GB_FB":   round(gb / fb, 3) if fb > 0 else None,
    }


def _hit_split_cols(st: dict, a: dict, suffix: str) -> dict:
    """Extract compact stat set from a hit split for L/R/RISP."""
    pa  = int(st.get("plateAppearances", 0))
    ab  = int(st.get("atBats", 0))
    h   = int(st.get("hits", 0))
    d   = int(st.get("doubles", 0))
    t   = int(st.get("triples", 0))
    hr  = int(st.get("homeRuns", 0))
    bb  = int(st.get("baseOnBalls", 0))
    so  = int(st.get("strikeOuts", 0))
    tb  = int(st.get("totalBases", 0))
    iso = (tb - h) / ab if ab > 0 else None
    sw  = int(a.get("totalSwings", 0))
    wm  = int(a.get("swingAndMisses", 0))
    return {
        f"PA{suffix}":      pa,
        f"AVG{suffix}":     float(st.get("avg", 0) or 0),
        f"OBP{suffix}":     float(st.get("obp", 0) or 0),
        f"SLG{suffix}":     float(st.get("slg", 0) or 0),
        f"K_pct{suffix}":   round(so / pa, 4) if pa > 0 else None,
        f"BB_pct{suffix}":  round(bb / pa, 4) if pa > 0 else None,
        f"ISO{suffix}":     round(iso, 4) if iso is not None else None,
        f"BABIP{suffix}":   float(st.get("babip", 0) or 0) or None,
        f"Whiff_pct{suffix}": round(wm / sw, 4) if sw > 0 else None,
        **{f"{k}{suffix}": v for k, v in _bip_rates(a).items()},
    }


def _pit_split_cols(st: dict, suffix: str) -> dict:
    """Extract compact stat set from a pitching split for L/R."""
    bf  = int(st.get("battersFaced", 0))
    k   = int(st.get("strikeOuts", 0))
    bb  = int(st.get("baseOnBalls", 0))
    return {
        f"BF{suffix}":     bf,
        f"AVG{suffix}":    float(st.get("avg", 0) or 0),
        f"OBP{suffix}":    float(st.get("obp", 0) or 0),
        f"SLG{suffix}":    float(st.get("slg", 0) or 0),
        f"K_pct{suffix}":  round(k / bf, 4) if bf > 0 else None,
        f"BB_pct{suffix}": round(bb / bf, 4) if bf > 0 else None,
        f"BABIP{suffix}":  float(st.get("babip", 0) or 0) or None,
    }


def _parse_ip(ip_str) -> float:
    """'4.2' -> 4.667  (MLB API outs-based notation)"""
    try:
        parts = str(ip_str).split(".")
        return int(parts[0]) + int(parts[1]) / 3 if len(parts) > 1 else float(parts[0])
    except Exception:
        return 0.0


def fetch_hitting(season: int) -> pd.DataFrame:
    base = "https://statsapi.mlb.com/api/v1/stats"

    # Season totals
    data = _get(base, stats="season", group="hitting", season=season,
                sportId=SPORT_ID, leagueId=LEAGUE_ID, limit=500)
    basic = {s["player"]["id"]: s for s in data.get("stats", [{}])[0].get("splits", [])}

    # Advanced (Whiff%, ISO, batted-ball breakdown)
    adv_data = _get(base, stats="seasonAdvanced", group="hitting", season=season,
                    sportId=SPORT_ID, leagueId=LEAGUE_ID, limit=500)
    adv = {s["player"]["id"]: s["stat"] for s in adv_data.get("stats", [{}])[0].get("splits", [])}

    # Situational splits
    vsl       = _fetch_splits(season, "hitting", "vsl")
    vsl_adv   = _fetch_adv_splits(season, "hitting", "vsl")
    vsr       = _fetch_splits(season, "hitting", "vsr")
    vsr_adv   = _fetch_adv_splits(season, "hitting", "vsr")
    risp      = _fetch_splits(season, "hitting", "risp")
    risp_adv  = _fetch_adv_splits(season, "hitting", "risp")

    rows = []
    for pid, s in basic.items():
        st  = s["stat"]
        pl  = s["player"]
        tm  = s.get("team", {})
        a   = adv.get(pid, {})

        pa   = int(st.get("plateAppearances", 0))
        ab   = int(st.get("atBats", 0))
        h    = int(st.get("hits", 0))
        d    = int(st.get("doubles", 0))
        t    = int(st.get("triples", 0))
        hr   = int(st.get("homeRuns", 0))
        r    = int(st.get("runs", 0))
        rbi  = int(st.get("rbi", 0))
        bb   = int(st.get("baseOnBalls", 0))
        ibb  = int(st.get("intentionalWalks", 0))
        hbp  = int(st.get("hitByPitch", 0))
        so   = int(st.get("strikeOuts", 0))
        sb   = int(st.get("stolenBases", 0))
        cs   = int(st.get("caughtStealing", 0))
        tb   = int(st.get("totalBases", 0))
        gidp = int(st.get("groundIntoDoublePlay", 0))
        sf   = int(st.get("sacFlies", 0))
        singles = h - d - t - hr

        tp = (singles + 2*d + 3*t + 4*hr + r + 2*rbi
              + bb + 1.5*ibb + tb - 2*so - 1.5*gidp + 3*sb - 1.5*cs)
        pppa = tp / pa if pa > 0 else None

        sw = int(a.get("totalSwings", 0))
        wm = int(a.get("swingAndMisses", 0))

        row = {
            "MLBAM_ID": pid,
            "Name":     pl["fullName"],
            "Team":     tm.get("name", ""),
            "Season":   season,
            # Counting
            "PA": pa, "AB": ab, "H": h, "1B": singles,
            "2B": d, "3B": t, "HR": hr,
            "R": r, "RBI": rbi, "BB": bb, "IBB": ibb,
            "HBP": hbp, "SO": so, "SB": sb, "CS": cs,
            "TB": tb, "GIDP": gidp, "SF": sf,
            # Computed
            "PPPA":      round(pppa, 4) if pppa is not None else None,
            "AVG":       float(st.get("avg", 0) or 0),
            "OBP":       float(st.get("obp", 0) or 0),
            "SLG":       float(st.get("slg", 0) or 0),
            "ISO":       round((tb - h) / ab, 4) if ab > 0 else None,
            "BABIP":     float(st.get("babip", 0) or 0) or None,
            "K_pct":     round(so / pa, 4) if pa > 0 else None,
            "BB_pct":    round(bb / pa, 4) if pa > 0 else None,
            "SB_pct":    round(sb / (sb + cs), 4) if (sb + cs) > 0 else None,
            "Whiff_pct": round(wm / sw, 4) if sw > 0 else None,
            **_bip_rates(a),
            # L/R/RISP splits
            **_hit_split_cols(vsl.get(pid, {}),  vsl_adv.get(pid, {}),  "_vsL"),
            **_hit_split_cols(vsr.get(pid, {}),  vsr_adv.get(pid, {}),  "_vsR"),
            **_hit_split_cols(risp.get(pid, {}), risp_adv.get(pid, {}), "_RISP"),
        }
        rows.append(row)

    return pd.DataFrame(rows).sort_values("PA", ascending=False).reset_index(drop=True)


def fetch_pitching(season: int) -> pd.DataFrame:
    base = "https://statsapi.mlb.com/api/v1/stats"

    data = _get(base, stats="season", group="pitching", season=season,
                sportId=SPORT_ID, leagueId=LEAGUE_ID, limit=500)
    basic = {s["player"]["id"]: s for s in data.get("stats", [{}])[0].get("splits", [])}

    adv_data = _get(base, stats="seasonAdvanced", group="pitching", season=season,
                    sportId=SPORT_ID, leagueId=LEAGUE_ID, limit=500)
    adv = {s["player"]["id"]: s["stat"] for s in adv_data.get("stats", [{}])[0].get("splits", [])}

    vsl = _fetch_splits(season, "pitching", "vsl")
    vsr = _fetch_splits(season, "pitching", "vsr")

    rows = []
    for pid, s in basic.items():
        st = s["stat"]
        pl = s["player"]
        tm = s.get("team", {})
        a  = adv.get(pid, {})

        ip  = _parse_ip(st.get("inningsPitched", "0"))
        k   = int(st.get("strikeOuts", 0))
        bb  = int(st.get("baseOnBalls", 0))
        er  = int(st.get("earnedRuns", 0))
        hra = int(st.get("homeRuns", 0))
        bf  = int(st.get("battersFaced", 0))
        gs  = int(st.get("gamesStarted", 0))
        g   = int(st.get("gamesPlayed", 0))
        w   = int(st.get("wins", 0))
        l   = int(st.get("losses", 0))
        sv  = int(st.get("saves", 0))
        hld = int(st.get("holds", 0))
        bs  = int(st.get("blownSaves", 0))
        qs  = int(a.get("qualityStarts", 0))

        era    = er / ip * 9 if ip > 0 else None
        k_pct  = k / bf if bf > 0 else None
        bb_pct = bb / bf if bf > 0 else None
        kbb    = (k - bb) / bf if bf > 0 else None
        ppi    = (2*(k/ip) - 0.5*(bb/ip) - 1*(er/ip) - 2*(hra/ip) + 0.75) if ip > 0 else None

        # Whiff% from advanced (direct field, more accurate than swing/miss counts)
        whiff_adv = a.get("whiffPercentage")
        sw = int(a.get("totalSwings", 0))
        wm = int(a.get("swingAndMisses", 0))
        whiff_pct = (float(whiff_adv) / 100 if whiff_adv is not None
                     else (wm / sw if sw > 0 else None))

        row = {
            "MLBAM_ID": pid,
            "Name":     pl["fullName"],
            "Team":     tm.get("name", ""),
            "Season":   season,
            "G": g, "GS": gs, "IP": round(ip, 2),
            "K": k, "BB": bb, "ER": er, "HRA": hra,
            "W": w, "L": l, "SV": sv, "HLD": hld, "BS": bs, "QS": qs,
            "BF": bf,
            "ERA":       round(era, 3) if era is not None else None,
            "BABIP":     float(a.get("babip", 0) or 0) or None,
            "K_pct":     round(k_pct, 4) if k_pct is not None else None,
            "BB_pct":    round(bb_pct, 4) if bb_pct is not None else None,
            "KBB_pct":   round(kbb, 4) if kbb is not None else None,
            "Whiff_pct": round(whiff_pct, 4) if whiff_pct is not None else None,
            "PPI_skill": round(ppi, 4) if ppi is not None else None,
            "Role":      "SP" if g > 0 and gs / g >= 0.5 else "RP",
            **_bip_rates(a),
            **_pit_split_cols(vsl.get(pid, {}), "_vsL"),
            **_pit_split_cols(vsr.get(pid, {}), "_vsR"),
        }
        rows.append(row)

    return pd.DataFrame(rows).sort_values("IP", ascending=False).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--full",   action="store_true", help="Fetch all seasons 2005–current")
    args = ap.parse_args()

    current_year = 2026

    if args.full:
        seasons = [yr for yr in range(FIRST_YEAR, current_year + 1) if yr not in NO_SEASON]
    else:
        seasons = [args.season or current_year]

    all_hit, all_pit = [], []

    # Load existing data for incremental updates
    if not args.full:
        if OUT_HIT.exists():
            existing_hit = pd.read_csv(OUT_HIT)
            seasons_to_skip_hit = set(existing_hit[existing_hit["Season"] < current_year]["Season"].unique())
        else:
            existing_hit = pd.DataFrame()
            seasons_to_skip_hit = set()
        if OUT_PIT.exists():
            existing_pit = pd.read_csv(OUT_PIT)
            seasons_to_skip_pit = set(existing_pit[existing_pit["Season"] < current_year]["Season"].unique())
        else:
            existing_pit = pd.DataFrame()
            seasons_to_skip_pit = set()
    else:
        existing_hit = existing_pit = pd.DataFrame()
        seasons_to_skip_hit = seasons_to_skip_pit = set()

    for season in seasons:
        # Hitting
        if season in seasons_to_skip_hit:
            yr_hit = existing_hit[existing_hit["Season"] == season]
            print(f"  {season} hitting: {len(yr_hit)} rows (cached)")
        else:
            print(f"  {season} hitting…", end=" ", flush=True)
            yr_hit = fetch_hitting(season)
            print(f"{len(yr_hit)} rows")
            if args.full:
                time.sleep(0.3)
        all_hit.append(yr_hit)

        # Pitching
        if season in seasons_to_skip_pit:
            yr_pit = existing_pit[existing_pit["Season"] == season]
            print(f"  {season} pitching: {len(yr_pit)} rows (cached)")
        else:
            print(f"  {season} pitching…", end=" ", flush=True)
            yr_pit = fetch_pitching(season)
            print(f"{len(yr_pit)} rows")
            if args.full:
                time.sleep(0.3)
        all_pit.append(yr_pit)

    hit = pd.concat(all_hit, ignore_index=True).sort_values(["Season","PA"], ascending=[False,False])
    pit = pd.concat(all_pit, ignore_index=True).sort_values(["Season","IP"], ascending=[False,False])

    OUT_HIT.parent.mkdir(parents=True, exist_ok=True)
    hit.to_csv(OUT_HIT, index=False)
    pit.to_csv(OUT_PIT, index=False)
    print(f"\nWrote {len(hit)} hitter rows → {OUT_HIT}")
    print(f"Wrote {len(pit)} pitcher rows → {OUT_PIT}")

    # Quick summary for current season only
    if len(seasons) == 1:
      try:
        scores = pd.read_csv(DATA_DIR / "rankings" / "prospect_scores.csv")
        p_scores = pd.read_csv(DATA_DIR / "rankings" / "pitcher_scores.csv")
        hit = hit[hit["Season"] == seasons[0]]
        pit = pit[pit["Season"] == seasons[0]]

        hit_prospects = hit.merge(
            scores[["MLBAM_ID","Combined_Rank","Level","Team"]].rename(columns={"Team":"MiLB_Team","Level":"MiLB_Level"}),
            on="MLBAM_ID", how="inner"
        ).sort_values("PA", ascending=False)
        print(f"\n  {len(hit_prospects)} hitters are current prospects:")
        for _, r in hit_prospects[hit_prospects["PA"] >= 5].head(15).iterrows():
            pppa_str = f"{r['PPPA']:.3f}" if pd.notna(r['PPPA']) else "N/A"
            print(f"    #{r['Combined_Rank']:>3}  {r['Name']:<25}  {r['PA']:>3} PA  PPPA {pppa_str}  ({r['MiLB_Level']})")

        pit_prospects = pit.merge(
            p_scores[["MLBAM_ID","Combined_Score","SP_Rank","RP_Rank"]],
            on="MLBAM_ID", how="inner"
        ).sort_values("IP", ascending=False)
        print(f"\n  {len(pit_prospects)} pitchers are current prospects:")
        for _, r in pit_prospects[pit_prospects["IP"] >= 1].head(15).iterrows():
            rank_str = f"SP#{int(r['SP_Rank'])}" if pd.notna(r['SP_Rank']) else f"RP#{int(r['RP_Rank'])}"
            era_str  = f"{r['ERA']:.2f}" if pd.notna(r['ERA']) else "N/A"
            k_str    = f"  K% {r['K_pct']:.1%}" if pd.notna(r['K_pct']) else ""
            print(f"    {rank_str:>6}  {r['Name']:<25}  {r['IP']:>5.1f} IP  ERA {era_str}{k_str}")

      except Exception as e:
        print(f"  (skipping prospect join: {e})")


if __name__ == "__main__":
    main()
