# firstModel.md

Reference documentation for prospectsMain **v0.1–v0.6** (the original Phase 1–4 model). Superseded by `CLAUDE.md` (v1.0 overhaul, August 2026).

---

## Version History

### v0.1 — Initial PPPA Model
PA-weighted PPPA z-scores blended with BW scout rankings. No regression; no skill decomposition. Two signals: raw PPPA z-score (60%) and BW scout ranking (40%). `compute_stats.py` computed TB, TP, PPG, PPPA, and z-scores from raw counting stats. `player_scores.csv` produced `50 + 10 × (0.6×PPPA_Z − 0.4×Age_Z_SL)` per player. BW_Rank was a blend input at this stage.

### v0.2 — Phase 1: Skill Score
Introduced regression-derived skill scoring to replace the raw PPPA z-score. WLS regression (`regression_predictive.csv`) predicts next-season PPPA from same-season process stats — BB/K, K%, GB/FB, SwStr%, ISO, Age_Z_SL — weighted by min(PA_T, PA_T+1). Each season gets its own coefficients so the weights adapt automatically to offensive-era drift. PA-weighted career average of era-appropriate scores → `skill_scores.csv` (50±10). wRC+ explicitly excluded — it is an outcome stat, not a process signal.

### v0.3 — Phase 2: MLB Outcomes Regression
WLS regression (`regression_mlb_outcomes.csv`) predicting first-year MLB PPPA_Z from PA-weighted pre-debut MiLB stats. 8 predictors selected via a 512-combination ablation study: BB/K, K%, GB/FB, Age_Z_SL, wRC+, ISO (level averages) + K%_delta (trajectory between first and last pre-debut season) + PPPA_Z_SL_best (arc — best peer-relative PPPA season, A-ball+ only). N=1,612; R²=0.128. Applied to current prospects → `prospect_mlb_proj.csv`. A 3× age boost applied to the Age_Z_SL coefficient when scoring current prospects (trained coefficient underestimates age importance due to survivorship bias). PA shrinkage by level: AAA/AA=150, A+/A=250, A-/R=400, DSL/CPX=550 full-confidence threshold.

### v0.4 — Phase 3: Four-Signal Blend
Unified ranking integrating all signals: `Combined_Score = 0.40×Skill_Score + 0.25×MLB_Proj_Score + 0.22×PPPA_Score + 0.13×Age_Score`. All signals re-standardized within the pool to 50±10 before blending. PPPA_Score used SB translation level discounts (AAA=1.00, AA=0.84, A+=0.71, A=0.57, R=0.39). Age_Score: two-component penalty — level-discounted Age_Z_SL + absolute age-for-level benchmark (0.5 × years above target age). BW_Rank demoted to reference-only column; excluded from blend. Output: `2026ProspectPool.csv` (~3,046 players). MLB exclusion threshold raised from any MLB appearance to ≥50 career MLB PA to allow cup-of-coffee players to remain prospect-eligible.

### v0.5 — Phase 4: Non-Linear Arc Features
Added peak and trajectory features to Phase 2. Best-season peer-relative PPPA (PPPA_Z_SL_best — peer-relative to control for SB environment inflation across levels) retained as the sole new arc predictor after ablation. K%_best, K%_slope, K%_arc_sd tested but non-significant once fixed predictors controlled for. SB_rate_best added then removed — statistically significant bivariate relationship (+0.512 WLS) masked by BB% suppression, not a true independent signal. Arc features restricted to A-ball and above — DSL/CPX Rookie peaks are environment artifacts. `milb_impact_analysis.py` upgraded to 34 features (30 original + 4 Phase 4 arc columns) with XGBoost 5-fold CV + SHAP.

### v0.6 — SB Translation Analysis + AAA/Rk Rankings
`sb_translation_analysis.py` derived chained level-to-level SB discount factors empirically (2023+ era): step medians R→A=0.69, A→A+=0.80, A+→AA=0.85, AA→AAA=0.84, AAA→MLB=0.69. Chained to MLB (normalized AAA=1.0): AA=0.84, A+=0.71, A=0.57, R=0.39. These replaced the earlier estimated level weights in PPPA_Score. Direct-jump validation: chained AAA→MLB (0.690) matched direct-jump median (0.689) within 0.001. Added `build_aaa_rankings.py` (ProspectSavant-based AAA power rankings, 295 players sorted by Overall composite) and `build_rk_rankings.py` (Rookie ball rankings). Added TBC org-level rankings (`build_tbc_rankings.py`, `_tbc_org_ranks_for_year()`). Added player cards (`generate_player_card.py`, `generate_player_card_cat.py`, `regen_all_cards.py`).

**Succeeded by v1.0 (TOOLS_Score + ABILITY_Score overhaul). See `CLAUDE.md`.**

---

## Project Overview

**prospectsMain** is a baseball prospect analysis tool built around minor league player statistics. It scores prospects using a combination of statistical performance (PA-weighted z-scores) and a trusted scout's rankings (BW rankings), geared toward points-based fantasy leagues.

**Critical framing: the target is PPPA, not MLB success.** This distinction matters throughout every phase of the model. wRC+ and WAR measure general MLB offensive value and are reasonable proxies, but they diverge from fantasy point production in important ways:
- Strikeouts cost -2 points in this scoring system — roughly 3× the penalty implied by wRC+. K% is the single most impactful negative signal.
- Stolen bases score +3 points each. SB rate is a major positive signal almost entirely absent from wRC+ and WAR.
- A player can rank #2 in overall fantasy value while ranking #21 in wRC+; a player can rank #6 overall while ranking #11 in wRC+. The divergence is real and significant.

When evaluating research, calibrating models, or choosing proxy metrics: always ask whether the signal translates to PPPA specifically, not just to general MLB success. Prefer PPPA trajectory over wRC+ trajectory, K% over OPS, and weight SB-related signals explicitly.

## Environment

- Python 3.11 (virtual environment configured via PyCharm)
- Dependencies: `requirements.txt` (pandas>=2.0, statsmodels>=0.14, xgboost>=2.0, shap>=0.44, scikit-learn>=1.3)

## Data

`data/minorLeagueData.csv` — **derived output** generated by `compute_stats.py` from the MLB Stats API pull (`data/api/milb_hitting.csv`). Do not edit manually. Contains counting stats + all derived columns (TB, TP, PPG, PPPA, z-scores). PlayerId uses numeric IDs (MLBAM or FanGraphs via Chadwick crosswalk). All other data sources (historical_ml_advanced, historical_ml_batted, missing_milb_data) use sa-prefix IDs for recent players — ID mismatch is expected and handled via Name+Season+Team+Level fallback joins in `build_ovr_hist_data()`.

**Schema (raw input columns — sourced from `data/api/milb_hitting.csv`):**

| Column   | Description                                      |
|----------|--------------------------------------------------|
| PlayerId | Unique player identifier                         |
| Season   | Year                                             |
| Name     | Player name                                      |
| Team     | MLB organization abbreviation                    |
| Level    | Minor league level: R, A, A+, AA, AAA            |
| Age      | Player age during season                         |
| G        | Games played                                     |
| PA       | Plate appearances                                |
| 1B–HR    | Hit types (singles, doubles, triples, home runs) |
| R        | Runs scored                                      |
| RBI      | Runs batted in                                   |
| BB       | Walks                                            |
| IBB      | Intentional walks                                |
| SO       | Strikeouts                                       |
| GIDP     | Grounded into double plays                       |
| SB       | Stolen bases                                     |
| CS       | Caught stealing                                  |

**Derived columns added by `compute_stats.py`:**

| Column      | Description                                                  |
|-------------|--------------------------------------------------------------|
| League      | League name (from Leagues/ CSVs); unmatched rows = DiscLeague |
| TB          | Total bases                                                  |
| TP          | Total points (fantasy scoring formula)                       |
| PPG         | Points per game                                              |
| PPPA        | Points per plate appearance                                  |
| PPG_Z       | PPG z-score vs. Season+League+Age peers                      |
| PPPA_Z      | PPPA z-score vs. Season+League+Age peers                     |
| PPG_Z_SL    | PPG z-score vs. Season+League peers                          |
| PPPA_Z_SL   | PPPA z-score vs. Season+League peers                         |
| Age_Z_SL    | Age z-score vs. Season+League peers (negative = younger)     |

A single player can appear multiple times per season (across levels/teams) and across multiple seasons. PlayerId is the stable key for tracking a player's career trajectory.

### Other data files

- `data/averages_season_league.csv` — Season+League baseline averages.
- `data/averages_season_league_age.csv` — Season+League+Age baseline averages.
- `data/Leagues/` — one CSV per league used to assign the League column. Pre-2021 rookie ball leagues are discontinued; unmatched rows are assigned DiscLeague.
- `data/rankings/bwRankings.csv` — 2026 prospect rankings from a trusted scout (tools-heavy, not performance-based). Columns: Rank, Name, Team, Pos, Age, Report. BW_Rank is **reference-only** — never included in the Combined_Score blend.
- `data/rankings/player_scores.csv` — one row per player; output of `compute_stats.py`. Columns: PlayerId, Name, Seasons, Season_Range, Total_PA, Score.
- `data/rankings/2026ProspectPool.csv` — output of `build_prospect_pool.py`. All players active in 2025 or 2026 (Season ≥ 2025). ~2,947 players. Reference columns (not in blend): BW_Rank, BA_Rank (national top-100), PIPE_Rank (national top-100), BA_Org_Rank (org-level franchise rank), PIPE_Org_Rank. BA/PIPE_Org_Rank covers ~367/406 players vs ~51 for national lists.
- `data/rankings/bw_vs_model.csv` — comparison of all 309 BW-ranked players against Combined_Rank. Columns: BW_Rank, BW_Name, BW_Team, Pos, BW_Age, In_Pool, Combined_Rank, Delta, Pool_Name, Pool_Team, Level, Skill_Score, MLB_Proj_Score, PPPA_Score, Age_Score, Combined_Score. Delta = BW_Rank − Combined_Rank (positive = we rank higher). 207 matched, 102 not in pool.
- `data/tbc_team_rankings_2026.csv` — TBC franchise (org-level) prospect lists scraped from thebaseballcube.com (requires premium account). 1,770 rows: 870 BA + 900 PIPE across all 30 MLB orgs for 2026. Columns: year, source, team_id, rank (org rank 1-N), mlb_rank (global/national rank if also on national list), player, pos, ht, wt, ba, th, born, place, hilvl, mlb_years, stat_years, draft_info, status, cur_org, cur_lev. Scraped via batched browser fetch with 1.5s delays. Regenerate for new season by scraping `prospects_team_year/{year}~{teamId}~{source}/` for all 30 team IDs. Team IDs: 2=ARI, 3=ATL, 4=BAL, 5=BOS, 6=CHC, 7=CWS, 8=CIN, 10=COL, 11=DET, 13=HOU, 14=KC, 15=LAD, 16=MIL, 17=MIN, 19=NYM, 20=NYY, 22=PHI, 23=PIT, 24=SD, 25=SF, 26=SEA, 27=STL, 28=TB, 29=TEX, 30=TOR, 32=WSH, 34=MIA, 35=CLE, 36=LAA, 37=OAK.
- `data/rankings/tbc_vs_model.csv` — all prospect pool players with any TBC ranking (national top-100 or org-level). ~449 players. Columns: Combined_Rank, Name, Team, Level, BA_Rank (national, null if not top-100), BA_Delta, PIPE_Rank (national), PIPE_Delta, BA_Org_Rank (within-org rank from franchise list), PIPE_Org_Rank, TBC_Name, Pos, Born, Age_At_Rank, Skill_Score, MLB_Proj_Score, PPPA_Score, Age_Score, Combined_Score. Delta = national_rank − Combined_Rank (positive = model ranks higher than scouts). Sorted by Combined_Rank.
- `data/ml_updated_data.csv` — FanGraphs-sourced 2026 season rate stats. 3,001 rows, all Season=2026. Columns: K%, BB/K, ISO, wRC+, SwStr%, GB/FB, PA, PlayerId, MinorMasterId, MLBAMID, plus others. Overrides missing_milb_data.csv for all 2026 rows in ovr_hist_data. Join is on accent-normalized Name+Season+Level (~2,947 rows matched).
- `data/ps_AAA_2026.csv` — ProspectSavant export of 2026 AAA hitters. 300 players, 69 columns. Statcast metrics not in main pipeline: xwOBA, xBA, xSLG, Barrel%, EV variants, Chase%, Hard Hit%, launch angle, wBsR, PS Score (0–1 percentile composite). Columns without `.1` suffix are percentile ranks; `.1` suffix are raw values. Primary source for `build_aaa_rankings.py`.
- `data/rankings/aaa_2026.csv` — output of `build_aaa_rankings.py`. 295 players (PS players minus MLB veterans with ≥50 career MLB PA). Sorted by Overall score descending. See `build_aaa_rankings.py` for full column list and composite score methodology.
- `data/historical/` — historical analysis files (see Historical Data section below).
- `data/historical_ml_advanced.csv` — historical MiLB advanced rate stats per player-season (FanGraphs-sourced). Columns: Season, Name, Team, Level, Age, PA, BB%, K%, BB/K, AVG, OBP, SLG, OPS, ISO, Spd, BABIP, wSB, wRC, wRAA, wOBA, wRC+, PlayerId. Pipeline reads BB/K, K%, ISO (wRC+ now sourced from missing_milb_data.csv).
- `data/historical_ml_batted.csv` — historical MiLB batted ball stats per player-season (FanGraphs-sourced). Columns: Season, Name, Team, Level, Age, PA, BABIP, GB/FB, LD%, GB%, FB%, IFFB%, HR/FB, Pull%, Cent%, Oppo%, SwStr%, Balls, Strikes, Pitches, PlayerId.
- `data/missing_milb_data.csv` — FanGraphs-sourced stats unavailable from the MLB Stats API. 77,805 rows, Seasons 2006–2026. Columns: Season, Name, Team, Level, Age, wRC+, wSB, BB/K, wOBA, wRAA, SwStr%, HR/FB, PlayerId. Standard levels (R/A/A+/AA/AAA) plus non-standard (A-, DSL, CPX — excluded from pipeline). Non-standard levels account for ~20.5% of rows. PlayerId uses FanGraphs numeric IDs. Pipeline reads wRC+, BB/K, K%, ISO, SwStr%, GB/FB (all primary sources — updated more frequently for current season than historical CSVs); wSB, wOBA, wRAA, HR/FB not yet integrated into the model.
- `data/api/` — MLB Stats API-sourced data (see API Data section below). Covers 2006–2026, all levels, PA ≥ 10. PlayerId uses FanGraphs ID where a Chadwick crosswalk exists, otherwise MLBAM ID.

## Scripts

### `run_pipeline.py` — **run this after updating minorLeagueData.csv**
Single entry point that runs the full pipeline in order:
1. `compute_stats.py` — recompute derived cols, z-scores, player_scores.csv
2. Rebuild `ovr_hist_data.csv` from minorLeagueData + advanced + batted + missing_milb sources
3. Override 2026 rows in ovr_hist_data with `data/ml_updated_data.csv` (FanGraphs 2026 stats, joined on accent-normalized Name+Season+Level)
4. Rebuild `regression_season_league.csv` (descriptive OLS, reference only)
5. Rebuild `regression_predictive.csv` (WLS predictive regression per season)
6. Rebuild `skill_scores.csv` (Phase 1 skill score)
7. Rebuild `regression_mlb_outcomes.csv` (Phase 2 WLS: MiLB skills → first-year MLB PPPA_Z)
8. Rebuild `prospect_mlb_proj.csv` (apply Phase 2 coefficients to all current players)
9. `build_prospect_pool.py` — Phase 3 unified ranking (Skill + MLB_Proj + PPPA + Age)
- `_name_fill()` uses accent-stripped normalized names for all Name-based fallback joins — fixes matching for players with accented names (e.g. Jesús Made, Luis Peña). Uses `unicodedata.NFD` decomposition + Mn category filter.

### `fetch_milb_data.py` — **run to refresh API-sourced data**
Standalone script that pulls MiLB batting stats from the MLB Stats API. Run independently of `run_pipeline.py` (operates on different data sources). It:
1. Downloads and caches the Chadwick Bureau register (`data/api/chadwick.csv`) for MLBAM ↔ FanGraphs ID crosswalk
2. **Incremental mode** (default when output files exist): strips the current season from saved files and re-fetches only that season (~30 seconds). Full history fetch only runs when no output files exist.
3. For each fetched season × level (AAA/AA/A+/A/R): fetches `stats=season` (counting) and `stats=seasonAdvanced` (rate/batted ball)
4. Builds MLB org abbreviation and league name lookups per season
5. Filters to PA ≥ 10 **and Age ≤ 26** — cap raised from 24 to include age 25-26 AAA players for `build_aaa_rankings.py`; `build_prospect_pool.py` has an explicit `MAX_PROSPECT_AGE = 24` guard so they never appear in prospect rankings
6. Crosswalks MLBAM IDs to FanGraphs IDs via Chadwick; MLBAM-only for unmapped players
7. Writes `data/api/milb_hitting.csv` and `data/api/milb_advanced.csv`
8. Fetches birth dates from the MLB Stats API (`/api/v1/people`) for all players in the current season pull; cached incrementally in `data/api/player_birthdays.csv` — only fetches IDs not already present
- 2020 season returns no data (MiLB cancelled due to COVID)
- Age filter is also applied retroactively to existing rows when loading in incremental mode
- To advance to a new season year: update `SEASONS = list(range(2006, {new_year+1}))` at the top of the script; `CURRENT_SEASON` derives from `SEASONS[-1]` automatically
- **Not available from API** (still require FanGraphs): wRC+, wOBA, wRAA, wRC, Spd, wSB, Pull%, Cent%, Oppo%

### `compute_stats.py`
Called by `run_pipeline.py`. Can also be run standalone. It:
1. Reads raw counting stats from `data/api/milb_hitting.csv` (automated API fetch)
2. Computes derived columns: TB, TP, PPG, PPPA
3. Preserves League directly from the API data (full league names e.g. "Arizona Complex League")
4. Computes Season+League and Season+League+Age averages/std devs
5. Attaches z-scores (all grouped by League, not Level)
6. Writes derived output to `data/minorLeagueData.csv` (this file is generated, not a source)
7. Writes `data/rankings/player_scores.csv`

### `milb_impact_analysis.py` — **standalone; run independently to study feature importance**
XGBoost + SHAP analysis of which MiLB stats predict MLB success. Not part of `run_pipeline.py`. Pulls from `historical_ml_advanced.csv`, `historical_ml_batted.csv`, and `ovr_hist_data.csv`; outcomes from `hist_mlb_data.csv`. It:
1. Builds PA-weighted MiLB feature averages across all pre-debut seasons per player (30 features total)
2. Computes trajectory deltas (first vs. last pre-debut season) for BB%, K%, wRC+, ISO, PPPA
3. Adds context features: total MiLB PA, seasons, max level reached, debut year
4. Trains XGBoost with 5-fold CV + early stopping to prevent overfitting (min_child_weight=20, L1/L2 regularization)
5. Computes SHAP values on the full model for feature importance
6. Runs two separate models — first-year and career — and saves ranked importance CSVs
- **Outcome 1**: first-year MLB PPPA (MLB PA ≥ 100); N≈920; OOF R²≈0.105
- **Outcome 2**: career MLB PPPA, PA-weighted across all seasons (career PA ≥ 400, seasons ≥ 2); N≈1,017; OOF R²≈0.222
- Outputs: `data/historical/stat_impact_firstyear.csv`, `data/historical/stat_impact_career.csv`
- 34 features total (30 original + 4 Phase 4: K%_best, PPPA_Z_SL_best, K%_slope, K%_arc_sd; SB_rate_best removed — see below)
- **Key findings (post-Phase 4, June 2026):**
  - Raw PPPA #1 both outcomes (level-compression effect — not noise; PPPA_Z_SL handles this)
  - **K%** #2 first-year, #4 career — single most consistent actionable predictor
  - **Age_Z_SL** #10 first-year, #2 career — far more important for career arc than first-year translation
  - **PPPA_best** (Phase 4) #4 first-year, #11 career — strongest new feature; peak fantasy season is a real ceiling signal. Now implemented as PPPA_Z_SL_best (peer-relative) rather than absolute PPPA to control for SB environment inflation across levels.
  - **K%_best** (Phase 4) #7 both outcomes — consistent across first-year and career
  - **K%_arc_sd** (Phase 4) #11 first-year, #8 career — K% volatility is a significant negative signal; inconsistent K% profiles predict worse outcomes
  - **K%_slope** (Phase 4) #17 first-year, #27 career — trend direction adds modest signal; negative direction = improving K% = better outcomes
  - **SB_rate_best** (Phase 4) #27 first-year, #14 career — removed from Phase 2 regression (see below)
  - Career outcomes remain ~2× more predictable than first-year (OOF R² 0.222 vs 0.105)
  - HR/FB remains the strongest consistently positive non-arc signal
- **SB_rate_best interpretation (moderation/mediation analysis, June 2026):**
  - The negative SHAP direction for SB_rate_best is a **multicollinearity/suppression artifact**, not a true negative relationship. The bivariate WLS coefficient for SB_rate_best → MLB PPPA_Z is +0.512 (positive); controlling for the full feature set makes it even more positive (+0.634).
  - **BB% is a suppressor, not a mediator.** High SB players also have high BB% (path a: p<0.0001), but BB% carries a negative coefficient in the full model because wRC+ already captures disciplined contact hitters. Adding BB% as a control suppresses the visible SB coefficient downward.
  - **No significant linear moderation** by K%_best (p=0.72) or Age_Z_SL (p=0.54) was found via WLS interaction terms.
  - **The K%/SB interaction is real but non-linear.** Among high-SB players (above median SB_rate_best), K%_best quartile predicts a 0.47 PPPA_Z gap: Q1 (low K%, best contact) = +0.135 PPPA_Z, Q4 (high K%) = −0.334 PPPA_Z. XGBoost tree splits capture this shape; linear WLS with N=1,608 cannot.
  - **Practical conclusion:** Low K% + high SB = elite fantasy ceiling. High K% + high SB = risky — the K% penalty (−2/SO) dominates any SB upside. SB_rate_best is a positive signal conditional on K%_best; it is not a standalone red flag.

### `build_prospect_pool.py`
Called by `run_pipeline.py` as the final step (after all scoring signals are built). It:
1. Takes all minorLeagueData players whose most recent season ≥ 2025 (active in 2025 or 2026)
2. Deduplicates to one row per player (most recent season, most PA if tied)
3. Left-joins BW_Rank — non-BW players get null BW_Rank but are included and scored normally
4. Loads Skill_Score (Phase 1) and MLB_Proj_Score (Phase 2)
5. Re-standardizes all signals within the pool to 50±10 scale
6. Computes PPPA_Score, Age_Score, Combined_Score, and Combined_Rank
7. Writes to `data/rankings/2026ProspectPool.csv` (~3,046 players)
- `normalize_name()` strips accents via `unicodedata.NFD` decomposition — fixes matching for players with accented names (e.g. Jesús Made, Luis Peña). Applied to both BW join and MLB exclusion filter.
- MLB exclusion threshold: ≥50 career MLB PA (via `hist_mlb_data.csv`). Filters on both PlayerId and normalized Name to handle ID system mismatches. Previously excluded any MLB appearance; raised to 50 PA to allow cup-of-coffee players to remain prospect-eligible.
- `MAX_PROSPECT_AGE = 24` guard explicitly filters out age 25-26 players — pipeline now fetches up to Age 26 for AAA rankings, but they must not appear in the prospect pool.
- `_born_name_lookup()` — shared helper that builds (hit_bday, hit_name) lookup frames from player_birthdays.csv + milb_hitting.csv; used by both `_tbc_ranks_for_year()` and `_tbc_org_ranks_for_year()`.
- `_tbc_org_ranks_for_year(year)` — reads `data/tbc_team_rankings_{year}.csv` (scraped franchise lists) and resolves org-level BA_Org_Rank / PIPE_Org_Rank to PlayerId via born+name primary / name-only fallback. Returns (ba_org_df, pipe_org_df).
- Output columns include `BA_Org_Rank` and `PIPE_Org_Rank` (org-level rankings within team's BA/PIPE list, 1-N) alongside `BA_Rank` and `PIPE_Rank` (national top-100 ranks).
- `tbc_vs_model.csv` expands to ~449 players (all pool players with any TBC ranking — national or org-level) vs 51 with national-only lists. Delta columns (BA_Delta, PIPE_Delta) only computed when national rank is available.

### `build_aaa_rankings.py` — **standalone; run independently to refresh AAA power rankings**
Builds `data/rankings/aaa_2026.csv` — a 2026 AAA power rankings table focused on current-season performance. Run independently of `run_pipeline.py`. It:
1. Loads `ps_AAA_2026.csv` (ProspectSavant, 300 players) as the base roster
2. Joins exact birth dates from `data/api/player_birthdays.csv` via `milb_hitting.csv` (MLBAM_ID lookup) to compute decimal ExactAge
3. Joins PPPA, SB, CS from `minorLeagueData.csv` (2026 AAA rows only) by normalized name
4. Computes derived pipeline stats: SB_rate (SB/PA), SB_succ (SB/(SB+CS)), BB%-K% (BB%.1 − K%.1)
5. Computes percentile ranks (0–100) for all stat columns; Chase% and SwStr% inverted (lower = better)
6. Builds four sub-composite scores (all inputs are percentile ranks on 0–100 scale):
   - **Discipline**: BB%-K%(50%) + Chase%(30%) + SwStr%(20%)
   - **Power**: MaxEV(14%) + EV90(14%) + Barrel%(27%) + PullAir%(10%) + xSLG(35%)
   - **Contact**: xBA(50%) + HardHit%(20%) + ZContact%(30%)
   - **Speed**: Spd(40%) + SB_rate(35%) + SB_succ(25%)
7. Builds Overall composite: PPPA_R(42%) + Discipline(22%) + Power(13%) + Speed(12%) + Contact(6%) + Age_R(5%)
8. Excludes players with ≥50 career MLB PA (joined from `hist_mlb_data.csv` by normalized name)
9. Writes 245 players sorted by Overall descending

**Column order:** Name, Team, Age, ExactAge, Age_R, PA, PPPA, PPPA_R, Overall, Overall_R, Discipline, Discipline_R, Power, Power_R, Contact, Contact_R, Speed, Speed_R, then individual stat columns each followed by their `_R` percentile rank.

**ProspectSavant column convention:** columns without `.1` suffix are PS percentile ranks (0–1); `.1` suffix are raw values. Exception: `PullAir%` is raw with no .1 version; `Barrel%` in PS is a percentile — raw barrel rate is stored as `PA/Barrels` (renamed to `Barrel%` in output).

**Overall score rationale (power rankings — current 2026 production, not prospect ceiling):** PPPA dominates at 42% because this is a snapshot ranking; Age is minimal at 5% because a 26yo raking deserves the same ranking credit as a 21yo with identical output in this context.

### `sb_translation_analysis.py` — **standalone; run independently to study SB level translation**
Chained level-to-level SB translation analysis. For each player, finds every one-step promotion (R→A, A→A+, A+→AA, AA→AAA, AAA→MLB) and compares SB/PA before and after. Chains step-level medians to derive discount factors from any level to MLB. Outputs:
- `data/historical/sb_translation_log.csv` — one row per player-promotion (unfiltered)
- `data/historical/sb_translation_rates.csv` — level-pair × season aggregates (From_SB ≥ 5 filter)
- `data/historical/sb_translation_eras.csv` — level-pair × era aggregates
- `data/historical/sb_translation_chain.csv` — chained discount factors level→MLB by era
**Key findings (2023+ era, From_SB ≥ 5):** Step medians: R→A=0.69, A→A+=0.80, A+→AA=0.85, AA→AAA=0.84, AAA→MLB=0.69. Chained to MLB: AAA=0.69, AA=0.58, A+=0.49, A=0.39, R=0.27. Normalized to AAA=1.0: AA=0.84, A+=0.71, A=0.57, R=0.39. These normalized factors are used as PPPA_Z_SL level weights in build_prospect_pool.py. Direct-jump validation: chained AAA→MLB (0.690) matches direct-jump median (0.689) within 0.001; lower-level direct estimates were inflated by selection bias.

## Scoring Methodology

### Fantasy points (TP)
| Stat  | Weight | Stat | Weight |
|-------|--------|------|--------|
| 1B    | +1     | SO   | -2     |
| 2B    | +2     | GIDP | -1.5   |
| 3B    | +3     | SB   | +3     |
| HR    | +4     | CS   | -1.5   |
| R     | +1     | TB   | +1     |
| RBI   | +2     |      |        |
| BB    | +1     |      |        |
| IBB   | +1.5   |      |        |

### Player Score (player_scores.csv)
PA-weighted mean of `(0.6 × PPPA_Z) − (0.4 × Age_Z_SL)` across all seasons.
- Age_Z_SL is subtracted because being younger than peers is a positive prospect signal.
- Scaled to `50 + 10 × raw_score` (50 = average, ±10 = ±1 std dev). Clipped at 0.
- **Weighting: 60% PPPA performance, 40% age relative to league peers.**

### Combined Score (2026ProspectPool.csv) — Phase 3
Four-signal blend. All signals re-standardized within the pool to 50±10 before blending:
- **Skill_Score** (40%): BB/K, K%, GB/FB, SwStr%, ISO, Age_Z_SL weighted by era-specific regression coefficients (Phase 1).
- **MLB_Proj_Score** (25%): Phase 2 WLS coefficients applied to each prospect's recent MiLB stats + trajectory, with 3× age boost and PA shrinkage applied (Phase 2).
- **PPPA_Score** (22%): PA-weighted career PPPA_Z_SL with level discounts derived from empirical SB translation factors (2023+ era, normalized to AAA=1.0): AAA=1.00, AA=0.84, A+=0.71, A=0.57, R=0.39. Weight per season = PA × level_weight. Players missing PPPA_Z_SL data receive 50 (pool neutral).
- **Age_Score** (13%): Two-component age penalty. (1) Level-discounted Age_Z_SL (R=0.40×, A=0.60×, A+=0.75×, AA/AAA=1.00×). (2) Absolute age-for-level penalty: 0.5 × years above target age (R=18, A=19, A+=20, AA=21, AAA=22). Combined, inverted, and standardized to 50±10.
- `Combined_Score = 0.40 × Skill_Score + 0.25 × MLB_Proj_Score + 0.22 × PPPA_Score + 0.13 × Age_Score`
- Players missing a signal (no advanced stat coverage) receive 50 (pool neutral) for that component.
- BW_Rank and Score (PPPA_Z-based) retained as reference columns but excluded from the blend. BW_Rank is null for players not in the BW scout list.
- Output columns: PlayerId, Name, Team, Level, Age, PA, BW_Rank, Score, Score_Rank, Skill_Score, MLB_Proj_Score, PPPA_Score, Age_Score, Combined_Score, Combined_Rank.
- Pool includes all players active in 2025 or 2026 (~3,046 players). Not gated by BW rankings.

## Historical Data

### data/historical/ovr_hist_data.csv
Combined historical MiLB dataset. One row per player-season-stint. Sources and join keys:
- minorLeagueData.csv — base rows (all columns). PlayerId is numeric (MLBAM/FanGraphs).
- missing_milb_data.csv — **primary source** for wRC+, BB/K, K%, ISO, SwStr%, GB/FB. Joined on PlayerId+Season+Team+Level; Name+Season+Team+Level fallback for rows where numeric vs sa-prefix IDs don't match.
- historical_ml_advanced.csv — BB/K, K%, ISO **fallbacks only** (joined on PlayerId+Season+Team+Level+PA; Name fallback applied)
- historical_ml_batted.csv — GB/FB, SwStr% **fallbacks only** (joined on PlayerId+Season+Team+Level+PA; Name fallback applied)
- **ID system note**: minorLeagueData uses numeric IDs; historical_ml_advanced/batted/missing_milb use sa-prefix IDs for recent players. PlayerId joins will miss ~2026 players; `_name_fill()` in run_pipeline.py retries on Name+Season+Team+Level for unmatched rows.

Columns: PlayerId, Season, Name, Level, League, Age, PA, SB, BB/K, K%, ISO, wRC+, GB/FB, SwStr%, PPPA, PPPA_Z_SL, Age_Z_SL

- All float columns rounded to 2 decimal places.
- ~4,192 rows missing GB%/SwStr% (mostly pre-2007 seasons where batted ball data is unavailable).
- ~1,166 rows missing BB/K/K%/ISO (from historical_ml_advanced.csv coverage gaps).
- wRC+ coverage depends on missing_milb_data.csv; non-standard levels (A-, DSL, CPX) produce null wRC+.

### data/historical/hist_mlb_data.csv
MLB batting statistics for graduated players. Columns include Season, Name, Team, G, PA, 1B–CS, TB, PPG, PPPA, PPPA_Z (z-scored within Season). GDP is the GIDP equivalent in this file. PPG/PPPA computed using the same scoring formula as compute_stats.py.

### data/historical/regression_mlb_outcomes.csv
Phase 2 WLS regression predicting first-year MLB PPPA_Z from PA-weighted MiLB skill predictors (last 2 pre-debut seasons) + Phase 4 arc features (all pre-debut seasons). Single-row output. Key design decisions:
- **Level predictors**: BB/K, K%, GB/FB, Age_Z_SL, wRC+, ISO (PA-weighted across last 2 MiLB seasons before debut). ISO differentiates contact-only profiles from well-rounded ones.
- **Trajectory predictors**: K%_delta only (delta between earliest and latest of the 2 seasons; 0 for single-season players). BB/K_delta and wRC+_delta removed — non-significant in ablation study.
- **Arc predictor**: PPPA_Z_SL_best only (best peer-relative PPPA season, PA ≥ 80). **Arc features restricted to A-ball and above** — R level (which includes remapped DSL/CPX/A-) is excluded to prevent inflated Rookie/DSL peaks from anchoring projections. PPPA_Z_SL_best uses peer-relative PPPA (z-scored vs Season+League) rather than absolute PPPA — controls for SB environment inflation across levels. K%_best, K%_slope, K%_arc_sd removed — non-significant in ablation study.
- **Predictor selection**: chosen via ablation study (512 combinations of 9 candidate variables on top of 5 fixed predictors). BB/K + GB/FB + PPPA_Z_SL_best maximized Adj_R² among all subsets tested.
- **Weight**: first-year MLB PA
- N=1,612; R²=0.128; 8 predictors total

### data/historical/regression_season_league.csv
Descriptive OLS regression of PPPA on all available predictors (BB/K, K%, ISO, wRC+, GB/FB, SwStr%, Age_Z_SL), run per Season+League group. **Not predictive — predictors and target are same-season.** Kept for reference but superseded by regression_predictive.csv.

### data/historical/regression_predictive.csv
WLS regression predicting next-season PPPA (T+1) from current-season skill predictors (T). One row per Season_T (17 seasons, 2007–2025). Key design decisions:
- **Weighted least squares**: weight = min(PA_T, PA_T+1) — down-weights small samples in either season.
- **level_diff** (level_T+1 − level_T, encoded R=1 through AAA=5) included as a control variable.
- **Predictors**: BB/K, K%, GB/FB, SwStr%, Age_Z_SL, ISO, level_diff. wRC+ excluded (outcome stat). ISO added as raw power skill signal.
- 2006 skipped (no advanced/batted data). 2019/2020 skipped (COVID cancelled 2020 season).

### data/historical/prospect_mlb_proj.csv
Phase 2 coefficients applied to every player in ovr_hist_data who has complete MLB_PREDS data. Two columns: PlayerId, MLB_Proj_Score. Standardized to 50±10 across all scored players. Built by `build_prospect_mlb_proj()` in run_pipeline.py; joined into 2026ProspectPool.csv by build_prospect_pool.py. Covers 15,074 players (2026 pipeline run).
- **Age_Z_SL coefficient boosted 3×** when applying regression to current prospects — the trained coefficient (-0.0638) underestimates age's importance due to survivorship bias (only age-mismatched players who still succeeded appear in the training data).
- **PA shrinkage**: raw projection is shrunk toward the mean based on qualifying PA (complete advanced-stat seasons). Full-confidence threshold scales by level quality (MLE-grounded — higher-discount levels need more PA to achieve the same information content):
  - AAA/AA: 150 PA
  - A+/A: 250 PA
  - A-/R: 400 PA (historical levels, no longer active post-2021 restructuring)
  - DSL/CPX: 550 PA (median DSL/CPX prospect at ~40% confidence; only multi-year DSL performers approach full weight)
  - Threshold lookup uses each player's most recent level from minorLeagueData.csv (pre-remap), defaulting to 250 PA for unknowns.

### data/historical/sb_translation_log.csv / sb_translation_rates.csv / sb_translation_eras.csv / sb_translation_chain.csv
Output of `sb_translation_analysis.py`. Chained level-to-level SB translation analysis covering all affiliated levels (R→A through AAA→MLB), 2006–2026. Key files:
- **sb_translation_log.csv** — one row per player-promotion, unfiltered (12,681 transitions total).
- **sb_translation_rates.csv** — level-pair × season aggregates (From_SB ≥ 5 filter).
- **sb_translation_eras.csv** — level-pair × era aggregates (era breaks: 2006-2014, 2015-2019, 2020-2022, 2023-2026).
- **sb_translation_chain.csv** — chained discount factors (level→MLB) by era.
2023+ era chained factors (normalized AAA=1.0): AA=0.84, A+=0.71, A=0.57, R=0.39. These are used as PPPA_Z_SL level weights in build_prospect_pool.py.

### data/historical/skill_scores.csv
Phase 1 prospect skill score. One row per player. Key methodology:
- **Predictors**: BB/K, K%, GB/FB, SwStr%, Age_Z_SL, ISO — pure process/skill stats. wRC+ removed (outcome stat; lives in Phase 2).
- ISO replaces wRC+ as the power component — measures raw extra-base hit production, less park/context-dependent than wRC+.
- For each player-season row, applies that season's regression coefficients from regression_predictive.csv as weights — coefficients adapt to the offensive era automatically.
- Raw skill score standardized within Season+League, then PA-weighted across all seasons per player.
- Scaled to 50 + 10 × raw_score (50 = average, ±10 = ±1 std dev). Clipped at 0.
- Excludes rows missing any skill predictor or from seasons without regression coefficients (2019, 2026).
- **Integrated into Combined_Score at 40% weight** (Phase 3).

## API Data (`data/api/`)

Sourced from the MLB Stats API. Covers all affiliated MiLB levels (AAA/AA/A+/A/R), 2006–2026, PA ≥ 10. Refreshed by running `fetch_milb_data.py`.

### data/api/milb_hitting.csv
Counting stats. Columns: PlayerId, MLBAM_ID, Season, Name, Team, Level, League, Age, G, PA, 1B, 2B, 3B, HR, R, RBI, BB, IBB, SO, GIDP, SB, CS
- 1B derived as Hits − 2B − 3B − HR (not a raw API field)
- Team = MLB org abbreviation (e.g. "CHC"), not affiliate name

### data/api/milb_advanced.csv
Rate and batted ball stats. Columns: PlayerId, MLBAM_ID, Season, Name, Team, Level, League, Age, PA, AVG, OBP, SLG, OPS, ISO, BABIP, BB%, K%, GB%, LD%, FB%, GB/FB, IFFB%, HR/FB, SwStr%, Pitches
- All rates stored as decimals (0–1 scale), matching FanGraphs convention
- BB% = walksPerPlateAppearance; K% = strikeoutsPerPlateAppearance (from API)
- Batted ball rates (GB%, LD%, FB%, IFFB%) derived from raw hit+out counts per type
- HR/FB derived by joining HR from milb_hitting on MLBAM_ID+Season+Team+Level
- SwStr% = swingAndMisses / numberOfPitches
- ~872 rows with null HR/FB (no recorded fly balls)

### data/api/player_birthdays.csv
Cached birth dates fetched from the MLB Stats API (`/api/v1/people`). Columns: MLBAM_ID, BirthDate (YYYY-MM-DD). Built and updated incrementally by `fetch_milb_data.py` — only fetches IDs not already in the cache. Used by `build_aaa_rankings.py` to compute decimal ExactAge for precise age percentile ranking within AAA cohorts. ~2,766 rows.

### data/api/chadwick.csv
Cached Chadwick Bureau register crosswalk (MLBAM ID ↔ FanGraphs ID). Downloaded from the Chadwick Bureau GitHub repo (16 split files). Auto-downloaded on first `fetch_milb_data.py` run; cached locally after that.
- 127,760 players with MLBAM IDs; 21,185 have FanGraphs IDs
- PlayerId in API files: FanGraphs ID where mapped, MLBAM ID string otherwise
- ~16.7% of unique players in the API data map to FanGraphs IDs (lower for recent seasons as register lags)

## Roadmap

### Phase 1 (MiLB skill model) — complete
Skill-weighted prospect score built from regression-derived, era-appropriate coefficients. Predictors: BB/K, K%, GB/FB, SwStr%, ISO, Age_Z_SL. wRC+ removed (outcome stat; lives in Phase 2). Stored in skill_scores.csv.

### Phase 2 (MLB outcomes model) — complete
WLS regression predicting first-year MLB PPPA_Z from PA-weighted MiLB skill predictors + trajectory + arc. 8 predictors selected via ablation study (512 combinations tested): BB/K, K%, GB/FB, Age_Z_SL, wRC+, ISO (level averages) + K%_delta (trajectory) + PPPA_Z_SL_best (arc). SwStr%, BB/K_delta, wRC+_delta, K%_best, K%_slope, K%_arc_sd all removed — non-significant once fixed predictors are controlled for. Applied to current prospects via prospect_mlb_proj.csv.

### Phase 3 (Integration) — complete
Four-signal unified ranking: `Combined_Score = 0.40 × Skill_Score + 0.25 × MLB_Proj_Score + 0.22 × PPPA_Score + 0.13 × Age_Score`. All signals re-standardized within the pool to 50±10 before blending. Pool includes all players active in 2025 or 2026 (~3,010 players) — not gated by BW rankings. Players with ≥50 career MLB PA are excluded. BW_Rank is a nullable reference column (null for non-BW players). MLB_Proj_Score applies a 3× age boost and level-adjusted PA shrinkage (AAA/AA: 150 PA, A+/A: 250 PA, A-/R: 400 PA, DSL/CPX: 550 PA full confidence). PPPA_Score uses empirical SB translation factors as level weights (AAA=1.00, AA=0.84, A+=0.71, A=0.57, R=0.39). Age_Score uses a two-component penalty (level-discounted Age_Z_SL + absolute benchmark penalty). Stored in 2026ProspectPool.csv.

---

### Phase 4 (Non-linear career trajectory modeling) — complete
**Motivation:** The current Phase 2 trajectory feature (delta between first and last pre-debut season) only captures a single linear direction of change. It misses players like Luis Lara who had an elite start, a prolonged slump, and then re-emerged as a top prospect — their net delta looks flat even though their career arc is highly meaningful.

**Goal:** Expand the trajectory representation in Phase 2 (and the milb_impact_analysis.py feature set) to capture non-linear career shapes.

**Research findings (June 2026):**
- Non-linear arc shape is a real signal — the effect exists and is acknowledged in PECOTA's comparable-player framework and developmental trajectory research (Frontiers in Psychology, 2019: 17 distinct MiLB pathways identified). However, extracting it cleanly requires sufficient longitudinal data.
- Most prospects have only 2–4 pre-debut seasons, making most arc-modeling approaches impractical for the majority of the pool.
- Interestingly, position players with *fewer* MiLB seasons before debut had better MLB outcomes on average — suggesting time-in-minors is not uniformly positive and arc complexity may be most meaningful for multi-year prospects who stall.

**Approved design (post-research):**
1. **Best-season K% and best-season peer-relative PPPA (PPPA_Z_SL_best)** as new features alongside the PA-weighted averages — highest confidence addition. K% has the strongest minor-to-major correlation (r=0.77 per KATOH/Dahl research) and is the single most impactful fantasy component (-2 per SO). PPPA_Z_SL_best captures peak fantasy production relative to peers, controlling for SB environment inflation (MiLB SB rates are structurally higher than MLB; using absolute PPPA overstates the ceiling of SB-heavy players). Low implementation cost.
2. ~~**Best-season SB rate**~~ — initially added; subsequently removed. WLS coefficient not statistically significant (p=0.107) and SB is already captured through PPPA_Z_SL_best. Entering through two channels was double-weighting SB without statistical support.
3. **Slope + residual SD for K%** restricted to players with 3+ pre-debut seasons at ≥100 PA each — theoretically grounded in aging-curve research but impacts a minority of the pool. Players with fewer seasons fall back to existing delta.
4. Any arc feature must have a defined fallback (pool neutral or delta) for players with fewer than 3 seasons — otherwise young/fast-moving prospects are systematically disadvantaged.
5. **Arc features restricted to A-ball and above** — R level (which includes remapped DSL/CPX/A-) excluded from best-season computation. DSL/CPX Rookie-ball peaks (e.g., 41 SB in DSL, elite K% in CPX) are environment artifacts that inflate projections when used as ceiling anchors. Players with no full-season data fall back to pool-neutral arc features.

**Dropped from original design (post-research):**
- **Best-season wRC+**: wRC+ does not penalize K% as harshly as the fantasy scoring system does and ignores SB entirely. PPPA is the correct target; wRC+ as a trajectory feature drifts toward general MLB success, not fantasy value.
- **Segmented trajectory (thirds)**: With 2 seasons (majority of prospects) thirds collapse to the same delta already computed. Consumes degrees of freedom without proportional signal gain.
- **Shape clustering**: Cluster assignments are unstable with 2–4 data points — a player's cluster changes with one new season. Useful for descriptive post-hoc analysis only, not live prediction.

**Key stats for non-linear trajectory (priority order):** K%, PPPA_Z_SL (peer-relative), ISO, BB/K. GB% replaced by GB/FB throughout the model. Post-ablation: only PPPA_Z_SL_best retained as arc predictor — K%_best, K%_slope, K%_arc_sd removed (non-significant once fixed predictors controlled for). SB rate deprioritized as an arc feature — not statistically significant in Phase 2 and captured through PPPA_Z_SL_best. wRC+ deprioritized as a trajectory feature; useful as a level-stat feature in Phase 2 but not as an arc signal for fantasy-specific projection.

---

### Phase 5 (Peak/Slump scoring — Heater/Slump scores) — planned
**Motivation:** Prospect evaluation generalizes too much across a player's full career average, masking how dominant they can be at their best or how fragile they look at their worst. Two players with identical averages may have very different upside and risk profiles.

**Goal:** For each prospect, compute two complementary scores — **Heater Score** (peak performance ceiling) and **Slump Score** (worst-stretch floor) — then benchmark both against same-age, same-level peers.

**Research findings (June 2026):**
- Stat stabilization thresholds (Russell Carleton / Pizza Cutter split-half reliability research, R²≥0.49 threshold):

  | Stat    | PA to stabilize (MLB) | Use for peaks/slumps? |
  |---------|----------------------|----------------------|
  | K%      | ~60 PA               | Yes — primary        |
  | BB%     | ~120 PA              | Marginal             |
  | ISO     | ~160 AB (~200 PA)    | Caution              |
  | BABIP   | ~820 BIP             | No — too noisy       |
  | HR/FB   | ~50 FBs              | No — single-season too noisy |
  | wRC+    | (composite)          | Yes — use with PA ≥80 minimum |

  MiLB thresholds are higher than MLB equivalents due to wider talent dispersion. Treat K% as reliable above ~80 PA, wRC+ above ~120 PA at the MiLB level.

- **Rolling PA windows are not feasible with current data.** The dataset is season-level rows only — no intra-season granularity exists. A "best 200 PA stretch" requires game-by-game or weekly data that is not available. This is a fundamental data constraint, not a design tweak.
- Age weighting is strongly supported: age relative to level is confirmed as "the single strongest proxy for prospect quality" (Dahl/KATOH research). A peak at 19 in A-ball carries substantially more signal than the same numbers at 23.

**Approved design (post-research):**
- Redesign around **season-level best and worst** rather than rolling windows — feasible with current data, statistically defensible with a PA minimum.
- **Primary stats: K% and PPPA** — K% is the most stable skill signal and the largest single negative drag on fantasy scoring (-2/SO); PPPA is the direct fantasy output and the correct optimization target for this system.
- **Secondary stat: SB rate (SB per PA)** — SB = +3 in the scoring formula, almost entirely absent from wRC+/WAR-based metrics, and a major driver of top fantasy finishers. Best-season SB rate is a meaningful ceiling indicator.
- **Tertiary stat: ISO** — meaningful for identifying outlier power seasons (HR = +4, TB = +1); less stable than K% at the season level but useful as a confirming signal.
- **Drop wRC+ as a primary Heater/Slump stat**: wRC+ does not penalize K% at the -2 rate and ignores SB. A player with a great wRC+ season could be a fantasy disappointment; PPPA captures this correctly.
- **Drop BABIP and HR/FB entirely** from this system — both have very high stabilization thresholds (820 BIP and 50+ FBs respectively) and cannot reliably distinguish skill from variance at the season level.
- Minimum 80 PA per season to qualify as a valid peak/slump season.
- Age-weight the Heater Score: multiply by a bonus factor for being young-for-level (using Age_Z_SL).
- Standardize Heater Score and Slump Score within same-age, same-level peers (z-score → 50±10 scale).
- Output: heater_slump_scores.csv (separate file, not merged into 2026ProspectPool.csv to avoid bloating the main rankings).
- Label Slump Score explicitly as a floor/risk indicator — it is not a disqualifier, it is a volatility flag.

---

### Phase 6 (Early-signal watchlist — small-sample prospects) — planned
**Motivation:** International signings and newly drafted high school players have very few PA, making them hard to rank using the full model. But this volatility is also an opportunity — identifying a breakout before the market does has high value. A simplified, age-weighted signal on the core essentials can surface these players before they appear in mainstream lists.

**Goal:** Build a separate watchlist (watchlist_prospects.csv) for players with very small sample sizes (career PA ≤ ~200) who are popping in the essentials. Explicitly not a rankings product — a scouting lead generator only.

**Research findings (June 2026):**
- DSL/CPX stats have modest but non-zero predictive validity for MLB outcomes (R²~0.03–0.08 from BP/FanGraphs research when raw stats used without age controls). Signal improves meaningfully once age-relative-to-league is included.
- **Age_Z_SL is the single most defensible signal at small samples** — it is exact (zero sampling error), does not regress to the mean, and is consistently the strongest cross-level predictor of eventual MLB value (Victor Wang SABR research; Russell Carleton BP work). At 100 PA, a K% reading is imprecise; the player's age is exact.
- K% at 100–150 PA is usable as a soft signal (stabilizes fastest, ~60–80 PA MLB equivalent). Meaningful for extreme outliers; imprecise in the middle of the distribution.
- BB% at 100–150 PA is at or below its reliability threshold. Not a primary signal at these sample sizes.
- PPPA at small samples has two distinct limitations that must be separated: (1) **level compression** — all MiLB PPPA will deflate at MLB, which is structural and already handled by PPPA_Z_SL z-scoring; (2) **component instability** — at 80–150 PA, HR and SB counts are small enough that one extra HR or a few stolen bases swings PPPA substantially. The concern in Phase 6 is exclusively (2), not (1). Use PPPA as a tertiary confirming signal at these sample sizes; weight K% and Age_Z_SL more heavily where component instability is the dominant risk.
- **Turnaround flag has ~60–75% false positive rate at sub-200 PA** (per Carleton's breakout research). A wRC+ swing of ±30 points is within normal sampling error at 70–100 PA. Retain flag but label it explicitly speculative.
- Survivorship bias caveat: all players in the data passed an organizational selection filter before appearing in games. The observable data overrepresents players organizations already believe in.

**Approved design (post-research):**
- **Formula reweighting:** Age_Z_SL 50% / K% 35% / PPPA 15%. BB% removed as a primary driver (below reliability threshold at target PA range); may be included as a binary flag for extreme values (BB% ≥ 12% in CPX/DSL). SB rate included as a confirming signal or secondary flag — SB = +3 in the scoring formula and is one of the first skills to show up in raw counting stats even at small samples.
- **Minimum PA floor:** 80 PA before any score is displayed. Below 80 PA, K% cannot carry meaningful weight even as a soft signal.
- **Restrict statistical watchlist to stateside appearances** (CPX, A-ball first season). DSL-only players listed separately as a purely organizational/scouting flag with an explicit note that no statistical validity is implied.
- **Turnaround flag definition:** K% drop of ≥5pp OR PPPA jump of ≥20% above career average in the most recent season, with ≥80 PA in that season. PPPA is the correct turnaround target for this system — not wRC+. Label as speculative in output.
- Output columns: PlayerId, Name, Team, Level, Age, Career_PA, K%, BB%, SB, PPPA, Age_Z_SL, Watchlist_Score, Watchlist_Rank, Turnaround_Flag.

**Data pipeline gap (resolved):** DSL, CPX, and A- rows are now remapped to Level "R" and included in the main pipeline (`ovr_hist_data.csv`, skill scores, MLB projections, and prospect pool). The prospect pool is no longer gated by BW rankings — all players active in 2025/2026 are included. Phase 6 can now use the existing pipeline data directly.

---

### Planned: Phase 3 Overhaul (3-component transparent model) — not yet started
**Motivation:** Current rankings produce poor results for elite prospects — consensus top-50 players (e.g., Condon, Montgomery, Emmanuel Rodriguez) rank 600–850 due to an age penalty that is too aggressive for 23-year-old AAA players. Additionally, 11 of the top 25 are A/R ball players, which is implausible even for a performance-focused model. The model needs to decompose PPPA into its drivers more transparently and trust lower-level results less.

**Issues identified:**
1. **Age penalty too aggressive**: The absolute benchmark penalty (0.5 × years above target age) tanks 23yo AAA players who are genuinely elite. Should be a soft continuous modifier, not an additive penalty cliff.
2. **Too many A/R players in top 25**: Lower-level stats are noisy. PA-based confidence shrinkage should reduce weight on small samples, not just via the Phase 2 PA threshold. Model needs a more explicit lower-level discount applied to PPPA_Score or overall.
3. **PPPA decomposition needed**: Rather than taking PPPA as a black box, decompose into its key fantasy drivers: strikeout avoidance (K%), stolen base production (SB rate × translation factor), and power production (TBD metric — not ISO).
4. **Transparency**: Each component should be interpretable and auditable independently.

**Proposed 3-component structure:**
- **Contact/Discipline**: K%, BB/K — directly captures the −2/SO penalty and walk value.
- **Power/Speed**: SB rate (level-translated) + TBD power metric. SB gets explicit empirical grounding via sb_translation factors. Power metric TBD — ISO rejected (too context-dependent); options under consideration: HR/PA, XBH/PA, weighted fantasy rate `(2B×2 + 3B×3 + HR×4)/PA`, or Statcast metrics (Barrel%, EV — AAA-only via ps_AAA_2026.csv).
- **Age Signal**: Continuous modifier, not additive penalty. Being young-for-level multiplies or boosts the score; no hard cliff for players above target age.

**TODO — Power metric decision**: ISO is not the right power measure for fantasy PPPA. Need to decide between:
- `HR/PA` — pure power ceiling, ignores XBH
- `XBH/PA` (2B+3B+HR per PA) — broader power rate
- Weighted fantasy rate: `(2×2B + 3×3B + 4×HR) / PA` — directly proportional to fantasy point value of extra-base hits
- Statcast metrics (Barrel%, Hard Hit%, EV) — available for AAA via ps_AAA_2026.csv, not available for lower levels
Decision pending: user will upload consensus rankings before overhaul begins.
