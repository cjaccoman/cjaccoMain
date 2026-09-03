# prospectsMain

Baseball prospect analysis tool targeting fantasy **PPPA** (points per plate appearance) in a points-based scoring league. Scores and ranks minor league hitters using a blend of demonstrated production (ABILITY_Score) and raw physical skills (TOOLS_Score).

**The original Phase 1–4 model is documented in `firstModel.md`.**

---

## Critical Framing: Target is PPPA, Not MLB Success

The target is **PPPA**, not wRC+ or WAR. This distinction shapes every modeling decision:
- Strikeouts cost −2 points — roughly 3× the penalty implied by wRC+. K% is the single most impactful negative signal.
- Stolen bases score +3 points each. SB rate is a major positive signal almost entirely absent from wRC+ and WAR.
- A player can rank #2 in overall fantasy value while ranking #21 in wRC+. The divergence is real and significant.

When evaluating research or choosing proxy metrics: always ask whether the signal translates to PPPA specifically, not just to general MLB success.

---

## Fantasy Scoring Formula

| Stat | Weight | Stat | Weight |
|------|--------|------|--------|
| 1B | +1 | SO | −2 |
| 2B | +2 | GIDP | −1.5 |
| 3B | +3 | SB | +3 |
| HR | +4 | CS | −1.5 |
| R | +1 | TB | +1 |
| RBI | +2 | | |
| BB | +1 | | |
| IBB | +1.5 | | |

`TP = 1B + 2×2B + 3×3B + 4×HR + R + 2×RBI + BB + 1.5×IBB + TB − 2×SO − 1.5×GIDP + 3×SB − 1.5×CS`

---

## Environment

- Python 3.11 (virtual environment configured via PyCharm)
- Dependencies: `requirements.txt` (pandas>=2.0, statsmodels>=0.14, xgboost>=2.0, shap>=0.44, scikit-learn>=1.3)

---

## Version History

### v0.1 — Initial PPPA Model
PA-weighted PPPA z-scores. No regression; no skill decomposition. Two weighted inputs: PPPA_Z (60%) and −Age_Z_SL (40%, subtracted because being younger than peers is a positive signal). Formula: `50 + 10 × (0.6×PPPA_Z − 0.4×Age_Z_SL)` per player. BW scout rankings were a reference column throughout — never part of the scoring formula. Documented in `firstModel.md`.

### v0.2 — Phase 1: Skill Score
Introduced regression-derived skill scoring. WLS regression of next-season PPPA on same-season process stats (BB/K, K%, GB/FB, SwStr%, ISO, Age_Z_SL). Era-adaptive: each season's coefficients from `regression_predictive.csv` used as weights. Output: `skill_scores.csv`.

### v0.3 — Phase 2: MLB Outcomes Regression
WLS regression predicting first-year MLB PPPA_Z from PA-weighted pre-debut MiLB stats. 8 predictors selected via 512-combination ablation study: BB/K, K%, GB/FB, Age_Z_SL, wRC+, ISO (level averages) + K%_delta (trajectory) + PPPA_Z_SL_best (arc). Output: `regression_mlb_outcomes.csv`, `prospect_mlb_proj.csv`.

### v0.4 — Phase 3: Four-Signal Blend
Unified ranking: `Combined_Score = 0.40×Skill_Score + 0.25×MLB_Proj_Score + 0.22×PPPA_Score + 0.13×Age_Score`. All signals re-standardized to 50±10 within the pool before blending. Output: `2026ProspectPool.csv`.

### v0.5 — Phase 4: Non-Linear Arc Features
Added best-season K% and peer-relative PPPA (PPPA_Z_SL_best) as arc features to Phase 2, plus K%_slope and K%_arc_sd for players with 3+ qualifying pre-debut seasons. Post-ablation: only PPPA_Z_SL_best retained (K%_best, K%_slope, K%_arc_sd non-significant when fixed predictors controlled for). SB_rate_best removed — not statistically significant and already captured through PPPA_Z_SL_best.

### v0.6 — AAA Rankings & SB Translation Analysis
Added `build_aaa_rankings.py` (ProspectSavant-based AAA power rankings) and `sb_translation_analysis.py` (chained level-to-level SB discount factors). 2023+ era chained factors (normalized AAA=1.0): AA=0.84, A+=0.71, A=0.57, R=0.39. These became the PPPA_Score level weights in Phase 3. Also added `build_rk_rankings.py` for Rookie ball. **Superseded in v1.0**: level discounts rederived from Skill_PPPA full-population study (`analysis/skill_pppa_translation.py`); see current values below.

### v1.1 — Pitcher Rankings (August 2026)
Added complete pitcher prospect scoring system with separate SP and RP rankings. Two new composites mirroring the hitter model: STUFF_Score (raw stuff / physical skills) and PERFORMANCE_Score (demonstrated production). Data fetched via MLB Stats API pitching endpoints. Key design decisions: age cutoff ≤ 25 (vs. ≤ 24 for hitters); IP shrinkage thresholds STUFF=167 IP, PERFORMANCE=120 IP; level discounts same as hitters; no pool re-standardization after career averaging (avoids double-standardization amplification). New files: `fetch/fetch_milb_pitching.py`, `pipeline/build_pitcher_features.py`, `pipeline/build_pitcher_scores.py`, `data/api/milb_pitching.csv`, `data/api/milb_pitching_advanced.csv`, `data/rankings/pitcher_features.csv`, `data/rankings/pitcher_scores.csv`.

### v1.0 — Overhaul: TOOLS_Score + ABILITY_Score (August 2026)
**Complete replacement of Phase 1–4 scoring.** The old Skill_Score, MLB_Proj_Score, PPPA_Score, and Age_Score are all retired. New model is two complementary composites — TOOLS (physical skills, era+level normalized) and ABILITY (demonstrated production) — blended with a PA-weighted OVR trajectory score. Era analysis derived empirically from MLB outcome data (2006–2026). Full details below.

**Key new data added this overhaul:**
- `milb_pitches_agg.csv`: 107,657 player-season-level rows (2006–2026, all 5 levels) with PullAir%, Chase%/Z-Contact% (AAA 2023–2026 only), built by `fetch_milb_pitches.py` via MLB Stats API game feeds.
- `data/prospectSavant/`: ProspectSavant exports with Spd, MaxEV, EV90, xBA, xSLG, Barrel%, Chase%, Z-Contact% for AAA (2023–2026), AA, A+, A, Rk (2026).
- `prospect_features.csv`: one row per player-season-level, all columns needed for TOOLS/ABILITY scoring.

---

## Current Model (v1.0)

### Architecture

```
Combined_Score = 0.50 × Current_Score + 0.50 × OVR_Score

Current_Score  = 0.30 × TOOLS_Score + 0.50 × ABILITY_Score + 0.20 × Age_Score
OVR_Score      = 0.40 × TOOLS_Score + 0.40 × ABILITY_Score + 0.20 × Slope_Score
```

- **Current_Score**: computed within the current prospect pool (most recent season ≥ 2025, age ≤ 24). Weights standardized to 50±10 within this pool.
- **OVR_Score**: computed across all historical player-seasons (2006+). Slope_Score is the PA-weighted PPPA_Z_SL trajectory across levels — positive = production improves as player advances. Players with < 2 qualifying levels (≥ 80 PA each) receive neutral 50.
- **Age_Score** (Current only): −Age_Z_SL standardized to 50±10. Uses most-recent-season Age_Z_SL. Age enters twice: here as 20% standalone, and inside TOOLS/ABILITY via per-component age multiplier.
- Players missing from OVR pool receive neutral 50 for that component.

---

### TOOLS_Score

Measures what a player *can do* — raw physical skills, normalized for era and level. Output: `data/rankings/tools_scores.csv`.

**Top-level weights:** Discipline 45% / Power 35% / Athleticism 20%

**Discipline:**

| Tier | Available when | Components |
|------|---------------|------------|
| Full | AAA, 2023–2026 (Chase%/Z-Contact% available) | −Chase%_adj (40%) + ZContact%_adj (35%) + −Whiff%_adj (25%) |
| Fallback | All other rows | −Whiff%_adj (100%) |

**Power:**

| Tier | Available when | Components |
|------|---------------|------------|
| Full | ProspectSavant rows (MaxEV + EV90 available) | MaxEV_z (35%) + EV90_z (35%) + HRFB_adj (30%) |
| Fallback | All other rows | HRFB_adj (100%) |

**Athleticism:**

| Tier | Available when | Components |
|------|---------------|------------|
| Full | ProspectSavant rows with Spd | Spd_z (50%) + 3B_PA_adj (50%) |
| Fallback | All other rows | 3B_PA_adj (100%) |

**ProspectSavant coverage by level (Spd / MaxEV / EV90):**

| Level | Seasons with PS data |
|-------|---------------------|
| AAA | 2023–2026 |
| AA | 2026 only |
| A+ | 2026 only |
| A | 2023–2026 |
| Rk | 2026 only |

Chase%/Z-Contact% are not available from ProspectSavant at AA/A+/A/Rk and are not available from game feeds below AAA. Those levels always use the Discipline fallback (Whiff% only).

**Age adjustment (Discipline + Power only):** Each component × (1 + 0.20 × −Age_Z_SL), clipped to ±2 SD. A player 2 SD younger gets ~40% boost; 2 SD older gets ~40% cut. Athleticism excluded — sprint speed is a physical tool not expected to improve with age.

**Normalization:** MaxEV, EV90, Spd are raw values z-scored within Level. Era-adjusted `_adj` columns are already z-scored within Level×era; used directly. Each sub-component standardized to mean=0, std=1 before blending. Missing component → filled with 0 (neutral). Final TOOLS_Score standardized to 50±10 across all scored rows, clipped at 0.

---

### ABILITY_Score

Measures what a player *has done* — demonstrated production, adjusted for era and level. Output: `data/rankings/ability_scores.csv`.

**Component weights:** Fantasy Output 45% / Discipline 25% / SB Talent 15% / Game Power 15%

| Component | Metric | Normalization |
|-----------|--------|---------------|
| Fantasy Output | PPPA_Z_SL × level_discount | Level discounts: AAA=1.00, AA=0.59, A+=0.34, A=0.23, R=0.10 |
| Discipline | BB% − 2×K% (BB_2K) | z-scored within Season+Level |
| SB Talent | SB_pct × (SB/PA) | z-scored within Season+Level |
| Game Power | 0.5×PullAir% + 0.5×HR_AB | z-scored within Season+Level; HR_AB = HR / AB |

**Level discount methodology:** Derived from `analysis/skill_pppa_translation.py` using a full-population approach: all non-prospect MiLB players (PA ≥ 50 per level), with non-graduates assigned MLB_Skill = 0. Metric: Skill_PPPA = −2×K% + 4×HR/PA + 3×SB/PA − 1.5×CS/PA (the three components with highest PPPA formula weights and strongest MiLB→MLB signal). Ratios normalized to AAA=1.0. Values reflect both translation quality AND probability of reaching MLB (graduation rates: AAA≈51%, AA≈33%, A+≈20%, A≈14%, R≈6%). Prior SB-only chain (AA=0.84, A+=0.71, A=0.57, R=0.39) superseded — it answered the wrong question and used a selection-biased sample.

**Age adjustment:** Each component × (1 + 0.20 × −Age_Z_SL), clipped to ±2 SD.

**Fallback:** Sparse Season+Level cells (< 10 qualifying rows with PA ≥ 50) fall back to Level-only z-scoring. Missing component → 0 (neutral). Final ABILITY_Score standardized to 50±10, clipped at 0.

**HR/PA is explicitly excluded.** Power uses HR_AB (HR / (PA−BB−IBB) — approximates AB by removing non-contact plate appearances) or HR/FB (TOOLS) — both strip non-power PA more cleanly than HR/PA.

---

### Per-Season Score Aggregation

TOOLS_Score and ABILITY_Score are built at the player-season-level. `pipeline/build_prospect_scores.py` aggregates to one value per prospect:

- For each player, compute PA × level_weight career average per level.
- **Per-level shrinkage**: `shrink = min(PA_eff / threshold, 1.0)`; `shrunk = 50 + shrink × (level_avg − 50)`. Thresholds (in AAA-equivalent PA): TOOLS=250, ABILITY=175. Small-sample levels contribute near-neutral; full-season data contributes at face value.
- Level weights (discount factors, normalized to AAA=1.0): AAA=1.00, AA=0.59, A+=0.34, A=0.23, R=0.10.
- Eligibility (current pool): most recent season ≥ 2025, age ≤ 24, career MLB PA < 50.

### Post-Blend Discipline Gate

Applied in `pipeline/build_prospect_scores.py` after `Combined_Score` is computed. Captures the empirical non-linear cliff in MLB outcomes below peer-relative BB_2K thresholds — a career-level finding, not a per-season one. Keeps ABILITY component scores clean (no contamination of power/speed signals).

**Composite:** PA-weighted career average BB_2K (non-AAA, to avoid AAA survivorship bias), z-scored within the current prospect pool. `Disc_Composite_Z` is the gate input.

**Slope:** PA-weighted OLS of BB_2K on Season across all career rows with PA ≥ 50. Requires ≥ 2 qualifying seasons; NaN otherwise (no slope modifier fires). Positive = improving discipline over time. Stored as `Disc_Slope`.

**Thresholds and penalties (deducted from Combined_Score):**
- `Disc_Composite_Z ≤ −1.00` (bottom ~16% of pool): −3.0 pts
- `Disc_Composite_Z ≤ −0.67` (bottom ~25% of pool): −1.5 pts
- Cap: −4.0 pts total

**Slope modifier** (fires only when a penalty exists and slope data available):
- `Disc_Slope ≥ +0.02/yr` (improving): penalty × 0.60
- `Disc_Slope ≤ −0.02/yr` (worsening): penalty × 1.25

**Output columns:** `Career_Disc_Flag` (label: hard/soft + improving/worsening suffix), `Disc_Composite_Z`, `Disc_Slope`.

---

### Dynasty Positional Scarcity Adjustment

Applied in `pipeline/build_prospect_scores.py` after the discipline gate. Optional layer for dynasty fantasy leagues where lineup slots create positional scarcity independent of raw production. Does **not** modify `Combined_Score` — it is a separate column so the unadjusted ranking is always preserved.

**Why fixed tiers, not empirical replacement levels:** The prospect pool has ~30% listed as SS (young players migrate to SS in the minors before positional assignment stabilizes) and only ~3% listed as 1B. Using pool distribution to set replacement levels inverts the true MLB scarcity ordering. Fixed tiers anchored to MLB starting-lineup scarcity avoid this.

**Position mapping:** Generic infield (`IF`) maps to the 2B tier. DH/unknown positions receive no bonus (0).

**Tiers (added to `Combined_Score` to produce `Pos_Adj_Score`):**

| Position | Bonus |
|----------|-------|
| C | +5.0 |
| SS | +3.0 |
| 2B | +1.5 |
| 3B | +0.5 |
| 1B | −1.0 |
| OF | −2.0 |

**Output columns:** `FantasyPos` (normalized position for tier lookup), `Pos_Bonus`, `Pos_Adj_Score`, `Pos_Adj_Rank`.

**HTML toggle:** The artifact's "Pos Adj: OFF/ON" button switches the rank column between `Combined_Rank` and `Pos_Adj_Rank` and annotates the Combined score cell with the position bonus when ON.

**League context:** Designed for a 12-team dynasty league with slots C(2), 1B(1), 2B(1), 3B(1), SS(1), INF(1), OF(4), UT(3), plus IL and MiLB reserve spots. MiLB eligibility = Career+Current AB ≤ 130.

---

### Archetypes

`build_archetypes.py` assigns a descriptive label to each prospect based on their TOOLS and ABILITY component profile. Output: `data/rankings/archetype_labels.csv`. Labels appear in `prospect_scores.csv` and the HTML artifact.

---

## Era Analysis (v1.0 Research Foundation)

All era breaks derived empirically from `hist_mlb_data.csv` (PA ≥ 100, N≈430/season, 2006–2026).

### Era Definitions — Key Variables

**K% — one break: pre-2015 / 2015+**
| Era | Median K% | Mean K% |
|-----|-----------|---------|
| 2006–2014 | 17.8% | 18.1% |
| 2015–2026 | 22.8% | 22.2% |

**HR/FB — two breaks: pre-2016 / 2016–2021 / 2022+**
| Era | Median HR/FB | Mean HR/FB |
|-----|-------------|-----------|
| 2006–2015 | 8.5% | 9.0% |
| 2016–2021 | 12.2% | 12.8% |
| 2022–2026 | 10.7% | 11.0% |

**SB/PA — one break: pre-2023 / 2023+**
| Era | Median SB/PA | Mean SB/PA |
|-----|-------------|-----------|
| 2006–2022 | 0.76% | 0.77% |
| 2023–2026 | 1.10% | 1.14% |

**PPPA — two breaks: pre-2010 / 2010–2020 / 2021+**
| Era | Median PPPA | Mean PPPA |
|-----|------------|----------|
| 2006–2009 | 0.830 | 0.850 |
| 2010–2020 | 0.745 | 0.740 |
| 2021–2026 | 0.690 | 0.687 |

**Key finding:** PPPA has fallen ~15 pts since 2006–2009 despite rising HR rates. The K% explosion (SO drag: −0.385 → −0.521 per PA) overwhelmed all HR gains. The 2023+ SB revival (+0.046 → +0.069 SB pts/PA) partially offsets K% drag but hasn't reversed the decline.

### MLB Outcome Analysis — First-Year Debuters (PA ≥ 100)

| Era | N | Median K% | Median HR/PA | Median SB/PA | Median PPPA |
|-----|---|-----------|-------------|-------------|------------|
| 2006–2014 | 1035 | 18.7% | 2.17% | 0.83% | 0.770 |
| 2015–2019 | 395 | 23.7% | 2.71% | 0.76% | 0.670 |
| 2020–2022 | 222 | 25.8% | 2.68% | 0.93% | 0.580 |
| 2023–2026 | 274 | 25.3% | 2.45% | 1.56% | 0.620 |

These era breaks are encoded in `pipeline/add_era_columns.py` and used for era-relative z-scoring throughout the TOOLS and ABILITY models.

---

## Pitch-Level Data — Zone Stats

### Source: MLB Stats API Game Feed

`fetch_milb_pitches.py` fetches per-pitch data from `/api/v1.1/game/{gamePk}/feed/live`. Outputs:
- `data/api/milb_pitches_games.csv` — raw game-level cache (one row per player × game)
- `data/api/milb_pitches_agg.csv` — aggregated player-season-level metrics (107,657 rows, 2006–2026)

**Metric definitions:**
- `Chase% = (swings on zones 11–14) / (all pitches in zones 11–14)`
- `Z-Contact% = (contact on zones 1–9) / (swings on zones 1–9)`
- `PullAir% = (pulled fly balls + pulled line drives) / all batted balls`

### Coverage

| Metric | Coverage |
|--------|----------|
| PullAir% | All 5 levels, all seasons 2006–2026 (100%) |
| Chase% / Z-Contact% | AAA only, 2023–2026 only (100% at AAA from 2023; zero coverage before 2023 at any level) |

**A-ball (2021–2026) shows 30–76% zone coverage — park-specific Trackman only, not level-wide. Not used.**

**Usage in model:** PullAir% feeds into ABILITY Game Power (all rows); Chase% and Z-Contact% feed into TOOLS Discipline (AAA 2023+ full tier; all others fall back to Whiff%).

**Incremental update:**
```
python fetch/fetch_milb_pitches.py           # incremental — new current-season games only
python fetch/fetch_milb_pitches.py --season 2026
```

---

## Ablation Studies — PPPA Translation

### MiLB → First-Year MLB (v1 — Cross-Environment)
N=1,346 players, R²=0.129. **Efficient core: SO_PA + HR_PA + SB_PA (3 predictors, Adj_R²=0.113).** Adding 7 more predictors gains only +0.009 Adj_R². R_PA and RBI_PA are context/lineup signals — not skill.

### Same-Season MLB (v2 — Within-Environment)
N=9,155 player-seasons, R²=0.9999. Coefficients recover the scoring formula exactly. **Cross-study insight:** within the same MLB environment PPPA is near-deterministic (v2 R²≈1.0). Across environments (MiLB→MLB), signal collapses to R²≈0.12. The 3 stats that survive that collapse cleanly (HR, SO, SB) are the three with the highest formula weights and the most individual-skill content.

---

## Data Reference

### `data/api/` — MLB Stats API

| File | Description |
|------|-------------|
| `milb_hitting.csv` | Raw counting stats from MLB Stats API (2006–2026, PA ≥ 10, Age ≤ 26) |
| `milb_advanced.csv` | Rate/batted ball stats from API seasonAdvanced endpoint; includes Whiff% (swingAndMisses/totalSwings, all levels) and SwStr% (swingAndMisses/numberOfPitches) |
| `milb_pitching.csv` | Raw pitching counting stats from MLB Stats API (2006–2026, IP ≥ 10, Age ≤ 26): G, GS, IP, ER, K, BB, HRA, W, L, SV, HLD, BS, CG, SHO, QS, WP |
| `milb_pitching_advanced.csv` | Rate/batted ball stats from pitching seasonAdvanced endpoint: K%, BB%, K-BB%, Whiff%, BABIP, GB%, LD%, FB%, GB/FB, QS |
| `milb_pitches_agg.csv` | Aggregated pitch-level metrics from game feeds: Chase%, Z-Contact% (AAA 2023–2026), PullAir% (all levels/seasons). **Whiff% column is broken — all 7,860 non-null rows are 0 (never computed). Do not use; `milb_advanced.csv` is authoritative for Whiff%.** |
| `milb_pitches_games.csv` | Raw game-level pitch cache (one row per player × game) |
| `milb_statcast_aaa.csv` | Baseball Savant statcast data for AAA |
| `mlb_statcast.csv` | Season-by-season MLB Statcast (2015–current, 9,822 rows): MaxEV, AvgEV, EV50, EV_FBLD, EV_GB, Brl%, SweetSpot%, EV95%, xBA, xSLG, xwOBA (and actuals + deltas), SprintSpeed, Bolts, HP_to_1B. Built by `fetch/fetch_mlb_statcast.py`. |
| `mlb_bat_tracking.csv` | Career-aggregate bat tracking for 647 MLB players: AvgBatSpeed, SwingLength, HardSwing%, SquaredUp/Swing, Blast/Swing, Whiff/Swing. Career aggregate only — Savant does not expose per-season bat tracking. |
| `mlb_statcast_ev.csv` | Legacy career-aggregate MaxEV only (2,570 players). Used by `analysis/build_player_comps.py` for MaxEV_mlb fallback. May be superseded by `mlb_statcast.csv`. |
| `player_birthdays.csv` | Cached birth dates (~2,766 MLBAM_IDs) |
| `player_positions.csv` | MLBAM_ID → Position lookup |
| `chadwick.csv` | MLBAM ↔ FanGraphs ID crosswalk (127,760 players) |

### `data/fangraphs/` — FanGraphs Exports

| File | Description |
|------|-------------|
| `missing_milb_data.csv` | FanGraphs-sourced rate stats (77,805 rows, 2006–2026) — primary source for wRC+, BB/K, K%, ISO, SwStr%, GB/FB |
| `ml_updated_data.csv` | FanGraphs 2026 season rate stats (3,001 rows, Season=2026); overrides missing_milb_data for 2026 rows |
| `historical_ml_advanced.csv` | FanGraphs historical advanced stats — BB/K, K%, ISO fallbacks |
| `historical_ml_batted.csv` | FanGraphs historical batted ball stats — GB/FB, SwStr% fallbacks |
| `misc_mlb.csv` | FanGraphs MLB season stats (HR/FB for era analysis) |
| `minorLeaguewrcp.csv` | wRC+ source for `cards/generate_player_card_cat.py` |

### `data/computed/` — Pipeline-Generated Intermediates

| File | Description |
|------|-------------|
| `minorLeagueData.csv` | Output of `pipeline/compute_stats.py` — counting stats + TB, TP, PPG, PPPA, z-scores |
| `averages_season_league.csv` | Season+League baseline averages |
| `averages_season_league_age.csv` | Season+League+Age baseline averages |
| `player_comps.csv` | One row per player (20,367 total); PA-weighted PPPA_Z_SL and age at each MiLB level, plus first-year and career MLB PPPA_Z outcomes. Used by `analysis/build_player_comps.py` comparator. |
| `babip_luck.csv` | One row per player-season (PA ≥ 80); BABIP and HR/FB deviation from each player's own career leave-one-out baseline. Columns: BABIP, BABIP_career, BABIP_delta, BABIP_delta_z, HR/FB, HRFB_career, HRFB_delta, HRFB_delta_z, Luck_Score (0.60×BABIP_delta_z + 0.40×HRFB_delta_z). Positive = lucky, negative = unlucky. Built by `analysis/build_babip_luck.py`. |

### `data/rankings/` — Scores & Rankings

| File | Description |
|------|-------------|
| `prospect_features.csv` | One row per player-season-level; all raw inputs for TOOLS/ABILITY scoring. Includes `career_FBs_est` (total career FB estimate across all qualifying seasons) and `prior_FBs_est` (same, excluding this row's own season — used for fallback-tier HRFB shrinkage in build_tools_score.py). |
| `tools_scores.csv` | Per-player-season TOOLS_Score (standardized to 50±10) |
| `ability_scores.csv` | Per-player-season ABILITY_Score (standardized to 50±10) |
| `pitcher_features.csv` | One row per pitcher-season-level; raw inputs for STUFF/PERFORMANCE scoring: ERA, K%, BB%, K-BB%, Whiff%, GB%, PPI_skill, era labels, era-adjusted z-scores. |
| `pitcher_scores.csv` | Current 2026 pitcher prospect pool: 1,258 SP + 2,491 RP. Columns: STUFF_Score, PERFORMANCE_Score, Age_Score, Current_Score, OVR_Score, Combined_Score, SP_Rank/RP_Rank. |
| `archetype_labels.csv` | Prospect archetype assignments |
| `prospect_scores_ovr.csv` | All-time OVR scores (2006+, all prospects including graduates) |
| `prospect_scores.csv` | Current 2026 prospect pool scores + Combined_Score + dynasty positional adjustment columns (`FantasyPos`, `Pos_Bonus`, `Pos_Adj_Score`, `Pos_Adj_Rank`) |
| `prospect_scores_web.json` | JSON export for HTML artifact |
| `aaa_2026.csv` | 2026 AAA power rankings (303 players, sorted by Overall) |
| `rk_2026.csv` | 2026 Rookie ball rankings |
| `player_scores.csv` | Legacy PPPA z-score rankings (PA-weighted, 50±10 scale) |
| `2026ProspectPool.csv` | Legacy Phase 3 Combined_Score pool (v0.4 model) |
| `bwRankings.csv` | 2026 BW scout rankings (reference only; not used in blend) |
| `bw_vs_model.csv` | BW rankings vs. current model comparison |
| `tbc_prospect_rankings.csv` | Historical TBC national rankings (input to `output/build_tbc_rankings.py`) |
| `tbc_team_rankings_2026.csv` | TBC franchise prospect lists (scraped 2026; 1,770 rows) |
| `tbc_rankings.csv` | Cleaned TBC rankings output |
| `tbc_vs_model.csv` | TBC rankings vs. current model comparison |
| `situational_splits.csv` | RISP_K_pct, Platoon_Gap, Count02_K_pct per player (computed; not yet wired into scoring) |

### `data/prospectSavant/` — ProspectSavant Exports

`ps_{Level}_{year}.csv` — fetched by `fetch/fetch_prospectsavant.py`. Available levels/seasons:
- AAA: 2023–2026
- AA, A+, A, Rk: 2026 only

Columns without `.1` suffix = PS percentile ranks (0–1). `.1` suffix = raw values. Exception: `PullAir%` is raw with no .1 version; `Barrel%` in PS is a percentile — raw barrel rate is `PA/Barrels` (renamed to `Barrel%` in output).

### `data/historical/` — Research & Model History

| File | Description |
|------|-------------|
| `ovr_hist_data.csv` | Combined historical MiLB dataset (one row per player-season-stint) |
| `hist_mlb_data.csv` | MLB batting stats for graduated players |
| `regression_predictive.csv` | WLS next-season PPPA regression (17 seasons, 2007–2025) |
| `regression_mlb_outcomes.csv` | Phase 2 WLS: MiLB skills → first-year MLB PPPA_Z (N=1,612, R²=0.128) |
| `skill_scores.csv` | Phase 1 skill scores (legacy; not used in v1.0 model) |
| `prospect_mlb_proj.csv` | Phase 2 projections (legacy; not used in v1.0 model) |
| `sb_translation_chain.csv` | Chained SB discount factors by era and level |
| `stat_impact_firstyear.csv` | XGBoost + SHAP feature importance — first-year MLB PPPA |
| `stat_impact_career.csv` | XGBoost + SHAP feature importance — career MLB PPPA |

### `data/Leagues/` — League Membership

One CSV per league; used by `pipeline/compute_stats.py` to assign the `League` column in minorLeagueData.

---

## Scripts

### Full Pipeline (run in order)

```
run_pipeline.py                           # Steps 1–8: compute_stats → build_prospect_pool (legacy v0.x)
pipeline/build_prospect_features.py       # Build prospect_features.csv (TOOLS/ABILITY input features)
pipeline/add_era_columns.py               # Add EraK%, EraHR/FB, EraPPPA, EraSB columns
pipeline/add_tools_era_adjustment.py      # Add era-adjusted _adj columns (Whiff%_adj, HRFB_adj, etc.)
pipeline/build_tools_score.py             # Compute TOOLS_Score per player-season
pipeline/build_ability_score.py           # Compute ABILITY_Score per player-season
pipeline/build_prospect_scores_ovr.py     # Aggregate OVR (all-time) scores
pipeline/build_archetypes.py              # Assign archetype labels
pipeline/build_prospect_scores.py         # Aggregate current-pool scores → prospect_scores.csv
output/build_aaa_rankings.py              # 2026 AAA power rankings → aaa_2026.csv
output/build_rk_rankings.py              # 2026 Rookie rankings → rk_2026.csv
output/build_rankings_html.py            # Build HTML artifact (prospects + AAA + SP + RP tabs)

# Pitcher pipeline (run after hitter pipeline)
pipeline/build_pitcher_features.py        # Build pitcher_features.csv (STUFF/PERFORMANCE input features)
pipeline/build_pitcher_scores.py          # Aggregate current-pool pitcher scores → pitcher_scores.csv
```

### Data Fetch / Refresh (run independently)

```
fetch/fetch_mlb_data.py                # Refresh hist_mlb_data.csv from MLB Stats API (replaces FanGraphs manual export)
                                       #   incremental by default (current season only); --full refetches 2006–current
fetch/fetch_milb_data.py               # Refresh milb_hitting.csv + milb_advanced.csv from MLB Stats API
fetch/fetch_milb_pitching.py           # Refresh milb_pitching.csv + milb_pitching_advanced.csv from MLB Stats API
fetch/fetch_milb_pitches.py            # Refresh milb_pitches_agg.csv (incremental by default)
fetch/fetch_mlb_statcast.py            # Refresh mlb_statcast.csv + mlb_bat_tracking.csv from Baseball Savant (incremental by default; --full refetches 2015–current)
fetch/fetch_milb_statcast.py           # Refresh milb_statcast_aaa.csv from Baseball Savant
fetch/fetch_prospectsavant.py          # Refresh data/prospectSavant/ CSV files
fetch/merge_pitches_history.py         # Merge historical pitch game feeds into milb_pitches_agg.csv
fetch/refresh_milb_advanced.py         # Refresh milb_advanced.csv rate stats
```

### Analysis / Standalone

```
analysis/milb_impact_analysis.py       # XGBoost + SHAP feature importance (first-year + career outcomes)
analysis/sb_translation_analysis.py    # Chained SB level-to-level translation analysis (legacy; superseded)
analysis/sb_speed_analysis.py          # SB/PA + Sprint Speed MiLB->MLB translation study
analysis/skill_pppa_translation.py     # Skill_PPPA full-population level discount study (current basis)
analysis/build_player_comps.py         # Build player_comps.csv + prospect comparator tool
pipeline/build_tbc_rankings.py         # Build tbc_rankings.csv from tbc_prospect_rankings.csv
pipeline/build_situational_splits.py   # Build situational_splits.csv (RISP, platoon, count splits)
analysis/build_babip_luck.py           # Build babip_luck.csv — per-season BABIP + HR/FB luck vs career baseline
```

### Player Cards

```
cards/generate_player_card.py          # Generate points-league radar card for one player
cards/generate_player_card_cat.py      # Generate category-league card (wRC+ replaces PPPA)
cards/regen_all_cards.py               # Regenerate all existing PNG cards in PlayerCards/2026/
```

**Card output:** `~/Documents/prospectFiles/PlayerCards/2026/{Team}/{Name}.png`
**Graduated players:** Moved to `PlayerCards/2026/Graduated/{Level}/` subfolders.
**Skip list in `cards/regen_all_cards.py`:** Henry Bolte, Travis Bazzana.

---

## Pitcher Model (v1.1)

### Pitching Fantasy Scoring Formula

`TP_pit = 5×W − 3×L − 1×ER − 2×HR + 2×K − 0.5×BB + 0.75×IP + 4.25×CG + 4×SHO + 5×SV + 3×HLD − 3×BS + 2.5×QS + 1×RW − 1×RL`

**PPI_skill** (controllable rate points per inning): `2×K/IP − 0.5×BB/IP − 1×ER/IP − 2×HRA/IP + 0.75`

### Architecture

Same structure as hitter model:
```
Combined_Score = 0.50 × Current_Score + 0.50 × OVR_Score
Current_Score  = 0.30 × STUFF_Score + 0.50 × PERFORMANCE_Score + 0.20 × Age_Score
OVR_Score      = 0.40 × STUFF_OVR + 0.40 × PERF_OVR + 0.20 × Slope_Score
```

### STUFF_Score (raw physical skills)

**Weights:** K/Whiff composite (45%) + −BB% command (35%) + GB% (20%)

- K/Whiff composite: `0.5×K%_adj + 0.5×Whiff%_adj`
- Command: `−BB%_adj` (inverted — lower BB% = better)
- Contact mgmt: `GB%_adj`

Age adjustment: each component × (1 + 0.20 × −Age_Z_SL), clipped ±2 SD.

### PERFORMANCE_Score (demonstrated production)

**Weights:** ERA_adj (40%) + KBB_adj (30%) + PPI_adj (30%)

- ERA_adj: inverted (lower ERA = better)
- KBB_adj: K-BB% era-adjusted z-score
- PPI_adj: PPI_skill era-adjusted z-score

Age adjustment: same as STUFF.

### Key Differences from Hitter Model

- **No pool re-standardization**: career averages are already on 50±10 scale from per-row standardization. Re-standardizing within the current pool would double-standardize and amplify outliers (shrinkage compresses the career distribution; pool re-standardization then over-expands it). Hitter model also does this but with a smaller, more selective current pool — the effect is milder there.
- **IP shrinkage thresholds**: STUFF=167 IP (~1 full SP season), PERFORMANCE=120 IP.
- **Age cutoff**: ≤ 25 (vs. ≤ 24 for hitters).
- **Role determination**: SP if GS/G ≥ 0.5 in most recent qualifying season.
- **IP parsing**: MLB Stats API returns IP as "45.2" where ".2" = 2 outs = 2/3 IP. `parse_ip("45.2") → 45.667`.
- **QS availability**: only in `seasonAdvanced` endpoint, not `season`. Both endpoints are fetched per season×level.

### Data Sources

- `milb_pitching.csv`: counting stats (IP, K, BB, ER, HRA, W, L, SV, HLD, BS, CG, SHO, QS, WP, G, GS, BF, H, HBP, IBB, SVO).
- `milb_pitching_advanced.csv`: rate stats (K%, BB%, K-BB%, Whiff%, BABIP, GB%, LD%, FB%, GB/FB, QS).
- Era labels: same MLB-derived breaks (K%: 2015; HR/FB: 2016, 2022; PPPA: 2010, 2021).
- Level discounts: same as hitters (AAA=1.00, AA=0.59, A+=0.34, A=0.23, R=0.10).

---

## Key Design Decisions & Notes

**PlayerId system:** `data/api/milb_hitting.csv` uses FanGraphs numeric IDs (via Chadwick crosswalk) where available, MLBAM ID otherwise. `missing_milb_data.csv` uses `sa-`prefix IDs for recent players. ID mismatches handled via `_name_fill()` (normalized Name+Season+Level fallback) throughout `run_pipeline.py`.

**Name normalization:** `unicodedata.NFD` decomposition strips accents. Applied to all Name-based joins and card filename slugs. Underscore→space replacement needed when matching card filenames.

**MLB exclusion threshold:** ≥ 50 career MLB PA (via `hist_mlb_data.csv`). Allows cup-of-coffee players to remain prospect-eligible. Applied by PlayerId + normalized Name to handle ID system mismatches.

**Age limits:** `fetch/fetch_milb_data.py` fetches Age ≤ 26. `pipeline/build_prospect_scores.py` enforces Age ≤ 24. The 25–26 window exists only for AAA rankings.

**ProspectSavant Chase%/Z-Contact% column convention:** Columns without `.1` are PS percentile ranks (0–1). `.1` suffix are raw values. `Chase%`, `ZContact%`, `Whiff%`, `Barrel%` are stored in percent form (0–100 scale) in `aaa_2026.csv` — use `fmt(v, 1)`, not `pctFmt(v)` (×100 wrong).

**ProspectSavant MaxEV/EV90/Spd zero convention:** PS outputs `0.0` (not null) for players it tracks by name but lacks batted-ball or sprint data for. These zeros are physically impossible (0 mph exit velocity, 0 ft/sec sprint speed) and must be treated as null. `load_ps()` in `pipeline/build_prospect_features.py` applies `ps[col] = ps[col].where(ps[col] > 0)` for all three columns before returning. Confirmed scope: AA 2026 and A+ 2026 have MaxEV/EV90=0 for every player (619/619 and 662/662 rows); A 2026 has 466/756 zeros; Rk 2026 has 769/1268 zeros. Without this fix, those zeros participate in `z_within_level()` via `dropna()` (0 ≠ NaN), contaminating entire Level pools and producing wildly incorrect z-scores for real players in the same Level group.

**HR/FB > 1.0 nulled:** FanGraphs returns HR/FB values like 2.0 or 3.0 for rows where the fly-ball denominator is mis-counted in very small samples (PA < ~70). These are physically impossible — you can't hit more HRs than fly balls. `pipeline/build_prospect_features.py` nulls any `HR/FB > 1.0` immediately after the `milb_advanced` join, before any career computation or era-adjustment. Affected players: Cauro 2025 A (3.0), Ruiz 2015 A (2.0), Garcia 2021 A (2.0), Peraza 2024 AA (2.0). Without this fix those rows produce HRFB_adj z-scores of 26–44.

**HR/FB fallback-tier shrinkage:** In `pipeline/build_tools_score.py`, when a player has no ProspectSavant exit velocity data (Power fallback tier), HRFB_adj carries 100% of Power. Single-season HR/FB tops out at r≈0.52 even at 100 fly balls in MiLB — without EV validation, extreme single-season values are dominated by small-sample noise. Fix: `score[hrfb_only] = (hrfb × career_shrink)[hrfb_only]` where `career_shrink = min(prior_FBs_est / 150, 1.0)`. `prior_FBs_est` is `career_FBs_est` minus this row's own FB estimate — using prior-season FBs only ensures the current season cannot supply both the extreme HRFB_adj signal and the bulk of its own shrinkage weight. `career_FBs_est` is the sum of FB estimates (HR / HR/FB) across all qualifying player-season rows (HR > 0, HR/FB > 0, FB_est ≥ 15), computed in `pipeline/build_prospect_features.py`. Threshold 150 FBs ≈ 3 solid MiLB seasons. A player whose breakout year is their entire career history gets near-zero prior FBs and therefore near-zero fallback Power weight. **This shrinkage applies to the fallback tier only** — the full tier (MaxEV + EV90 available) provides independent EV validation of power, so HRFB at its 30% weight there is correct at full strength.

**2020 season:** MiLB cancelled; returns no data.

**Situational splits:** `situational_splits.csv` (RISP_K_pct, Platoon_Gap, Count02_K_pct) is built and wired into `prospect_features.csv` but not yet incorporated into TOOLS_Score or ABILITY_Score. Decision on integration is pending.

---

## Roadmap

### Situational Splits Integration (planned)
`RISP_K_pct`, `Platoon_Gap`, `Count02_K_pct` are in `prospect_features.csv` but not wired into any scoring component. Platoon_Gap and Count02_K_pct are the most likely candidates to add signal — both capture plate discipline under pressure, which complements the existing BB_2K and Whiff% discipline signals.

### Phase 5 — Heater / Slump Scores (planned)
Per-prospect best-season and worst-season scores, age-weighted and standardized within same-age same-level peers. Primary stats: K% and PPPA. Minimum 80 PA per season. Output: `heater_slump_scores.csv`. See `firstModel.md` for full research notes.

### Phase 6 — Early-Signal Watchlist (planned)
Watchlist for players with career PA ≤ 200. Formula: Age_Z_SL (50%) + K% (35%) + PPPA (15%). Minimum 80 PA floor. Restricts statistical watchlist to stateside appearances (CPX, A-ball first season). See `firstModel.md` for full research notes.

### `fetch/fetch_milb_statcast.py`
Currently has `LAST_SEASON = 2025`. Update to 2026 when Baseball Savant MiLB endpoint adds 2026 data.

---

## Model Summary

**prospectsMain v1.0** ranks minor league hitters for points-league fantasy using two complementary composites — TOOLS (physical skills) and ABILITY (demonstrated production) — blended with a career trajectory score.

**TOOLS_Score** captures what a player *can do*: plate discipline (Chase%/Z-Contact% at AAA; Whiff% elsewhere), raw power (MaxEV/EV90 where available; HR/FB fallback), and athleticism (Sprint Speed + 3B/PA). All metrics are era-adjusted within Level×era cells and age-boosted by 0.20×−Age_Z_SL per component. Weight: Discipline 45% / Power 35% / Athleticism 20%.

**ABILITY_Score** captures what a player *has done*: fantasy output (PPPA_Z_SL with level discounts), contact/discipline (BB%−2×K%), stolen base talent (SB_success × SB_rate), and game power (PullAir% + HR/AB). Same age-boost applied to all components. Weight: Output 45% / Discipline 25% / Speed 15% / Power 15%.

**Final ranking:** `Combined_Score = 0.50×Current_Score + 0.50×OVR_Score`, then a post-blend discipline gate (−1.5 to −4.0 pts) fires for prospects in the bottom 16–25% of career BB_2K, with a PA-weighted OLS slope modifier that softens the penalty for improving trajectories and hardens it for worsening ones. Current_Score weights today's pool (season ≥ 2025) with ABILITY outweighing TOOLS (50/30/20 ABILITY/TOOLS/Age) — reflecting that demonstrated PPPA-relevant production is more predictive than physical proxies, consistent with the MiLB→MLB ablation (R²≈0.13). OVR_Score weights historical all-time performance equally between TOOLS, ABILITY, and PPPA trajectory slope (40/40/20). The 50/50 blend ensures that a player's career arc (OVR) is as important as their current standing — rewarding players who have been consistently excellent and penalizing players who looked good recently but have a weaker historical profile.

**What the model prioritizes, in order:** (1) Strikeout avoidance — K% is the single largest negative lever and appears in both TOOLS (Discipline) and ABILITY (BB_2K). (2) Stolen base talent — SB=+3 per stolen base with no wRC+ analog; both SB_talent (ABILITY) and Sprint Speed (TOOLS) capture it. (3) Real power production — HR/AB and PullAir% (ABILITY Game Power) + MaxEV/HR/FB (TOOLS Raw Power). (4) Age relative to peers — baked into every component via the per-component age multiplier, plus a standalone Age_Score at 20% in Current_Score.
