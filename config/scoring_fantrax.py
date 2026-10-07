"""Fantrax competitive dynasty scoring constants.

Targeting the competitive Fantrax dynasty standard used in high-stakes leagues.
Key differences from personal model:
  SO:   -0.5  (vs -2)   -- K penalty is 4x smaller; power/TTO players much less penalized
  SB:   +2    (vs +3)   -- speed still rewarded but 33% less
  RBI:  +1    (vs +2)   -- context-driven stat; less rewarded
  HBP:  +1    (new)     -- not yet in milb_hitting.csv; treated as 0 until fetch added
  CS:   -1    (vs -1.5)
  TB, IBB, GIDP: removed entirely
"""

PROFILE_NAME = "fantrax"

# ---------------------------------------------------------------------------
# Scoring weights applied in compute_stats.py
# ---------------------------------------------------------------------------
# HBP skipped: not in milb_hitting.csv. Adds ~0.01 FPPPA per PA when populated.
SCORING_WEIGHTS = {
    "1B": 1, "2B": 2, "3B": 3, "HR": 4,
    "R": 1, "RBI": 1, "BB": 1,
    "SO": -0.5, "SB": 2, "CS": -1,
}

# ---------------------------------------------------------------------------
# ABILITY Discipline component: BB% - DISC_K_MULT * K%
# Ratio follows the scoring formula: |SO_weight| / |BB_weight| = 0.5 / 1.0
# ---------------------------------------------------------------------------
DISC_K_MULT = 0.5

# ---------------------------------------------------------------------------
# Level discounts for ABILITY Fantasy Output (PPPA_Z_SL weighting)
# These should be re-derived via analysis/skill_pppa_translation_fantrax.py.
# Placeholder: same as personal model until empirically re-derived.
# Expectation: K%-heavy levels (AA, AAA) may shift slightly since K% matters less.
# ---------------------------------------------------------------------------
LEVEL_DISCOUNT = {"AAA": 1.00, "AA": 0.77, "A+": 0.39, "A": 0.29, "R": 0.12}

# ---------------------------------------------------------------------------
# Post-blend discipline gate (deducted from Combined_Score)
# Much softer than personal: K% at -0.5 is not the same cliff as at -2.0.
# Thresholds: (Disc_Composite_Z threshold, penalty_points)
# ---------------------------------------------------------------------------
DISC_GATE = [
    (-0.67, -1.25),  # bottom ~25% of pool: single tier (personal: two tiers at -1.00/-0.67)
    # Two-tier structure not empirically justified: hard (-1.00) and soft (-0.67) zones
    # produced identical Annual_TP_Z means (-0.303 vs -0.302). Real cliff is at -0.67.
]
DISC_GATE_CAP = -1.25   # single tier = cap equals penalty

# ---------------------------------------------------------------------------
# Archetype score adjustments (added to Combined_Score after gate)
# TTO penalty dramatically reduced: high-K power hitters lose ~0.5/PA not 2/PA.
# Pure Contact bonus slightly reduced: low-K is less uniquely valuable.
# ---------------------------------------------------------------------------
ARCHETYPE_ADJ = {
    "Pure Contact": +0.5,   # personal: +2.0; empirically confirmed (p=0.092, Annual_TP_Z)
    # TTO (-1.0) and Power/K-Risk (-0.6) removed: penalties dissolve when volume-weighted
    # (p=0.58 and p=0.94 vs Annual_TP_Z — TTO hitters still play and accumulate HR/BB)
}

# ---------------------------------------------------------------------------
# Blend weights for Current_Score and OVR_Score
# Empirically derived: TOOLS has near-zero independent Fantrax signal (p=0.91
# unconstrained OLS) because SO=-0.5 removes K-avoidance as TOOLS' main channel.
# ABILITY dominates; Slope carries more signal than current 0.20 weight.
#
# Personal model:  Current = 0.30/0.50/0.20 (TOOLS/ABILITY/Age)
#                  OVR     = 0.40/0.40/0.20 (TOOLS/ABILITY/Slope)
# Fantrax empirical (OVR, N=1,124): TOOLS=0.17 [0.10,0.24], ABILITY=0.53 [0.46,0.60], Slope=0.30 [0.26,0.35]
# ---------------------------------------------------------------------------
CURRENT_WEIGHTS = {"tools": 0.30, "ability": 0.70}   # personal: 0.50/0.50
OVR_WEIGHTS     = {"tools": 0.20, "ability": 0.50, "slope": 0.30}  # personal: 0.40/0.40/0.20

# TOOLS sub-component weights (Discipline / Power / Athleticism)
# Empirically derived vs Annual_TP_Z (N=1,462 graduates, constrained OLS + bootstrap):
#   Discipline: 0.45 → 0.35 (SO=-0.5 removes K-avoidance channel; emp 0.360 [0.32,0.40])
#   Power:      0.35 → 0.40 (HR=+4 unchanged; emp 0.407 [0.37,0.44])
#   Athleticism:0.20 → 0.25 (marginal gain; emp 0.233 [0.19,0.27])
TOOLS_WEIGHTS = {"disc": 0.35, "power": 0.40, "ath": 0.25}  # personal: 0.45/0.35/0.20

# ---------------------------------------------------------------------------
# Output directory names (relative to data/)
# ---------------------------------------------------------------------------
COMPUTED_DIR = "computed_fantrax"
RANKINGS_DIR = "rankings_fantrax"
