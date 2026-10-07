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
    (-1.00, -1.5),   # bottom ~16% of pool: -1.5 pts  (personal: -3.0)
    (-0.67, -0.75),  # bottom ~25% of pool: -0.75 pts (personal: -1.5)
]
DISC_GATE_CAP = -2.0   # personal: -4.0

# ---------------------------------------------------------------------------
# Archetype score adjustments (added to Combined_Score after gate)
# TTO penalty dramatically reduced: high-K power hitters lose ~0.5/PA not 2/PA.
# Pure Contact bonus slightly reduced: low-K is less uniquely valuable.
# ---------------------------------------------------------------------------
ARCHETYPE_ADJ = {
    "Three True Outcomes": -1.0,   # personal: -3.0
    "Pure Contact":        +1.5,   # personal: +2.0
}

# ---------------------------------------------------------------------------
# Output directory names (relative to data/)
# ---------------------------------------------------------------------------
COMPUTED_DIR = "computed_fantrax"
RANKINGS_DIR = "rankings_fantrax"
