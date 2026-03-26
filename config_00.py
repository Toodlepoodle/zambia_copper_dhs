"""
config_00.py  —  SINGLE SOURCE OF TRUTH
=========================================
All shared constants for the Zambia copper paper pipeline.
Imported by every other script — never duplicated.

To run on a different machine, change BASE_DIR only here,
or set environment variable ZAMBIA_BASE_DIR.
"""

import os
from pathlib import Path

# =============================================================================
# PATH — edit here or set env var ZAMBIA_BASE_DIR
# =============================================================================
BASE_DIR = Path(os.environ.get("ZAMBIA_BASE_DIR",
                               r"C:\Users\Sayan\Desktop\ZAMBIA_V1"))
OUT_DIR  = BASE_DIR / "OUTPUT"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# COPPER PRICES  (LME annual average, nominal USD/metric ton)
# Sources: ICSG LME (1992-2002); IMF/FRED PCOPPUSDA (2003-2024)
# =============================================================================
COPPER_PRICES = {
    1992: 2297,
    1996: 2289,
    2002: 1558,
    2007: 7132,
    2014: 6863,
    2018: 6530,
    2024: 9142,
}

# =============================================================================
# PROVINCE HARMONISATION
# Key design decisions:
#   1. Muchinga → "northern" because Zambia split Muchinga from Northern in 2011.
#      It appears ONLY in DHS waves 2014, 2018, 2024.  Keeping it as a separate
#      province makes the panel unbalanced and breaks province FE pre/post 2014
#      (the FE for "northern" pre-2014 ≠ post-2014 if the geographic unit changed).
#      Merging back to "northern" gives 9 consistent provinces across all 7 waves.
#
#   2. All three spellings of North-Western are collapsed to "northwestern".
#      The harmonise() function also collapses internal whitespace before lookup,
#      so "north  western" (double-space) and similar variants are safe.
# =============================================================================
PROVINCE_HARMONISE = {
    "central":       "central",
    "copperbelt":    "copperbelt",
    "eastern":       "eastern",
    "luapula":       "luapula",
    "lusaka":        "lusaka",
    "muchinga":      "northern",   # Split from Northern 2011 — merge back for panel consistency
    "northern":      "northern",
    "north-western": "northwestern",
    "northwestern":  "northwestern",
    "north western": "northwestern",
    "southern":      "southern",
    "western":       "western",
}

MINING_PROVINCES = {"copperbelt", "northwestern"}

# =============================================================================
# WAVE FILES
# =============================================================================
WAVE_FILES = {
    1992: {
        "HR": "ZAMBIA1992/ZMHR21DT/ZMHR21FL.DTA",
        "KR": "ZAMBIA1992/ZMKR21DT/ZMKR21FL.DTA",
        "IR": "ZAMBIA1992/ZMIR21DT/ZMIR21FL.DTA",
    },
    1996: {
        "HR": "ZAMBIA1996/ZMHR31DT/ZMHR31FL.DTA",
        "KR": "ZAMBIA1996/ZMKR31DT/ZMKR31FL.DTA",
        "IR": "ZAMBIA1996/ZMIR31DT/ZMIR31FL.DTA",
        "MR": "ZAMBIA1996/ZMMR31DT/ZMMR31FL.DTA",
    },
    2002: {
        "HR": "ZAMBIA2002/ZMHR42DT/ZMHR42FL.DTA",
        "KR": "ZAMBIA2002/ZMKR42DT/ZMKR42FL.DTA",
        "IR": "ZAMBIA2002/ZMIR42DT/ZMIR42FL.DTA",
        "MR": "ZAMBIA2002/ZMMR41DT/ZMMR41FL.DTA",
    },
    2007: {
        "HR": "ZAMBIA2007/ZMHR51DT/ZMHR51FL.DTA",
        "KR": "ZAMBIA2007/ZMKR51DT/ZMKR51FL.DTA",
        "IR": "ZAMBIA2007/ZMIR51DT/ZMIR51FL.DTA",
        "MR": "ZAMBIA2007/ZMMR51DT/ZMMR51FL.DTA",
    },
    2014: {
        "HR": "ZAMBIA2014/ZMHR61DT/ZMHR61FL.DTA",
        "KR": "ZAMBIA2014/ZMKR61DT/ZMKR61FL.DTA",
        "IR": "ZAMBIA2014/ZMIR61DT/ZMIR61FL.DTA",
        "MR": "ZAMBIA2014/ZMMR61DT/ZMMR61FL.DTA",
    },
    2018: {
        "HR": "ZAMBIA2018/ZMHR71DT/ZMHR71FL.DTA",
        "KR": "ZAMBIA2018/ZMKR71DT/ZMKR71FL.DTA",
        "IR": "ZAMBIA2018/ZMIR71DT/ZMIR71FL.DTA",
        "MR": "ZAMBIA2018/ZMMR71DT/ZMMR71FL.DTA",
    },
    2024: {
        "HR": "ZAMBIA2024/ZMHR81DT/ZMHR81FL.dta",
        "KR": "ZAMBIA2024/ZMKR81DT/ZMKR81FL.dta",
        "IR": "ZAMBIA2024/ZMIR81DT/ZMIR81FL.dta",
        "MR": "ZAMBIA2024/ZMMR81DT/ZMMR81FL.dta",
    },
}

# =============================================================================
# DHS MISSING / FLAG CODES
# Values that must NEVER be coded as valid 0 or 1.
# Standard DHS: 8=DK(1-digit), 9=missing(1-digit),
#               96=other, 97=inconsistent, 98=DK(2-digit), 99=missing(2-digit)
# WASH-specific: 96="other source" and 99="not stated" must be NaN, not 0
# =============================================================================
DHS_MISSING_CODES = {8, 9, 96, 97, 98, 99, 997, 998, 999, 9997, 9998, 9999}

# =============================================================================
# CLASSIFICATION CODES
# =============================================================================
IMPROVED_WATER_CODES = {10, 11, 12, 13, 20, 21, 30, 31, 40, 41, 51, 71}
IMPROVED_SANIT_CODES = {11, 12, 13, 14, 15, 21, 22, 41}
IMPROVED_FLOOR_CODES = {30, 31, 32, 33, 34, 35, 36}

MIN_OBS      = 50   # minimum observations for a regression to run
MIN_CLUSTERS = 5    # minimum province clusters for reliable CR1S SE
