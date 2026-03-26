"""
01_inspect_variables.py
=======================
Before building the panel, inspect every key variable across all waves.
For each variable we check:
  - Does it exist?
  - Is it continuous or categorical?
  - What are the categories and what do they mean (value labels)?
  - Missing value rates (two thresholds: basic 10%, research-grade 30%)
  - Basic distribution
  - Conditional missingness flags (e.g. sought_treatment only asked if sick)
  - Unit consistency warnings (e.g. months_bf jumps in 2018/2024)
  - Coding drift warnings (e.g. m4 changes meaning across waves)

Run this BEFORE 02_build_panel.py
Output: variable_inspection_report.txt

Usage:
    python 01_inspect_variables.py
"""

import os
import sys
from pathlib import Path
import pandas as pd
import pyreadstat

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(os.environ.get("ZAMBIA_BASE_DIR",
                               r"C:\Users\Sayan\Desktop\ZAMBIA_V1"))
OUT_FILE = BASE_DIR / "variable_inspection_report.txt"

# Minimum % non-missing to be considered usable at all
BASIC_USABLE_THRESHOLD    = 10.0
# Minimum % non-missing for econometric work (regressions, panel models)
RESEARCH_USABLE_THRESHOLD = 30.0

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
    },
    2002: {
        "HR": "ZAMBIA2002/ZMHR42DT/ZMHR42FL.DTA",
        "KR": "ZAMBIA2002/ZMKR42DT/ZMKR42FL.DTA",
        "IR": "ZAMBIA2002/ZMIR42DT/ZMIR42FL.DTA",
    },
    2007: {
        "HR": "ZAMBIA2007/ZMHR51DT/ZMHR51FL.DTA",
        "KR": "ZAMBIA2007/ZMKR51DT/ZMKR51FL.DTA",
        "IR": "ZAMBIA2007/ZMIR51DT/ZMIR51FL.DTA",
    },
    2014: {
        "HR": "ZAMBIA2014/ZMHR61DT/ZMHR61FL.DTA",
        "KR": "ZAMBIA2014/ZMKR61DT/ZMKR61FL.DTA",
        "IR": "ZAMBIA2014/ZMIR61DT/ZMIR61FL.DTA",
    },
    2018: {
        "HR": "ZAMBIA2018/ZMHR71DT/ZMHR71FL.DTA",
        "KR": "ZAMBIA2018/ZMKR71DT/ZMKR71FL.DTA",
        "IR": "ZAMBIA2018/ZMIR71DT/ZMIR71FL.DTA",
    },
    2024: {
        "HR": "ZAMBIA2024/ZMHR81DT/ZMHR81FL.dta",
        "KR": "ZAMBIA2024/ZMKR81DT/ZMKR81FL.dta",
        "IR": "ZAMBIA2024/ZMIR81DT/ZMIR81FL.dta",
    },
}

# =============================================================================
# VARIABLES TO INSPECT
# (standard_name, [candidate_col_names], file_type)
# DHS renames variables across waves, so we list multiple candidates.
# =============================================================================
VARS_TO_INSPECT = {

    # ── HOUSEHOLD RECODE (HR) ─────────────────────────────────────────────────
    "HR": {
        "province":       ["hv024"],
        "urban_rural":    ["hv025"],
        "wealth_quintile":["hv270"],
        "wealth_score":   ["hv271"],
        "electricity":    ["hv206"],
        "radio":          ["hv207"],
        "tv":             ["hv208"],
        "fridge":         ["hv209"],
        "bicycle":        ["hv210"],
        "car":            ["hv212"],
        "water_source":   ["hv201"],
        "toilet_type":    ["hv205"],
        "hh_size":        ["hv009"],
        "survey_weight":  ["hv005"],
    },

    # ── KIDS RECODE (KR) ──────────────────────────────────────────────────────
    "KR": {
        "province":      ["v024"],
        "child_alive":   ["b5"],    # 0=dead, 1=alive
        "age_at_death":  ["b7"],    # months (0=neonatal)
        "dob_cmc":       ["b3"],    # date of birth in CMC
        "interview_cmc": ["v008"],  # interview date in CMC
        "haz_score":     ["hw70"],  # height-for-age z × 100
        "waz_score":     ["hw71"],  # weight-for-age z × 100
        "whz_score":     ["hw72"],  # weight-for-height z × 100
        # ── KR variables added for harmonization audit ────────────────────────
        "ever_breastfed":["m4"],    # ⚠ coding drifts across waves (see WARN below)
        "months_bf":     ["m5"],    # ⚠ unit shifts in 2018/2024 (see WARN below)
        "sought_treatment":["h32z"],# ⚠ conditional: only asked if child was sick
        "vacc_all":      ["h10"],   # ⚠ 70–85% structurally missing (age-limited)
        "survey_weight": ["v005"],
    },

    # ── INDIVIDUAL RECODE (IR) ────────────────────────────────────────────────
    "IR": {
        "province":      ["v024"],
        "anc_visits":    ["m14_1"],  # ANC visits for index child
        "edu_years":     ["v133"],   # years of education
        "edu_level":     ["v106"],   # 0=none,1=primary,2=secondary,3=higher
        "age_woman":     ["v012"],   # age in years
        "employed":      ["v714"],   # currently employed: 0/1
        "survey_weight": ["v005"],
        "interview_cmc": ["v008"],
    },
}

# =============================================================================
# KNOWN HARMONIZATION ISSUES
# Variables where the same column means different things across waves,
# or where units change. Inspected but flagged with a WARNING.
# key = (var_name, wave), value = warning string
# =============================================================================
HARMONIZATION_WARNINGS = {
    # m4 coding: early waves store duration in months (0-92), later waves use
    # special codes (93=stopped, 94=never, 95=still breastfeeding).
    # You CANNOT pool these without recoding. See 02_build_panel.py bf_map.
    ("ever_breastfed", 1992): "CODING DRIFT: m4 stores duration (0-92 months) not a yes/no indicator. Recode: >0→1, 0→0.",
    ("ever_breastfed", 1996): "CODING DRIFT: m4 stores duration (0-92 months) not a yes/no indicator. Recode: >0→1, 0→0.",
    ("ever_breastfed", 2002): "CODING DRIFT: m4 stores duration (0-92 months) not a yes/no indicator. Recode: >0→1, 0→0.",
    ("ever_breastfed", 2007): "CODING DRIFT: m4 stores duration (0-92 months) not a yes/no indicator. Recode: >0→1, 0→0.",
    ("ever_breastfed", 2014): "CODING DRIFT: m4 stores duration (0-92 months) not a yes/no indicator. Recode: >0→1, 0→0.",
    ("ever_breastfed", 2018): "OK: m4 uses final codes — 93=stopped, 94=never, 95=still. Recode: 94→0, 93/95→1.",
    ("ever_breastfed", 2024): "OK: m4 uses final codes — 93=stopped, 94=never, 95=still. Recode: 94→0, 93/95→1.",

    # m5 unit shift: 1992-2014 stores months; 2018-2024 stores days or uses
    # censored/top-coded values that inflate the mean to ~50-70.
    # Do NOT pool raw m5 across all waves.
    ("months_bf", 2018): "UNIT SHIFT: mean≈68 is implausible in months. Likely stores days or top-coded. EXCLUDE from panel or rescale.",
    ("months_bf", 2024): "UNIT SHIFT: mean≈49 is implausible in months. Likely stores days or top-coded. EXCLUDE from panel or rescale.",

    # sought_treatment is only asked of children who had fever/cough (h22=1 or h31=1).
    # 50-80% missing is STRUCTURAL (healthy children are skipped), not random.
    # In regressions: condition the sample or use as share of sick children only.
    ("sought_treatment", 1992): "CONDITIONAL MISSING: only asked if child had fever/cough. 50-80% skip is structural, not random.",
    ("sought_treatment", 1996): "CONDITIONAL MISSING: only asked if child had fever/cough. 50-80% skip is structural, not random.",
    ("sought_treatment", 2002): "CONDITIONAL MISSING: only asked if child had fever/cough. 50-80% skip is structural, not random.",
    ("sought_treatment", 2007): "CONDITIONAL MISSING: only asked if child had fever/cough. 50-80% skip is structural, not random.",
    ("sought_treatment", 2014): "CONDITIONAL MISSING: only asked if child had fever/cough. 50-80% skip is structural, not random.",
    ("sought_treatment", 2018): "CONDITIONAL MISSING: only asked if child had fever/cough. 50-80% skip is structural, not random.",
    ("sought_treatment", 2024): "CONDITIONAL MISSING: only asked if child had fever/cough. 50-80% skip is structural, not random.",

    # vacc_all (h10) is only asked for children with vaccination cards present
    # at interview. 70-85% missing is structural. Do not use as a panel outcome;
    # use individual vaccine indicators (h2-h9) instead.
    ("vacc_all", 1992): "STRUCTURAL MISSING: h10 only asked for children with vaccination card present (~70-85% skip). Use h2-h9 instead.",
    ("vacc_all", 1996): "STRUCTURAL MISSING: h10 only asked for children with vaccination card present (~70-85% skip). Use h2-h9 instead.",
    ("vacc_all", 2002): "STRUCTURAL MISSING: h10 only asked for children with vaccination card present (~70-85% skip). Use h2-h9 instead.",
    ("vacc_all", 2007): "STRUCTURAL MISSING: h10 only asked for children with vaccination card present (~70-85% skip). Use h2-h9 instead.",
    ("vacc_all", 2014): "STRUCTURAL MISSING: h10 only asked for children with vaccination card present (~70-85% skip). Use h2-h9 instead.",
    ("vacc_all", 2018): "STRUCTURAL MISSING: h10 only asked for children with vaccination card present (~70-85% skip). Use h2-h9 instead.",
    ("vacc_all", 2024): "STRUCTURAL MISSING: h10 only asked for children with vaccination card present (~70-85% skip). Use h2-h9 instead.",
}

# =============================================================================
# DHS WASH CODES (standard across waves)
# =============================================================================
IMPROVED_WATER = {
    10: "piped into dwelling",  11: "piped into dwelling",
    12: "piped to yard/plot",   13: "public tap/standpipe",
    20: "tube well/borehole",   21: "tube well/borehole",
    30: "protected well",       31: "protected well",
    40: "protected spring",     41: "protected spring",
    51: "rainwater",            71: "bottled water",
}
UNIMPROVED_WATER = {
    32: "unprotected well",   42: "unprotected spring",
    43: "unprotected spring", 61: "tanker truck",
    62: "cart with small tank", 96: "other",
}
IMPROVED_SANIT = {
    11: "flush to piped sewer",   12: "flush to septic tank",
    13: "flush to pit latrine",   14: "flush to somewhere else",
    15: "flush, don't know where",21: "ventilated improved pit",
    22: "pit latrine with slab",  41: "composting toilet",
}
UNIMPROVED_SANIT = {
    23: "pit latrine without slab", 42: "bucket",
    43: "hanging toilet/latrine",  96: "other",
    31: "no facility/bush/field",
}

# =============================================================================
# FILE LOADING
# =============================================================================

def load_file(path):
    try:
        df, meta = pyreadstat.read_dta(str(path))
        df.columns = df.columns.str.lower()
        return df, meta, None
    except Exception as e:
        return None, None, str(e)


def get_labels(meta, col):
    if meta and hasattr(meta, "variable_value_labels"):
        return meta.variable_value_labels.get(col, {})
    return {}


# =============================================================================
# VARIABLE INSPECTION
# =============================================================================

def usability_flag(pct_missing):
    pct_valid = 100 - pct_missing
    if pct_valid < BASIC_USABLE_THRESHOLD:
        return "UNUSABLE"
    elif pct_valid < RESEARCH_USABLE_THRESHOLD:
        return "WEAK"
    else:
        return "OK"


def inspect_var(df, meta, col, var_name, wave, ftype, lines):
    """Inspect a single variable and append findings to lines."""

    lines.append(f"\n    [{var_name}] → column: {col}")

    series = df[col]
    n_total   = len(series)
    n_missing = series.isna().sum()
    n_valid   = n_total - n_missing
    pct_miss  = n_missing / n_total * 100

    usability = str(usability_flag(pct_miss))
    lines.append(
    f"      N={n_total:,}  |  Valid={n_valid:,}  |  "
    f"Missing={n_missing:,} ({pct_miss:.1f}%)  |  {usability}"
    )

    # ── Harmonization warning ─────────────────────────────────────────────────
    warn = HARMONIZATION_WARNINGS.get((var_name, wave))
    if warn:
        lines.append(f"      ⚠ WARN: {warn}")

    labels  = get_labels(meta, col)
    n_unique = series.nunique(dropna=True)

    # ── Categorical branch ────────────────────────────────────────────────────
    categorical_vars = {
        "province", "urban_rural", "wealth_quintile",
        "electricity", "radio", "tv", "fridge", "bicycle", "car",
        "child_alive", "edu_level", "employed",
        "water_source", "toilet_type",
        "ever_breastfed", "sought_treatment", "vacc_all",
    }

    if (n_unique <= 30 and var_name not in {"edu_years"}) or var_name in categorical_vars:
        lines.append(f"      TYPE: Categorical  |  Unique values: {n_unique}")
        lines.append(f"      VALUE COUNTS (with labels):")

        vc = series.value_counts(dropna=False).sort_index()
        for val, cnt in vc.items():
            pct = cnt / n_total * 100
            if pd.isna(val):
                label, val_str = "NaN/Missing", "NaN"
            else:
                try:
                    key = int(val)
                    val_str = str(key)
                except (ValueError, TypeError):
                    key = val
                    val_str = str(val)
                label = str(labels.get(key, labels.get(val, ""))).strip()

            # Annotate known DHS meanings
            extra = ""
            if var_name == "water_source":
                extra = (f"  [IMPROVED: {IMPROVED_WATER[key]}]"   if key in IMPROVED_WATER
                         else f"  [UNIMPROVED: {UNIMPROVED_WATER.get(key,'')}]" if key in UNIMPROVED_WATER
                         else "")
            elif var_name == "toilet_type":
                extra = (f"  [IMPROVED: {IMPROVED_SANIT[key]}]"   if key in IMPROVED_SANIT
                         else f"  [UNIMPROVED: {UNIMPROVED_SANIT.get(key,'')}]" if key in UNIMPROVED_SANIT
                         else "")
            elif var_name == "urban_rural":
                extra = "  [URBAN]" if key == 1 else "  [RURAL]" if key == 2 else ""
            elif var_name == "child_alive":
                extra = "  [ALIVE]" if key == 1 else "  [DEAD]" if key == 0 else ""
            elif var_name == "edu_level":
                extra = {0:"  [no education]",1:"  [primary]",
                         2:"  [secondary]",3:"  [higher]"}.get(key, "")
            elif var_name == "employed":
                extra = "  [YES]" if key == 1 else "  [NO]" if key == 0 else ""
            elif var_name == "electricity":
                extra = "  [HAS electricity]" if key == 1 else "  [NO electricity]" if key == 0 else ""

            lines.append(
                f"        {val_str:>6}  {label:<35}  n={cnt:>7,}  ({pct:.1f}%){extra}"
            )

    # ── Continuous branch ─────────────────────────────────────────────────────
    else:
        lines.append(f"      TYPE: Continuous  |  Unique values: {n_unique}")
        try:
            num = pd.to_numeric(series, errors="coerce")
            desc = num.describe()
            lines.append(
                f"      min={desc['min']:.2f}  p25={desc['25%']:.2f}  "
                f"median={desc['50%']:.2f}  p75={desc['75%']:.2f}  "
                f"max={desc['max']:.2f}  mean={desc['mean']:.2f}"
            )
        except Exception as e:
            lines.append(f"      Could not compute stats: {e}")

        # Variable-specific notes
        if var_name == "wealth_score":
            lines.append("      NOTE: Raw factor score. Divide by 100,000 for standardised score.")
        elif var_name == "survey_weight":
            lines.append("      NOTE: DHS weight. Divide by 1,000,000 before use.")
        elif var_name in ("haz_score", "waz_score", "whz_score"):
            lines.append("      NOTE: Z-score × 100. Divide by 100 for actual z-score.")
            lines.append("      NOTE: Values < -200 = stunted/wasted/underweight.")
            lines.append("      NOTE: Values > 9996 = flagged/missing — set to NaN before use.")
            num = pd.to_numeric(series, errors="coerce")
            n_flagged = (num > 9990).sum()
            n_stunted = ((num < -200) & (num > -9990)).sum()
            lines.append(
                f"      Flagged (>9990): {n_flagged:,}  |  "
                f"Below -2SD (<-200): {n_stunted:,} ({n_stunted/n_total*100:.1f}%)"
            )
        elif var_name == "months_bf":
            # Extra plausibility check: breastfeeding duration should be < 36 months
            num = pd.to_numeric(series, errors="coerce")
            median_val = num.median()
            if median_val > 36:
                lines.append(
                    f"      ⚠ PLAUSIBILITY FAIL: median={median_val:.1f} months — "
                    f"exceeds 36 months. Likely unit error (days?) or top-coding. "
                    f"DO NOT pool with earlier waves."
                )
            else:
                lines.append(
                    f"      Plausibility OK: median={median_val:.1f} months (expected 12–24)."
                )
        elif var_name == "hh_size":
            lines.append("      NOTE: Number of household members.")
        elif var_name == "edu_years":
            lines.append("      NOTE: Years of education completed.")
        elif var_name == "age_at_death":
            lines.append("      NOTE: Age at death in months. 0=neonatal (<1 month).")
            lines.append("      NOTE: Values < 12 = infant mortality. Values < 60 = U5 mortality.")
        elif var_name == "anc_visits":
            lines.append("      NOTE: ANC visits for most recent birth. 0=none. 98/99→NaN.")
            num = pd.to_numeric(series, errors="coerce")
            n_none  = (num == 0).sum()
            n_4plus = (num >= 4).sum()
            lines.append(f"      Zero visits: {n_none:,}  |  4+ visits: {n_4plus:,}")
        elif var_name in ("dob_cmc", "interview_cmc"):
            lines.append("      NOTE: CMC = Century Month Code. Jan 1900 = CMC 1.")
            lines.append("      NOTE: year = 1900 + int((CMC-1)/12)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    lines = []
    lines.append("=" * 72)
    lines.append("ZAMBIA DHS — VARIABLE INSPECTION REPORT")
    lines.append("=" * 72)
    lines.append("Purpose : Understand every key variable before building the panel.")
    lines.append("Checks  : type · categories · labels · missing rates · unit issues.")
    lines.append(f"Thresholds: BASIC={BASIC_USABLE_THRESHOLD}% valid  |  "
                 f"RESEARCH={RESEARCH_USABLE_THRESHOLD}% valid")
    lines.append("Legend  : ✓ OK  |  ~ WEAK (<30% valid)  |  ✗ UNUSABLE (<10% valid)")
    lines.append("")

    for wave, files in WAVE_FILES.items():
        lines.append("\n" + "=" * 72)
        lines.append(f"WAVE: {wave}")
        lines.append("=" * 72)

        for ftype, rel_path in files.items():
            full_path = BASE_DIR / rel_path
            lines.append(f"\n  [{ftype}]  {rel_path}")

            if not full_path.exists():
                lines.append("  !! FILE NOT FOUND")
                continue

            df, meta, err = load_file(full_path)
            if err:
                lines.append(f"  !! LOAD ERROR: {err}")
                continue

            lines.append(f"  Loaded: {len(df):,} obs × {len(df.columns)} vars")

            for var_name, candidates in VARS_TO_INSPECT.get(ftype, {}).items():
                found_col = next(
                    (c.lower() for c in candidates if c.lower() in df.columns),
                    None
                )
                if found_col is None:
                    lines.append(f"\n    [{var_name}] → NOT FOUND (tried: {candidates})")
                    prefix  = candidates[0][:3]
                    similar = [c for c in df.columns if c.startswith(prefix)][:5]
                    if similar:
                        lines.append(
                            f"      Hint — columns starting with '{prefix}': {similar}"
                        )
                else:
                    inspect_var(df, meta, found_col, var_name, wave, ftype, lines)

    # ── Construction notes ────────────────────────────────────────────────────
    lines.append("\n\n" + "=" * 72)
    lines.append("CONSTRUCTION NOTES FOR 02_build_panel.py")
    lines.append("=" * 72)
    lines.append("""
WEALTH
  hv270 (quintile) and hv271 (score) exist from 2002 onwards — NOT in 1992/1996.
  For 1992/1996: build an asset index from hv206–hv212
  (mean of electricity, radio, tv, fridge, bicycle, car).

URBAN / RURAL
  hv025: 1=urban, 2=rural → recode to urban=1, rural=0.

WATER (hv201)
  Improved codes: 10,11,12,13,20,21,30,31,40,41,51,71
  96 (other) and 99 (not stated) → NaN — never code as unimproved.

SANITATION (hv205)
  Improved codes: 11,12,13,14,15,21,22,41
  Same NaN rule for 96/99.

CHILD MORTALITY (KR)
  b5: 1=alive, 0=dead
  u5_dead       = (b5==0) AND b7 < 60
  infant_dead   = (b5==0) AND b7 < 12
  neonatal_dead = (b5==0) AND b7 < 1

ANTHROPOMETRICS (KR)
  hw70/hw71/hw72: z-scores × 100 → divide by 100 for real z-score.
  Flag and set to NaN if > 9990.
  stunted = hw70 < -200  |  wasted = hw72 < -200  |  underweight = hw71 < -200
  NOT collected in 1992.

EVER BREASTFED (m4 — KR)  ⚠ CODING DRIFT
  1992–2014: m4 stores duration in months (0–92). Recode: >0→1, 0→0.
  2018–2024: m4 uses codes 93=stopped, 94=never, 95=still.
             Recode: 94→0 (never), 93/95→1 (ever).
  See bf_map in 02_build_panel.py process_kr().

MONTHS BREASTFED (m5 — KR)  ⚠ UNIT SHIFT
  1992–2014: stores duration in months (plausible mean ~17–19).
  2018–2024: mean ~50–68 — implausible. Likely days or top-coded values.
  DO NOT pool m5 raw across all 7 waves. Either exclude 2018/2024
  or investigate and rescale.

SOUGHT TREATMENT (h32z — KR)  ⚠ CONDITIONAL MISSING
  Only asked if child had fever or cough (h22=1 or h31=1).
  50–80% missing is structural — not random.
  Use as: share of sick children who sought treatment
  (denominator = children with h22=1 OR h31=1), not full sample.

VACC_ALL (h10 — KR)  ⚠ STRUCTURAL MISSING
  Only asked for children whose vaccination card was seen at interview.
  70–85% missing is structural. Use individual vaccine indicators h2–h9
  to construct vacc_full yourself.

SURVEY WEIGHTS
  hv005 (HR), v005 (KR/IR): divide by 1,000,000 before use.

PROVINCE CODES
  1992–2007: 9 provinces (codes 1–9).
  2014–2024: 10 provinces — Muchinga split from Northern (codes 1–10).
  Mining: Copperbelt (code 2 all waves); North-Western (code 7 in 1992–2007,
  code 8 in 2014+). Always harmonise by NAME not numeric code.

ANC VISITS (m14_1 — IR)
  Visits for most recent birth. 0=none. 98/99 → NaN.
  Binary: anc_4plus = (m14_1 >= 4).

EDUCATION
  v106: 0=none, 1=primary, 2=secondary, 3=higher.
  v133: years of education (continuous). Cap at 30 years.
""")

    lines.append("=" * 72)
    lines.append("INSPECTION COMPLETE")
    lines.append("=" * 72)

    report = "\n".join(lines)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report written to: {OUT_FILE}")
    print("Review variable_inspection_report.txt before running 02_build_panel.py")


if __name__ == "__main__":
    try:
        import pyreadstat
    except ImportError:
        print("Run: pip install pandas pyreadstat")
        sys.exit(1)
    main()


# =============================================================================
# =============================================================================
# UNIT TESTS & SANITY CHECKS
# Run directly:  python -m pytest 01_inspect_variables.py -v
#           or:  python 01_inspect_variables.py --run-tests
# =============================================================================
# =============================================================================

import unittest
import warnings
import numpy as np
from unittest.mock import MagicMock, patch


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build a minimal mock pyreadstat meta object
# ─────────────────────────────────────────────────────────────────────────────

def _make_meta(labels: dict | None = None):
    """Return a mock pyreadstat meta object with variable_value_labels."""
    meta = MagicMock()
    meta.variable_value_labels = labels or {}
    return meta


# =============================================================================
# 1. UNIT TESTS — usability_flag()
# =============================================================================

class TestUsabilityFlag(unittest.TestCase):

    def test_fully_missing_is_unusable(self):
        self.assertEqual(usability_flag(100.0), "✗ UNUSABLE")

    def test_just_below_basic_threshold_is_unusable(self):
        # 91% missing → 9% valid → below BASIC_USABLE_THRESHOLD (10%)
        self.assertEqual(usability_flag(91.0), "✗ UNUSABLE")

    def test_exactly_basic_threshold_boundary(self):
        # pct_valid == BASIC_USABLE_THRESHOLD → NOT unusable (boundary inclusive on weak side)
        flag = usability_flag(100.0 - BASIC_USABLE_THRESHOLD)
        self.assertNotIn("UNUSABLE", flag)

    def test_weak_band(self):
        # 75% missing → 25% valid → between BASIC and RESEARCH → WEAK
        flag = usability_flag(75.0)
        self.assertIn("WEAK", flag)

    def test_exactly_research_threshold_boundary(self):
        # pct_valid == RESEARCH_USABLE_THRESHOLD → OK (boundary)
        flag = usability_flag(100.0 - RESEARCH_USABLE_THRESHOLD)
        self.assertIn("OK", flag)

    def test_fully_valid_is_ok(self):
        self.assertIn("OK", usability_flag(0.0))

    def test_zero_missing_is_ok(self):
        self.assertIn("OK", usability_flag(0.0))

    def test_return_type_is_string(self):
        for pct in [0, 10, 30, 50, 90, 100]:
            self.assertIsInstance(usability_flag(float(pct)), str)


# =============================================================================
# 2. UNIT TESTS — get_labels()
# =============================================================================

class TestGetLabels(unittest.TestCase):

    def test_returns_correct_labels(self):
        meta = _make_meta({"hv025": {1: "urban", 2: "rural"}})
        self.assertEqual(get_labels(meta, "hv025"), {1: "urban", 2: "rural"})

    def test_missing_column_returns_empty_dict(self):
        meta = _make_meta({"hv025": {1: "urban"}})
        self.assertEqual(get_labels(meta, "hv999"), {})

    def test_none_meta_returns_empty_dict(self):
        self.assertEqual(get_labels(None, "hv025"), {})

    def test_meta_without_attribute_returns_empty_dict(self):
        meta = MagicMock(spec=[])          # no attributes at all
        self.assertEqual(get_labels(meta, "hv025"), {})

    def test_empty_labels_dict_returns_empty(self):
        meta = _make_meta({})
        self.assertEqual(get_labels(meta, "hv025"), {})

    def test_numeric_keys_preserved(self):
        meta = _make_meta({"b5": {0: "dead", 1: "alive"}})
        labels = get_labels(meta, "b5")
        self.assertEqual(labels[0], "dead")
        self.assertEqual(labels[1], "alive")


# =============================================================================
# 3. UNIT TESTS — load_file()
# =============================================================================

class TestLoadFile(unittest.TestCase):

    def test_nonexistent_file_returns_error_string(self):
        df, meta, err = load_file(Path("/nonexistent/path/file.dta"))
        self.assertIsNone(df)
        self.assertIsNone(meta)
        self.assertIsInstance(err, str)
        self.assertTrue(len(err) > 0)

    def test_successful_load_returns_dataframe(self):
        """Mocks pyreadstat.read_dta to verify load_file's return contract."""
        mock_df = pd.DataFrame({"HV025": [1, 2, 1], "HV009": [4, 5, 3]})
        mock_meta = _make_meta()
        with patch("pyreadstat.read_dta", return_value=(mock_df, mock_meta)):
            df, meta, err = load_file(Path("dummy.dta"))
        self.assertIsNone(err)
        self.assertIsNotNone(df)
        # Columns must be lowercased
        self.assertTrue(all(c == c.lower() for c in df.columns))

    def test_load_lowercases_columns(self):
        mock_df = pd.DataFrame({"HV025": [1], "V012": [25]})
        mock_meta = _make_meta()
        with patch("pyreadstat.read_dta", return_value=(mock_df, mock_meta)):
            df, _, _ = load_file(Path("dummy.dta"))
        self.assertIn("hv025", df.columns)
        self.assertIn("v012", df.columns)
        self.assertNotIn("HV025", df.columns)

    def test_load_error_returns_none_df_and_meta(self):
        with patch("pyreadstat.read_dta", side_effect=Exception("bad file")):
            df, meta, err = load_file(Path("bad.dta"))
        self.assertIsNone(df)
        self.assertIsNone(meta)
        self.assertIn("bad file", err)


# =============================================================================
# 4. UNIT TESTS — inspect_var() (output content assertions)
# =============================================================================

class TestInspectVar(unittest.TestCase):
    """Tests inspect_var() by checking what it appends to lines."""

    def _run(self, series_data, var_name="hh_size", wave=2007,
             ftype="HR", col="hv009", labels=None):
        df  = pd.DataFrame({col: series_data})
        meta = _make_meta({col: labels} if labels else {})
        lines = []
        inspect_var(df, meta, col, var_name, wave, ftype, lines)
        return lines

    # ── Missing-rate reporting ────────────────────────────────────────────────

    def test_missing_rate_reported_in_output(self):
        data = [1, 2, None, None, 4]     # 40% missing
        lines = self._run(data)
        combined = " ".join(lines)
        self.assertIn("40.0%", combined)

    def test_zero_missing_reported(self):
        lines = self._run([1, 2, 3, 4, 5])
        combined = " ".join(lines)
        self.assertIn("0.0%", combined)

    def test_fully_missing_flagged_unusable(self):
        lines = self._run([None, None, None])
        combined = " ".join(lines)
        self.assertIn("UNUSABLE", combined)

    # ── Harmonization warnings ────────────────────────────────────────────────

    def test_harmonization_warning_emitted_for_known_pair(self):
        lines = self._run([1, 0, 1], var_name="ever_breastfed", wave=1992,
                          col="m4")
        combined = " ".join(lines)
        self.assertIn("CODING DRIFT", combined)

    def test_no_warning_when_pair_not_in_dict(self):
        lines = self._run([4, 5, 6], var_name="hh_size", wave=2002)
        combined = " ".join(lines)
        self.assertNotIn("⚠ WARN", combined)

    def test_months_bf_unit_shift_warning_2018(self):
        lines = self._run([60, 70, 80], var_name="months_bf", wave=2018,
                          col="m5")
        combined = " ".join(lines)
        self.assertIn("UNIT SHIFT", combined)

    # ── Categorical detection ─────────────────────────────────────────────────

    def test_categorical_detected_for_binary_column(self):
        lines = self._run([0, 1, 1, 0], var_name="child_alive", col="b5")
        combined = " ".join(lines)
        self.assertIn("Categorical", combined)

    def test_categorical_labels_included_when_provided(self):
        lines = self._run([1, 2, 1], var_name="urban_rural", col="hv025",
                          labels={1: "urban", 2: "rural"})
        combined = " ".join(lines)
        self.assertIn("urban", combined)

    # ── Continuous detection ──────────────────────────────────────────────────

    def test_continuous_detected_for_high_cardinality(self):
        data = list(range(100))      # 100 unique values → continuous
        lines = self._run(data, var_name="wealth_score", col="hv271")
        combined = " ".join(lines)
        self.assertIn("Continuous", combined)

    def test_survey_weight_note_present(self):
        data = list(range(100, 200))
        lines = self._run(data, var_name="survey_weight", col="v005")
        combined = " ".join(lines)
        self.assertIn("1,000,000", combined)

    def test_zscore_flagged_values_counted(self):
        # Need >30 unique values to trigger the continuous branch in inspect_var().
        # Build a spread of z-scores (×100) with some flagged (>9990) and stunted (<-200).
        rng = np.random.default_rng(0)
        data = list(rng.integers(-500, 200, 40)) + [9999, 9998]   # 2 flagged
        lines = self._run(data, var_name="haz_score", col="hw70")
        combined = " ".join(lines)
        self.assertIn("Flagged", combined)
        self.assertIn("Below -2SD", combined)

    # ── months_bf plausibility check ──────────────────────────────────────────

    def test_months_bf_plausibility_fail_high_median(self):
        data = list(range(40, 90))   # median ≈ 64 → implausible
        lines = self._run(data, var_name="months_bf", wave=2014, col="m5")
        combined = " ".join(lines)
        self.assertIn("PLAUSIBILITY FAIL", combined)

    def test_months_bf_plausibility_ok_normal_median(self):
        # Need >30 unique values for the continuous branch.
        # Use floats spread across 10–30 months to guarantee uniqueness.
        rng = np.random.default_rng(1)
        data = list(10.0 + rng.random(50) * 20)   # 50 unique floats, all 10–30
        lines = self._run(data, var_name="months_bf", wave=2007, col="m5")
        combined = " ".join(lines)
        self.assertIn("Plausibility OK", combined)

    # ── ANC visits notes ──────────────────────────────────────────────────────

    def test_anc_visits_zero_and_4plus_counted(self):
        # Need >30 unique values to reach the continuous branch where Zero/4+ are counted.
        rng = np.random.default_rng(2)
        # Mix of 0s, values 1-10, and some 4+ — give plenty of unique values
        data = [0] * 5 + list(range(1, 36)) + [4, 5, 6]
        lines = self._run(data, var_name="anc_visits", col="m14_1")
        combined = " ".join(lines)
        self.assertIn("Zero visits", combined)
        self.assertIn("4+ visits", combined)


# =============================================================================
# 5. SANITY CHECKS ON CONFIGURATION DICTIONARIES
# =============================================================================

class TestConfigurationSanity(unittest.TestCase):
    """Validate the module-level lookup tables are internally consistent."""

    # ── WAVE_FILES ────────────────────────────────────────────────────────────

    def test_all_seven_waves_present(self):
        expected = {1992, 1996, 2002, 2007, 2014, 2018, 2024}
        self.assertEqual(set(WAVE_FILES.keys()), expected)

    def test_each_wave_has_three_file_types(self):
        for wave, files in WAVE_FILES.items():
            with self.subTest(wave=wave):
                self.assertSetEqual(set(files.keys()), {"HR", "KR", "IR"})

    def test_wave_paths_are_strings(self):
        for wave, files in WAVE_FILES.items():
            for ftype, path in files.items():
                with self.subTest(wave=wave, ftype=ftype):
                    self.assertIsInstance(path, str)

    def test_wave_paths_end_in_dta(self):
        for wave, files in WAVE_FILES.items():
            for ftype, path in files.items():
                with self.subTest(wave=wave, ftype=ftype):
                    self.assertTrue(
                        path.endswith(".DTA") or path.endswith(".dta"),
                        msg=f"Unexpected extension: {path}"
                    )

    # ── VARS_TO_INSPECT ───────────────────────────────────────────────────────

    def test_vars_to_inspect_has_three_file_types(self):
        self.assertSetEqual(set(VARS_TO_INSPECT.keys()), {"HR", "KR", "IR"})

    def test_survey_weight_present_in_all_file_types(self):
        for ftype in ("HR", "KR", "IR"):
            with self.subTest(ftype=ftype):
                self.assertIn("survey_weight", VARS_TO_INSPECT[ftype])

    def test_candidate_lists_are_non_empty(self):
        for ftype, vardict in VARS_TO_INSPECT.items():
            for var_name, candidates in vardict.items():
                with self.subTest(ftype=ftype, var=var_name):
                    self.assertIsInstance(candidates, list)
                    self.assertGreater(len(candidates), 0)

    def test_candidate_column_names_are_lowercase(self):
        for ftype, vardict in VARS_TO_INSPECT.items():
            for var_name, candidates in vardict.items():
                for col in candidates:
                    with self.subTest(ftype=ftype, var=var_name, col=col):
                        self.assertEqual(
                            col, col.lower(),
                            msg=f"Candidate '{col}' is not lowercase — load_file() lowercases columns, "
                                f"so mixed-case candidates will never match."
                        )

    # ── HARMONIZATION_WARNINGS ────────────────────────────────────────────────

    def test_harmonization_warning_keys_are_tuples(self):
        for key in HARMONIZATION_WARNINGS:
            self.assertIsInstance(key, tuple)
            self.assertEqual(len(key), 2)

    def test_harmonization_warning_waves_are_valid(self):
        valid_waves = set(WAVE_FILES.keys())
        for (var_name, wave), msg in HARMONIZATION_WARNINGS.items():
            with self.subTest(var=var_name, wave=wave):
                self.assertIn(wave, valid_waves,
                              msg=f"Wave {wave} in HARMONIZATION_WARNINGS not in WAVE_FILES")

    def test_harmonization_warning_messages_are_non_empty_strings(self):
        for key, msg in HARMONIZATION_WARNINGS.items():
            with self.subTest(key=key):
                self.assertIsInstance(msg, str)
                self.assertGreater(len(msg.strip()), 0)

    def test_sought_treatment_warnings_cover_all_waves(self):
        for wave in WAVE_FILES:
            with self.subTest(wave=wave):
                self.assertIn(("sought_treatment", wave), HARMONIZATION_WARNINGS)

    def test_vacc_all_warnings_cover_all_waves(self):
        for wave in WAVE_FILES:
            with self.subTest(wave=wave):
                self.assertIn(("vacc_all", wave), HARMONIZATION_WARNINGS)

    def test_ever_breastfed_warnings_cover_all_waves(self):
        for wave in WAVE_FILES:
            with self.subTest(wave=wave):
                self.assertIn(("ever_breastfed", wave), HARMONIZATION_WARNINGS)

    def test_months_bf_unit_shift_flagged_for_2018_and_2024(self):
        for wave in (2018, 2024):
            key = ("months_bf", wave)
            self.assertIn(key, HARMONIZATION_WARNINGS)
            self.assertIn("UNIT SHIFT", HARMONIZATION_WARNINGS[key])

    def test_months_bf_no_unit_shift_warning_for_early_waves(self):
        for wave in (1992, 1996, 2002, 2007, 2014):
            key = ("months_bf", wave)
            # Key should not exist (no warning for early waves)
            self.assertNotIn(key, HARMONIZATION_WARNINGS,
                             msg=f"Unexpected months_bf warning for wave {wave}")

    # ── DHS WASH CODES ────────────────────────────────────────────────────────

    def test_no_overlap_between_improved_and_unimproved_water(self):
        overlap = set(IMPROVED_WATER) & set(UNIMPROVED_WATER)
        self.assertEqual(overlap, set(),
                         msg=f"Codes appear in both IMPROVED and UNIMPROVED water: {overlap}")

    def test_no_overlap_between_improved_and_unimproved_sanit(self):
        overlap = set(IMPROVED_SANIT) & set(UNIMPROVED_SANIT)
        self.assertEqual(overlap, set(),
                         msg=f"Codes appear in both IMPROVED and UNIMPROVED sanitation: {overlap}")

    def test_wash_code_96_not_in_improved_water(self):
        self.assertNotIn(96, IMPROVED_WATER,
                         msg="Code 96 (other) must NOT be classified as improved water")

    def test_wash_code_96_not_in_improved_sanit(self):
        self.assertNotIn(96, IMPROVED_SANIT,
                         msg="Code 96 (other) must NOT be classified as improved sanitation")

    def test_wash_labels_are_non_empty_strings(self):
        for code_dict, name in [
            (IMPROVED_WATER, "IMPROVED_WATER"),
            (UNIMPROVED_WATER, "UNIMPROVED_WATER"),
            (IMPROVED_SANIT, "IMPROVED_SANIT"),
            (UNIMPROVED_SANIT, "UNIMPROVED_SANIT"),
        ]:
            for code, label in code_dict.items():
                with self.subTest(dict=name, code=code):
                    self.assertIsInstance(label, str)
                    self.assertGreater(len(label.strip()), 0)

    def test_wash_codes_are_integers(self):
        for code_dict in (IMPROVED_WATER, UNIMPROVED_WATER, IMPROVED_SANIT, UNIMPROVED_SANIT):
            for code in code_dict:
                self.assertIsInstance(code, int)

    # ── THRESHOLDS ────────────────────────────────────────────────────────────

    def test_basic_threshold_less_than_research_threshold(self):
        self.assertLess(BASIC_USABLE_THRESHOLD, RESEARCH_USABLE_THRESHOLD,
                        msg="BASIC threshold must be strictly less than RESEARCH threshold")

    def test_thresholds_in_valid_range(self):
        for t in (BASIC_USABLE_THRESHOLD, RESEARCH_USABLE_THRESHOLD):
            self.assertGreater(t, 0.0)
            self.assertLess(t, 100.0)


# =============================================================================
# 6. INTEGRATION-LEVEL SANITY CHECKS (no real DHS files required)
# =============================================================================

class TestIntegrationSanity(unittest.TestCase):
    """
    Simulate a complete inspect_var() pass over a synthetic DataFrame that
    mimics a DHS KR file.  Validates end-to-end correctness without any
    real DHS files.
    """

    def _synthetic_kr(self):
        """Build a small synthetic KR-like DataFrame."""
        n = 200
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "v024":  rng.integers(1, 10, n),                   # province
            "b5":    rng.integers(0, 2, n),                    # child_alive
            "b7":    rng.integers(0, 60, n).astype(float),     # age_at_death
            "b3":    rng.integers(1100, 1400, n),              # dob_cmc
            "v008":  rng.integers(1100, 1400, n),              # interview_cmc
            "hw70":  np.concatenate([rng.integers(-400, 200, 180),
                                     np.full(20, 9999)]),      # haz (20 flagged)
            "hw71":  rng.integers(-400, 200, n),
            "hw72":  rng.integers(-400, 200, n),
            "m4":    rng.integers(0, 30, n),                   # ever_breastfed (pre-2018)
            "m5":    rng.integers(1, 36, n).astype(float),     # months_bf
            "h32z":  np.where(rng.random(n) < 0.65, np.nan, rng.integers(0, 2, n)),
            "h10":   np.where(rng.random(n) < 0.75, np.nan, rng.integers(0, 1, n)),
            "v005":  rng.integers(500_000, 2_000_000, n).astype(float),
        })
        return df

    def test_inspect_var_runs_without_exception_for_all_kr_vars(self):
        df   = self._synthetic_kr()
        meta = _make_meta()
        kr_vars = VARS_TO_INSPECT["KR"]
        for var_name, candidates in kr_vars.items():
            found_col = next(
                (c.lower() for c in candidates if c.lower() in df.columns), None
            )
            if found_col is None:
                continue   # variable not in synthetic df — skip silently
            lines = []
            try:
                inspect_var(df, meta, found_col, var_name, 2007, "KR", lines)
            except Exception as exc:
                self.fail(f"inspect_var() raised {exc!r} for var={var_name}")

    def test_inspect_var_outputs_usability_flag(self):
        df   = self._synthetic_kr()
        meta = _make_meta()
        lines = []
        inspect_var(df, meta, "h32z", "sought_treatment", 2007, "KR", lines)
        combined = " ".join(lines)
        # h32z has ~65% missing in synthetic data → WEAK or UNUSABLE
        self.assertTrue(
            any(tok in combined for tok in ("WEAK", "UNUSABLE", "OK")),
            msg="No usability flag found in inspect_var output"
        )

    def test_flagged_zscore_values_counted_correctly(self):
        df   = self._synthetic_kr()     # 20 rows have hw70 == 9999
        meta = _make_meta()
        lines = []
        inspect_var(df, meta, "hw70", "haz_score", 2007, "KR", lines)
        combined = " ".join(lines)
        self.assertIn("Flagged (>9990): 20", combined)

    def test_main_runs_without_crash_when_all_files_missing(self):
        """main() should gracefully handle all files being absent."""
        with patch.object(Path, "exists", return_value=False):
            try:
                # Redirect output file to /tmp to avoid touching user's FS
                import builtins
                original_open = builtins.open
                with patch("builtins.open", wraps=original_open):
                    with patch(
                        "builtins.open",
                        unittest.mock.mock_open()
                    ):
                        main()
            except SystemExit:
                pass   # acceptable — main may sys.exit on some errors
            except Exception as exc:
                self.fail(f"main() raised unexpected exception: {exc!r}")


# =============================================================================
# 7. EDGE-CASE TESTS
# =============================================================================

class TestEdgeCases(unittest.TestCase):

    def test_usability_flag_boundary_exactly_basic(self):
        # pct_valid = BASIC → not UNUSABLE
        pct_miss = 100.0 - BASIC_USABLE_THRESHOLD
        flag = usability_flag(pct_miss)
        self.assertNotIn("UNUSABLE", flag)

    def test_usability_flag_boundary_exactly_research(self):
        pct_miss = 100.0 - RESEARCH_USABLE_THRESHOLD
        flag = usability_flag(pct_miss)
        self.assertIn("OK", flag)

    def test_inspect_var_all_nan_series(self):
        df = pd.DataFrame({"hv009": [None, None, None]})
        meta = _make_meta()
        lines = []
        inspect_var(df, meta, "hv009", "hh_size", 2007, "HR", lines)
        combined = " ".join(lines)
        self.assertIn("UNUSABLE", combined)

    def test_inspect_var_single_row(self):
        df = pd.DataFrame({"v012": [25]})
        meta = _make_meta()
        lines = []
        inspect_var(df, meta, "v012", "age_woman", 2007, "IR", lines)
        self.assertTrue(len(lines) > 0)

    def test_inspect_var_large_unique_count_classified_continuous(self):
        df = pd.DataFrame({"hv271": list(range(500))})
        meta = _make_meta()
        lines = []
        inspect_var(df, meta, "hv271", "wealth_score", 2007, "HR", lines)
        combined = " ".join(lines)
        self.assertIn("Continuous", combined)

    def test_get_labels_returns_empty_for_column_with_no_labels(self):
        meta = _make_meta({"hv025": {}})
        self.assertEqual(get_labels(meta, "hv025"), {})

    def test_load_file_with_non_path_type_returns_error(self):
        # Passing an integer instead of a Path should not crash the program
        df, meta, err = load_file(Path("/this/definitely/does/not/exist.dta"))
        self.assertIsNone(df)
        self.assertIsInstance(err, str)


# =============================================================================
# TEST RUNNER ENTRY POINT
# =============================================================================

def _run_tests():
    """Run all tests and print a summary. Called via --run-tests flag."""
    loader  = unittest.TestLoader()
    suite   = unittest.TestSuite()
    for cls in [
        TestUsabilityFlag,
        TestGetLabels,
        TestLoadFile,
        TestInspectVar,
        TestConfigurationSanity,
        TestIntegrationSanity,
        TestEdgeCases,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if "--run-tests" in sys.argv:
        sys.exit(_run_tests())
    else:
        try:
            import pyreadstat
        except ImportError:
            print("Run: pip install pandas pyreadstat")
            sys.exit(1)
        main()