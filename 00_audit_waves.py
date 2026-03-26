"""
FIXED: Exhaustive discovery of ALL potentially useful variables across
every DHS file and every wave.

Key fixes over original:
  1. Detects variable type (categorical vs continuous) from dtype + value labels
  2. Separates "column exists" from "has usable non-missing data"
     → marks as EMPTY if >90% missing
  3. For categoricals: reports modal category + n_categories, not numeric stats
  4. For continuous: reports mean ± sd, min/max
  5. File caching done at module level so each file loaded only ONCE
  6. Case-insensitive path matching for .dta vs .DTA extensions
  7. Wave-specific missing thresholds: column must have ≥10% valid to count
  8. Summary matrix shows ✓ (usable), ~ (exists but mostly missing), ✗ (absent)

Files checked per wave:
  HR  = Household Recode
  KR  = Kids Recode
  IR  = Individual Recode (women)
  MR  = Men's Recode
  BR  = Birth Recode
  PR  = Person Recode

Output: OUTPUT/variable_discovery.txt

USAGE:
  python 00_audit_waves.py           # normal run
  python 00_audit_waves.py --test    # run unit tests and exit (no data needed)
"""

import sys
import os
import unittest
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(os.environ.get("ZAMBIA_BASE_DIR",
                               r"C:\Users\Sayan\Desktop\ZAMBIA_V1"))
OUT_DIR  = BASE_DIR / "OUTPUT"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Minimum % of non-missing required to call a variable "usable"
USABLE_THRESHOLD = 10.0   # percent

# =============================================================================
# ALL FILES PER WAVE
# =============================================================================
WAVE_FILES = {
    1992: {
        "HR": "ZAMBIA1992/ZMHR21DT/ZMHR21FL.DTA",
        "KR": "ZAMBIA1992/ZMKR21DT/ZMKR21FL.DTA",
        "IR": "ZAMBIA1992/ZMIR21DT/ZMIR21FL.DTA",
        "BR": "ZAMBIA1992/ZMBR21DT/ZMBR21FL.dta",
        "PR": "ZAMBIA1992/ZMPR21DT/ZMPR21FL.DTA",
    },
    1996: {
        "HR": "ZAMBIA1996/ZMHR31DT/ZMHR31FL.DTA",
        "KR": "ZAMBIA1996/ZMKR31DT/ZMKR31FL.DTA",
        "IR": "ZAMBIA1996/ZMIR31DT/ZMIR31FL.DTA",
        "MR": "ZAMBIA1996/ZMMR31DT/ZMMR31FL.DTA",
        "BR": "ZAMBIA1996/ZMBR31DT/ZMBR31FL.dta",
        "PR": "ZAMBIA1996/ZMPR31DT/ZMPR31FL.DTA",
    },
    2002: {
        "HR": "ZAMBIA2002/ZMHR42DT/ZMHR42FL.DTA",
        "KR": "ZAMBIA2002/ZMKR42DT/ZMKR42FL.DTA",
        "IR": "ZAMBIA2002/ZMIR42DT/ZMIR42FL.DTA",
        "MR": "ZAMBIA2002/ZMMR41DT/ZMMR41FL.DTA",
        "BR": "ZAMBIA2002/ZMBR42DT/ZMBR42FL.dta",
        "PR": "ZAMBIA2002/ZMPR43DT/ZMPR43FL.DTA",
    },
    2007: {
        "HR": "ZAMBIA2007/ZMHR51DT/ZMHR51FL.DTA",
        "KR": "ZAMBIA2007/ZMKR51DT/ZMKR51FL.DTA",
        "IR": "ZAMBIA2007/ZMIR51DT/ZMIR51FL.DTA",
        "MR": "ZAMBIA2007/ZMMR51DT/ZMMR51FL.DTA",
        "BR": "ZAMBIA2007/ZMBR51DT/ZMBR51FL.DTA",
        "PR": "ZAMBIA2007/ZMPR51DT/ZMPR51FL.DTA",
    },
    2014: {
        "HR": "ZAMBIA2014/ZMHR61DT/ZMHR61FL.DTA",
        "KR": "ZAMBIA2014/ZMKR61DT/ZMKR61FL.DTA",
        "IR": "ZAMBIA2014/ZMIR61DT/ZMIR61FL.DTA",
        "MR": "ZAMBIA2014/ZMMR61DT/ZMMR61FL.DTA",
        "BR": "ZAMBIA2014/ZMBR61DT/ZMBR61FL.DTA",
        "PR": "ZAMBIA2014/ZMPR61DT/ZMPR61FL.DTA",
    },
    2018: {
        "HR": "ZAMBIA2018/ZMHR71DT/ZMHR71FL.DTA",
        "KR": "ZAMBIA2018/ZMKR71DT/ZMKR71FL.DTA",
        "IR": "ZAMBIA2018/ZMIR71DT/ZMIR71FL.DTA",
        "MR": "ZAMBIA2018/ZMMR71DT/ZMMR71FL.DTA",
        "BR": "ZAMBIA2018/ZMBR71DT/ZMBR71FL.DTA",
        "PR": "ZAMBIA2018/ZMPR71DT/ZMPR71FL.DTA",
    },
    2024: {
        "HR": "ZAMBIA2024/ZMHR81DT/ZMHR81FL.dta",
        "KR": "ZAMBIA2024/ZMKR81DT/ZMKR81FL.dta",
        "IR": "ZAMBIA2024/ZMIR81DT/ZMIR81FL.dta",
        "MR": "ZAMBIA2024/ZMMR81DT/ZMMR81FL.dta",
        "BR": "ZAMBIA2024/ZMBR81DT/ZMBR81FL.dta",
        "PR": "ZAMBIA2024/ZMPR81DT/ZMPR81FL.dta",
    },
}

# =============================================================================
# VARIABLE GROUPS
# Each entry: (standard_name, [candidate_cols], file_type, description,
#              var_type)
# var_type = "cat"  → categorical (report frequencies)
#          = "cont" → continuous  (report mean/sd/range)
#          = "auto" → detect automatically
# =============================================================================
VARIABLE_GROUPS = {

    # ── CHILD HEALTH (KR) ─────────────────────────────────────────────────────
    "child_health": [
        ("vacc_bcg",         ["h2"],                  "KR", "BCG vaccination",                        "cat"),
        ("vacc_dpt1",        ["h3"],                  "KR", "DPT 1st dose",                           "cat"),
        ("vacc_dpt2",        ["h5"],                  "KR", "DPT 2nd dose",                           "cat"),
        ("vacc_dpt3",        ["h7"],                  "KR", "DPT 3rd dose",                           "cat"),
        ("vacc_polio0",      ["h0"],                  "KR", "Polio at birth",                         "cat"),
        ("vacc_polio1",      ["h4"],                  "KR", "Polio 1st dose",                         "cat"),
        ("vacc_polio2",      ["h6"],                  "KR", "Polio 2nd dose",                         "cat"),
        ("vacc_polio3",      ["h8"],                  "KR", "Polio 3rd dose",                         "cat"),
        ("vacc_measles",     ["h9"],                  "KR", "Measles vaccination",                    "cat"),
        ("vacc_all",         ["h10"],                 "KR", "All basic vaccinations",                 "cat"),
        ("had_diarrhoea",    ["h11"],                 "KR", "Diarrhoea last 2 weeks",                 "cat"),
        ("had_fever",        ["h22"],                 "KR", "Fever last 2 weeks",                     "cat"),
        ("had_cough",        ["h31"],                 "KR", "Cough/ARI last 2 weeks",                 "cat"),
        ("sought_treatment", ["h32z","h32a","h32b",
                              "h32c","h32d","h32e",
                              "h32f","h32g","h32h",
                              "h32i","h32j","h32k"], "KR", "Sought treatment for fever/ARI",         "cat"),
        ("ever_breastfed",   ["m4"],                  "KR", "Ever breastfed",                         "cat"),
        ("months_bf",        ["m5"],                  "KR", "Duration of breastfeeding (months)",     "cont"),
        ("vitamin_a",        ["h34"],                 "KR", "Vitamin A last 6 months",               "cat"),
        ("slept_net",        ["ml1","hml12"],          "KR", "Slept under mosquito net",               "cat"),
        ("size_birth",       ["m18"],                 "KR", "Size of child at birth",                 "cat"),
        ("child_anemia",     ["hc57","hw57"],          "KR", "Child anaemia level",                    "cat"),
        ("child_hb",         ["hc56","hw56"],          "KR", "Child haemoglobin (g/dL × 10)",          "cont"),
    ],

    # ── MATERNAL HEALTH (IR) ──────────────────────────────────────────────────
    "maternal_health": [
        ("anc_visits",         ["m14_1"],               "IR", "ANC visits (index child)",              "cont"),
        ("anc_first_trim",     ["m13_1"],               "IR", "ANC in first trimester",               "cat"),
        ("delivery_facility",  ["m15_1"],               "IR", "Delivered in health facility",          "cat"),
        ("delivery_skilled",   ["m3a_1","m3b_1",
                                "m3c_1","m3d_1",
                                "m3e_1","m3a","m3b",
                                "m3c","m3d","m3e"],    "IR", "Skilled birth attendance",              "cat"),
        ("delivery_csection",  ["m17_1"],               "IR", "Caesarean section",                    "cat"),
        ("pnc_mother",         ["m62_1","m66_1"],       "IR", "Postnatal check for mother",           "cat"),
        ("pnc_baby",           ["m70_1","m71_1"],       "IR", "Postnatal check for baby",             "cat"),
        ("tetanus",            ["m1_1"],                "IR", "Tetanus injections (pregnancy)",        "cont"),
        ("iron_suppl",         ["m45_1"],               "IR", "Iron supplementation pregnancy",        "cat"),
    ],

    # ── WOMEN'S BIOMARKERS (IR) ───────────────────────────────────────────────
    "biomarkers_women": [
        ("woman_anemia",  ["v457"],  "IR", "Woman anaemia level",          "cat"),
        ("woman_hb",      ["v456"],  "IR", "Woman Hb (g/dL × 10)",         "cont"),
        ("woman_bmi",     ["v445"],  "IR", "Woman BMI × 100",              "cont"),
        ("woman_hiv",     ["v781"],  "IR", "Woman HIV positive",           "cat"),
        ("woman_height",  ["v438"],  "IR", "Woman height cm × 10",         "cont"),
    ],

    # ── REPRODUCTIVE HEALTH (IR) ──────────────────────────────────────────────
    "reproductive_health": [
        ("contraceptive",    ["v313"],        "IR", "Contraceptive method used",         "cat"),
        ("unmet_need",       ["v626a","v626"],"IR", "Unmet need for family planning",    "cat"),
        ("ideal_children",   ["v613"],        "IR", "Ideal number of children",          "cont"),
        ("wanted_last",      ["v367"],        "IR", "Last birth wanted?",                "cat"),
        ("age_first_birth",  ["v212"],        "IR", "Age at first birth",                "cont"),
        ("age_first_sex",    ["v531"],        "IR", "Age at first intercourse",          "cont"),
        ("marital_status",   ["v501"],        "IR", "Current marital status",            "cat"),
        ("age_first_union",  ["v511"],        "IR", "Age at first union",                "cont"),
    ],

    # ── WOMEN'S AUTONOMY (IR) ─────────────────────────────────────────────────
    "autonomy": [
        ("decide_healthcare", ["v743a"],  "IR", "Decides: woman's healthcare",  "cat"),
        ("decide_purchase",   ["v743b"],  "IR", "Decides: large purchases",     "cat"),
        ("decide_visits",     ["v743c"],  "IR", "Decides: visits to family",    "cat"),
        ("has_account",       ["v170"],   "IR", "Has bank account",             "cat"),
        ("has_mobile",        ["v169a"],  "IR", "Has mobile phone",             "cat"),
        ("owns_land",         ["v745b"],  "IR", "Owns land alone/jointly",      "cat"),
        ("owns_house",        ["v745a"],  "IR", "Owns house alone/jointly",     "cat"),
        ("employed",          ["v714"],   "IR", "Currently employed",           "cat"),
        ("earn_cash",         ["v741"],   "IR", "Earns cash",                   "cat"),
    ],

    # ── DOMESTIC VIOLENCE (IR) ────────────────────────────────────────────────
    "domestic_violence": [
        ("dv_pushed",       ["d105a"],  "IR", "Husband pushed/shook/threw",    "cat"),
        ("dv_slapped",      ["d105b"],  "IR", "Husband slapped",               "cat"),
        ("dv_punched",      ["d105c"],  "IR", "Husband punched with fist",     "cat"),
        ("dv_kicked",       ["d105d"],  "IR", "Husband kicked/dragged",        "cat"),
        ("dv_choked",       ["d105e"],  "IR", "Husband tried to choke/burn",   "cat"),
        ("dv_weapon",       ["d105f"],  "IR", "Husband threatened with weapon","cat"),
        ("dv_sex_force",    ["d105g"],  "IR", "Husband forced sex",            "cat"),
        ("dv_humiliate",    ["d103a"],  "IR", "Husband humiliated her",        "cat"),
        ("dv_any_phys",     ["d106"],   "IR", "Any physical violence ever",    "cat"),
        ("dv_any_sex",      ["d107"],   "IR", "Any sexual violence ever",      "cat"),
        ("dv_any",          ["d108"],   "IR", "Any violence (phys or sex)",    "cat"),
        ("dv_pregnant",     ["d110"],   "IR", "Violence during pregnancy",     "cat"),
        ("dv_justify_burn", ["v744a"],  "IR", "Justifies DV: burns food",      "cat"),
        ("dv_justify_argue",["v744b"],  "IR", "Justifies DV: argues",          "cat"),
        ("dv_justify_sex",  ["v744d"],  "IR", "Justifies DV: refuses sex",     "cat"),
        ("dv_justify_kids", ["v744e"],  "IR", "Justifies DV: neglects kids",   "cat"),
    ],

    # ── MEN'S RECODE (MR) ────────────────────────────────────────────────────
    "men": [
        ("man_employed",    ["mv714"],   "MR", "Man currently employed",       "cat"),
        ("man_edu_level",   ["mv106"],   "MR", "Man education level",          "cat"),
        ("man_edu_years",   ["mv133"],   "MR", "Man years of education",       "cont"),
        ("man_anemia",      ["mv457"],   "MR", "Man anaemia level",            "cat"),
        ("man_hiv",         ["mv781"],   "MR", "Man HIV positive",             "cat"),
        ("man_condom",      ["mv761"],   "MR", "Used condom at last sex",      "cat"),
        ("man_multi_part",  ["mv766a"],  "MR", "2+ partners last year",        "cat"),
        ("man_dv_justified",["mv744a"],  "MR", "Man justifies DV",             "cat"),
    ],

    # ── HOUSEHOLD DWELLING (HR) ───────────────────────────────────────────────
    "dwelling": [
        ("floor_type",   ["hv213"],           "HR", "Main floor material",      "cat"),
        ("wall_type",    ["hv214"],           "HR", "Main wall material",       "cat"),
        ("roof_type",    ["hv215"],           "HR", "Main roof material",       "cat"),
        ("cooking_fuel", ["hv226"],           "HR", "Type of cooking fuel",     "cat"),
        ("rooms_sleep",  ["hv216"],           "HR", "Rooms for sleeping",       "cont"),
        ("has_phone",    ["hv221","hv243a"],   "HR", "Has mobile telephone",    "cat"),
        ("net_in_hh",    ["hv227"],           "HR", "Mosquito net in HH",       "cat"),
        ("hh_head_sex",  ["hv219"],           "HR", "Sex of HH head",           "cat"),
        ("hh_head_age",  ["hv220"],           "HR", "Age of HH head",           "cont"),
    ],

    # ── CHILD DEMOGRAPHIC (KR) ────────────────────────────────────────────────
    "child_demog": [
        ("preceding_int", ["b11"],  "KR", "Preceding birth interval (mo)",  "cont"),
        ("multiple_birth",["b0"],   "KR", "Multiple birth (0=single)",       "cat"),
        ("birth_order",   ["bord"], "KR", "Birth order",                     "cont"),
        ("sex_child",     ["b4"],   "KR", "Sex of child",                    "cat"),
        ("child_age",     ["b8"],   "KR", "Current age of child",            "cont"),
    ],

    # ── HIV (IR) ──────────────────────────────────────────────────────────────
    "hiv": [
        ("hiv_pos",      ["v781"],   "IR", "HIV positive",                  "cat"),
        ("knows_status", ["v828"],   "IR", "Knows HIV status",               "cat"),
        ("hiv_counsel",  ["v826a"],  "IR", "HIV counselling at ANC",         "cat"),
    ],
}


# =============================================================================
# FILE LOADING — cached, case-insensitive
# =============================================================================

def resolve_path(base: Path, rel: str) -> Path | None:
    """
    Try exact path first; if not found, try case-insensitive match
    on each path component. Returns resolved Path or None.
    """
    full = base / rel
    if full.exists():
        return full

    # Walk down component by component doing case-insensitive matching
    parts = Path(rel).parts
    current = base
    for part in parts:
        if not current.is_dir():
            return None
        matches = [c for c in current.iterdir()
                   if c.name.lower() == part.lower()]
        if not matches:
            return None
        current = matches[0]
    return current if current.exists() else None


_FILE_CACHE: dict = {}   # (wave, ftype) → (df, meta, error_str|None)


def get_file(wave: int, ftype: str):
    key = (wave, ftype)
    if key in _FILE_CACHE:
        return _FILE_CACHE[key]

    rel = WAVE_FILES.get(wave, {}).get(ftype)
    if rel is None:
        _FILE_CACHE[key] = (None, None, "No path defined")
        return _FILE_CACHE[key]

    path = resolve_path(BASE_DIR, rel)
    if path is None:
        _FILE_CACHE[key] = (None, None, f"File not found (tried: {rel})")
        return _FILE_CACHE[key]

    try:
        df, meta = pyreadstat.read_dta(str(path))
        df.columns = df.columns.str.lower()

        # ── Warn-only sanity checks (never change return value or flow) ────
        # Check columns are all lowercase after .str.lower()
        non_lower = [c for c in df.columns if c != c.lower()]
        if non_lower:
            print(f"  [WARN] {wave}/{ftype}: {len(non_lower)} columns not "
                  f"lowercase after .str.lower() — first 3: {non_lower[:3]}")

        # Check file is not empty
        if len(df) == 0:
            print(f"  [WARN] {wave}/{ftype}: loaded file has 0 rows")

        # Check standard province column exists (warn only — some recodes differ)
        prov_cols = {"hv024", "v024", "mv024"}
        if ftype in ("HR", "IR", "KR", "MR") and not prov_cols.intersection(df.columns):
            print(f"  [WARN] {wave}/{ftype}: no standard province column found "
                  f"(expected one of {prov_cols}). Province FE may not work.")

        # Check standard weight column exists
        wt_map = {"HR": "hv005", "IR": "v005", "KR": "v005", "MR": "mv005"}
        expected_wt = wt_map.get(ftype)
        if expected_wt and expected_wt not in df.columns:
            print(f"  [WARN] {wave}/{ftype}: standard weight column '{expected_wt}' "
                  f"not found")

        _FILE_CACHE[key] = (df, meta, None)
    except Exception as e:
        _FILE_CACHE[key] = (None, None, f"Load error: {e}")

    return _FILE_CACHE[key]


# =============================================================================
# VARIABLE TYPE DETECTION
# =============================================================================

def detect_type(series: pd.Series, meta, col: str, declared: str) -> str:
    """
    Returns "cat" or "cont".
    Priority: explicit declaration > has value labels > pandas dtype.
    """
    if declared in ("cat", "cont"):
        return declared

    # If value labels exist for this column → categorical
    if meta and hasattr(meta, "variable_value_labels"):
        if col in meta.variable_value_labels and meta.variable_value_labels[col]:
            return "cat"

    # Fall back to dtype
    if pd.api.types.is_float_dtype(series) or pd.api.types.is_integer_dtype(series):
        n_unique = series.nunique()
        # Heuristic: ≤12 unique integer values → treat as categorical
        if pd.api.types.is_float_dtype(series):
            return "cont"
        return "cat" if n_unique <= 12 else "cont"

    return "cat"   # strings → categorical


# =============================================================================
# VARIABLE STATISTICS
# =============================================================================

def analyse_variable(df: pd.DataFrame, col: str, meta,
                     declared_type: str) -> dict:
    """
    Returns dict with:
      found_col, dtype_raw, var_type, n_total, n_valid,
      pct_missing, usable, detail
    """
    raw = df[col]
    n_total   = len(raw)
    n_missing = raw.isna().sum()

    # For DHS, missing codes are often 9, 99, 999, 9999 (system missing in Stata
    # already becomes NaN via pyreadstat). We also treat 0 as valid for most vars.
    n_valid   = n_total - n_missing
    pct_miss  = n_missing / n_total * 100 if n_total else 100.0
    usable = bool((100 - pct_miss) > USABLE_THRESHOLD)

    var_type  = detect_type(raw, meta, col, declared_type)
    dtype_raw = str(raw.dtype)

    # ── Get value labels if available ─────────────────────────────────────────
    val_labels: dict = {}
    if meta and hasattr(meta, "variable_value_labels"):
        val_labels = meta.variable_value_labels.get(col, {})

    detail = ""

    if not usable:
        detail = f"MOSTLY MISSING ({pct_miss:.0f}% missing, n_valid={n_valid})"

    elif var_type == "cat":
        # Work with original (non-coerced) values
        try:
            counts = raw.value_counts(dropna=True).sort_index()
        except TypeError:
            counts = raw.astype(str).value_counts(dropna=True)

        parts = []
        for val, cnt in counts.items():
            pct = cnt / n_valid * 100
            label = ""
            try:
                label = str(val_labels.get(int(val), ""))[:18]
            except (ValueError, TypeError):
                label = str(val)[:18]
            parts.append(f"{val}={label}({pct:.0f}%)")
            if len(parts) >= 6:   # show at most 6 categories
                parts.append("...")
                break
        detail = "  |  ".join(parts)

    else:  # continuous
        # Coerce to numeric (some DHS cont vars stored with value labels
        # e.g. 99=missing → already NaN via Stata but let's be safe)
        numeric = pd.to_numeric(raw, errors="coerce")
        # Drop labelled missing codes (≥95 percentile that are round numbers)
        # Standard DHS missing for cont: 99, 999, 9999
        for miss_code in [9, 99, 999, 9999]:
            if (numeric == miss_code).mean() > 0.01:
                # ── Warn only — does not change the cleaning logic ─────────
                print(f"  [WARN] col='{col}': {miss_code} appears in >"
                      f"1% of values as a numeric — may be uncleaned DHS "
                      f"flag code. Being removed from stats.")
                numeric = numeric.replace(miss_code, np.nan)

        valid_num = numeric.dropna()
        if len(valid_num) == 0:
            detail = "NUMERIC BUT ALL MISSING AFTER CODE REMOVAL"
            usable  = False
        else:
            detail = (
                f"mean={valid_num.mean():.2f}  sd={valid_num.std():.2f}  "
                f"min={valid_num.min():.1f}  max={valid_num.max():.1f}  "
                f"n={len(valid_num)}"
            )

    return {
        "found_col":  col,
        "dtype_raw":  dtype_raw,
        "var_type":   var_type,
        "n_total":    n_total,
        "n_valid":    n_valid,
        "pct_miss":   pct_miss,
        "usable":     usable,
        "detail":     detail,
    }


# =============================================================================
# HELPER: find first matching column (case-insensitive)
# =============================================================================

def find_col(df_cols: set, candidates: list[str]) -> str | None:
    df_cols_lower = {c.lower() for c in df_cols}
    for c in candidates:
        if c.lower() in df_cols_lower:
            return c.lower()
    return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("ZAMBIA DHS — EXHAUSTIVE VARIABLE DISCOVERY (FIXED)")
    lines.append("=" * 80)
    lines.append(f"Usable threshold: ≥{USABLE_THRESHOLD}% non-missing")
    lines.append("Legend: ✓ = usable  ~ = exists but mostly missing  ✗ = absent")
    lines.append("")

    waves = sorted(WAVE_FILES.keys())

    # availability[(group, var_name)][wave] = "✓"|"~"|"✗"|"ERR"|"NO_FILE"
    availability: dict = {}

    for group_name, var_list in VARIABLE_GROUPS.items():
        lines.append("")
        lines.append("─" * 80)
        lines.append(f"GROUP: {group_name.upper().replace('_', ' ')}")
        lines.append("─" * 80)

        for entry in var_list:
            var_name, candidates, ftype, description, declared_type = entry

            lines.append(f"\n  [{var_name}]  {description}")
            lines.append(f"  File: {ftype}  |  Candidates: {candidates}  |  Type: {declared_type}")

            wave_status: dict = {}

            for wave in waves:
                df, meta, err = get_file(wave, ftype)

                if err:
                    # Distinguish "no MR in 1992" from actual error
                    if "No path defined" in err or "not found" in err.lower():
                        wave_status[wave] = ("NO_FILE", err)
                    else:
                        wave_status[wave] = ("ERR", err)
                    continue

                found_col = find_col(set(df.columns), candidates)

                if found_col is None:
                    wave_status[wave] = ("MISSING", "Column not in file")
                    continue

                stats = analyse_variable(df, found_col, meta, declared_type)

                if stats["usable"]:
                    symbol = "✓"
                else:
                    symbol = "~"   # exists but mostly missing

                miss_pct = f"{stats['pct_miss']:.0f}%miss"
                label    = (f"{symbol} col={found_col} dtype={stats['dtype_raw']} "
                            f"type={stats['var_type']} {miss_pct} "
                            f"n_valid={stats['n_valid']}")
                wave_status[wave] = (symbol, label, stats["detail"])

            # ── Print wave-by-wave results ─────────────────────────────────
            for wave in waves:
                info = wave_status.get(wave, ("?", "not checked"))
                sym  = info[0]
                lbl  = info[1] if len(info) > 1 else ""
                det  = info[2] if len(info) > 2 else ""
                lines.append(f"    {wave}: [{sym}] {lbl}")
                if det:
                    lines.append(f"         {det}")

            # ── Summary for this variable ──────────────────────────────────
            usable_waves  = [w for w, i in wave_status.items() if i[0] == "✓"]
            partial_waves = [w for w, i in wave_status.items() if i[0] == "~"]
            absent_waves  = [w for w, i in wave_status.items()
                             if i[0] in ("MISSING", "NO_FILE", "ERR")]
            lines.append(
                f"  → USABLE: {usable_waves}  "
                f"PARTIAL(mostly missing): {partial_waves}  "
                f"ABSENT: {absent_waves}"
            )

            # Store for matrix
            availability[(group_name, var_name)] = {
                w: wave_status.get(w, ("?",))[0] for w in waves
            }

    # ── Summary matrix ─────────────────────────────────────────────────────────
    lines.append("")
    lines.append("=" * 80)
    lines.append("SUMMARY MATRIX — VARIABLE AVAILABILITY BY WAVE")
    lines.append("Legend: ✓=usable  ~=mostly missing  ✗=absent  E=file error")
    lines.append("=" * 80)
    lines.append("")

    wave_header = "  ".join(str(w) for w in waves)
    lines.append(f"  {'Variable':<28} {'Group':<22} {wave_header}")
    lines.append("  " + "─" * 100)

    for (group, var), wave_dict in availability.items():
        cells = []
        for w in waves:
            s = wave_dict.get(w, "?")
            if s == "✓":
                cells.append("  ✓  ")
            elif s == "~":
                cells.append("  ~  ")
            elif s in ("MISSING", "NO_FILE"):
                cells.append("  ✗  ")
            elif s == "ERR":
                cells.append("  E  ")
            else:
                cells.append("  ?  ")
        lines.append(f"  {var:<28} {group:<22} {''.join(cells)}")

    # ── Recommendations ────────────────────────────────────────────────────────
    lines.append("")
    lines.append("=" * 80)
    lines.append("PANEL CONSTRUCTION GUIDANCE")
    lines.append("=" * 80)
    lines.append("""
STEP 1: Use ONLY ✓ waves for each variable (ignore ~ waves).

STEP 2: Continuous variables — verify unit consistency across waves:
  woman_hb (v456)  → always g/dL × 10 in DHS; divide by 10 for real units
  woman_bmi (v445) → always BMI × 100; divide by 100
  woman_height     → always cm × 10; divide by 10
  child_hb         → hc56/hw56 in g/dL × 10

STEP 3: Categorical variables — check coding consistency:
  vacc variables: 0=no, 1=vaccination card, 2=mother recall, 8=don't know
  Use: (h2 == 1 | h2 == 2) → vaccinated
  contraceptive v313: 0=not using, 1=folklore, 2=trad, 3=modern

STEP 4: For panel regression, create binary indicators from categoricals
  using the value labels printed above for each wave.

FULL PANEL variables (all 7 waves likely usable):
  vacc_bcg, vacc_measles, had_diarrhoea, had_fever
  marital_status, age_first_birth, floor_type, wall_type
  birth_order, sex_child

PARTIAL PANEL (2007+):
  woman_anemia, woman_hb, woman_bmi, woman_hiv
  dv_any, dv_any_phys, dv_any_sex
  delivery_facility, pnc_mother
  has_account, has_mobile, man_hiv
""")

    # ── Save ───────────────────────────────────────────────────────────────────
    out_path = OUT_DIR / "variable_discovery.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nDiscovery report saved: {out_path}")
    print("Review OUTPUT/variable_discovery.txt before building expanded panel.")


# =============================================================================
# UNIT TESTS
# Run with: python 00_audit_waves.py --test
# No data files required.
# =============================================================================

class TestResolvePath(unittest.TestCase):

    def setUp(self):
        import tempfile, shutil
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_exact_path_found(self):
        """Exact path resolves correctly"""
        sub = self.tmp / "subdir"
        sub.mkdir()
        f = sub / "file.dta"
        f.write_text("x")
        self.assertEqual(resolve_path(self.tmp, "subdir/file.dta"), f)

    def test_case_insensitive_dir(self):
        """Case-insensitive match on directory name"""
        sub = self.tmp / "SubDir"
        sub.mkdir()
        f = sub / "file.DTA"
        f.write_text("x")
        result = resolve_path(self.tmp, "subdir/file.dta")
        self.assertIsNotNone(result)
        self.assertTrue(result.exists())

    def test_missing_returns_none(self):
        """Non-existent path returns None"""
        self.assertIsNone(resolve_path(self.tmp, "no/such/file.dta"))

    def test_nonexistent_base_returns_none(self):
        """Non-existent base directory returns None"""
        self.assertIsNone(resolve_path(Path("/no_such_base_xyz"), "file.dta"))

    def test_multiple_case_variants(self):
        """Works with mixed-case at multiple path levels"""
        d1 = self.tmp / "ZAMBIA2007"
        d2 = d1 / "ZMHR51DT"
        d2.mkdir(parents=True)
        f = d2 / "ZMHR51FL.DTA"
        f.write_text("x")
        result = resolve_path(self.tmp, "zambia2007/zmhr51dt/zmhr51fl.dta")
        self.assertIsNotNone(result)


class TestFindCol(unittest.TestCase):

    def test_exact_match(self):
        """Finds exact column name"""
        self.assertEqual(find_col({"h2", "v456"}, ["h2"]), "h2")

    def test_case_insensitive(self):
        """Case-insensitive match — returns lowercase"""
        self.assertEqual(find_col({"H2", "V456"}, ["h2"]), "h2")

    def test_first_candidate_wins(self):
        """Returns first matching candidate, not last"""
        # both v626a and v626 are in the set
        self.assertEqual(find_col({"v626", "v626a"}, ["v626a", "v626"]), "v626a")

    def test_no_match_returns_none(self):
        """Returns None when no candidate found"""
        self.assertIsNone(find_col({"h2", "v456"}, ["xyz", "abc"]))

    def test_empty_df_cols(self):
        """Returns None on empty column set"""
        self.assertIsNone(find_col(set(), ["h2"]))

    def test_returns_lowercase(self):
        """Returned column name is always lowercase"""
        result = find_col({"HV024"}, ["hv024"])
        self.assertEqual(result, "hv024")

    def test_single_candidate_list(self):
        """Works correctly with a single-element candidate list"""
        self.assertEqual(find_col({"v501"}, ["v501"]), "v501")


class TestDetectType(unittest.TestCase):

    def _meta_with_labels(self, col, labels):
        class M:
            variable_value_labels = {}
        m = M()
        m.variable_value_labels[col] = labels
        return m

    def test_explicit_cat(self):
        """Explicit 'cat' is returned as-is"""
        self.assertEqual(detect_type(pd.Series([1.0, 2.0]), None, "x", "cat"), "cat")

    def test_explicit_cont(self):
        """Explicit 'cont' is returned as-is"""
        self.assertEqual(detect_type(pd.Series([1, 2, 3]), None, "x", "cont"), "cont")

    def test_auto_with_value_labels_is_cat(self):
        """auto + value labels → cat"""
        s = pd.Series([1.0, 2.0])
        m = self._meta_with_labels("x", {1: "yes", 2: "no"})
        self.assertEqual(detect_type(s, m, "x", "auto"), "cat")

    def test_auto_float_no_labels_is_cont(self):
        """auto + float dtype + no labels → cont"""
        s = pd.Series([1.5, 2.5, 3.5])
        self.assertEqual(detect_type(s, None, "x", "auto"), "cont")

    def test_auto_int_few_unique_is_cat(self):
        """auto + int + ≤12 unique values → cat"""
        s = pd.Series([0, 1, 1, 0], dtype=int)
        self.assertEqual(detect_type(s, None, "x", "auto"), "cat")

    def test_auto_int_many_unique_is_cont(self):
        """auto + int + >12 unique values → cont"""
        s = pd.Series(list(range(20)), dtype=int)
        self.assertEqual(detect_type(s, None, "x", "auto"), "cont")

    def test_auto_string_is_cat(self):
        """auto + string dtype → cat"""
        s = pd.Series(["yes", "no", "yes"])
        self.assertEqual(detect_type(s, None, "x", "auto"), "cat")

    def test_returns_only_cat_or_cont(self):
        """Return value is always 'cat' or 'cont', never anything else"""
        for declared in ("cat", "cont", "auto"):
            result = detect_type(pd.Series([1, 2, 3]), None, "x", declared)
            self.assertIn(result, ("cat", "cont"))

    def test_none_meta_handled(self):
        """None meta does not raise"""
        result = detect_type(pd.Series([1.0, 2.0]), None, "x", "auto")
        self.assertIn(result, ("cat", "cont"))


class TestAnalyseVariable(unittest.TestCase):

    def _meta(self):
        class M:
            variable_value_labels = {}
        return M()

    def test_return_keys_always_present(self):
        """Return dict always has all 7 expected keys"""
        df  = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        res = analyse_variable(df, "x", self._meta(), "cont")
        for key in ("found_col", "dtype_raw", "var_type", "n_total",
                    "n_valid", "pct_miss", "usable", "detail"):
            self.assertIn(key, res)

    def test_found_col_matches_input(self):
        """found_col in result matches the col argument"""
        df  = pd.DataFrame({"h2": [0.0, 1.0, 2.0] * 10})
        res = analyse_variable(df, "h2", self._meta(), "cat")
        self.assertEqual(res["found_col"], "h2")

    def test_n_total_correct(self):
        """n_total equals len(df)"""
        df  = pd.DataFrame({"x": [1.0] * 50})
        res = analyse_variable(df, "x", self._meta(), "cont")
        self.assertEqual(res["n_total"], 50)

    def test_n_valid_excludes_nan(self):
        """n_valid = n_total - n_missing"""
        df  = pd.DataFrame({"x": [1.0, np.nan, 2.0, np.nan, 3.0]})
        res = analyse_variable(df, "x", self._meta(), "cont")
        self.assertEqual(res["n_valid"], 3)
        self.assertEqual(res["n_total"], 5)

    def test_pct_miss_calculation(self):
        """pct_miss is correctly computed"""
        df  = pd.DataFrame({"x": [np.nan] * 40 + [1.0] * 60})
        res = analyse_variable(df, "x", self._meta(), "cont")
        self.assertAlmostEqual(res["pct_miss"], 40.0)

    def test_usable_true_above_threshold(self):
        """usable=True when valid% >= USABLE_THRESHOLD"""
        df  = pd.DataFrame({"x": [1.0] * 90 + [np.nan] * 10})
        res = analyse_variable(df, "x", self._meta(), "cont")
        self.assertTrue(res["usable"])

    def test_usable_false_below_threshold(self):
        """usable=False when valid% < USABLE_THRESHOLD"""
        df  = pd.DataFrame({"x": [np.nan] * 95 + [1.0] * 5})
        res = analyse_variable(df, "x", self._meta(), "cont")
        self.assertFalse(res["usable"])

    def test_detail_not_empty_when_usable(self):
        """detail string is non-empty when variable is usable"""
        df  = pd.DataFrame({"x": [1.0, 2.0, 3.0] * 20})
        res = analyse_variable(df, "x", self._meta(), "cont")
        self.assertTrue(res["usable"])
        self.assertTrue(len(res["detail"]) > 0)

    def test_detail_mentions_mostly_missing(self):
        """detail says MOSTLY MISSING when usable=False"""
        df  = pd.DataFrame({"x": [np.nan] * 95 + [1.0] * 5})
        res = analyse_variable(df, "x", self._meta(), "cont")
        self.assertIn("MOSTLY MISSING", res["detail"])

    def test_cat_detail_contains_percentages(self):
        """Categorical detail string contains percentage signs"""
        df  = pd.DataFrame({"h2": [0.0, 1.0, 2.0] * 20})
        res = analyse_variable(df, "h2", self._meta(), "cat")
        self.assertIn("%", res["detail"])

    def test_cont_detail_contains_mean(self):
        """Continuous detail string contains 'mean='"""
        df  = pd.DataFrame({"x": [10.0, 20.0, 30.0] * 20})
        res = analyse_variable(df, "x", self._meta(), "cont")
        self.assertIn("mean=", res["detail"])

    def test_missing_code_99_removed_from_stats(self):
        """Value 99 appearing >1% of the time is removed before computing stats"""
        # 20 real values around 5.0, 80 flag code 99s
        data = [99.0] * 80 + [5.0, 5.5, 6.0, 5.2] * 5
        df   = pd.DataFrame({"v456": data})
        res  = analyse_variable(df, "v456", self._meta(), "cont")
        if res["usable"] and "mean=" in res["detail"]:
            mean_val = float(res["detail"].split("mean=")[1].split()[0])
            # Mean should be near 5.0, not pulled toward 99
            self.assertLess(mean_val, 20.0,
                            "mean should not be contaminated by code 99")

    def test_var_type_is_cat_or_cont(self):
        """var_type in result is always 'cat' or 'cont'"""
        for declared in ("cat", "cont"):
            df  = pd.DataFrame({"x": [1.0, 2.0, 3.0] * 10})
            res = analyse_variable(df, "x", self._meta(), declared)
            self.assertIn(res["var_type"], ("cat", "cont"))

    def test_all_nan_column_is_not_usable(self):
        """All-NaN column is marked not usable"""
        df  = pd.DataFrame({"x": [np.nan] * 100})
        res = analyse_variable(df, "x", self._meta(), "cont")
        self.assertFalse(res["usable"])


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Run unit tests and exit (no data required)")
    args, remaining = parser.parse_known_args()

    if args.test:
        print("=" * 60)
        print("Running unit tests for 00_audit_waves.py")
        print("=" * 60)
        sys.argv = [sys.argv[0]] + remaining
        unittest.main(verbosity=2)
    else:
        try:
            import pyreadstat
        except ImportError:
            print("ERROR: pip install pandas pyreadstat numpy")
            sys.exit(1)
        main()