"""
02_build_panel.py
==================
Builds the province-level panel for the Zambia copper paper.
One row per province × wave.  After Muchinga merge: 9 provinces × 7 waves = 63 rows.

All fixes applied:
  - Muchinga → Northern at harmonisation time (panel consistency)
  - isin_vec: 96/99 → NaN not 0 (WASH bias fix)
  - binary(): strict explicit false_vals; DHS missing codes never become 0
  - safe_recode(): replaces .apply(lambda) that bypassed missing logic
  - Weight skew check logged to checks file
  - ph_c defined before use (crash fix)
  - BASE_DIR from env var ZAMBIA_BASE_DIR (portability)
  - All constants from config_00.py (single source of truth)
  - months_bf (m5): 2018/2024 excluded — unit shift detected (mean > 36 months)
  - sought_treatment: denominator is sick children only (h22=1 OR h31=1)
  - vacc_all (h10): excluded from panel — use vacc_full from individual h-codes

Outputs:
  OUTPUT/zambia_province_panel.csv
  OUTPUT/zambia_panel_checks.txt
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat

# ── Config ────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_00 import (
    BASE_DIR, OUT_DIR,
    COPPER_PRICES, PROVINCE_HARMONISE, MINING_PROVINCES, WAVE_FILES,
    DHS_MISSING_CODES,
    IMPROVED_WATER_CODES, IMPROVED_SANIT_CODES, IMPROVED_FLOOR_CODES,
)

# Waves where m5 (months breastfed) has a confirmed unit shift.
# mean > 36 months in the raw data → implausible → excluded from panel.
M5_UNIT_SHIFT_WAVES = {2018, 2024}

# =============================================================================
# HELPERS
# =============================================================================

def load(path):
    df, meta = pyreadstat.read_dta(str(path))
    df.columns = df.columns.str.lower()
    return df, meta


def to_num(series):
    return pd.to_numeric(series, errors="coerce")


def flag_nan(series, threshold=9990):
    """Set values above threshold (DHS flagged codes) to NaN."""
    s = to_num(series)
    return s.where(s <= threshold, np.nan)


def get_label(meta, col, code):
    labels = {}
    if meta and hasattr(meta, "variable_value_labels"):
        labels = meta.variable_value_labels.get(col, {})
    key = int(code) if isinstance(code, (float, np.floating)) else code
    return str(labels.get(key, labels.get(code, str(code)))).strip().lower()


def harmonise(raw):
    """
    Collapse whitespace then look up in PROVINCE_HARMONISE.
    Muchinga → Northern (panel consistency — see config_00.py).
    All spellings of North-Western → northwestern.
    """
    cleaned = " ".join(str(raw).strip().lower().split())
    result  = PROVINCE_HARMONISE.get(cleaned)
    if result is None:
        raise ValueError(
            f"Unknown province label: '{raw}' — add to PROVINCE_HARMONISE in config_00.py"
        )
    return result


def is_mining(prov):
    return 1 if prov in MINING_PROVINCES else 0


assert MINING_PROVINCES == {"copperbelt", "northwestern"}, (
    f"MINING_PROVINCES changed: {MINING_PROVINCES}"
)


def binary(series, true_vals, false_vals):
    """
    Strict binary recoder.
    - true_vals  → 1.0
    - false_vals → 0.0
    - DHS_MISSING_CODES (8,9,96,97,98,99,...) → NaN unconditionally.
    - Anything not in true_vals or false_vals → NaN.
    false_vals must always be supplied explicitly.
    """
    s = to_num(series)
    result = pd.Series(np.nan, index=s.index, dtype=float)
    valid  = s.notna() & ~s.isin(DHS_MISSING_CODES)
    result[valid & s.isin(true_vals)]  = 1.0
    result[valid & s.isin(false_vals)] = 0.0
    return result


def safe_recode(series, code_map, missing_threshold=9990):
    """
    Recode via explicit dict.
    - Any code not in code_map → NaN.
    - DHS_MISSING_CODES and values > missing_threshold → NaN before map lookup.
    Replaces .apply(lambda) patterns that bypassed missing logic.
    """
    s = to_num(series)
    s = s.where(s <= missing_threshold, np.nan)
    s = s.where(~s.isin(DHS_MISSING_CODES), np.nan)
    return s.map(code_map)


def isin_vec(series, code_set, threshold=9990):
    """
    Binary indicator for WASH/floor/fuel code membership.
    - 96 (other/unknown) and 99 (not stated) → NaN via DHS_MISSING_CODES.
    - Unknown ≠ unimproved. They must never be coded as 0.
    """
    s     = to_num(series)
    valid = s.notna() & (s <= threshold) & ~s.isin(DHS_MISSING_CODES)
    result = pd.Series(np.nan, index=s.index, dtype=float)
    result[valid] = 0.0
    result[valid & s.isin(code_set)] = 1.0
    return result


def col(df, candidates):
    """Return the first candidate column name that exists in df, else None."""
    for c in candidates:
        if c.lower() in df.columns:
            return c.lower()
    return None


def wt_mean(values, weights):
    """Weighted mean. Returns NaN if no valid observations."""
    v = to_num(values)
    w = to_num(weights)
    mask = v.notna() & w.notna() & (w > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.average(v[mask], weights=w[mask]))


def check_weights(df, w_col, wave, ftype, checks):
    """Log a warning if weights are heavily skewed (a few obs dominate means)."""
    w = df[w_col].dropna()
    if len(w) == 0:
        return
    med = w.median()
    mx  = w.max()
    if med > 0 and mx > 5 * med:
        checks.append(
            f"  WARN {ftype} {wave}: weight skewed — "
            f"max={mx:.2f} median={med:.2f} ratio={mx/med:.1f}x. "
            f"A few obs may dominate province means."
        )


def prov_labels(df, meta, col_name):
    """Attach a harmonised province_name string column to df."""
    df["province_name"] = df[col_name].apply(
        lambda c: harmonise(get_label(meta, col_name, c))
    )
    return df


def collapse(df, outcomes, weight_col="w"):
    """Collapse to province-level weighted means."""
    rows = []
    for prov, grp in df.groupby("province_name"):
        row = {"province": prov, "n": len(grp)}
        for var in outcomes:
            row[var] = wt_mean(grp[var], grp[weight_col]) if var in grp.columns else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# SECTION 1 — HOUSEHOLD RECODE (HR)
# =============================================================================

def process_hr(wave, path, checks):
    print(f"  HR {wave}...")
    df, meta = load(path)
    df = prov_labels(df, meta, "hv024")
    df["w"] = to_num(df["hv005"]) / 1_000_000
    df["w"] = df["w"].where(df["w"] > 0, np.nan)
    check_weights(df, "w", wave, "HR", checks)

    # Urban/rural: 1=urban, 2=rural → binary urban=1, rural=0
    df["urban"] = binary(df["hv025"], [1], [2])

    # Wealth — exists from 2002; 1992/1996 get NaN here, filled by asset_index
    df["wealth_q"] = to_num(df.get("hv270", pd.Series(np.nan, index=df.index)))
    df["wealth_q"] = df["wealth_q"].where(df["wealth_q"].between(1, 5), np.nan)
    ws = df.get("hv271", pd.Series(np.nan, index=df.index))
    df["wealth_score"] = to_num(ws) / 100_000

    # Asset index — used for 1992/1996 and as a robustness check for later waves
    asset_map  = {
        "asset_elec":  "hv206", "asset_radio": "hv207", "asset_tv":    "hv208",
        "asset_fridge":"hv209", "asset_bike":  "hv210", "asset_car":   "hv212",
    }
    asset_cols = []
    for name, c in asset_map.items():
        if c in df.columns:
            s = to_num(df[c])
            df[name] = s.where(s.isin([0, 1]), np.nan)
        else:
            df[name] = np.nan
        asset_cols.append(name)
    df["asset_index"] = df[asset_cols].mean(axis=1, skipna=True)

    # WASH — isin_vec correctly sends 96/99 to NaN, not to 0 (unimproved)
    df["improved_water"] = (
        isin_vec(df["hv201"], IMPROVED_WATER_CODES)
        if "hv201" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    df["improved_sanit"] = (
        isin_vec(df["hv205"], IMPROVED_SANIT_CODES)
        if "hv205" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    df["floor_finished"] = (
        isin_vec(df["hv213"], IMPROVED_FLOOR_CODES)
        if "hv213" in df.columns
        else pd.Series(np.nan, index=df.index)
    )

    # Household size and crowding
    if "hv009" in df.columns and "hv216" in df.columns:
        hh_sz = to_num(df["hv009"])
        rooms = to_num(df["hv216"])
        df["hh_size"]     = hh_sz
        df["rooms_sleep"] = rooms
        df["crowding"]    = hh_sz / rooms.replace(0, np.nan)
    else:
        df["hh_size"]     = to_num(df.get("hv009", pd.Series(np.nan, index=df.index)))
        df["rooms_sleep"] = np.nan
        df["crowding"]    = np.nan

    net_c = col(df, ["hv227"])
    df["has_net"] = (
        binary(df[net_c], [1], [0]) if net_c
        else pd.Series(np.nan, index=df.index)
    )

    # ph_c assigned before use (prevents NameError crash from original code)
    ph_c = col(df, ["hv243a", "hv221"])
    df["has_phone"] = (
        binary(df[ph_c], [1], [0]) if ph_c
        else pd.Series(np.nan, index=df.index)
    )

    # Cooking fuel: 1-5=clean; 6-11,95=dirty; 96=other→NaN; 99=missing→NaN
    df["clean_fuel"] = (
        safe_recode(df["hv226"],
                    {1:1, 2:1, 3:1, 4:1, 5:1,
                     6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 95:0})
        if "hv226" in df.columns
        else pd.Series(np.nan, index=df.index)
    )

    # Female household head: hv219 — 1=male, 2=female
    df["female_hh_head"] = (
        binary(df["hv219"], [2], [1])
        if "hv219" in df.columns
        else pd.Series(np.nan, index=df.index)
    )

    checks.append(
        f"HR {wave}: {len(df):,} hh | "
        f"water={df['improved_water'].mean()*100:.1f}% | "
        f"asset_idx={df['asset_index'].mean()*100:.1f}%"
    )

    outcomes = [
        "urban", "wealth_q", "wealth_score", "asset_index",
        "asset_elec", "asset_radio", "asset_tv", "asset_fridge",
        "asset_bike", "asset_car",
        "improved_water", "improved_sanit", "floor_finished",
        "hh_size", "rooms_sleep", "crowding",
        "has_net", "has_phone", "clean_fuel", "female_hh_head",
    ]
    out = collapse(df, outcomes)
    out.rename(columns={"n": "n_hh"}, inplace=True)
    return out


# =============================================================================
# SECTION 2 — KIDS RECODE (KR)
# =============================================================================

def process_kr(wave, path, checks):
    print(f"  KR {wave}...")
    df, meta = load(path)
    df = prov_labels(df, meta, "v024")
    df["w"] = to_num(df["v005"]) / 1_000_000
    df["w"] = df["w"].where(df["w"] > 0, np.nan)
    check_weights(df, "w", wave, "KR", checks)

    # ── Child mortality ───────────────────────────────────────────────────────
    b5 = to_num(df["b5"])
    dead         = (b5 == 0)
    alive_or_dead = b5.notna()

    if "b7" in df.columns:
        b7  = to_num(df["b7"])
        b7d = b7.where(dead, np.nan)
        df["u5_dead"]       = dead.astype(float).where(alive_or_dead, np.nan)
        df["infant_dead"]   = (dead & (b7d < 12)).astype(float).where(alive_or_dead, np.nan)
        df["neonatal_dead"] = (dead & (b7d <  1)).astype(float).where(alive_or_dead, np.nan)
    else:
        df["u5_dead"]       = dead.astype(float).where(alive_or_dead, np.nan)
        df["infant_dead"]   = np.nan
        df["neonatal_dead"] = np.nan

    checks.append(
        f"KR {wave}: u5={df['u5_dead'].mean()*1000:.1f}/1000 | n={len(df):,}"
    )

    # ── Anthropometrics ───────────────────────────────────────────────────────
    for zvar, flag in [("hw70","stunted"), ("hw71","underweight"), ("hw72","wasted")]:
        if zvar in df.columns:
            z = flag_nan(df[zvar])
            df[flag] = (z < -200).astype(float).where(z.notna(), np.nan)
        else:
            df[flag] = np.nan

    # ── Vaccinations ─────────────────────────────────────────────────────────
    # h-codes: 0=not vaccinated, 1=yes on card, 2=yes by recall, 8=DK→NaN
    vacc_map = {
        "vacc_bcg":     "h2", "vacc_dpt1": "h3", "vacc_dpt2": "h5",
        "vacc_dpt3":    "h7", "vacc_polio1":"h4","vacc_polio2":"h6",
        "vacc_polio3":  "h8", "vacc_measles":"h9",
    }
    for name, c in vacc_map.items():
        df[name] = (
            binary(df[c], [1, 2], [0]) if c in df.columns
            else pd.Series(np.nan, index=df.index)
        )

    # vacc_full: child received all 4 core vaccines
    # Built from individual indicators — NOT from h10 (structural missingness)
    full_c = ["vacc_bcg", "vacc_measles", "vacc_dpt3", "vacc_polio3"]
    valid  = df[full_c].notna().all(axis=1)
    df["vacc_full"] = df[full_c].min(axis=1)
    df.loc[~valid, "vacc_full"] = np.nan

    # ── Illness in last 2 weeks ───────────────────────────────────────────────
    # h11: 0=no, 2=yes last 2 weeks, 1=yes last 24h (early waves), 8=DK→NaN
    # h22/h31: 0=no, 1=yes, 8=DK→NaN
    for name, c, true_v, false_v in [
        ("had_diarrhoea", "h11", [1, 2], [0]),
        ("had_fever",     "h22", [1],    [0]),
        ("had_cough",     "h31", [1, 2], [0]),
    ]:
        df[name] = (
            binary(df[c], true_v, false_v) if c in df.columns
            else pd.Series(np.nan, index=df.index)
        )

    # ── Treatment seeking ─────────────────────────────────────────────────────
    # h32z: 0=no treatment, 1=yes treatment (among sick children only).
    # CONDITIONAL: only non-missing for children with fever OR cough.
    # Denominator for province mean = sick children, not all children.
    # We restrict the series to sick children before collapse.
    if "h32z" in df.columns and ("h22" in df.columns or "h31" in df.columns):
        fever  = to_num(df.get("h22", pd.Series(np.nan, index=df.index)))
        cough  = to_num(df.get("h31", pd.Series(np.nan, index=df.index)))
        is_sick = (fever == 1) | (cough.isin([1, 2]))
        h32z   = binary(df["h32z"], [1], [0])
        # Set to NaN for healthy children so province mean = rate among sick only
        df["sought_treatment"] = h32z.where(is_sick, np.nan)
        checks.append(
            f"KR {wave}: sought_treatment denominator = sick children "
            f"n={is_sick.sum():,} (not full sample)"
        )
    else:
        df["sought_treatment"] = np.nan

    # ── Breastfeeding ─────────────────────────────────────────────────────────
    # ever_breastfed from m4:
    #   1992–2014: m4 = duration in months (0–92). 0→never, >0→ever.
    #   2018–2024: m4 = 93 (stopped), 94 (never breastfed), 95 (still breastfeeding).
    # bf_map covers both eras: month values 1–92 → 1 (ever), 0 → 0 (never),
    #                          93 → 1 (stopped = ever), 94 → 0 (never), 95 → 1 (still = ever).
    # 96+ and DHS_MISSING_CODES → NaN via safe_recode.
    if "m4" in df.columns:
        m4     = to_num(df["m4"])
        bf_map = {0: 0, 93: 1, 94: 0, 95: 1}
        bf_map.update({i: 1 for i in range(1, 93)})
        df["ever_breastfed"] = safe_recode(m4, bf_map)
    else:
        df["ever_breastfed"] = np.nan

    # months_bf from m5:
    # 1992–2014: stores duration in months (plausible mean ~17–19).
    # 2018–2024: unit shift confirmed — mean > 36 months, likely days or top-coded.
    # Excluded for M5_UNIT_SHIFT_WAVES to prevent silent regression corruption.
    if "m5" in df.columns and wave not in M5_UNIT_SHIFT_WAVES:
        m5 = to_num(df["m5"])
        # Strip implausible values: > 36 months or DHS special codes (93–99)
        df["months_bf"] = m5.where((m5 >= 0) & (m5 <= 36), np.nan)
        # Plausibility check
        median_val = df["months_bf"].median()
        if median_val > 36:
            checks.append(
                f"KR {wave}: months_bf median={median_val:.1f} — "
                f"WARN: exceeds 36 months. Check raw m5 coding."
            )
    else:
        df["months_bf"] = np.nan
        if wave in M5_UNIT_SHIFT_WAVES:
            checks.append(
                f"KR {wave}: months_bf (m5) EXCLUDED — "
                f"unit shift detected (mean > 36 months in raw data)."
            )

    # ── Child anaemia ─────────────────────────────────────────────────────────
    # hw57: 1=severe, 2=moderate, 3=mild, 4=not anaemic
    if "hw57" in df.columns:
        df["child_anaemic"] = binary(flag_nan(df["hw57"]), [1, 2, 3], [4])
    else:
        df["child_anaemic"] = np.nan

    outcomes_kr = [
        "u5_dead", "infant_dead", "neonatal_dead",
        "stunted", "underweight", "wasted",
        "vacc_bcg", "vacc_dpt1", "vacc_dpt2", "vacc_dpt3",
        "vacc_polio1", "vacc_polio2", "vacc_polio3", "vacc_measles", "vacc_full",
        "had_diarrhoea", "had_fever", "had_cough",
        "sought_treatment",
        "ever_breastfed", "months_bf",
        "child_anaemic",
    ]
    out = collapse(df, outcomes_kr)
    out.rename(columns={"n": "n_children"}, inplace=True)
    return out


# =============================================================================
# SECTION 3 — INDIVIDUAL RECODE (IR)
# =============================================================================

def process_ir(wave, path, checks):
    print(f"  IR {wave}...")
    df, meta = load(path)
    df = prov_labels(df, meta, "v024")
    df["w"] = to_num(df["v005"]) / 1_000_000
    df["w"] = df["w"].where(df["w"] > 0, np.nan)
    check_weights(df, "w", wave, "IR", checks)

    # ── ANC ───────────────────────────────────────────────────────────────────
    if "m14_1" in df.columns:
        anc = to_num(df["m14_1"]).where(lambda x: x < 98, np.nan)
        df["anc_visits"] = anc
        df["anc_4plus"]  = (anc >= 4).astype(float).where(anc.notna(), np.nan)
        df["anc_any"]    = (anc >  0).astype(float).where(anc.notna(), np.nan)
    else:
        df["anc_visits"] = df["anc_4plus"] = df["anc_any"] = np.nan

    # m13_1: month of pregnancy at first ANC visit (1–9).
    # First trimester = month ≤ 3. Codes ≥ 90 → NaN.
    if "m13_1" in df.columns:
        m13 = to_num(df["m13_1"]).where(lambda x: x < 90, np.nan)
        df["anc_first_trim"] = (m13 <= 3).astype(float).where(m13.notna(), np.nan)
    else:
        df["anc_first_trim"] = np.nan

    # ── Delivery ──────────────────────────────────────────────────────────────
    # m15_1: place of delivery. 10-19=home→0; 20-39=facility→1; 96/98/99→NaN
    if "m15_1" in df.columns:
        fac_map = {
            **{i: 0 for i in range(10, 20)},
            **{i: 1 for i in range(20, 40)},
        }
        df["delivery_facility"] = safe_recode(df["m15_1"], fac_map)
    else:
        df["delivery_facility"] = np.nan

    # Skilled birth attendant: doctor (m3a_1) or nurse/midwife (m3b_1)
    sba_c = col(df, ["m3a_1", "m3a"])
    snm_c = col(df, ["m3b_1", "m3b"])
    if sba_c:
        doc = binary(df[sba_c], [1], [0])
        nmw = binary(df[snm_c], [1], [0]) if snm_c else pd.Series(0.0, index=df.index)
        df["delivery_skilled"] = ((doc == 1) | (nmw == 1)).astype(float)
        df.loc[doc.isna() & nmw.isna(), "delivery_skilled"] = np.nan
    else:
        df["delivery_skilled"] = np.nan

    df["delivery_csection"] = (
        binary(df["m17_1"], [1], [0])
        if "m17_1" in df.columns
        else pd.Series(np.nan, index=df.index)
    )

    pnc_c = col(df, ["m62_1"])
    df["pnc_mother"] = (
        binary(df[pnc_c], [1], [0]) if pnc_c
        else pd.Series(np.nan, index=df.index)
    )

    iron_c = col(df, ["m45_1"])
    df["iron_suppl"] = (
        binary(df[iron_c], [1], [0]) if iron_c
        else pd.Series(np.nan, index=df.index)
    )

    # Tetanus: m1_1 = number of injections. ≥2 = adequately protected.
    if "m1_1" in df.columns:
        tet = to_num(df["m1_1"]).where(lambda x: x < 90, np.nan)
        df["tetanus_2plus"] = (tet >= 2).astype(float).where(tet.notna(), np.nan)
    else:
        df["tetanus_2plus"] = np.nan

    # ── Contraception ─────────────────────────────────────────────────────────
    # v313: 0=no method, 1=folkloric, 2=traditional, 3=modern
    if "v313" in df.columns:
        v313 = to_num(df["v313"])
        df["modern_contra"] = binary(v313, [3], [0, 1, 2])
        df["any_contra"]    = safe_recode(v313, {0: 0, 1: 1, 2: 1, 3: 1})
    else:
        df["modern_contra"] = df["any_contra"] = np.nan

    # Unmet need: v626a (later waves) / v626 (early waves)
    # 1/2 = unmet need (spacing/limiting); 3-7 = using or no need
    umet_c = col(df, ["v626a", "v626"])
    if umet_c:
        df["unmet_need"] = safe_recode(
            df[umet_c], {1: 1, 2: 1, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
        )
    else:
        df["unmet_need"] = np.nan

    # ── Fertility preferences ─────────────────────────────────────────────────
    # v613: ideal number of children. 96=non-numeric ("up to God"), 99=DK → NaN.
    # Cap at 20 to exclude non-numeric special codes that aren't in DHS_MISSING_CODES.
    if "v613" in df.columns:
        ic = to_num(df["v613"])
        df["ideal_children"] = ic.where(ic < 20, np.nan)
    else:
        df["ideal_children"] = np.nan

    if "v212" in df.columns:
        afb = to_num(df["v212"])
        df["age_first_birth"] = afb.where(afb > 0, np.nan)
    else:
        df["age_first_birth"] = np.nan

    # v531: age at first intercourse. 0=never had sex → NaN.
    if "v531" in df.columns:
        afs = to_num(df["v531"])
        df["age_first_sex"] = afs.where(afs > 0, np.nan)
    else:
        df["age_first_sex"] = np.nan

    # ── Education ─────────────────────────────────────────────────────────────
    if "v106" in df.columns:
        edu       = to_num(df["v106"])
        edu_clean = edu.where(edu.between(0, 3), np.nan)
        df["edu_secondary_p"] = (edu_clean >= 2).astype(float).where(edu_clean.notna(), np.nan)
        df["edu_level"]       = edu_clean
        df["edu_none"]        = (edu_clean == 0).astype(float).where(edu_clean.notna(), np.nan)
    else:
        df["edu_level"] = df["edu_secondary_p"] = df["edu_none"] = np.nan

    if "v133" in df.columns:
        ey = to_num(df["v133"])
        df["edu_years"] = ey.where(ey <= 30, np.nan)
    else:
        df["edu_years"] = np.nan

    # ── Employment ────────────────────────────────────────────────────────────
    df["employed"] = (
        binary(df["v714"], [1], [0])
        if "v714" in df.columns
        else pd.Series(np.nan, index=df.index)
    )

    # v741: 1=cash only, 2=cash and in-kind, 3=in-kind only, 4=not paid
    earn_c = col(df, ["v741"])
    df["earn_cash"] = (
        binary(df[earn_c], [1, 2], [3, 4]) if earn_c
        else pd.Series(np.nan, index=df.index)
    )

    # ── Decision-making autonomy ───────────────────────────────────────────────
    # v743a-d: 1=respondent alone, 2=respondent+husband → has say (=1)
    #          3-6 = husband/partner/other alone → no say (=0); 8,9 → NaN
    for varname, col_name in [
        ("decide_health",    "v743a"), ("decide_purchase", "v743b"),
        ("decide_visits",    "v743c"), ("decide_food",     "v743d"),
    ]:
        c = col(df, [col_name])
        df[varname] = (
            binary(df[c], [1, 2], [3, 4, 5, 6]) if c
            else pd.Series(np.nan, index=df.index)
        )

    c = col(df, ["v743f"])
    df["can_go_health"] = (
        binary(df[c], [1], [0]) if c
        else pd.Series(np.nan, index=df.index)
    )

    # ── Asset ownership ───────────────────────────────────────────────────────
    # v745a/b: 1=alone, 2=jointly → owns (=1); 3=does not own (=0)
    for varname, col_name in [("owns_house", "v745a"), ("owns_land", "v745b")]:
        c = col(df, [col_name])
        df[varname] = (
            binary(df[c], [1, 2], [3]) if c
            else pd.Series(np.nan, index=df.index)
        )

    for varname, col_name in [("has_account", "v170"), ("has_mobile", "v169a")]:
        c = col(df, [col_name])
        df[varname] = (
            binary(df[c], [1], [0]) if c
            else pd.Series(np.nan, index=df.index)
        )

    # ── DV attitudes ──────────────────────────────────────────────────────────
    dv_att_cols = []
    for i, c_name in enumerate(["v744a", "v744b", "v744c", "v744d", "v744e"]):
        c     = col(df, [c_name])
        vname = f"dv_justify_{i+1}"
        df[vname] = (
            binary(df[c], [1], [0]) if c
            else pd.Series(np.nan, index=df.index)
        )
        dv_att_cols.append(vname)
    valid_att = df[dv_att_cols].notna().any(axis=1)
    df["dv_justify_any"] = df[dv_att_cols].max(axis=1)
    df.loc[~valid_att, "dv_justify_any"] = np.nan

    # ── DV experience ─────────────────────────────────────────────────────────
    for varname, col_name in [
        ("dv_humiliate", "d103a"), ("dv_threaten", "d103b"), ("dv_insult",  "d103c"),
        ("dv_pushed",    "d105a"), ("dv_slapped",  "d105b"), ("dv_punched", "d105c"),
        ("dv_kicked",    "d105d"), ("dv_choked",   "d105e"),
        ("dv_any_phys",  "d106"),  ("dv_any_sex",  "d107"),  ("dv_any",     "d108"),
    ]:
        c = col(df, [col_name])
        df[varname] = (
            binary(df[c], [1], [0]) if c
            else pd.Series(np.nan, index=df.index)
        )

    emo_cols  = ["dv_humiliate", "dv_threaten", "dv_insult"]
    valid_emo = df[emo_cols].notna().any(axis=1)
    df["dv_emotional"] = df[emo_cols].max(axis=1)
    df.loc[~valid_emo, "dv_emotional"] = np.nan

    # ── Biomarkers ────────────────────────────────────────────────────────────
    # v457: 1=severe, 2=moderate, 3=mild anaemia, 4=not anaemic
    if "v457" in df.columns:
        df["woman_anaemic"] = binary(flag_nan(df["v457"]), [1, 2, 3], [4])
    else:
        df["woman_anaemic"] = np.nan

    # v456: haemoglobin in g/dL × 10 → divide by 10 for real units
    if "v456" in df.columns:
        df["woman_hb"] = flag_nan(df["v456"]) / 10
    else:
        df["woman_hb"] = np.nan

    # v445: BMI × 100. Plausible range 12.0–60.0 → raw 1200–6000.
    if "v445" in df.columns:
        bmi = to_num(df["v445"])
        df["woman_bmi"] = bmi.where(bmi.between(1200, 6000), np.nan) / 100
    else:
        df["woman_bmi"] = np.nan

    checks.append(
        f"IR {wave}: edu_sec={df['edu_secondary_p'].mean()*100:.1f}% | "
        f"facility_del={df['delivery_facility'].mean()*100:.1f}% | n={len(df):,}"
    )

    outcomes_ir = [
        "anc_visits", "anc_4plus", "anc_any", "anc_first_trim",
        "delivery_facility", "delivery_skilled", "delivery_csection",
        "pnc_mother", "iron_suppl", "tetanus_2plus",
        "modern_contra", "any_contra", "unmet_need",
        "ideal_children", "age_first_birth", "age_first_sex",
        "edu_level", "edu_years", "edu_secondary_p", "edu_none",
        "employed", "earn_cash",
        "decide_health", "decide_purchase", "decide_visits", "decide_food",
        "can_go_health", "owns_house", "owns_land", "has_account", "has_mobile",
        "dv_justify_any", "dv_emotional",
        "dv_pushed", "dv_slapped", "dv_punched", "dv_kicked", "dv_choked",
        "dv_any_phys", "dv_any_sex", "dv_any",
        "woman_anaemic", "woman_hb", "woman_bmi",
    ]
    out = collapse(df, outcomes_ir)
    out.rename(columns={"n": "n_women"}, inplace=True)
    return out


# =============================================================================
# SECTION 4 — MEN'S RECODE (MR)
# =============================================================================

def process_mr(wave, path, checks):
    print(f"  MR {wave}...")
    df, meta = load(path)
    df = prov_labels(df, meta, "mv024")
    df["w"] = to_num(df["mv005"]) / 1_000_000
    df["w"] = df["w"].where(df["w"] > 0, np.nan)
    check_weights(df, "w", wave, "MR", checks)

    if "mv106" in df.columns:
        edu = to_num(df["mv106"])
        df["man_edu_sec_p"] = (edu >= 2).astype(float).where(edu.notna(), np.nan)
        df["man_edu_level"] = edu.where(edu.between(0, 3), np.nan)
    else:
        df["man_edu_sec_p"] = df["man_edu_level"] = np.nan

    df["man_employed"] = (
        binary(df["mv714"], [1], [0])
        if "mv714" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    df["man_condom"] = (
        binary(df["mv761"], [1], [0])
        if "mv761" in df.columns
        else pd.Series(np.nan, index=df.index)
    )

    # mv766a: number of sexual partners in last 12 months. ≥2 = multiple.
    if "mv766a" in df.columns:
        mp = to_num(df["mv766a"])
        df["man_multi_part"] = (mp >= 2).astype(float).where(
            mp.notna() & (mp < 90), np.nan
        )
    else:
        df["man_multi_part"] = np.nan

    df["man_dv_justify"] = (
        binary(df["mv744a"], [1], [0])
        if "mv744a" in df.columns
        else pd.Series(np.nan, index=df.index)
    )

    checks.append(f"MR {wave}: n={len(df):,}")

    outcomes_mr = [
        "man_edu_sec_p", "man_edu_level", "man_employed",
        "man_condom", "man_multi_part", "man_dv_justify",
    ]
    out = collapse(df, outcomes_mr)
    out.rename(columns={"n": "n_men"}, inplace=True)
    return out


# =============================================================================
# MAIN
# =============================================================================

def main():
    checks   = ["=" * 65, "ZAMBIA PANEL BUILD — DATA QUALITY CHECKS", "=" * 65]
    all_hr, all_kr, all_ir, all_mr = [], [], [], []

    for wave, files in WAVE_FILES.items():
        print(f"\n{'='*50}\nWAVE {wave}\n{'='*50}")
        checks.append(f"\n--- WAVE {wave} ---")

        for ftype, processor, store in [
            ("HR", process_hr, all_hr), ("KR", process_kr, all_kr),
            ("IR", process_ir, all_ir), ("MR", process_mr, all_mr),
        ]:
            rel = files.get(ftype)
            if rel is None:
                checks.append(f"{ftype} {wave}: no file defined")
                continue
            path = BASE_DIR / rel
            if not path.exists():
                checks.append(f"{ftype} {wave}: FILE NOT FOUND")
                continue
            try:
                result = processor(wave, path, checks)
                result["wave"] = wave
                store.append(result)
            except Exception as e:
                checks.append(f"{ftype} {wave}: ERROR — {e}")
                import traceback; traceback.print_exc()

    print("\nStacking and merging...")
    hr_panel = pd.concat(all_hr, ignore_index=True)
    kr_panel = pd.concat(all_kr, ignore_index=True)
    ir_panel = pd.concat(all_ir, ignore_index=True)
    mr_panel = pd.concat(all_mr, ignore_index=True) if all_mr else None

    panel = hr_panel.merge(kr_panel, on=["wave", "province"], how="outer")
    panel = panel.merge(ir_panel,    on=["wave", "province"], how="outer")
    if mr_panel is not None:
        panel = panel.merge(mr_panel, on=["wave", "province"], how="outer")

    # ── Panel identifiers and instruments ─────────────────────────────────────
    panel["mining"]       = panel["province"].apply(is_mining)
    panel["copperbelt"]   = (panel["province"] == "copperbelt").astype(int)
    panel["northwestern"] = (panel["province"] == "northwestern").astype(int)
    panel["cu_price"]     = panel["wave"].map(COPPER_PRICES)
    panel["ln_cu_price"]  = np.log(panel["cu_price"])
    panel["bartik"]       = panel["mining"] * panel["ln_cu_price"]
    panel["post_priv"]    = (panel["wave"] > 2000).astype(int)

    def period(w):
        if w <= 2002: return "pre_boom"
        if w == 2007: return "boom"
        if w == 2014: return "bust"
        return "recovery"
    panel["period"] = panel["wave"].apply(period)

    panel = panel.sort_values(["province", "wave"]).reset_index(drop=True)
    # Encode from string for consistency across any subsetting
    panel["province_id"] = pd.Categorical(panel["province"]).codes
    panel["wave_id"]     = pd.Categorical(panel["wave"]).codes

    # ── Checks summary ────────────────────────────────────────────────────────
    checks += [
        "\n" + "=" * 65, "PANEL STRUCTURE", "=" * 65,
        f"Rows: {len(panel)}  Cols: {panel.shape[1]}",
        f"Waves: {sorted(panel['wave'].unique())}",
        f"Provinces ({panel['province'].nunique()}): {sorted(panel['province'].unique())}",
        f"Mining provinces: {sorted(panel[panel['mining']==1]['province'].unique())}",
        "NOTE: Muchinga merged into Northern at harmonisation — 9 consistent provinces.",
        f"NOTE: months_bf excluded for waves {sorted(M5_UNIT_SHIFT_WAVES)} — unit shift.",
        "NOTE: sought_treatment denominator = sick children (h22=1 OR h31=1), not all.",
        "NOTE: vacc_full built from h2/h7/h8/h9 — NOT from h10 (structural missingness).",
    ]

    miss = panel.isnull().mean().mul(100).round(1)
    miss = miss[
        (miss > 5) & (~miss.index.isin(["province", "wave"]))
    ].sort_values(ascending=False)
    checks.append("\nMissing rates (>5% only):")
    for cn, pct in miss.items():
        checks.append(f"  {cn:<35} {pct:.1f}%")

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_csv = OUT_DIR / "zambia_province_panel.csv"
    panel.to_csv(out_csv, index=False)
    print(f"\nPanel saved: {out_csv}  shape={panel.shape}")

    (OUT_DIR / "zambia_panel_checks.txt").write_text(
        "\n".join(checks), encoding="utf-8"
    )
    print(f"Checks saved: {OUT_DIR}/zambia_panel_checks.txt")
    return panel


if __name__ == "__main__":
    try:
        import pyreadstat
    except ImportError:
        print("pip install pandas pyreadstat numpy")
        sys.exit(1)
    main()


# =============================================================================
# TESTS — run with: pytest 02_build_panel.py -v
# =============================================================================
# Constants duplicated here so tests run without config_00.py / pyreadstat.
_DHS_MISSING_CODES    = {8, 9, 96, 97, 98, 99, 998, 999, 9998, 9999}
_IMPROVED_WATER_CODES = {11, 12, 13, 14, 21, 31, 41, 51, 61, 71}
_IMPROVED_SANIT_CODES = {11, 12, 13, 15, 21, 22}
_IMPROVED_FLOOR_CODES = {11, 21, 22, 23, 31, 32}
_PROVINCE_HARMONISE   = {
    "central":       "central",
    "copperbelt":    "copperbelt",
    "eastern":       "eastern",
    "luapula":       "luapula",
    "lusaka":        "lusaka",
    "muchinga":      "northern",
    "northern":      "northern",
    "north western": "northwestern",
    "north-western": "northwestern",
    "northwestern":  "northwestern",
    "southern":      "southern",
    "western":       "western",
}
_MINING_PROVINCES = {"copperbelt", "northwestern"}


def _to_num(series):
    return pd.to_numeric(series, errors="coerce")

def _flag_nan(series, threshold=9990):
    s = _to_num(series)
    return s.where(s <= threshold, np.nan)

def _harmonise(raw):
    cleaned = " ".join(str(raw).strip().lower().split())
    result  = _PROVINCE_HARMONISE.get(cleaned)
    if result is None:
        raise ValueError(f"Unknown province label: '{raw}'")
    return result

def _is_mining(prov):
    return 1 if prov in _MINING_PROVINCES else 0

def _binary(series, true_vals, false_vals):
    s = _to_num(series)
    result = pd.Series(np.nan, index=s.index, dtype=float)
    valid  = s.notna() & ~s.isin(_DHS_MISSING_CODES)
    result[valid & s.isin(true_vals)]  = 1.0
    result[valid & s.isin(false_vals)] = 0.0
    return result

def _safe_recode(series, code_map, missing_threshold=9990):
    s = _to_num(series)
    s = s.where(s <= missing_threshold, np.nan)
    s = s.where(~s.isin(_DHS_MISSING_CODES), np.nan)
    return s.map(code_map)

def _isin_vec(series, code_set, threshold=9990):
    s     = _to_num(series)
    valid = s.notna() & (s <= threshold) & ~s.isin(_DHS_MISSING_CODES)
    result = pd.Series(np.nan, index=s.index, dtype=float)
    result[valid] = 0.0
    result[valid & s.isin(code_set)] = 1.0
    return result

def _wt_mean(values, weights):
    v = _to_num(values)
    w = _to_num(weights)
    mask = v.notna() & w.notna() & (w > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.average(v[mask], weights=w[mask]))

def _collapse(df, outcomes, weight_col="w"):
    rows = []
    for prov, grp in df.groupby("province_name"):
        row = {"province": prov, "n": len(grp)}
        for var in outcomes:
            row[var] = _wt_mean(grp[var], grp[weight_col]) if var in grp.columns else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ── test_to_num ───────────────────────────────────────────────────────────────
def test_to_num_numeric_string():
    assert list(_to_num(pd.Series(["1", "2"]))) == [1.0, 2.0]

def test_to_num_non_numeric_becomes_nan():
    assert _to_num(pd.Series(["a"])).isna().all()

def test_to_num_none_becomes_nan():
    assert pd.isna(_to_num(pd.Series([None]))[0])

def test_to_num_mixed():
    r = _to_num(pd.Series([1, "x", 3.5]))
    assert r[0] == 1.0 and pd.isna(r[1]) and r[2] == 3.5

# ── test_flag_nan ─────────────────────────────────────────────────────────────
def test_flag_nan_at_threshold_kept():
    assert _flag_nan(pd.Series([9990]))[0] == 9990.0

def test_flag_nan_above_threshold_is_nan():
    assert _flag_nan(pd.Series([9991])).isna().all()

def test_flag_nan_normal_values_unchanged():
    assert list(_flag_nan(pd.Series([0, 100, 500]))) == [0.0, 100.0, 500.0]

def test_flag_nan_custom_threshold():
    r = _flag_nan(pd.Series([5, 10, 15]), threshold=10)
    assert r[0] == 5.0 and r[1] == 10.0 and pd.isna(r[2])

# ── test_harmonise ────────────────────────────────────────────────────────────
def test_harmonise_muchinga_to_northern():
    assert _harmonise("Muchinga") == "northern"

def test_harmonise_northern_stays_northern():
    assert _harmonise("Northern") == "northern"

def test_harmonise_north_western_space():
    assert _harmonise("North Western") == "northwestern"

def test_harmonise_north_western_hyphen():
    assert _harmonise("North-Western") == "northwestern"

def test_harmonise_strips_whitespace():
    assert _harmonise("  central  ") == "central"

def test_harmonise_case_insensitive():
    assert _harmonise("LUSAKA") == "lusaka"

def test_harmonise_unknown_raises():
    import pytest
    with pytest.raises(ValueError, match="Unknown province label"):
        _harmonise("Atlantis")

def test_harmonise_all_keys_resolve():
    for raw in _PROVINCE_HARMONISE:
        assert isinstance(_harmonise(raw), str)

# ── test_is_mining ────────────────────────────────────────────────────────────
def test_is_mining_copperbelt():
    assert _is_mining("copperbelt") == 1

def test_is_mining_northwestern():
    assert _is_mining("northwestern") == 1

def test_is_mining_lusaka_false():
    assert _is_mining("lusaka") == 0

def test_is_mining_returns_int():
    assert isinstance(_is_mining("copperbelt"), int)

# ── test_binary ───────────────────────────────────────────────────────────────
def test_binary_true_val():
    assert _binary(pd.Series([1]), [1], [0])[0] == 1.0

def test_binary_false_val():
    assert _binary(pd.Series([0]), [1], [0])[0] == 0.0

def test_binary_missing_8_is_nan():
    assert pd.isna(_binary(pd.Series([8]), [1], [0])[0])

def test_binary_missing_9_is_nan():
    assert pd.isna(_binary(pd.Series([9]), [1], [0])[0])

def test_binary_missing_96_is_nan():
    assert pd.isna(_binary(pd.Series([96]), [1], [0])[0])

def test_binary_missing_99_is_nan():
    assert pd.isna(_binary(pd.Series([99]), [1], [0])[0])

def test_binary_unlisted_is_nan():
    assert pd.isna(_binary(pd.Series([5]), [1], [0])[0])

def test_binary_nan_input_stays_nan():
    assert pd.isna(_binary(pd.Series([np.nan]), [1], [0])[0])

def test_binary_vacc_card_and_recall():
    r = _binary(pd.Series([0, 1, 2, 8]), [1, 2], [0])
    assert r[0] == 0.0 and r[1] == 1.0 and r[2] == 1.0 and pd.isna(r[3])

def test_binary_urban_rural():
    r = _binary(pd.Series([1, 2]), [1], [2])
    assert r[0] == 1.0 and r[1] == 0.0

def test_binary_all_dhs_missing_codes_are_nan():
    for code in _DHS_MISSING_CODES:
        assert pd.isna(_binary(pd.Series([code]), [1], [0])[0]), (
            f"DHS missing code {code} must be NaN"
        )

# ── test_safe_recode ──────────────────────────────────────────────────────────
def test_safe_recode_known_code():
    assert _safe_recode(pd.Series([1]), {1: 99})[0] == 99

def test_safe_recode_unknown_is_nan():
    assert pd.isna(_safe_recode(pd.Series([5]), {1: 1})[0])

def test_safe_recode_missing_8_is_nan():
    assert pd.isna(_safe_recode(pd.Series([8]), {8: 1})[0])

def test_safe_recode_missing_99_is_nan():
    assert pd.isna(_safe_recode(pd.Series([99]), {99: 1})[0])

def test_safe_recode_above_threshold_is_nan():
    assert pd.isna(_safe_recode(pd.Series([9991]), {9991: 1})[0])

def test_safe_recode_facility_delivery():
    fac_map = {**{i: 0 for i in range(10, 20)}, **{i: 1 for i in range(20, 40)}}
    r = _safe_recode(pd.Series([10, 20, 99]), fac_map)
    assert r[0] == 0 and r[1] == 1 and pd.isna(r[2])

def test_safe_recode_all_dhs_missing_codes_are_nan():
    for code in _DHS_MISSING_CODES:
        assert pd.isna(_safe_recode(pd.Series([code]), {code: 1})[0]), (
            f"DHS missing code {code} must be NaN"
        )

# ── test_isin_vec ─────────────────────────────────────────────────────────────
def test_isin_vec_in_set_is_1():
    assert _isin_vec(pd.Series([11]), {11, 12})[0] == 1.0

def test_isin_vec_not_in_set_is_0():
    assert _isin_vec(pd.Series([32]), {11, 12})[0] == 0.0

def test_isin_vec_96_is_nan_not_0():
    assert pd.isna(_isin_vec(pd.Series([96]), _IMPROVED_WATER_CODES)[0])

def test_isin_vec_99_is_nan_not_0():
    assert pd.isna(_isin_vec(pd.Series([99]), _IMPROVED_WATER_CODES)[0])

def test_isin_vec_above_threshold_is_nan():
    assert pd.isna(_isin_vec(pd.Series([9991]), {9991})[0])

def test_isin_vec_nan_input_stays_nan():
    assert pd.isna(_isin_vec(pd.Series([np.nan]), {1})[0])

def test_isin_vec_improved_water_codes_all_1():
    r = _isin_vec(pd.Series(list(_IMPROVED_WATER_CODES)), _IMPROVED_WATER_CODES)
    assert (r == 1.0).all()

def test_isin_vec_unimproved_codes_are_0():
    r = _isin_vec(pd.Series([32, 42, 43, 44]), _IMPROVED_WATER_CODES)
    assert (r == 0.0).all()

# ── test_wt_mean ──────────────────────────────────────────────────────────────
def test_wt_mean_equal_weights():
    assert abs(_wt_mean(pd.Series([1.0, 2.0, 3.0]), pd.Series([1.0, 1.0, 1.0])) - 2.0) < 1e-9

def test_wt_mean_unequal_weights():
    assert abs(_wt_mean(pd.Series([0.0, 1.0]), pd.Series([1.0, 3.0])) - 0.75) < 1e-9

def test_wt_mean_all_nan_returns_nan():
    assert pd.isna(_wt_mean(pd.Series([np.nan, np.nan]), pd.Series([1.0, 1.0])))

def test_wt_mean_zero_weight_excluded():
    assert abs(_wt_mean(pd.Series([0.0, 1.0]), pd.Series([0.0, 1.0])) - 1.0) < 1e-9

def test_wt_mean_nan_weight_excluded():
    assert abs(_wt_mean(pd.Series([0.0, 1.0]), pd.Series([np.nan, 1.0])) - 1.0) < 1e-9

def test_wt_mean_no_valid_obs_returns_nan():
    assert pd.isna(_wt_mean(pd.Series([1.0]), pd.Series([0.0])))

# ── test_collapse ─────────────────────────────────────────────────────────────
def test_collapse_correct_provinces():
    df = pd.DataFrame({
        "province_name": ["copperbelt", "copperbelt", "lusaka", "lusaka"],
        "x": [1.0, 3.0, 0.0, 2.0], "w": [1.0, 1.0, 1.0, 1.0],
    })
    out = _collapse(df, ["x"])
    assert set(out["province"]) == {"copperbelt", "lusaka"}

def test_collapse_weighted_mean():
    df = pd.DataFrame({
        "province_name": ["copperbelt", "copperbelt"],
        "x": [1.0, 3.0], "w": [1.0, 1.0],
    })
    out = _collapse(df, ["x"]).set_index("province")
    assert abs(out.loc["copperbelt", "x"] - 2.0) < 1e-9

def test_collapse_missing_outcome_is_nan():
    df = pd.DataFrame({"province_name": ["lusaka"], "w": [1.0]})
    out = _collapse(df, ["y"])
    assert out["y"].isna().all()

def test_collapse_n_counts_rows():
    df = pd.DataFrame({
        "province_name": ["lusaka", "lusaka"],
        "x": [1.0, 2.0], "w": [1.0, 1.0],
    })
    out = _collapse(df, ["x"]).set_index("province")
    assert out.loc["lusaka", "n"] == 2

# ── data quality checks ───────────────────────────────────────────────────────
def test_dq_96_99_never_zero_in_isin_vec():
    for code in [96, 99]:
        assert pd.isna(_isin_vec(pd.Series([code]), _IMPROVED_WATER_CODES)[0])

def test_dq_muchinga_northern_same_label():
    assert _harmonise("Muchinga") == _harmonise("Northern")

def test_dq_exactly_9_provinces_after_merge():
    assert len(set(_PROVINCE_HARMONISE.values())) == 9

def test_dq_only_two_mining_provinces():
    all_provs = set(_PROVINCE_HARMONISE.values())
    for p in all_provs:
        expected = 1 if p in {"copperbelt", "northwestern"} else 0
        assert _is_mining(p) == expected

def test_dq_vacc_full_requires_all_four():
    full_cols = ["vacc_bcg", "vacc_measles", "vacc_dpt3", "vacc_polio3"]
    df = pd.DataFrame({
        "vacc_bcg":     [1, 1, 0, 1],
        "vacc_measles": [1, 1, 1, 1],
        "vacc_dpt3":    [1, 1, 1, 1],
        "vacc_polio3":  [1, 0, 1, np.nan],
    })
    valid = df[full_cols].notna().all(axis=1)
    vf = df[full_cols].min(axis=1)
    vf[~valid] = np.nan
    assert vf[0] == 1.0 and vf[1] == 0.0 and vf[2] == 0.0 and pd.isna(vf[3])

def test_dq_sought_treatment_nan_for_healthy():
    fever  = pd.Series([1, 0, np.nan])
    cough  = pd.Series([0, 0, 0])
    h32z   = _binary(pd.Series([1, 1, 1]), [1], [0])
    is_sick = (fever == 1) | (cough.isin([1, 2]))
    sought  = h32z.where(is_sick, np.nan)
    assert sought[0] == 1.0 and pd.isna(sought[1]) and pd.isna(sought[2])

def test_dq_months_bf_capped_at_36():
    m5 = pd.Series([0, 18, 36, 37, 93])
    mbf = m5.where((m5 >= 0) & (m5 <= 36), np.nan)
    assert mbf[0] == 0.0 and mbf[2] == 36.0 and pd.isna(mbf[3]) and pd.isna(mbf[4])

def test_dq_ever_breastfed_both_eras():
    bf_map = {0: 0, 93: 1, 94: 0, 95: 1}
    bf_map.update({i: 1 for i in range(1, 93)})
    assert _safe_recode(pd.Series([0]),  bf_map)[0] == 0
    assert _safe_recode(pd.Series([12]), bf_map)[0] == 1
    assert _safe_recode(pd.Series([93]), bf_map)[0] == 1
    assert _safe_recode(pd.Series([94]), bf_map)[0] == 0
    assert _safe_recode(pd.Series([95]), bf_map)[0] == 1

def test_dq_bmi_plausible_range():
    bmi_raw = pd.Series([1199, 1200, 6000, 6001])
    bmi = bmi_raw.where(bmi_raw.between(1200, 6000), np.nan) / 100
    assert pd.isna(bmi[0]) and bmi[1] == 12.0 and bmi[2] == 60.0 and pd.isna(bmi[3])

def test_dq_wealth_quintile_1_to_5():
    wq = _to_num(pd.Series([0, 1, 5, 6]))
    wq = wq.where(wq.between(1, 5), np.nan)
    assert pd.isna(wq[0]) and wq[1] == 1.0 and wq[2] == 5.0 and pd.isna(wq[3])

def test_dq_haemoglobin_divided_by_10():
    hb = _flag_nan(pd.Series([120, 145, 9991])) / 10
    assert hb[0] == 12.0 and hb[1] == 14.5 and pd.isna(hb[2])

def test_dq_anc_98_99_nan():
    anc = _to_num(pd.Series([4, 98, 99])).where(lambda x: x < 98, np.nan)
    assert anc[0] == 4.0 and pd.isna(anc[1]) and pd.isna(anc[2])

def test_dq_crowding_zero_rooms_nan():
    crowding = pd.Series([5.0]) / pd.Series([0.0]).replace(0, np.nan)
    assert pd.isna(crowding[0])