"""
03_regressions.py  —  INDIVIDUAL-LEVEL VERSION  (OPTIMISED)
=============================================================
Runs individual-level DiD regressions for the Zambia copper paper.

Model:
  Y_igt = α + β(mining_g × ln_cu_price_t) + γ_g + δ_t + X_igt'ψ + ε_igt

  i = individual (woman / child / household / man)
  g = province
  t = wave
  β = Bartik DiD coefficient
  γ_g = province fixed effects  (via dummies)
  δ_t = wave fixed effects      (via dummies)
  X_igt = individual-level controls
  SE clustered at province level

Outputs:
  OUTPUT/ind_results.csv     — all coefficients
  OUTPUT/ind_tables.txt      — formatted paper tables
  OUTPUT/ind_checks.txt      — diagnostics

OPTIMISATION CHANGES vs original:
  1. Vectorised binary/flag_nan helpers — no row-by-row apply()
  2. Dummies built ONCE per dataset (not re-attached on every run_table call)
  3. Per-outcome progress printing so you can see work is happening
  4. Early skip if outcome has < MIN_OBS valid rows (saves matrix work)

FIXES applied:
  FIX 1 — ever_breastfed (m4): code 93 = "stopped breastfeeding" = EVER breastfed.
           Previous code set 93 → 0 (wrong). Now uses safe_recode with bf_map
           matching 02_build_panel.py exactly: {93:1, 94:0, 95:1, 1-92:1}.
  FIX 2 — had_diarrhoea (h11): early waves (1992) use code 1 = "last 24h" AND
           code 2 = "last 2 weeks". Previous code only set v==1 → 1, dropping
           code-2 responses as NaN. Now: v.isin([1, 2]) → 1.
           Same fix applied to had_cough (h31) which uses the same dual coding.
"""

import sys, warnings, time
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
from scipy import stats

# ── All shared constants from config_00.py (single source of truth) ─────────
import os as _os
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from config_00 import (
    BASE_DIR, OUT_DIR, MIN_OBS, MIN_CLUSTERS,
    COPPER_PRICES, PROVINCE_HARMONISE, MINING_PROVINCES, WAVE_FILES,
    DHS_MISSING_CODES,
    IMPROVED_WATER_CODES, IMPROVED_SANIT_CODES, IMPROVED_FLOOR_CODES,
)

# ── Outcome tables ─────────────────────────────────────────────────────────────
OUTCOME_TABLES = {
    "Table 1: Wealth and Assets": [
        ("asset_index",    "Asset index",               100, True),
        ("asset_elec",     "Has electricity",           100, True),
        ("asset_tv",       "Has television",            100, True),
        ("floor_finished", "Finished floor",            100, True),
        ("improved_water", "Improved water source",     100, True),
        ("improved_sanit", "Improved sanitation",       100, True),
        ("has_net",        "Has mosquito net",          100, True),
        ("clean_fuel",     "Clean cooking fuel",        100, True),
    ],
    "Table 2: Child Health": [
        ("u5_dead",        "U5 mortality (per 1000)",  1000, False),
        ("infant_dead",    "Infant mortality (per 1000)",1000,False),
        ("neonatal_dead",  "Neonatal mortality (per 1000)",1000,False),
        ("stunted",        "Stunted",                   100, False),
        ("underweight",    "Underweight",               100, False),
        ("wasted",         "Wasted",                    100, False),
        ("vacc_bcg",       "BCG vaccinated",            100, True),
        ("vacc_measles",   "Measles vaccinated",        100, True),
        ("vacc_dpt3",      "DPT3 vaccinated",           100, True),
        ("vacc_full",      "Fully vaccinated",          100, True),
        ("had_diarrhoea",  "Diarrhoea last 2 weeks",    100, False),
        ("had_fever",      "Fever last 2 weeks",        100, False),
        ("had_cough",      "Cough/ARI last 2 weeks",    100, False),
        ("ever_breastfed", "Ever breastfed",            100, True),
    ],
    "Table 3: Maternal and Reproductive Health": [
        ("delivery_facility","Facility delivery",       100, True),
        ("delivery_skilled", "Skilled birth attendance",100, True),
        ("delivery_csection","Caesarean section",       100, True),
        ("anc_4plus",        "ANC 4+ visits",           100, True),
        ("anc_first_trim",   "ANC first trimester",     100, True),
        ("pnc_mother",       "Postnatal care (mother)",  100, True),
        ("tetanus_2plus",    "Tetanus 2+ injections",   100, True),
        ("iron_suppl",       "Iron supplementation",    100, True),
        ("modern_contra",    "Modern contraception",    100, True),
        ("unmet_need",       "Unmet need for FP",       100, False),
        ("age_first_birth",  "Age at first birth",        1, True),
        ("age_first_sex",    "Age at first sex",          1, True),
    ],
    "Table 4: Women's Empowerment": [
        ("edu_secondary_p",  "Secondary+ education",    100, True),
        ("edu_years",        "Years of education",        1, True),
        ("employed",         "Currently employed",      100, True),
        ("earn_cash",        "Earns cash",              100, True),
        ("decide_health",    "Decides own healthcare",  100, True),
        ("decide_purchase",  "Decides large purchases", 100, True),
        ("decide_food",      "Decides daily food",      100, True),
        ("can_go_health",    "Can visit facility alone",100, True),
        ("owns_house",       "Owns house",              100, True),
        ("owns_land",        "Owns land",               100, True),
        ("has_account",      "Has bank account",        100, True),
        ("has_mobile",       "Has mobile phone",        100, True),
    ],
    "Table 5: Domestic Violence": [
        ("dv_any",           "Any violence (ever)",     100, False),
        ("dv_any_phys",      "Physical violence (ever)",100, False),
        ("dv_emotional",     "Emotional violence (ever)",100,False),
        ("dv_slapped",       "Slapped (ever)",          100, False),
        ("dv_punched",       "Punched (ever)",          100, False),
        ("dv_kicked",        "Kicked/dragged (ever)",   100, False),
        ("dv_choked",        "Choked/burned (ever)",    100, False),
        ("dv_justify_any",   "Justifies DV any reason", 100, False),
    ],
    "Table 6: Men's Outcomes": [
        ("man_edu_sec_p",    "Secondary+ education",    100, True),
        ("man_edu_level",    "Education level (0-3)",     1, True),
        ("man_employed",     "Currently employed",      100, True),
        ("man_condom",       "Condom at last sex",      100, True),
        ("man_multi_part",   "2+ partners last year",   100, False),
        ("man_dv_justify",   "Justifies DV",            100, False),
    ],
}

# =============================================================================
# HELPERS  (vectorised — no row-by-row apply)
# =============================================================================

def to_num(series):
    """Safe numeric conversion."""
    return pd.to_numeric(series, errors="coerce")

def flag_nan(series, threshold=9990):
    """Replace values above threshold with NaN."""
    s = to_num(series)
    return s.where(s <= threshold, np.nan)

def binary(series, true_vals, false_vals):
    """
    Strict binary. false_vals always required — no implicit mode.
    DHS_MISSING_CODES (8,9,96,97,98,99,...) always → NaN, never 0.
    Any code not in true_vals or false_vals → NaN.
    """
    s = to_num(series)
    result = pd.Series(np.nan, index=s.index, dtype=float)
    valid = s.notna() & ~s.isin(DHS_MISSING_CODES)
    result[valid & s.isin(true_vals)]  = 1.0
    result[valid & s.isin(false_vals)] = 0.0
    return result

def safe_recode(series, code_map, missing_threshold=9990):
    """Recode via explicit dict. Codes not in map → NaN. DHS_MISSING_CODES → NaN first."""
    s = to_num(series)
    s = s.where(s <= missing_threshold, np.nan)
    s = s.where(~s.isin(DHS_MISSING_CODES), np.nan)
    return s.map(code_map)

def isin_vec(series, code_set, threshold=9990):
    """
    For WASH/floor/fuel codes. DHS 96 (other) and 99 (missing) → NaN, NOT 0.
    Unknown source ≠ unimproved — coding 96/99 as 0 biases WASH outcomes downward.
    """
    s = to_num(series)
    valid = s.notna() & (s <= threshold) & ~s.isin(DHS_MISSING_CODES)
    result = pd.Series(np.nan, index=s.index, dtype=float)
    result[valid] = 0.0
    result[valid & s.isin(code_set)] = 1.0
    return result

def find_col(df, candidates):
    for c in candidates:
        if c.lower() in df.columns:
            return c.lower()
    return None

def stars(p):
    if pd.isna(p): return ""
    if p < 0.01:   return "***"
    if p < 0.05:   return "**"
    if p < 0.10:   return "*"
    return ""

# =============================================================================
# OLS WITH CLUSTERED SE
# =============================================================================

def ols_cluster(y_arr, X_arr, clust_arr):
    """OLS with CR1S clustered SE."""
    n, k = X_arr.shape
    try:
        XX    = X_arr.T @ X_arr
        XXinv = np.linalg.inv(XX)
    except np.linalg.LinAlgError:
        XXinv = np.linalg.pinv(X_arr.T @ X_arr)

    coef  = XXinv @ (X_arr.T @ y_arr)
    resid = y_arr - X_arr @ coef

    unique_c = np.unique(clust_arr)
    G = len(unique_c)
    meat = np.zeros((k, k))
    for c in unique_c:
        idx = np.where(clust_arr == c)[0]
        sc  = X_arr[idx].T @ resid[idx]
        meat += np.outer(sc, sc)

    correction = (G / (G - 1)) * ((n - 1) / (n - k))
    V    = correction * (XXinv @ meat @ XXinv)
    se   = np.sqrt(np.maximum(np.diag(V), 0.0))

    tstat = np.where(se > 0, coef / se, np.nan)
    pval  = np.where(se > 0,
                     2 * stats.t.sf(np.abs(tstat), df=G - 1),
                     np.nan)
    return coef, se, tstat, pval


def run_one(df, y_var, treat_var, control_cols, cluster_col, scale):
    """Run one regression; return result dict for treat_var coefficient."""
    cols_needed = list(dict.fromkeys(
        [y_var, treat_var, cluster_col] + control_cols))
    cols_needed = [c for c in cols_needed if c in df.columns]
    sub = df[cols_needed].dropna().copy()

    if len(sub) < MIN_OBS:
        return None

    y   = sub[y_var].to_numpy(dtype=float)
    ctrl_present = [c for c in control_cols if c in sub.columns]
    X   = sub[[treat_var] + ctrl_present].to_numpy(dtype=float)
    X   = np.column_stack([np.ones(len(X)), X])
    cl  = pd.Categorical(sub[cluster_col].values).codes.astype(int)

    G = len(np.unique(cl))
    if G < 2:
        return None

    try:
        coef, se, tstat, pval = ols_cluster(y, X, cl)
    except Exception:
        return None

    tidx  = 1
    ci_lo = (coef[tidx] - stats.t.ppf(0.975, df=G - 1) * se[tidx]) * scale
    ci_hi = (coef[tidx] + stats.t.ppf(0.975, df=G - 1) * se[tidx]) * scale

    yhat   = X @ coef
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "coef":       coef[tidx],
        "coef_s":     coef[tidx] * scale,
        "se":         se[tidx],
        "se_s":       se[tidx] * scale,
        "tstat":      tstat[tidx],
        "pval":       pval[tidx],
        "ci_lo":      ci_lo,
        "ci_hi":      ci_hi,
        "nobs":       len(sub),
        "n_clusters": G,
        "r2":         r2,
    }

# =============================================================================
# DATA BUILDERS  (vectorised)
# =============================================================================

def load(path):
    df, meta = pyreadstat.read_dta(str(path))
    df.columns = df.columns.str.lower()
    return df, meta

def get_label(meta, col_name, code):
    labels = {}
    if meta and hasattr(meta, "variable_value_labels"):
        labels = meta.variable_value_labels.get(col_name, {})
    key = int(code) if isinstance(code, (float, np.floating)) else code
    return str(labels.get(key, labels.get(code, str(code)))).strip().lower()

def harmonise_prov(raw):
    """Collapse whitespace then look up. Muchinga→Northern (see config_00.py)."""
    cleaned = " ".join(str(raw).strip().lower().split())
    result = PROVINCE_HARMONISE.get(cleaned)
    if result is None:
        raise ValueError(
            f"Unknown province label: '{raw}' — add to PROVINCE_HARMONISE in config_00.py"
        )
    return result


def build_hr(df, meta, wave):
    asset_map = {"asset_elec":"hv206","asset_radio":"hv207","asset_tv":"hv208",
                 "asset_fridge":"hv209","asset_bike":"hv210","asset_car":"hv212"}
    acols = []
    for nm, c in asset_map.items():
        if c in df.columns:
            s = to_num(df[c])
            df[nm] = s.where(s.isin([0, 1]), np.nan)
        else:
            df[nm] = np.nan
        acols.append(nm)
    df["asset_index"] = df[acols].mean(axis=1, skipna=True)

    wq = to_num(df.get("hv270", pd.Series(np.nan, index=df.index)))
    df["wealth_q"] = wq.where(wq.between(1, 5), np.nan)

    df["improved_water"] = isin_vec(df.get("hv201"), IMPROVED_WATER_CODES) \
        if "hv201" in df.columns else pd.Series(np.nan, index=df.index)
    df["improved_sanit"] = isin_vec(df.get("hv205"), IMPROVED_SANIT_CODES) \
        if "hv205" in df.columns else pd.Series(np.nan, index=df.index)
    df["floor_finished"] = isin_vec(df.get("hv213"), IMPROVED_FLOOR_CODES) \
        if "hv213" in df.columns else pd.Series(np.nan, index=df.index)

    nc = find_col(df, ["hv227"])
    df["has_net"]   = binary(df[nc], [1], [0]) if nc else pd.Series(np.nan, index=df.index)
    df["urban"]     = binary(df.get("hv025", pd.Series(np.nan, index=df.index)), [1], [2])
    df["hh_size"]   = to_num(df.get("hv009", pd.Series(np.nan, index=df.index)))
    df["female_hh"] = binary(df.get("hv219", pd.Series(np.nan, index=df.index)), [2], [1])

    if "hv226" in df.columns:
        df["clean_fuel"] = safe_recode(df["hv226"],
            {1:1,2:1,3:1,4:1,5:1,6:0,7:0,8:0,9:0,10:0,11:0,95:0})
    else:
        df["clean_fuel"] = np.nan

    return df


def build_kr(df, meta, wave):
    b5 = to_num(df.get("b5", pd.Series(np.nan, index=df.index)))
    dead          = (b5 == 0)
    alive_or_dead = b5.notna()
    if "b7" in df.columns:
        b7          = to_num(df["b7"])
        b7_for_dead = b7.where(dead, np.nan)
        df["u5_dead"]       = dead.astype(float).where(alive_or_dead, np.nan)
        df["infant_dead"]   = (dead & (b7_for_dead < 12)).astype(float).where(alive_or_dead, np.nan)
        df["neonatal_dead"] = (dead & (b7_for_dead <  1)).astype(float).where(alive_or_dead, np.nan)
    else:
        df["u5_dead"]       = dead.astype(float).where(alive_or_dead, np.nan)
        df["infant_dead"]   = df["neonatal_dead"] = np.nan

    for zvar, flag in [("hw70", "stunted"), ("hw71", "underweight"), ("hw72", "wasted")]:
        if zvar in df.columns:
            z = flag_nan(df[zvar])
            df[flag] = (z < -200).astype(float).where(z.notna(), np.nan)
        else:
            df[flag] = np.nan

    for nm, c in [("vacc_bcg","h2"), ("vacc_dpt3","h7"),
                  ("vacc_measles","h9"), ("vacc_polio3","h8")]:
        if c in df.columns:
            v = to_num(df[c])
            r = pd.Series(np.nan, index=v.index, dtype=float)
            r[v == 0]         = 0.0
            r[v.isin([1, 2])] = 1.0
            df[nm] = r
        else:
            df[nm] = np.nan

    full_c = ["vacc_bcg","vacc_measles","vacc_dpt3","vacc_polio3"]
    valid  = df[full_c].notna().all(axis=1)
    df["vacc_full"] = df[full_c].min(axis=1)
    df.loc[~valid, "vacc_full"] = np.nan

    # ── FIX 2: illness variables ──────────────────────────────────────────────
    # h11 (diarrhoea) and h31 (cough): early waves use 1=last 24h + 2=last 2wks.
    # Later waves use only 2=last 2 weeks. Both codes mean "child was sick".
    # Previous code: r[v == 1] = 1.0  →  dropped all code-2 responses as NaN.
    for nm, c, sick_codes in [
        ("had_diarrhoea", "h11", [1, 2]),   # FIX: both 1 and 2 = sick
        ("had_fever",     "h22", [1]),       # h22 uses only code 1 across all waves
        ("had_cough",     "h31", [1, 2]),    # FIX: same dual-code issue as h11
    ]:
        if c in df.columns:
            v = to_num(df[c])
            r = pd.Series(np.nan, index=v.index, dtype=float)
            r[v == 0]             = 0.0
            r[v.isin(sick_codes)] = 1.0
            df[nm] = r
        else:
            df[nm] = np.nan

    # ── FIX 1: ever_breastfed ─────────────────────────────────────────────────
    # m4 coding across waves:
    #   1992-2014: 0=never, 1-92=months breastfed (all positive values = ever)
    #   2018-2024: 93=stopped (=ever), 94=never breastfed, 95=still breastfeeding (=ever)
    # Previous code set m4==93 → 0, which was wrong (93 = stopped = ever breastfed).
    # bf_map matches 02_build_panel.py exactly.
    if "m4" in df.columns:
        m4     = to_num(df["m4"])
        bf_map = {0: 0, 93: 1, 94: 0, 95: 1}
        bf_map.update({i: 1 for i in range(1, 93)})
        df["ever_breastfed"] = safe_recode(m4, bf_map)
    else:
        df["ever_breastfed"] = np.nan

    df["child_age"]   = to_num(df.get("b8",   pd.Series(np.nan, index=df.index)))
    df["child_male"]  = binary(df.get("b4",   pd.Series(np.nan, index=df.index)), [1], [2])
    df["birth_order"] = to_num(df.get("bord", pd.Series(np.nan, index=df.index)))
    df["urban"]       = binary(df.get("v025", pd.Series(np.nan, index=df.index)), [1], [2])
    return df


def build_ir(df, meta, wave):
    if "m14_1" in df.columns:
        anc = to_num(df["m14_1"]).where(lambda x: x < 98, np.nan)
        df["anc_4plus"] = (anc >= 4).astype(float).where(anc.notna(), np.nan)
    else:
        df["anc_4plus"] = np.nan

    if "m13_1" in df.columns:
        m13 = to_num(df["m13_1"]).where(lambda x: x < 90, np.nan)
        df["anc_first_trim"] = (m13 <= 3).astype(float).where(m13.notna(), np.nan)
    else:
        df["anc_first_trim"] = np.nan

    if "m15_1" in df.columns:
        m15 = to_num(df["m15_1"])
        valid = m15.notna() & ~m15.isin({98, 99})
        df["delivery_facility"] = pd.Series(np.nan, index=m15.index, dtype=float)
        df.loc[valid,                       "delivery_facility"] = 0.0
        df.loc[valid & m15.between(20, 39), "delivery_facility"] = 1.0
    else:
        df["delivery_facility"] = np.nan

    sba_c = find_col(df, ["m3a_1","m3a"])
    snm_c = find_col(df, ["m3b_1","m3b"])
    if sba_c:
        doc = binary(df[sba_c], [1], [0])
        nmw = binary(df[snm_c], [1], [0]) if snm_c else pd.Series(0.0, index=df.index)
        df["delivery_skilled"] = ((doc == 1) | (nmw == 1)).astype(float)
        df.loc[doc.isna() & nmw.isna(), "delivery_skilled"] = np.nan
    else:
        df["delivery_skilled"] = np.nan

    df["delivery_csection"] = binary(
        df.get("m17_1", pd.Series(np.nan, index=df.index)), [1], [0])

    pnc_c = find_col(df, ["m62_1"])
    df["pnc_mother"] = binary(df[pnc_c], [1], [0]) if pnc_c \
        else pd.Series(np.nan, index=df.index)

    iron_c = find_col(df, ["m45_1"])
    df["iron_suppl"] = binary(df[iron_c], [1], [0]) if iron_c \
        else pd.Series(np.nan, index=df.index)

    if "m1_1" in df.columns:
        tet = to_num(df["m1_1"]).where(lambda x: x < 90, np.nan)
        df["tetanus_2plus"] = (tet >= 2).astype(float).where(tet.notna(), np.nan)
    else:
        df["tetanus_2plus"] = np.nan

    if "v313" in df.columns:
        df["modern_contra"] = binary(to_num(df["v313"]), [3], [0,1,2])
    else:
        df["modern_contra"] = np.nan

    umet_c = find_col(df, ["v626a","v626"])
    if umet_c:
        df["unmet_need"] = safe_recode(df[umet_c], {1:1,2:1,3:0,4:0,5:0,6:0,7:0})
    else:
        df["unmet_need"] = np.nan

    if "v212" in df.columns:
        afb = to_num(df["v212"])
        df["age_first_birth"] = afb.where(afb > 0, np.nan)
    else:
        df["age_first_birth"] = np.nan

    if "v531" in df.columns:
        afs = to_num(df["v531"])
        df["age_first_sex"] = afs.where(afs > 0, np.nan)
    else:
        df["age_first_sex"] = np.nan

    if "v106" in df.columns:
        edu = to_num(df["v106"])
        df["edu_secondary_p"] = (edu >= 2).astype(float).where(edu.notna(), np.nan)
        df["edu_level"]       = edu.where(edu.between(0, 3), np.nan)
    else:
        df["edu_secondary_p"] = df["edu_level"] = np.nan

    if "v133" in df.columns:
        ey = to_num(df["v133"])
        df["edu_years"] = ey.where(ey <= 30, np.nan)
    else:
        df["edu_years"] = np.nan

    df["employed"]  = binary(df.get("v714", pd.Series(np.nan, index=df.index)), [1], [0])
    earn_c = find_col(df, ["v741"])
    df["earn_cash"] = binary(df[earn_c], [1,2], [3,4]) if earn_c \
        else pd.Series(np.nan, index=df.index)

    for vn, cn in [("decide_health","v743a"), ("decide_purchase","v743b"),
                   ("decide_food","v743d")]:
        c = find_col(df, [cn])
        if c:
            v = to_num(df[c])
            r = pd.Series(np.nan, index=v.index, dtype=float)
            r[v.isin({3, 4, 5, 6})] = 0.0
            r[v.isin({1, 2})]       = 1.0
            df[vn] = r
        else:
            df[vn] = np.nan

    c = find_col(df, ["v743f"])
    df["can_go_health"] = binary(df[c], [1], [0]) if c \
        else pd.Series(np.nan, index=df.index)

    for vn, cn in [("owns_house","v745a"), ("owns_land","v745b")]:
        c = find_col(df, [cn])
        df[vn] = binary(df[c], [1,2], [3]) if c else pd.Series(np.nan, index=df.index)

    for vn, cn in [("has_account","v170"), ("has_mobile","v169a")]:
        c = find_col(df, [cn])
        df[vn] = binary(df[c], [1], [0]) if c else pd.Series(np.nan, index=df.index)

    dv_att = []
    for i, cn in enumerate(["v744a","v744b","v744c","v744d","v744e"]):
        c = find_col(df, [cn])
        vn = f"_dvj{i}"
        df[vn] = binary(df[c], [1], [0]) if c else pd.Series(np.nan, index=df.index)
        dv_att.append(vn)
    valid_att = df[dv_att].notna().any(axis=1)
    df["dv_justify_any"] = df[dv_att].max(axis=1)
    df.loc[~valid_att, "dv_justify_any"] = np.nan

    for vn, cn in [("dv_humiliate","d103a"), ("dv_threaten","d103b"), ("dv_insult","d103c"),
                   ("dv_pushed","d105a"),   ("dv_slapped","d105b"),  ("dv_punched","d105c"),
                   ("dv_kicked","d105d"),   ("dv_choked","d105e"),
                   ("dv_any_phys","d106"),  ("dv_any_sex","d107"),   ("dv_any","d108")]:
        c = find_col(df, [cn])
        if c:
            v = to_num(df[c])
            r = pd.Series(np.nan, index=v.index, dtype=float)
            r[v == 0] = 0.0
            r[v == 1] = 1.0
            df[vn] = r
        else:
            df[vn] = np.nan

    emo = ["dv_humiliate","dv_threaten","dv_insult"]
    valid_emo = df[emo].notna().any(axis=1)
    df["dv_emotional"] = df[emo].max(axis=1)
    df.loc[~valid_emo, "dv_emotional"] = np.nan

    if "v445" in df.columns:
        bmi = to_num(df["v445"])
        df["woman_bmi"] = bmi.where(bmi.between(1200, 6000), np.nan) / 100
    else:
        df["woman_bmi"] = np.nan

    df["age_woman"] = to_num(df.get("v012", pd.Series(np.nan, index=df.index)))
    df["urban"]     = binary(df.get("v025", pd.Series(np.nan, index=df.index)), [1], [2])
    df["married"]   = binary(df.get("v501", pd.Series(np.nan, index=df.index)), [1,2], [0,3,4,5])

    if "v104" in df.columns:
        v104 = to_num(df["v104"])
        df["v104"] = v104.where(v104 <= 95, np.nan)  # 96=visitor, 97=NS → NaN
    else:
        df["v104"] = np.nan
    return df


def build_mr(df, meta, wave):
    if "mv106" in df.columns:
        edu = to_num(df["mv106"])
        df["man_edu_sec_p"] = (edu >= 2).astype(float).where(edu.notna(), np.nan)
        df["man_edu_level"] = edu.where(edu.between(0, 3), np.nan)
    else:
        df["man_edu_sec_p"] = df["man_edu_level"] = np.nan

    df["man_employed"] = binary(df.get("mv714", pd.Series(np.nan, index=df.index)), [1], [0])

    df["man_condom"] = binary(df.get("mv761", pd.Series(np.nan, index=df.index)), [1], [0]) \
        if "mv761" in df.columns else pd.Series(np.nan, index=df.index)

    if "mv766a" in df.columns:
        mp = to_num(df["mv766a"])
        df["man_multi_part"] = (mp >= 2).astype(float).where(mp.notna() & (mp < 90), np.nan)
    else:
        df["man_multi_part"] = np.nan

    df["man_dv_justify"] = binary(df.get("mv744a", pd.Series(np.nan, index=df.index)), [1], [0]) \
        if "mv744a" in df.columns else pd.Series(np.nan, index=df.index)

    df["man_age"]   = to_num(df.get("mv012", pd.Series(np.nan, index=df.index)))
    df["man_urban"] = binary(df.get("mv025", pd.Series(np.nan, index=df.index)), [1], [2])
    return df


# =============================================================================
# LOAD ALL WAVES FOR ONE FILE TYPE
# =============================================================================

BUILDERS  = {"HR": build_hr, "KR": build_kr, "IR": build_ir, "MR": build_mr}
PROV_COLS = {"HR": "hv024",  "KR": "v024",   "IR": "v024",   "MR": "mv024"}
WGHT_COLS = {"HR": "hv005",  "KR": "v005",   "IR": "v005",   "MR": "mv005"}

def load_stack(ftype, checks):
    builder  = BUILDERS[ftype]
    prov_col = PROV_COLS[ftype]
    wgt_col  = WGHT_COLS[ftype]
    frames   = []

    for wave, files in WAVE_FILES.items():
        rel = files.get(ftype)
        if not rel:
            continue
        path = BASE_DIR / rel
        if not path.exists():
            checks.append(f"  {ftype} {wave}: NOT FOUND — {path.name}")
            continue
        try:
            t0 = time.time()
            df, meta = load(path)

            if prov_col not in df.columns:
                checks.append(f"  {ftype} {wave}: province col missing")
                continue

            df["province"] = df[prov_col].apply(
                lambda c: harmonise_prov(get_label(meta, prov_col, c)))

            w = to_num(df.get(wgt_col, pd.Series(1, index=df.index))) / 1_000_000
            df["w"] = w.where(w > 0, np.nan)

            df = builder(df, meta, wave)

            df["wave"]        = wave
            df["ln_cu_price"] = np.log(COPPER_PRICES[wave])
            df["mining"]      = df["province"].apply(
                lambda p: 1.0 if p in MINING_PROVINCES else 0.0)
            df["bartik"]      = df["mining"] * df["ln_cu_price"]
            df["post_priv"]   = 1 if wave > 2000 else 0

            frames.append(df)
            elapsed = time.time() - t0
            checks.append(f"  {ftype} {wave}: {len(df):,} obs  ({elapsed:.1f}s)")
            print(f"    loaded {ftype} {wave}: {len(df):,} rows  ({elapsed:.1f}s)")

        except Exception as e:
            checks.append(f"  {ftype} {wave}: ERROR — {e}")
            import traceback; traceback.print_exc()

    if not frames:
        return None

    out = pd.concat(frames, ignore_index=True)
    out["province_id"] = pd.Categorical(out["province"]).codes.astype(int)
    out["wave_id"]     = pd.Categorical(out["wave"]).codes.astype(int)

    keep = {"province","province_id","wave","wave_id",
            "bartik","mining","ln_cu_price","post_priv",
            "w"}   # survey weight — needed for weighted means in 08_figures.py
    for tbl_outcomes in OUTCOME_TABLES.values():
        for (var, _, _, _) in tbl_outcomes:
            keep.add(var)
    for c in ["urban","hh_size","female_hh",
              "child_age","child_male","birth_order",
              "age_woman","married","edu_level","edu_secondary_p",
              "man_age","man_urban",
              "v104"]:
        keep.add(c)
    out = out[[c for c in keep if c in out.columns]].copy()
    print(f"    slimmed to {len(out.columns)} columns")
    return out


# =============================================================================
# TWO-WAY FE WITH CLUSTERED SE
# =============================================================================

def demean_by_fast(arr, group_ids):
    """Vectorised group-demeaning via np.add.at. Works for 1-D and 2-D arrays."""
    out   = arr.astype(float)
    G     = int(group_ids.max()) + 1
    if out.ndim == 1:
        sums   = np.zeros(G)
        counts = np.zeros(G, dtype=int)
        np.add.at(sums,   group_ids, out)
        np.add.at(counts, group_ids, 1)
        means  = sums / np.maximum(counts, 1)
        out   -= means[group_ids]
    else:
        k      = out.shape[1]
        sums   = np.zeros((G, k))
        counts = np.zeros(G, dtype=int)
        np.add.at(sums,   group_ids, out)
        np.add.at(counts, group_ids, 1)
        means  = sums / np.maximum(counts[:, None], 1)
        out   -= means[group_ids]
    return out


def within_demean(arr, prov_ids, wave_ids, max_iter=50, tol=1e-10):
    """Iterative two-way FE demeaning with convergence check (tol=1e-10)."""
    out  = arr.copy().astype(float)
    prev = out.copy()
    for i in range(max_iter):
        out  = demean_by_fast(out, prov_ids)
        out  = demean_by_fast(out, wave_ids)
        diff = np.max(np.abs(out - prev))
        if diff < tol:
            return out, i+1, True
        prev = out.copy()
    return out, max_iter, False


def run_one_fe(df, y_var, treat_var, control_cols, scale, _warn_log=None):
    """OLS with two-way FE (within-demeaning). Clustered SE at province level."""
    if y_var not in df.columns:
        return None
    def _w(msg):
        if _warn_log is not None: _warn_log.append(f"  WARN {y_var}: {msg}")

    cols = [y_var, treat_var, "province", "province_id", "wave_id"] + control_cols
    cols = list(dict.fromkeys(c for c in cols if c in df.columns))
    sub  = df[cols].dropna().copy()

    if len(sub) < MIN_OBS:
        return None

    prov = sub["province_id"].to_numpy(dtype=int)
    wave = sub["wave_id"].to_numpy(dtype=int)
    ctrl_cols = [c for c in control_cols if c in sub.columns]

    Xraw = sub[[treat_var] + ctrl_cols].to_numpy(dtype=float)
    yraw = sub[y_var].to_numpy(dtype=float)

    yd, _ni, _cv = within_demean(yraw, prov, wave)
    Xd, _,   _   = within_demean(Xraw, prov, wave)
    if not _cv:
        _w("demeaning did not converge in 50 iters")

    try:
        XX    = Xd.T @ Xd
        XXinv = np.linalg.inv(XX)
    except np.linalg.LinAlgError:
        XXinv = np.linalg.pinv(Xd.T @ Xd)

    coef  = XXinv @ (Xd.T @ yd)
    resid = yd - Xd @ coef

    cl       = pd.Categorical(sub["province"].values).codes.astype(int)
    unique_c = np.unique(cl)
    G        = len(unique_c)
    if G < MIN_CLUSTERS:
        _w(f"G={G} < MIN_CLUSTERS={MIN_CLUSTERS}")
        return None
    k = Xd.shape[1]
    n = len(yd)

    meat = np.zeros((k, k))
    for c in unique_c:
        idx = cl == c
        sc  = Xd[idx].T @ resid[idx]
        meat += np.outer(sc, sc)

    correction = (G / (G - 1)) * ((n - 1) / (n - k))
    V    = correction * (XXinv @ meat @ XXinv)
    se   = np.sqrt(np.maximum(np.diag(V), 0.0))

    tstat = np.where(se > 0, coef / se, np.nan)
    pval  = np.where(se > 0,
                     2 * stats.t.sf(np.abs(tstat), df=G - 1),
                     np.nan)

    tidx  = 0
    ci_lo = (coef[tidx] - stats.t.ppf(0.975, df=G - 1) * se[tidx]) * scale
    ci_hi = (coef[tidx] + stats.t.ppf(0.975, df=G - 1) * se[tidx]) * scale

    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((yd - yd.mean()) ** 2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "coef":       coef[tidx],
        "coef_s":     coef[tidx] * scale,
        "se":         se[tidx],
        "se_s":       se[tidx] * scale,
        "tstat":      tstat[tidx],
        "pval":       pval[tidx],
        "ci_lo":      ci_lo,
        "ci_hi":      ci_hi,
        "nobs":       n,
        "n_clusters": G,
        "r2":         r2,
    }


def run_table(data, table_name, outcomes, base_controls, checks, all_rows):
    if data is None:
        checks.append(f"\nSKIP {table_name} — no data loaded")
        return

    checks.append(f"\n{table_name}  (N_total={len(data):,})")
    print(f"\n  {table_name}  ({len(data):,} obs)")

    n_outcomes = len(outcomes)
    for i, (var, label, scale, higher_better) in enumerate(outcomes, 1):
        t0 = time.time()
        print(f"    [{i}/{n_outcomes}] {var} ...", end=" ", flush=True)

        if var not in data.columns:
            checks.append(f"  {var}: NOT IN DATA — skip")
            print("NOT IN DATA")
            continue

        n_valid = data[var].notna().sum()
        if n_valid < 100:
            checks.append(f"  {var}: only {n_valid} non-missing — skip")
            print(f"only {n_valid} non-missing — skip")
            continue

        controls = [c for c in base_controls if c in data.columns]
        need = list({"province","province_id","wave","wave_id",
                     "bartik","mining",var} | set(controls))
        need = [c for c in need if c in data.columns]
        d = data[need]

        warn_log = []
        r1 = run_one_fe(d, var, "bartik", controls, scale, warn_log)
        r2 = run_one_fe(d[d["wave"] >= 2002], var, "bartik", controls, scale, warn_log)
        r3 = run_one_fe(d[d["wave"] <= 2002], var, "bartik", controls, scale, warn_log)
        sub4 = d[d["wave"].isin([2002, 2007])].copy()
        sub4["boom_treat"] = sub4["mining"] * (sub4["wave"] == 2007).astype(float)
        r4 = run_one_fe(sub4, var, "boom_treat", controls, scale, warn_log)
        r5 = run_one_fe(d[d["province"] != "muchinga"], var, "bartik", controls, scale, warn_log)
        for w in warn_log:
            checks.append(w)

        elapsed = time.time() - t0
        if r1:
            msg = (f"β={r1['coef_s']:+.3f} se={r1['se_s']:.3f} "
                   f"p={r1['pval']:.3f}{stars(r1['pval'])}  N={r1['nobs']:,}  ({elapsed:.1f}s)")
            checks.append(f"  {var:<28} {msg}")
            print(msg)
        else:
            checks.append(f"  {var:<28} FAILED ({elapsed:.1f}s)")
            print(f"FAILED ({elapsed:.1f}s)")

        for spec_name, r in [("main", r1), ("post_priv", r2), ("pre_priv", r3),
                              ("boom_2007", r4), ("balanced", r5)]:
            if r is None:
                continue
            all_rows.append({
                "table":         table_name,
                "outcome":       var,
                "label":         label,
                "spec":          spec_name,
                "scale":         scale,
                "higher_better": higher_better,
                **r,
            })


# =============================================================================
# MAIN
# =============================================================================

def main():
    t_start = time.time()
    np.random.seed(42)
    checks   = []
    all_rows = []

    checks.append("=" * 65)
    checks.append("ZAMBIA COPPER — INDIVIDUAL-LEVEL REGRESSION DIAGNOSTICS")
    checks.append("=" * 65)
    checks.append(f"  BASE_DIR      : {BASE_DIR}")
    checks.append(f"  MINING_PROVS  : {sorted(MINING_PROVINCES)}")
    checks.append(f"  DHS_MISSING   : {sorted(DHS_MISSING_CODES)}")
    checks.append(f"  MUCHINGA      : merged into Northern (panel consistency)")
    checks.append(f"  numpy seed    : 42")
    checks.append(f"  MIN_OBS       : {MIN_OBS}  MIN_CLUSTERS: {MIN_CLUSTERS}")
    checks.append(f"  FIX 1         : ever_breastfed — m4==93 now → 1 (was 0)")
    checks.append(f"  FIX 2         : had_diarrhoea/had_cough — h11/h31 codes [1,2] → sick (was only [1])")

    datasets = {}
    for ftype in ["HR", "KR", "IR", "MR"]:
        print(f"\nLoading {ftype}...")
        checks.append(f"\n--- {ftype} ---")
        datasets[ftype] = load_stack(ftype, checks)

    hr_data = datasets["HR"]
    kr_data = datasets["KR"]
    ir_data = datasets["IR"]
    mr_data = datasets["MR"]

    for nm, d in [("HR", hr_data), ("KR", kr_data), ("IR", ir_data), ("MR", mr_data)]:
        if d is not None:
            print(f"  {nm}: {len(d):,} obs | {d['province'].nunique()} provinces")

    print("\n" + "=" * 60)
    print("RUNNING REGRESSIONS")
    print("=" * 60)

    run_table(hr_data, "Table 1: Wealth and Assets",
              OUTCOME_TABLES["Table 1: Wealth and Assets"],
              ["urban", "hh_size"], checks, all_rows)

    run_table(kr_data, "Table 2: Child Health",
              OUTCOME_TABLES["Table 2: Child Health"],
              ["child_age", "child_male", "birth_order", "urban"], checks, all_rows)

    run_table(ir_data, "Table 3: Maternal and Reproductive Health",
              OUTCOME_TABLES["Table 3: Maternal and Reproductive Health"],
              ["age_woman", "urban", "married", "edu_level"], checks, all_rows)

    run_table(ir_data, "Table 4: Women's Empowerment",
              OUTCOME_TABLES["Table 4: Women's Empowerment"],
              ["age_woman", "urban", "married"], checks, all_rows)

    run_table(ir_data, "Table 5: Domestic Violence",
              OUTCOME_TABLES["Table 5: Domestic Violence"],
              ["age_woman", "urban", "edu_secondary_p"], checks, all_rows)

    run_table(mr_data, "Table 6: Men's Outcomes",
              OUTCOME_TABLES["Table 6: Men's Outcomes"],
              ["man_age", "man_urban"], checks, all_rows)

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(OUT_DIR / "ind_results.csv", index=False)

    tlines = []
    tlines.append("=" * 95)
    tlines.append("ZAMBIA COPPER — INDIVIDUAL-LEVEL REGRESSION TABLES")
    tlines.append("Y_igt = α + β(mining_g × ln_cu_price_t) + province_FE + wave_FE + controls + ε")
    tlines.append("SE clustered at province level (G=9 after Muchinga merge).  *** p<0.01  ** p<0.05  * p<0.10")
    tlines.append("INFERENCE: With G=9 clusters (2 treated), bootstrap p-values (04_robustness.py) are primary.")
    tlines.append("Columns: (1) Full panel  (2) Post-privatisation  "
                  "(3) Pre-privatisation  (4) 2002 vs 2007 boom  (5) No Muchinga")
    tlines.append("=" * 95)

    for tbl_name in OUTCOME_TABLES:
        sub = results_df[results_df["table"] == tbl_name]
        if sub.empty:
            continue
        tlines.append(f"\n{'='*95}")
        tlines.append(tbl_name)
        tlines.append(f"{'='*95}")
        tlines.append(
            f"  {'Outcome':<38} {'(1) Main':>12} {'(2) Post':>12} "
            f"{'(3) Pre':>12} {'(4) Boom':>12} {'(5) Bal':>12}   N")
        tlines.append("  " + "─" * 95)

        for (var, label, scale, _) in OUTCOME_TABLES[tbl_name]:
            vsub = sub[sub["outcome"] == var]
            if vsub.empty:
                continue
            row_c = f"  {label:<38}"
            row_s = f"  {'':38}"
            n_obs = ""
            for sp in ["main","post_priv","pre_priv","boom_2007","balanced"]:
                r = vsub[vsub["spec"] == sp]
                if r.empty:
                    row_c += f" {'—':>12}"
                    row_s += f" {'':>12}"
                else:
                    r = r.iloc[0]
                    cs = f"{r['coef_s']:+.3f}{stars(r['pval'])}"
                    ss = f"({r['se_s']:.3f})"
                    row_c += f" {cs:>12}"
                    row_s += f" {ss:>12}"
                    if sp == "main":
                        n_obs = f"  {int(r['nobs']):,}"
            tlines.append(row_c + n_obs)
            tlines.append(row_s)

    main_r = results_df[results_df["spec"] == "main"].copy()
    tlines.append(f"\n{'='*95}")
    tlines.append("SIGNIFICANT RESULTS — MAIN SPEC (p<0.10), sorted by p-value")
    tlines.append(f"{'='*95}")
    sig = main_r[main_r["pval"] < 0.10].sort_values("pval")
    tlines.append(f"  {'Outcome':<30} {'β (scaled)':>12} {'SE':>8} {'p':>8}  {'N':>8}")
    tlines.append("  " + "─" * 65)
    for _, row in sig.iterrows():
        tlines.append(
            f"  {row['outcome']:<30} {row['coef_s']:>+12.3f} "
            f"{row['se_s']:>8.3f} {row['pval']:>8.3f}{stars(row['pval'])}  "
            f"{int(row['nobs']):>8,}")

    tlines.append(f"\n{'='*95}")
    tlines.append("NULL RESULTS — MAIN SPEC (p>=0.10)")
    tlines.append(f"{'='*95}")
    null = main_r[main_r["pval"] >= 0.10].sort_values("pval")
    for _, row in null.iterrows():
        tlines.append(
            f"  {row['outcome']:<30} {row['coef_s']:>+12.3f} "
            f"{row['se_s']:>8.3f} {row['pval']:>8.3f}  "
            f"{int(row['nobs']):>8,}")

    with open(OUT_DIR / "ind_tables.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(tlines))
    with open(OUT_DIR / "ind_checks.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(checks))

    total = time.time() - t_start
    print(f"\nDone in {total/60:.1f} min.")
    print(f"  ind_results.csv — {len(results_df)} rows")
    print(f"  ind_tables.txt  — formatted tables")
    print(f"  ind_checks.txt  — diagnostics")

    print("\n=== SIGNIFICANT RESULTS (main spec p<0.10) ===")
    if len(sig):
        print(sig[["outcome","coef_s","se_s","pval","nobs"]].to_string(index=False))
    else:
        print("  None at p<0.10")

    return results_df


if __name__ == "__main__":
    try:
        import pyreadstat, scipy
    except ImportError:
        print("pip install pandas numpy scipy pyreadstat")
        sys.exit(1)
    main()