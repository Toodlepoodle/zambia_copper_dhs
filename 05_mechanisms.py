"""
05_mechanisms.py  —  MECHANISMS, HETEROGENEITY & ADDITIONAL ROBUSTNESS
========================================================================
Builds on 03_regressions.py data loading.

Checks implemented:
  1. Urban vs Rural heterogeneity within mining provinces
  2. Pre vs Post privatisation (ZCCM privatised 2000)
  3. Mortality outcomes — fixed DHS coding
  4. Placebo tests (child sex ratio, mother's age, birth order, HH size)
  5. Triple interaction: mining × price × urban
  6. Dose-response: continuous mining employment share
  7. Composition stability test (migration check)
  8. Long-resident subsample robustness (v104 >= 5 years)

FIXES applied vs previous version:
  FIX 1  — within_demean: max_iter=3 → 50, tol=1e-10, convergence check,
            returns (arr, n_iter, converged) — matches 03/04 scripts exactly.
  FIX 2  — All within_demean calls unpacked: yd, _, cv = within_demean(...).
  FIX 3  — Convergence warning logged whenever FE does not converge.
  FIX 4  — Triple interaction: added np.linalg.pinv fallback on LinAlgError.
  FIX 5  — Condition number check added to triple interaction (was missing).
  FIX 6  — Privatisation split: 2002 removed from BOTH samples to avoid
            overlap bias. Pre = {1992, 1996}, Post = {2007, 2014, 2018, 2024}.
            2002 is kept as a separate "boundary" note only.
  FIX 7  — Zero-variance treatment guard added to run_fe.
  FIX 8  — n <= k guard added to run_fe.
  FIX 9  — G < 5 minimum cluster guard (was G < 2).
  FIX 10 — NaN/inf guard on coef and SE in run_fe.
  FIX 11 — Dose-response: clear disclaimer that MINING_DOSE shares are
            approximate LFS proxies, not verified administrative data.
  FIX 12 — COPPER_PRICES, WAVE_FILES imported directly from config_00,
            not from reg (reg does not export these).
  FIX 13 — Triple interaction manual re-estimation replaced by a shared
            run_multi_fe() helper — eliminates code duplication and
            ensures consistency with run_fe().

Outputs:
  OUTPUT/urban_rural_heterogeneity.csv
  OUTPUT/privatisation_split.csv
  OUTPUT/mortality_results.csv
  OUTPUT/placebo_results.csv
  OUTPUT/triple_interaction.csv
  OUTPUT/dose_response.csv
  OUTPUT/composition_stability.csv
  OUTPUT/longresident_robustness.csv
  OUTPUT/mechanisms_tables.txt

Usage:
  python 07_mechanisms.py
"""

import sys, warnings, time
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ── Config ────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# FIX 12: import COPPER_PRICES and WAVE_FILES directly from config_00
from config_00 import BASE_DIR, OUT_DIR, COPPER_PRICES, WAVE_FILES, MINING_PROVINCES
OUT_DIR.mkdir(exist_ok=True)

try:
    from importlib import import_module
    reg            = import_module("03_regressions")
    load_stack     = reg.load_stack
    OUTCOME_TABLES = reg.OUTCOME_TABLES
    MIN_OBS        = reg.MIN_OBS
except Exception as e:
    print(f"Could not import 03_regressions: {e}")
    sys.exit(1)

# ── Focus outcomes ─────────────────────────────────────────────────────────────
MATERNAL = {
    "delivery_facility": (["age_woman","urban","married","edu_level"], 100),
    "delivery_skilled":  (["age_woman","urban","married","edu_level"], 100),
    "anc_4plus":         (["age_woman","urban","married","edu_level"], 100),
    "pnc_mother":        (["age_woman","urban","married","edu_level"], 100),
}
CHILD = {
    "had_fever":   (["child_age","child_male","birth_order","urban"], 100),
    "stunted":     (["child_age","child_male","birth_order","urban"], 100),
    "underweight": (["child_age","child_male","birth_order","urban"], 100),
}
MEN = {
    "man_employed": (["man_age","man_urban"], 100),
}
ALL_FOCUS = {**MATERNAL, **CHILD, **MEN}

# FIX 11: these are approximate LFS proxies, NOT verified administrative data.
# Source: Zambia Labour Force Survey averages (various years).
# Must be replaced with province-specific administrative employment shares
# before submission. Copperbelt and North-Western are the only provinces
# where mining is economically meaningful.
MINING_DOSE = {
    "copperbelt":   0.35,
    "northwestern": 0.12,
    "central":      0.02,
    "eastern":      0.01,
    "luapula":      0.01,
    "lusaka":       0.02,
    "northern":     0.01,
    "southern":     0.01,
    "western":      0.01,
}
MINING_DOSE_NOTE = (
    "IMPORTANT: MINING_DOSE shares are approximate LFS proxies. "
    "Replace with verified administrative employment data before publication."
)

# FIX 6: privatisation split — 2002 excluded from BOTH samples to avoid overlap.
PRE_PRIV_WAVES  = [1992, 1996]            # strictly pre-privatisation
POST_PRIV_WAVES = [2007, 2014, 2018, 2024] # strictly post-privatisation


def stars(p):
    if pd.isna(p): return ""
    if p < 0.01:   return "***"
    if p < 0.05:   return "**"
    if p < 0.10:   return "*"
    return ""


# =============================================================================
# SHARED REGRESSION MACHINERY
# =============================================================================

def demean_by_fast(arr, group_ids):
    """Vectorised group demeaning via np.add.at. 1-D and 2-D arrays."""
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
    """
    FIX 1: Iterative two-way FE demeaning.
    max_iter=50, tol=1e-10, returns (arr, n_iter, converged).
    Matches 03_regressions.py and 04_robustness.py exactly.
    """
    out  = arr.copy().astype(float)
    prev = out.copy()
    for i in range(max_iter):
        out  = demean_by_fast(out, prov_ids)
        out  = demean_by_fast(out, wave_ids)
        diff = np.max(np.abs(out - prev))
        if diff < tol:
            return out, i + 1, True
        prev = out.copy()
    return out, max_iter, False


def _ols_clustered(yd, Xd, cl, n, k, G):
    """
    OLS on demeaned arrays + CR1S clustered SE.
    Returns (coef, se, XXinv) or raises on singular matrix.
    FIX 4: uses pinv fallback.
    """
    try:
        XXinv = np.linalg.inv(Xd.T @ Xd)
    except np.linalg.LinAlgError:
        XXinv = np.linalg.pinv(Xd.T @ Xd)

    coef  = XXinv @ (Xd.T @ yd)
    resid = yd - Xd @ coef

    unique_c = np.unique(cl)
    meat = np.zeros((k, k))
    for c in unique_c:
        idx = cl == c
        sc  = Xd[idx].T @ resid[idx]
        meat += np.outer(sc, sc)

    correction = (G / (G - 1)) * ((n - 1) / (n - k))
    V  = correction * (XXinv @ meat @ XXinv)
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    return coef, se, XXinv, resid


def _prep_and_demean(df, y_var, treat_cols, control_cols,
                     cluster_col, min_obs, checks=None):
    """
    Shared prep: dropna, re-encode FE, demean, guard checks.
    Returns (yd, Xd, cl, n, k, G, sub) or None.
    treat_cols is a list; first element is the primary treatment.
    FIX 2/3/7/8/9/10 all applied here.
    """
    # FIX: Return None if outcome column doesn't exist
    if y_var not in df.columns:
        return None
    all_treat = treat_cols if isinstance(treat_cols, list) else [treat_cols]

    cols = list(dict.fromkeys(
        [y_var] + all_treat + [cluster_col, "province_id", "wave_id"] +
        [c for c in control_cols if c in df.columns]
    ))
    cols = [c for c in cols if c in df.columns]
    sub  = df[cols].dropna().copy()

    if len(sub) < min_obs:
        return None

    sub["_prov"] = pd.Categorical(sub["province_id"].values).codes.astype(int)
    sub["_wave"] = pd.Categorical(sub["wave_id"].values).codes.astype(int)
    prov = sub["_prov"].to_numpy(dtype=int)
    wave = sub["_wave"].to_numpy(dtype=int)

    ctrl = [c for c in control_cols if c in sub.columns]
    Xraw = sub[all_treat + ctrl].to_numpy(dtype=float)
    yraw = sub[y_var].to_numpy(dtype=float)

    # FIX 2: unpack tuple; FIX 3: log convergence warning
    yd, _, cv_y = within_demean(yraw, prov, wave)
    Xd, _, cv_X = within_demean(Xraw, prov, wave)
    if not cv_y or not cv_X:
        msg = f"  WARN {y_var}: FE demeaning did not converge in 50 iters"
        if checks is not None:
            checks.append(msg)

    # FIX 5: condition number guard
    cond = np.linalg.cond(Xd.T @ Xd)
    if cond > 1e12:
        return None

    # FIX 7: zero-variance treatment guard
    if np.std(Xd[:, 0]) < 1e-12:
        return None

    cl   = pd.Categorical(sub[cluster_col].values).codes.astype(int)
    n, k = Xd.shape
    G    = len(np.unique(cl))

    # FIX 8: n <= k guard; FIX 9: G < 5 minimum
    if G < 5 or n <= k:
        return None

    return yd, Xd, cl, n, k, G, sub


def run_fe(df, y_var, treat_var, control_cols, scale=1,
           min_obs=None, cluster_col="province_id", checks=None):
    """
    Core single-treatment FE regression.
    FIX 7/8/9/10 applied via _prep_and_demean and result validation.
    """
    if min_obs is None:
        min_obs = MIN_OBS

    out = _prep_and_demean(df, y_var, [treat_var], control_cols,
                           cluster_col, min_obs, checks)
    if out is None:
        return None

    yd, Xd, cl, n, k, G, sub = out
    coef, se, XXinv, resid = _ols_clustered(yd, Xd, cl, n, k, G)

    tidx = 0
    # FIX 10: NaN/inf guard
    if not np.isfinite(coef[tidx]) or not np.isfinite(se[tidx]) or se[tidx] <= 0:
        return None

    tstat = coef[tidx] / se[tidx]
    pval  = float(2 * stats.t.sf(abs(tstat), df=G - 1))
    ci_lo = (coef[tidx] - stats.t.ppf(0.975, df=G-1) * se[tidx]) * scale
    ci_hi = (coef[tidx] + stats.t.ppf(0.975, df=G-1) * se[tidx]) * scale

    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((yd - yd.mean()) ** 2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "coef":       coef[tidx],
        "coef_s":     coef[tidx] * scale,
        "se_s":       se[tidx]   * scale,
        "tstat":      tstat,
        "pval":       pval,
        "ci_lo":      ci_lo,
        "ci_hi":      ci_hi,
        "nobs":       n,
        "n_clusters": G,
        "r2":         r2,
    }


def run_multi_fe(df, y_var, treat_cols, control_cols, scale=1,
                 min_obs=None, cluster_col="province_id", checks=None):
    """
    FIX 13: shared helper for multi-treatment regressions (e.g. triple interaction).
    Returns (coef_array, se_array, pval_array, n, G) or None.
    Eliminates the manual re-estimation block in run_triple_interaction.
    """
    if min_obs is None:
        min_obs = MIN_OBS

    out = _prep_and_demean(df, y_var, treat_cols, control_cols,
                           cluster_col, min_obs, checks)
    if out is None:
        return None

    yd, Xd, cl, n, k, G, sub = out
    coef, se, XXinv, resid = _ols_clustered(yd, Xd, cl, n, k, G)

    # FIX 10: NaN/inf guard
    if not np.all(np.isfinite(coef)) or not np.all(np.isfinite(se)):
        return None

    tstat = np.where(se > 0, coef / se, np.nan)
    pval  = np.where(se > 0, 2 * stats.t.sf(np.abs(tstat), df=G - 1), np.nan)

    return coef, se, pval, n, G


def slim(df, y_var, treat_var, controls, extra=None):
    """Keep only columns needed for one regression."""
    need = set(
        [y_var, treat_var, "province_id", "wave_id",
         "province", "wave", "mining", "urban", "post_priv", "ln_cu_price"]
        + controls + (extra or [])
    )
    return df[[c for c in need if c in df.columns]]


# =============================================================================
# 1. URBAN / RURAL HETEROGENEITY
# =============================================================================

def run_urban_rural(datasets, tlines, checks):
    print("\n" + "="*60)
    print("1. URBAN / RURAL HETEROGENEITY")
    print("="*60)

    rows = []
    ftype_map = {
        "delivery_facility": "IR", "delivery_skilled": "IR",
        "anc_4plus": "IR",         "pnc_mother": "IR",
        "had_fever": "KR",         "stunted": "KR",
        "underweight": "KR",       "man_employed": "MR",
    }

    for var, (controls, scale) in ALL_FOCUS.items():
        ftype = ftype_map.get(var)
        if not ftype:
            continue
        data = datasets.get(ftype)
        if data is None or var not in data.columns:
            continue

        urb_col = "man_urban" if ftype == "MR" else "urban"
        d = slim(data, var, "bartik", controls, extra=[urb_col])
        if urb_col not in d.columns:
            print(f"  {var}: no urban column — skip")
            continue

        # urban is constant within split — remove from controls
        ctrl_no_urb = [c for c in controls if c not in ("urban","man_urban")]

        for label, urb_val in [("urban", 1.0), ("rural", 0.0)]:
            sub = d[d[urb_col] == urb_val].copy()
            if len(sub) < MIN_OBS:
                continue
            r = run_fe(sub, var, "bartik", ctrl_no_urb, scale, checks=checks)
            if r:
                rows.append({"outcome": var, "sample": label, **r})
                print(f"  {var:<28} {label:<8} "
                      f"β={r['coef_s']:+.3f}{stars(r['pval']):<3} "
                      f"p={r['pval']:.3f}  N={r['nobs']:,}")

    df_out = pd.DataFrame(rows)

    tlines.append("\nTABLE M1: Urban vs Rural Heterogeneity")
    tlines.append("─"*70)
    tlines.append(f"  {'Outcome':<28} {'Urban β':>10} {'p':>7}   {'Rural β':>10} {'p':>7}")
    tlines.append("  " + "─"*65)
    for var in ALL_FOCUS:
        u = df_out[(df_out.outcome==var) & (df_out.sample=="urban")]
        r = df_out[(df_out.outcome==var) & (df_out.sample=="rural")]
        if u.empty and r.empty:
            continue
        us = f"{u.iloc[0].coef_s:+.3f}{stars(u.iloc[0].pval)}" if not u.empty else "—"
        up = f"{u.iloc[0].pval:.3f}" if not u.empty else "—"
        rs = f"{r.iloc[0].coef_s:+.3f}{stars(r.iloc[0].pval)}" if not r.empty else "—"
        rp = f"{r.iloc[0].pval:.3f}" if not r.empty else "—"
        tlines.append(f"  {var:<28} {us:>10} {up:>7}   {rs:>10} {rp:>7}")

    return df_out


# =============================================================================
# 2. PRE / POST PRIVATISATION SPLIT
# =============================================================================

def run_privatisation(datasets, tlines, checks):
    """
    FIX 6: 2002 excluded from BOTH samples to avoid overlap bias.
    Pre  = {1992, 1996}        — state-owned mines, social mandate
    Post = {2007, 2014, 2018, 2024} — private mines, no social mandate
    """
    print("\n" + "="*60)
    print("2. PRE / POST PRIVATISATION (ZCCM privatised 2000)")
    print(f"   Pre  = {PRE_PRIV_WAVES}  (state-owned)")
    print(f"   Post = {POST_PRIV_WAVES}  (private)")
    print("   Note: 2002 excluded from both to avoid boundary overlap.")
    print("="*60)

    rows = []
    ftype_map = {
        "delivery_facility": "IR", "delivery_skilled": "IR",
        "anc_4plus": "IR",         "pnc_mother": "IR",
        "had_fever": "KR",         "stunted": "KR",
        "man_employed": "MR",
    }

    for var, (controls, scale) in ALL_FOCUS.items():
        ftype = ftype_map.get(var)
        if not ftype:
            continue
        data = datasets.get(ftype)
        if data is None or var not in data.columns:
            continue

        d = slim(data, var, "bartik", controls)

        for label, waves in [("pre_priv", PRE_PRIV_WAVES), ("post_priv", POST_PRIV_WAVES)]:
            sub = d[d["wave"].isin(waves)].copy()
            if len(sub) < MIN_OBS:
                continue
            r = run_fe(sub, var, "bartik", controls, scale, checks=checks)
            if r:
                rows.append({"outcome": var, "sample": label, **r})
                print(f"  {var:<28} {label:<12} "
                      f"β={r['coef_s']:+.3f}{stars(r['pval']):<3} "
                      f"p={r['pval']:.3f}  N={r['nobs']:,}")

    df_out = pd.DataFrame(rows)

    tlines.append("\nTABLE M2: Pre vs Post Privatisation (ZCCM privatised 2000)")
    tlines.append(f"  Pre:  waves {PRE_PRIV_WAVES}  (state-owned mines, social mandate)")
    tlines.append(f"  Post: waves {POST_PRIV_WAVES}  (private mines)")
    tlines.append("  Note: 2002 excluded from both samples to avoid overlap bias.")
    tlines.append("─"*70)
    tlines.append(f"  {'Outcome':<28} {'Pre β':>10} {'p':>7}   {'Post β':>10} {'p':>7}")
    tlines.append("  " + "─"*65)
    for var in ALL_FOCUS:
        pre  = df_out[(df_out.outcome==var) & (df_out.sample=="pre_priv")]
        post = df_out[(df_out.outcome==var) & (df_out.sample=="post_priv")]
        if pre.empty and post.empty:
            continue
        pres = f"{pre.iloc[0].coef_s:+.3f}{stars(pre.iloc[0].pval)}" if not pre.empty else "—"
        prep = f"{pre.iloc[0].pval:.3f}" if not pre.empty else "—"
        pos  = f"{post.iloc[0].coef_s:+.3f}{stars(post.iloc[0].pval)}" if not post.empty else "—"
        pop  = f"{post.iloc[0].pval:.3f}" if not post.empty else "—"
        tlines.append(f"  {var:<28} {pres:>10} {prep:>7}   {pos:>10} {pop:>7}")

    return df_out


# =============================================================================
# 3. MORTALITY — FIXED CODING
# =============================================================================

def build_mortality_data():
    """
    Directly loads KR files with correct DHS mortality coding.
    b5: 1=alive, 0=dead. b7: age at death in months (dead only).
    """
    import pyreadstat
    frames = []

    for wave, files in WAVE_FILES.items():
        rel = files.get("KR")
        if not rel:
            continue
        path = BASE_DIR / rel
        if not path.exists():
            continue

        try:
            df, meta = pyreadstat.read_dta(str(path))
            df.columns = df.columns.str.lower()

            prov_col = "v024"
            if prov_col not in df.columns:
                continue

            df["province"]    = df[prov_col].apply(
                lambda c: reg.harmonise_prov(reg.get_label(meta, prov_col, c)))
            df["wave"]        = wave
            df["ln_cu_price"] = np.log(COPPER_PRICES[wave])
            df["mining"]      = df["province"].apply(
                lambda p: 1.0 if p in MINING_PROVINCES else 0.0)
            df["bartik"]      = df["mining"] * df["ln_cu_price"]

            v025 = pd.to_numeric(
                df.get("v025", pd.Series(np.nan, index=df.index)), errors="coerce")
            df["urban"] = v025.map({1: 1.0, 2: 0.0})

            b5   = pd.to_numeric(
                df.get("b5", pd.Series(np.nan, index=df.index)), errors="coerce")
            dead = (b5 == 0)
            alive_or_dead = b5.notna()

            if "b7" in df.columns:
                b7          = pd.to_numeric(df["b7"], errors="coerce")
                b7_for_dead = b7.where(dead, np.nan)
                df["u5_dead"]       = dead.astype(float).where(alive_or_dead, np.nan)
                df["infant_dead"]   = (dead & (b7_for_dead < 12)).astype(float).where(alive_or_dead, np.nan)
                df["neonatal_dead"] = (dead & (b7_for_dead <  1)).astype(float).where(alive_or_dead, np.nan)
            else:
                df["u5_dead"]     = dead.astype(float).where(alive_or_dead, np.nan)
                df["infant_dead"] = df["neonatal_dead"] = np.nan

            b4 = pd.to_numeric(
                df.get("b4", pd.Series(np.nan, index=df.index)), errors="coerce")
            df["child_male_ratio"] = (b4 == 1).astype(float).where(b4.notna(), np.nan)
            df["child_age"]        = pd.to_numeric(df.get("b8",  pd.Series(np.nan, index=df.index)), errors="coerce")
            df["child_male"]       = (b4 == 1).astype(float).where(b4.notna(), np.nan)
            df["birth_order"]      = pd.to_numeric(df.get("bord",pd.Series(np.nan, index=df.index)), errors="coerce")

            keep = ["province","wave","ln_cu_price","mining","bartik","urban",
                    "u5_dead","infant_dead","neonatal_dead","child_male_ratio",
                    "child_age","child_male","birth_order"]
            df = df[[c for c in keep if c in df.columns]]
            frames.append(df)
            print(f"    mortality KR {wave}: {len(df):,} rows  "
                  f"u5_dead mean={df['u5_dead'].mean()*1000:.1f}/1000")

        except Exception as e:
            print(f"    mortality KR {wave}: ERROR — {e}")
            import traceback; traceback.print_exc()

    if not frames:
        return None

    out = pd.concat(frames, ignore_index=True)
    out["province_id"] = pd.Categorical(out["province"]).codes.astype(int)
    out["wave_id"]     = pd.Categorical(out["wave"]).codes.astype(int)
    return out


def run_mortality(mort_data, tlines, checks):
    print("\n" + "="*60)
    print("3. MORTALITY OUTCOMES (fixed coding)")
    print("="*60)

    rows    = []
    controls = ["child_age","child_male","birth_order","urban"]

    for var, scale, label in [
        ("u5_dead",       1000, "U5 mortality (per 1000)"),
        ("infant_dead",   1000, "Infant mortality (per 1000)"),
        ("neonatal_dead", 1000, "Neonatal mortality (per 1000)"),
    ]:
        if var not in mort_data.columns:
            continue
        n_valid   = mort_data[var].notna().sum()
        dead_mean = mort_data[var].mean() * 1000
        print(f"  {var}: n_valid={n_valid:,}  mean={dead_mean:.1f}/1000")

        r = run_fe(mort_data, var, "bartik", controls, scale, checks=checks)
        if r:
            rows.append({"outcome": var, "label": label, **r})
            print(f"    β={r['coef_s']:+.3f}{stars(r['pval']):<3} "
                  f"p={r['pval']:.3f}  N={r['nobs']:,}")
        else:
            print(f"    FAILED (too few obs, singular, or G<5)")

    df_out = pd.DataFrame(rows)

    tlines.append("\nTABLE M3: Child Mortality Outcomes (Fixed DHS Coding)")
    tlines.append("  b5=0 → dead. b7 = age at death (months). Scale: per 1,000 births.")
    tlines.append("─"*70)
    tlines.append(f"  {'Outcome':<32} {'β':>8} {'SE':>7} {'p':>8}  N")
    tlines.append("  " + "─"*65)
    for _, r in df_out.iterrows():
        tlines.append(
            f"  {r['label']:<32} {r['coef_s']:>+8.3f} {r['se_s']:>7.3f} "
            f"{r['pval']:>8.3f}{stars(r['pval'])}  {int(r['nobs']):,}")

    return df_out


# =============================================================================
# 4. PLACEBO TESTS
# =============================================================================

def run_placebo(datasets, mort_data, tlines, checks):
    print("\n" + "="*60)
    print("4. PLACEBO TESTS")
    print("="*60)
    print("  β ≈ 0 on all → research design is credible.")

    rows = []

    if mort_data is not None and "child_male_ratio" in mort_data.columns:
        r = run_fe(mort_data, "child_male_ratio", "bartik",
                   ["child_age","birth_order","urban"], scale=100, checks=checks)
        if r:
            rows.append({"outcome":"child_sex_ratio",
                         "label":"Child sex ratio (% male)", **r})
            print(f"  child_sex_ratio  β={r['coef_s']:+.3f}{stars(r['pval']):<3} "
                  f"p={r['pval']:.3f}  N={r['nobs']:,}")

    ir = datasets.get("IR")
    if ir is not None and "age_woman" in ir.columns:
        r = run_fe(ir, "age_woman", "bartik", ["urban","married"],
                   scale=1, checks=checks)
        if r:
            rows.append({"outcome":"mother_age",
                         "label":"Mother's age at survey (years)", **r})
            print(f"  mother_age       β={r['coef_s']:+.3f}{stars(r['pval']):<3} "
                  f"p={r['pval']:.3f}  N={r['nobs']:,}")

    if mort_data is not None and "birth_order" in mort_data.columns:
        r = run_fe(mort_data, "birth_order", "bartik",
                   ["child_age","urban"], scale=1, checks=checks)
        if r:
            rows.append({"outcome":"birth_order",
                         "label":"Birth order", **r})
            print(f"  birth_order      β={r['coef_s']:+.3f}{stars(r['pval']):<3} "
                  f"p={r['pval']:.3f}  N={r['nobs']:,}")

    hr = datasets.get("HR")
    if hr is not None and "hh_size" in hr.columns:
        r = run_fe(hr, "hh_size", "bartik", ["urban"], scale=1, checks=checks)
        if r:
            rows.append({"outcome":"hh_size",
                         "label":"Household size (persons)", **r})
            print(f"  hh_size          β={r['coef_s']:+.3f}{stars(r['pval']):<3} "
                  f"p={r['pval']:.3f}  N={r['nobs']:,}")

    df_out = pd.DataFrame(rows)

    tlines.append("\nTABLE M4: Placebo Tests")
    tlines.append("  Outcomes copper prices cannot causally affect.")
    tlines.append("  β ≈ 0 → identification strategy is credible.")
    tlines.append("─"*70)
    tlines.append(f"  {'Outcome':<35} {'β':>8} {'SE':>7} {'p':>8}  N")
    tlines.append("  " + "─"*65)
    for _, r in df_out.iterrows():
        tlines.append(
            f"  {r['label']:<35} {r['coef_s']:>+8.3f} {r['se_s']:>7.3f} "
            f"{r['pval']:>8.3f}{stars(r['pval'])}  {int(r['nobs']):,}")
    tlines.append("  Significant placebo results would undermine identification.")

    return df_out


# =============================================================================
# 5. TRIPLE INTERACTION: mining × price × urban
# =============================================================================

def run_triple_interaction(datasets, tlines, checks):
    """
    Model: Y = β1(bartik) + β2(bartik×urban) + β3(urban) + FE + X + ε
    β1         = rural effect of mining boom
    β1 + β2    = urban effect of mining boom
    β2         = urban-rural differential (test of mechanism)

    FIX 13: uses run_multi_fe() — no manual re-estimation.
    FIX 4/5: pinv fallback and condition number check inside run_multi_fe.
    """
    print("\n" + "="*60)
    print("5. TRIPLE INTERACTION: mining × price × urban")
    print("="*60)

    rows = []
    ftype_map = {
        "delivery_facility": "IR",
        "delivery_skilled":  "IR",
        "anc_4plus":         "IR",
        "had_fever":         "KR",
        "man_employed":      "MR",
    }

    for var, ftype in ftype_map.items():
        controls_base, scale = ALL_FOCUS.get(var, ([], 100))
        data = datasets.get(ftype)
        if data is None or var not in data.columns:
            continue

        d = slim(data, var, "bartik", controls_base,
                 extra=["ln_cu_price"]).copy()
        if "urban" not in d.columns:
            continue

        # Build interaction term
        d["bartik_urban"] = d["mining"] * d["ln_cu_price"] * d["urban"]

        # Controls: everything except urban (enters separately)
        ctrl = [c for c in controls_base if c not in ("urban",)]

        # FIX 13: run_multi_fe returns all coefficients
        result = run_multi_fe(
            d, var,
            treat_cols=["bartik", "bartik_urban"],
            control_cols=ctrl + ["urban"],
            scale=scale,
            checks=checks,
        )
        if result is None:
            print(f"  {var}: FAILED")
            continue

        coef, se, pval, n, G = result
        # coef[0]=bartik (rural), coef[1]=bartik_urban (differential)
        b_rural = float(coef[0]) * scale
        b_diff  = float(coef[1]) * scale
        b_urban = b_rural + b_diff
        p_rural = float(pval[0])
        p_diff  = float(pval[1])

        rows.append({
            "outcome":    var,
            "beta_rural": b_rural, "pval_rural": p_rural,
            "beta_urban": b_urban,
            "beta_diff":  b_diff,  "pval_diff":  p_diff,
            "nobs": n, "n_clusters": G,
        })
        print(f"  {var:<28} rural={b_rural:+.3f}{stars(p_rural):<3}  "
              f"urban={b_urban:+.3f}  diff={b_diff:+.3f}{stars(p_diff):<3}  "
              f"p_diff={p_diff:.3f}")

    df_out = pd.DataFrame(rows)

    tlines.append("\nTABLE M5: Triple Interaction (mining × price × urban)")
    tlines.append("  β_rural = effect in rural areas of mining provinces")
    tlines.append("  β_urban = β_rural + β_diff = effect in urban areas")
    tlines.append("  β_diff  = urban-rural differential (mechanism test)")
    tlines.append("─"*75)
    tlines.append(f"  {'Outcome':<28} {'β_rural':>10} {'β_urban':>10} "
                  f"{'β_diff':>10} {'p_diff':>8}")
    tlines.append("  " + "─"*70)
    for _, r in df_out.iterrows():
        tlines.append(
            f"  {r['outcome']:<28} {r['beta_rural']:>+10.3f} "
            f"{r['beta_urban']:>+10.3f} {r['beta_diff']:>+10.3f} "
            f"{r['pval_diff']:>8.3f}{stars(r['pval_diff'])}")

    return df_out


# =============================================================================
# 6. DOSE-RESPONSE: continuous mining exposure
# =============================================================================

def run_dose_response(datasets, tlines, checks):
    """
    Replace binary mining (0/1) with continuous mining employment share.
    Treatment: mine_share_g × ln_price_t.

    FIX 11: MINING_DOSE shares are approximate LFS proxies.
    Must be replaced with verified administrative data before publication.
    """
    print("\n" + "="*60)
    print("6. DOSE-RESPONSE (continuous mining employment share)")
    print(f"   WARNING: {MINING_DOSE_NOTE}")
    print("="*60)

    rows = []
    ftype_map = {
        "delivery_facility": "IR",
        "delivery_skilled":  "IR",
        "anc_4plus":         "IR",
        "had_fever":         "KR",
        "man_employed":      "MR",
    }

    for var, ftype in ftype_map.items():
        controls_base, scale = ALL_FOCUS.get(var, ([], 100))
        data = datasets.get(ftype)
        if data is None or var not in data.columns:
            continue

        d = slim(data, var, "bartik", controls_base,
                 extra=["ln_cu_price"]).copy()

        d["mine_share"]  = d["province"].map(MINING_DOSE).fillna(0.01)
        d["bartik_cont"] = d["mine_share"] * d["ln_cu_price"]

        r = run_fe(d, var, "bartik_cont", controls_base, scale, checks=checks)
        if r:
            rows.append({"outcome": var, **r})
            print(f"  {var:<28} β={r['coef_s']:+.3f}{stars(r['pval']):<3} "
                  f"p={r['pval']:.3f}  N={r['nobs']:,}")
        else:
            print(f"  {var:<28} FAILED")

    df_out = pd.DataFrame(rows)

    tlines.append("\nTABLE M6: Dose-Response (Continuous Mining Employment Share)")
    tlines.append("  Treatment: mine_employment_share_g × ln_copper_price_t")
    tlines.append(f"  {MINING_DOSE_NOTE}")
    tlines.append("  Shares: Copperbelt=0.35, Northwestern=0.12, others≈0.01")
    tlines.append("─"*70)
    tlines.append(f"  {'Outcome':<28} {'β':>8} {'SE':>7} {'p':>8}  N")
    tlines.append("  " + "─"*65)
    for _, r in df_out.iterrows():
        tlines.append(
            f"  {r['outcome']:<28} {r['coef_s']:>+8.3f} {r['se_s']:>7.3f} "
            f"{r['pval']:>8.3f}{stars(r['pval'])}  {int(r['nobs']):,}")

    return df_out


# =============================================================================
# 7. COMPOSITION STABILITY TEST
# =============================================================================

def run_composition_stability(datasets, tlines, checks):
    print("\n" + "="*60)
    print("7. COMPOSITION STABILITY TEST (migration check)")
    print("   β ≈ 0 on all → composition stable → good.")
    print("="*60)

    rows = []
    ir = datasets.get("IR")
    hr = datasets.get("HR")

    specs = [
        (ir, "age_woman",       ["married"],             1,   "Mean age of women (years)"),
        (ir, "edu_secondary_p", ["age_woman","married"], 100, "Secondary+ education (pp)"),
        (ir, "married",         ["age_woman"],           100, "Currently married (pp)"),
        (hr, "urban",           ["hh_size"],             100, "Urban share (pp)"),
        (hr, "hh_size",         ["urban"],               1,   "Mean household size (persons)"),
    ]

    for df, outcome, controls, scale, label in specs:
        if df is None or outcome not in df.columns:
            print(f"  {outcome:<22} SKIP")
            continue
        r = run_fe(df, outcome, "bartik", controls, scale, checks=checks)
        if r is None:
            print(f"  {outcome:<22} SKIP (insufficient obs or G<5)")
            continue
        rows.append({"outcome": outcome, "label": label, **r})
        flag = " *** CONCERN" if r["pval"] < 0.05 else (" * marginal" if r["pval"] < 0.10 else "")
        print(f"  {outcome:<22} β={r['coef_s']:+7.3f}  SE={r['se_s']:.3f}  "
              f"p={r['pval']:.3f}{stars(r['pval'])}  N={r['nobs']:,}{flag}")

    df_out = pd.DataFrame(rows)

    tlines.append("\nTABLE M7: Composition Stability Test")
    tlines.append("  Tests whether copper booms shift provincial demographics.")
    tlines.append("  β ≈ 0 on all → population composition stable → good.")
    tlines.append("─"*72)
    tlines.append(f"  {'Outcome':<42} {'β':>8} {'SE':>7} {'p':>8}  N")
    tlines.append("  " + "─"*68)
    for _, row in df_out.iterrows():
        flag = "  ← CONCERN" if row["pval"] < 0.05 else ""
        tlines.append(
            f"  {row['label']:<42} {row['coef_s']:>+8.3f} {row['se_s']:>7.3f} "
            f"{row['pval']:>8.3f}{stars(row['pval'])}  {int(row['nobs']):,}{flag}")

    n_sig = (df_out["pval"] < 0.10).sum() if len(df_out) > 0 else 0
    if n_sig == 0:
        verdict = "PASS — no demographic characteristic responds to Bartik treatment."
    elif n_sig == 1:
        verdict = f"MARGINAL — {n_sig} of {len(df_out)} characteristics significant at p<0.10."
    else:
        verdict = f"CONCERN — {n_sig} of {len(df_out)} characteristics shift significantly."
    tlines.append(f"  Verdict: {verdict}")
    print(f"  Verdict: {verdict}")

    return df_out


# =============================================================================
# 8. LONG-RESIDENT SUBSAMPLE
# =============================================================================

def run_longresident_robustness(datasets, tlines, checks):
    """
    Restricts to women resident in province 5+ years (v104 >= 5).
    If results hold → in-migrants not driving main estimates.
    """
    print("\n" + "="*60)
    print("8. LONG-RESIDENT SUBSAMPLE (v104 >= 5 years)")
    print("="*60)

    ir = datasets.get("IR")
    if ir is None:
        print("  SKIP — IR data not loaded")
        return pd.DataFrame()

    if "v104" not in ir.columns:
        print("  SKIP — v104 not in stacked IR data")
        print("  Add v104 to keep-columns in 03_regressions.py load_stack().")
        tlines.append("\nTABLE M8: Long-Resident Subsample — SKIPPED (v104 not available)")
        return pd.DataFrame()

    v104 = pd.to_numeric(ir["v104"], errors="coerce")
    v104 = v104.where(v104 <= 95, np.nan)   # 96=visitor, 97=NS → NaN
    ir   = ir.copy()
    ir["longres"] = ((v104 >= 5) | (v104 == 95)).astype(float)
    ir.loc[v104.isna(), "longres"] = np.nan

    n_total   = ir["longres"].notna().sum()
    n_longres = (ir["longres"] == 1).sum()
    pct       = 100 * n_longres / n_total if n_total > 0 else 0
    print(f"  v104 available: {n_total:,}  long-residents: {n_longres:,} ({pct:.1f}%)")

    if n_longres < 500:
        print("  SKIP — too few long-resident observations")
        return pd.DataFrame()

    ir_lr = ir[ir["longres"] == 1].copy()

    outcomes = [
        ("delivery_facility", ["age_woman","urban","married","edu_level"], 100, "Facility delivery"),
        ("delivery_skilled",  ["age_woman","urban","married","edu_level"], 100, "Skilled birth attend."),
        ("anc_4plus",         ["age_woman","urban","married","edu_level"], 100, "ANC 4+ visits"),
        ("pnc_mother",        ["age_woman","urban","married","edu_level"], 100, "Postnatal care"),
    ]

    rows = []
    for var, controls, scale, label in outcomes:
        if var not in ir.columns:
            continue
        r_full = run_fe(ir,    var, "bartik", controls, scale, checks=checks)
        r_lr   = run_fe(ir_lr, var, "bartik", controls, scale, checks=checks)
        if r_full is None:
            continue
        rows.append({
            "outcome":   var, "label": label,
            "beta_full": r_full["coef_s"], "se_full": r_full["se_s"],
            "p_full":    r_full["pval"],   "n_full":  r_full["nobs"],
            "beta_lr":   r_lr["coef_s"]  if r_lr else np.nan,
            "se_lr":     r_lr["se_s"]    if r_lr else np.nan,
            "p_lr":      r_lr["pval"]    if r_lr else np.nan,
            "n_lr":      r_lr["nobs"]    if r_lr else np.nan,
        })
        b_lr = f"{r_lr['coef_s']:+.2f}{stars(r_lr['pval'])}" if r_lr else "N/A"
        print(f"  {label:<25} full={r_full['coef_s']:+.2f}{stars(r_full['pval']):<3}  "
              f"long-res={b_lr}")

    df_out = pd.DataFrame(rows)

    tlines.append("\nTABLE M8: Long-Resident Subsample (v104 >= 5 years)")
    tlines.append("  Removes recent in-migrants. If results hold → composition bias bounded.")
    tlines.append("─"*78)
    tlines.append(f"  {'Outcome':<25} {'Full β':>8} {'p':>6}  {'N (full)':>9}  "
                  f"{'LR β':>8} {'p':>6}  {'N (LR)':>8}")
    tlines.append("  " + "─"*74)
    for _, row in df_out.iterrows():
        lr_b = f"{row['beta_lr']:>+8.2f}" if pd.notna(row["beta_lr"]) else "     N/A"
        lr_p = f"{row['p_lr']:>6.3f}{stars(row['p_lr'])}" if pd.notna(row["p_lr"]) else "    —"
        lr_n = f"{int(row['n_lr']):>8,}"                  if pd.notna(row["n_lr"]) else "       —"
        tlines.append(
            f"  {row['label']:<25} {row['beta_full']:>+8.2f} {row['p_full']:>6.3f}"
            f"{stars(row['p_full']):<3}  {int(row['n_full']):>9,}  "
            f"{lr_b} {lr_p}  {lr_n}")

    return df_out


# =============================================================================
# MAIN
# =============================================================================

def main():
    t_start = time.time()
    tlines  = []
    checks  = []
    tlines.append("=" * 70)
    tlines.append("ZAMBIA COPPER — MECHANISMS AND ADDITIONAL ROBUSTNESS")
    tlines.append("=" * 70)
    tlines.append(f"  G = 9 provinces (Muchinga merged into Northern)")
    tlines.append(f"  FE: max_iter=50, tol=1e-10, min_clusters=5")
    tlines.append(f"  Privatisation split: pre={PRE_PRIV_WAVES}, post={POST_PRIV_WAVES}")
    tlines.append(f"  {MINING_DOSE_NOTE}")

    print("Loading main datasets...")
    datasets = {}
    for ftype in ["HR","KR","IR","MR"]:
        print(f"  {ftype}...", end=" ", flush=True)
        datasets[ftype] = load_stack(ftype, [])
        n = len(datasets[ftype]) if datasets[ftype] is not None else 0
        print(f"{n:,} obs")

    print("\nBuilding mortality data (fixed coding)...")
    mort_data = build_mortality_data()

    ur_df      = run_urban_rural(datasets, tlines, checks)
    priv_df    = run_privatisation(datasets, tlines, checks)
    mort_df    = run_mortality(mort_data, tlines, checks) if mort_data is not None else pd.DataFrame()
    plac_df    = run_placebo(datasets, mort_data, tlines, checks)
    triple_df  = run_triple_interaction(datasets, tlines, checks)
    dose_df    = run_dose_response(datasets, tlines, checks)
    comp_df    = run_composition_stability(datasets, tlines, checks)
    longres_df = run_longresident_robustness(datasets, tlines, checks)

    ur_df.to_csv(     OUT_DIR / "urban_rural_heterogeneity.csv",  index=False)
    priv_df.to_csv(   OUT_DIR / "privatisation_split.csv",        index=False)
    mort_df.to_csv(   OUT_DIR / "mortality_results.csv",          index=False)
    plac_df.to_csv(   OUT_DIR / "placebo_results.csv",            index=False)
    triple_df.to_csv( OUT_DIR / "triple_interaction.csv",         index=False)
    dose_df.to_csv(   OUT_DIR / "dose_response.csv",              index=False)
    comp_df.to_csv(   OUT_DIR / "composition_stability.csv",      index=False)
    longres_df.to_csv(OUT_DIR / "longresident_robustness.csv",    index=False)

    with open(OUT_DIR / "mechanisms_tables.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(tlines))
    with open(OUT_DIR / "mechanisms_checks.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(checks))

    total = time.time() - t_start
    print(f"\nDone in {total/60:.1f} min.")
    for fn in ["urban_rural_heterogeneity.csv","privatisation_split.csv",
               "mortality_results.csv","placebo_results.csv",
               "triple_interaction.csv","dose_response.csv",
               "composition_stability.csv","longresident_robustness.csv",
               "mechanisms_tables.txt","mechanisms_checks.txt"]:
        print(f"  {fn}")


if __name__ == "__main__":
    try:
        import pyreadstat, scipy
    except ImportError:
        print("pip install pandas numpy scipy pyreadstat")
        sys.exit(1)
    main()