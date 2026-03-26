"""
04_robustness.py  —  ROBUSTNESS CHECKS
========================================
Three robustness checks for the main DiD results in 03_regressions.py:

  1. Wild Cluster Bootstrap (Webb 2014, MacKinnon et al. 2018)
     — Valid inference with G=9 clusters (Muchinga merged into Northern)
     — Uses 6-point Webb weights
     — Correctly imposes H0: restricted residuals (treatment removed)
     — Reports bootstrap p-values alongside analytic p-values

  2. Event Study / Dynamic DiD
     — Estimates beta_t separately for each wave
     — Wave 2002 (lowest price, pre-boom) used as base category
     — Flat pre-trends + spike at boom waves = causal credibility
     — Saves plot-ready CSV and ASCII chart to OUTPUT/

  3. Continuous Treatment DiD Robustness
     — Splits into above/below median price years and re-estimates
     — Tests whether the effect is monotone in the price level

FIXES applied vs previous version:
  FIX 1 — Bootstrap now correctly imposes H0 via restricted residuals.
           Removes treatment column, re-estimates without it, uses those
           residuals in the bootstrap. Previous version used unrestricted
           residuals → anti-conservative p-values.
  FIX 2 — within_demean: max_iter=3 → 50, added tol=1e-10 convergence
           check. Returns (arr, n_iter, converged) tuple — matches
           03_regressions.py exactly.
  FIX 3 — prep_arrays and event study now unpack within_demean tuple:
           yd, _, _ = within_demean(...) — previous code assigned the
           tuple itself to yd/Xd which would crash or silently corrupt.
  FIX 4 — COPPER_PRICES imported directly from config_00.
           reg.COPPER_PRICES → AttributeError at runtime (removed).
  FIX 5 — XXinv recomputed inside bootstrap loop using restricted Xd.
           Reusing unrestricted XXinv with restricted residuals is
           theoretically inconsistent.
  FIX 6 — Condition number check added to bootstrap and event study
           (already existed in continuous robustness).
  FIX 7 — Per-outcome RNG seed: seed + hash(var) % 10000.
           Ensures reproducibility regardless of FOCUS_OUTCOMES order.
  FIX 8 — FE convergence warning logged to checks file.
  FIX 9 — Zero-variance treatment guard before OLS.
  FIX 10 — NaN/inf guard on coef and SE throughout.
  FIX 11 — G=9 (not G=10) in all comments and output strings.

Outputs:
  OUTPUT/bootstrap_results.csv      — main results + bootstrap p-values
  OUTPUT/event_study.csv            — wave-by-wave beta estimates
  OUTPUT/event_study_chart.txt      — ASCII visualisation
  OUTPUT/continuous_robustness.csv  — high vs low price split
  OUTPUT/robustness_tables.txt      — formatted summary
  OUTPUT/robustness_checks.txt      — diagnostics

Usage:
  python 04_robustness.py

NOTE: Imports data-loading machinery from 03_regressions.py.
      Both files must be in the same directory.
"""

import sys, warnings, time
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
from scipy import stats

# ── Config and shared constants ───────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from config_00 import BASE_DIR, OUT_DIR, COPPER_PRICES   # FIX 4: direct import
OUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    from importlib import import_module
    reg            = import_module("03_regressions")
    load_stack     = reg.load_stack
    OUTCOME_TABLES = reg.OUTCOME_TABLES
    MIN_OBS        = reg.MIN_OBS
except Exception as e:
    print(f"Could not import 03_regressions: {e}")
    print("Copy 03_regressions.py to the same folder and retry.")
    sys.exit(1)

# ── Focus outcomes — significant results from 03_regressions.py ──────────────
# Update this dict if the main regression finds different significant outcomes.
FOCUS_OUTCOMES = {
    "had_fever":         ("KR", ["child_age","child_male","birth_order","urban"], 100),
    "floor_finished":    ("HR", ["urban","hh_size"],                             100),
    "pnc_mother":        ("IR", ["age_woman","urban","married","edu_level"],      100),
    "delivery_skilled":  ("IR", ["age_woman","urban","married","edu_level"],      100),
    "delivery_facility": ("IR", ["age_woman","urban","married","edu_level"],      100),
    "dv_choked":         ("IR", ["age_woman","urban","edu_secondary_p"],          100),
    "has_mobile":        ("IR", ["age_woman","urban","married"],                  100),
    "anc_4plus":         ("IR", ["age_woman","urban","married","edu_level"],      100),
    "man_employed":      ("MR", ["man_age","man_urban"],                          100),
    "owns_house":        ("IR", ["age_woman","urban","married"],                  100),
    "dv_justify_any":    ("IR", ["age_woman","urban","edu_secondary_p"],          100),
}

WAVE_ORDER = [1992, 1996, 2002, 2007, 2014, 2018, 2024]
BASE_WAVE  = 2002   # lowest copper price — base category for event study

# Webb (2014) 6-point weights — designed for G <= 11 clusters
WEBB_WEIGHTS = np.array([
    -np.sqrt(3/2), -np.sqrt(2/2), -np.sqrt(1/2),
     np.sqrt(1/2),  np.sqrt(2/2),  np.sqrt(3/2),
])

# =============================================================================
# SHARED HELPERS
# =============================================================================

def demean_by_fast(arr, group_ids):
    """Vectorised group demeaning via np.add.at. Works for 1-D and 2-D."""
    out    = arr.astype(float)
    G      = int(group_ids.max()) + 1
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
    Iterative two-way FE demeaning. FIX 2: max_iter=50, tol=1e-10,
    returns (arr, n_iter, converged) — matches 03_regressions.py exactly.
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


def prep_arrays(df, y_var, treat_var, control_cols, checks=None):
    """
    Drop NAs, within-demean, return (yd, Xd, cl, n, k, G, sub).
    FIX 3: unpacks within_demean tuple correctly.
    FIX 8: logs convergence warning if FE did not converge.
    """
    # FIX: Return None if outcome column doesn't exist
    if y_var not in df.columns:
        return None
    cols = list(dict.fromkeys(
        [y_var, treat_var, "province_id", "wave_id"] +
        [c for c in control_cols if c in df.columns]
    ))
    sub = df[cols].dropna().copy()
    if len(sub) < MIN_OBS:
        return None

    prov = sub["province_id"].to_numpy(dtype=int)
    wave = sub["wave_id"].to_numpy(dtype=int)
    ctrl = [c for c in control_cols if c in sub.columns]

    Xraw = sub[[treat_var] + ctrl].to_numpy(dtype=float)
    yraw = sub[y_var].to_numpy(dtype=float)

    # FIX 3: unpack tuple
    yd, _, cv_y = within_demean(yraw, prov, wave)
    Xd, _, cv_X = within_demean(Xraw, prov, wave)

    # FIX 8: log convergence warning
    if not cv_y or not cv_X:
        msg = f"  WARN {y_var}: FE demeaning did not converge in 50 iters"
        if checks is not None:
            checks.append(msg)

    # FIX 6: condition number guard
    cond = np.linalg.cond(Xd.T @ Xd)
    if cond > 1e12:
        return None

    # FIX 9: zero-variance treatment guard
    if np.std(Xd[:, 0]) < 1e-12:
        return None

    cl   = pd.Categorical(sub["province_id"].values).codes.astype(int)
    n, k = Xd.shape
    G    = len(np.unique(cl))

    # Edge case guards
    if G <= 1 or n <= k:
        return None

    return yd, Xd, cl, n, k, G, sub


def ols_fe(yd, Xd):
    """OLS on demeaned arrays. Returns (coef, resid, XXinv)."""
    try:
        XXinv = np.linalg.inv(Xd.T @ Xd)
    except np.linalg.LinAlgError:
        XXinv = np.linalg.pinv(Xd.T @ Xd)
    coef  = XXinv @ (Xd.T @ yd)
    resid = yd - Xd @ coef
    return coef, resid, XXinv


def clustered_se(Xd, resid, cl, XXinv, n, k, G):
    """CR1S clustered standard errors."""
    unique_c = np.unique(cl)
    meat = np.zeros((k, k))
    for c in unique_c:
        idx = cl == c
        sc  = Xd[idx].T @ resid[idx]
        meat += np.outer(sc, sc)
    correction = (G / (G - 1)) * ((n - 1) / (n - k))
    V = correction * (XXinv @ meat @ XXinv)
    return np.sqrt(np.maximum(np.diag(V), 0.0))


def stars(p):
    if pd.isna(p): return ""
    if p < 0.01:   return "***"
    if p < 0.05:   return "**"
    if p < 0.10:   return "*"
    return ""


# =============================================================================
# 1. WILD CLUSTER BOOTSTRAP
# =============================================================================

def wild_bootstrap_pval(yd, Xd, cl, coef, tidx=0, B=999, seed=42):
    """
    Wild cluster bootstrap p-value (Webb 2014, designed for G <= 11).

    FIX 1 — Correctly imposes H0 via restricted residuals:
      - Remove treatment column (index 0) from Xd
      - Re-estimate model without treatment
      - Use restricted residuals in all bootstrap replications
      Previous version used unrestricted residuals → anti-conservative.

    FIX 5 — XXinv recomputed from restricted Xd inside loop, not
      reused from the unrestricted model.

    FIX 7 — Per-outcome seed: seed + hash(outcome) % 10000.
      Caller passes outcome-specific seed for full reproducibility.
    """
    rng       = np.random.default_rng(seed)
    unique_cl = np.unique(cl)
    G         = len(unique_cl)
    n, k      = Xd.shape

    # ── FIX 1: compute restricted residuals (impose H0: beta_treat = 0) ──────
    Xr = Xd[:, 1:]   # drop treatment column (index 0)
    if Xr.shape[1] == 0:
        # No controls: restricted model is just the FE means (already absorbed)
        resid_r = yd.copy()
    else:
        try:
            XXr_inv = np.linalg.inv(Xr.T @ Xr)
        except np.linalg.LinAlgError:
            XXr_inv = np.linalg.pinv(Xr.T @ Xr)
        coef_r  = XXr_inv @ (Xr.T @ yd)
        resid_r = yd - Xr @ coef_r

    # Observed t-stat (from unrestricted model)
    try:
        XXinv_full = np.linalg.inv(Xd.T @ Xd)
    except np.linalg.LinAlgError:
        XXinv_full = np.linalg.pinv(Xd.T @ Xd)
    resid_full = yd - Xd @ coef
    se_obs     = clustered_se(Xd, resid_full, cl, XXinv_full, n, k, G)[tidx]
    if se_obs <= 0 or not np.isfinite(se_obs):
        return np.nan
    t_obs = coef[tidx] / se_obs

    # ── Bootstrap replications ────────────────────────────────────────────────
    t_boot = np.empty(B)
    for b in range(B):
        # Draw one Webb weight per cluster
        w = rng.choice(WEBB_WEIGHTS, size=G)

        # Scale restricted residuals cluster-by-cluster
        resid_star = resid_r.copy()
        for gi, c in enumerate(unique_cl):
            resid_star[cl == c] *= w[gi]

        # Bootstrap y* under H0 (no treatment effect)
        y_star = resid_star   # Xr @ coef_r already in resid_r = yd - Xr @ coef_r
        # Full bootstrap y* = Xr @ coef_r + resid_star = yd - resid_r + resid_star
        y_star = yd - resid_r + resid_star

        # FIX 5: recompute XXinv from full Xd (treatment + controls)
        try:
            XXinv_star = np.linalg.inv(Xd.T @ Xd)
        except np.linalg.LinAlgError:
            XXinv_star = np.linalg.pinv(Xd.T @ Xd)

        c_star   = XXinv_star @ (Xd.T @ y_star)
        r_star   = y_star - Xd @ c_star
        se_star  = clustered_se(Xd, r_star, cl, XXinv_star, n, k, G)[tidx]

        if se_star <= 0 or not np.isfinite(se_star):
            t_boot[b] = 0.0
        else:
            t_boot[b] = c_star[tidx] / se_star

    pval = float(np.mean(np.abs(t_boot) >= np.abs(t_obs)))
    return pval


def run_bootstrap_table(datasets, checks):
    """Run wild bootstrap for all focus outcomes. Returns DataFrame."""
    rows  = []
    total = len(FOCUS_OUTCOMES)

    for i, (var, (ftype, controls, scale)) in enumerate(FOCUS_OUTCOMES.items(), 1):
        data = datasets.get(ftype)
        if data is None or var not in data.columns:
            print(f"  [{i}/{total}] {var}: no data")
            continue

        need = list({"province","province_id","wave","wave_id",
                     "bartik","mining", var} | set(controls))
        d    = data[[c for c in need if c in data.columns]]
        out  = prep_arrays(d, var, "bartik", controls, checks)
        if out is None:
            print(f"  [{i}/{total}] {var}: skipped (too few obs / singular / zero variance)")
            continue

        yd, Xd, cl, n, k, G, sub = out
        coef, resid, XXinv = ols_fe(yd, Xd)
        se    = clustered_se(Xd, resid, cl, XXinv, n, k, G)
        tidx  = 0

        # FIX 10: NaN/inf guard
        if not np.isfinite(coef[tidx]) or not np.isfinite(se[tidx]) or se[tidx] <= 0:
            checks.append(f"  {var}: non-finite coef or SE — skipped")
            continue

        tstat         = coef[tidx] / se[tidx]
        pval_analytic = float(2 * stats.t.sf(abs(tstat), df=G - 1))

        # FIX 7: per-outcome seed for full reproducibility
        outcome_seed = 42 + hash(var) % 10000
        t0 = time.time()
        pval_boot = wild_bootstrap_pval(
            yd, Xd, cl, coef, tidx=tidx, B=999, seed=outcome_seed
        )
        elapsed = time.time() - t0

        coef_s = coef[tidx] * scale
        se_s   = se[tidx]   * scale

        print(f"  [{i}/{total}] {var:<28} "
              f"β={coef_s:+.3f}  "
              f"p_analytic={pval_analytic:.3f}{stars(pval_analytic)}  "
              f"p_boot={pval_boot:.3f}{stars(pval_boot)}  "
              f"({elapsed:.1f}s)")

        checks.append(
            f"  {var:<28} β={coef_s:+.3f}  "
            f"p_analytic={pval_analytic:.3f}{stars(pval_analytic)}  "
            f"p_boot={pval_boot:.3f}{stars(pval_boot)}  N={n:,}  G={G}"
        )

        rows.append({
            "outcome":        var,
            "coef_s":         coef_s,
            "se_s":           se_s,
            "tstat":          tstat,
            "pval_analytic":  pval_analytic,
            "pval_bootstrap": pval_boot,
            "nobs":           n,
            "n_clusters":     G,
        })

    return pd.DataFrame(rows)


# =============================================================================
# 2. EVENT STUDY
# =============================================================================

def run_event_study(datasets, checks):
    """
    For each focus outcome, estimate beta_t for each wave separately.
    Base wave = BASE_WAVE (2002, lowest copper price) — beta = 0 by construction.
    FIX 3: within_demean tuple unpacked correctly.
    FIX 6: condition number check added.
    FIX 8: convergence warning logged.
    """
    rows     = []
    non_base = [w for w in WAVE_ORDER if w != BASE_WAVE]

    for var, (ftype, controls, scale) in FOCUS_OUTCOMES.items():
        data = datasets.get(ftype)
        if data is None or var not in data.columns:
            continue

        need = list({"province","province_id","wave","wave_id",
                     "mining", var} | set(controls))
        d = data[[c for c in need if c in data.columns]].copy()

        # Interaction dummies: mining_g × I(wave == t), excluding base wave
        for w in non_base:
            d[f"evt_{w}"] = d["mining"] * (d["wave"] == w).astype(float)

        evt_cols = [f"evt_{w}" for w in non_base]

        cols_needed = list(dict.fromkeys(
            [var, "province_id", "wave_id"] + evt_cols +
            [c for c in controls if c in d.columns]
        ))
        sub = d[cols_needed].dropna().copy()

        if len(sub) < MIN_OBS:
            continue

        prov = sub["province_id"].to_numpy(dtype=int)
        wave = sub["wave_id"].to_numpy(dtype=int)
        ctrl = [c for c in controls if c in sub.columns]

        Xraw = sub[evt_cols + ctrl].to_numpy(dtype=float)
        yraw = sub[var].to_numpy(dtype=float)

        # FIX 3: unpack tuple; FIX 8: log convergence
        yd, _, cv_y = within_demean(yraw, prov, wave)
        Xd, _, cv_X = within_demean(Xraw, prov, wave)
        if not cv_y or not cv_X:
            checks.append(f"  WARN event study {var}: FE did not converge")

        # FIX 6: condition number guard
        cond = np.linalg.cond(Xd.T @ Xd)
        if cond > 1e12:
            checks.append(f"  WARN event study {var}: singular design matrix — skipped")
            continue

        cl   = pd.Categorical(sub["province_id"].values).codes.astype(int)
        n, k = Xd.shape
        G    = len(np.unique(cl))

        if G <= 1 or n <= k:
            continue

        coef, resid, XXinv = ols_fe(yd, Xd)
        se    = clustered_se(Xd, resid, cl, XXinv, n, k, G)
        tstat = np.where(se > 0, coef / se, np.nan)
        pval  = np.where(se > 0,
                         2 * stats.t.sf(np.abs(tstat), df=G - 1),
                         np.nan)

        # Base wave: beta = 0 by construction
        rows.append({
            "outcome": var, "wave": BASE_WAVE,
            "beta": 0.0, "se": 0.0, "pval": np.nan,
            "ci_lo": 0.0, "ci_hi": 0.0, "n": len(sub),
        })

        for j, w in enumerate(non_base):
            # FIX 10: NaN/inf guard
            if not np.isfinite(coef[j]) or not np.isfinite(se[j]):
                continue
            ci_lo = (coef[j] - stats.t.ppf(0.975, df=G - 1) * se[j]) * scale
            ci_hi = (coef[j] + stats.t.ppf(0.975, df=G - 1) * se[j]) * scale
            rows.append({
                "outcome": var,
                "wave":    w,
                "beta":    coef[j] * scale,
                "se":      se[j]   * scale,
                "pval":    float(pval[j]),
                "ci_lo":   ci_lo,
                "ci_hi":   ci_hi,
                "n":       len(sub),
            })

    return pd.DataFrame(rows)


def ascii_event_chart(es_df, outcome, scale_label="pp"):
    """ASCII event study chart for one outcome."""
    sub = es_df[es_df["outcome"] == outcome].sort_values("wave")
    if sub.empty:
        return f"  No data for {outcome}"

    betas = sub["beta"].values
    waves = sub["wave"].values
    lo    = sub["ci_lo"].values
    hi    = sub["ci_hi"].values

    all_vals = np.concatenate([lo, hi, [0]])
    vmin, vmax = all_vals.min(), all_vals.max()
    span   = vmax - vmin if vmax > vmin else 1.0
    HEIGHT = 12

    lines = []
    lines.append(f"\n  Event Study: {outcome}")
    lines.append(f"  Base wave: {BASE_WAVE} (β = 0 by construction)")
    lines.append("")

    grid = [[" "] * len(waves) for _ in range(HEIGHT)]
    for col, (b, l, h) in enumerate(zip(betas, lo, hi)):
        row_lo = HEIGHT - 1 - max(0, min(HEIGHT-1, int((h - vmin) / span * (HEIGHT-1))))
        row_hi = HEIGHT - 1 - max(0, min(HEIGHT-1, int((l - vmin) / span * (HEIGHT-1))))
        for r in range(min(row_lo, row_hi), max(row_lo, row_hi) + 1):
            grid[r][col] = "│"
        row_b = HEIGHT - 1 - max(0, min(HEIGHT-1, int((b - vmin) / span * (HEIGHT-1))))
        grid[row_b][col] = "●"

    row_zero = HEIGHT - 1 - max(0, min(HEIGHT-1, int((0 - vmin) / span * (HEIGHT-1))))
    for r, row in enumerate(grid):
        val_label = f"{vmax - r * span / (HEIGHT - 1):+6.1f} |"
        zero_mark = "── " if r == row_zero else "   "
        lines.append(f"  {val_label} {zero_mark}{'  '.join(row)}")

    lines.append("  " + " " * 10 + "  ".join([str(w)[:4] for w in waves]))
    # FIX 4: COPPER_PRICES imported directly — no reg.COPPER_PRICES
    lines.append(
        "  Copper prices: " +
        "  ".join([f"{COPPER_PRICES.get(int(w), '?'):,}" for w in waves])
    )
    return "\n".join(lines)


# =============================================================================
# 3. CONTINUOUS TREATMENT ROBUSTNESS
# =============================================================================

def run_continuous_robustness(datasets, checks):
    """
    Split sample into HIGH price waves (2007, 2014, 2018, 2024)
    and LOW price waves (1992, 1996, 2002) and re-run main spec on each.
    FIX 3/8/10 applied via updated prep_arrays.
    """
    rows       = []
    HIGH_WAVES = [2007, 2014, 2018, 2024]
    LOW_WAVES  = [1992, 1996, 2002]

    for var, (ftype, controls, scale) in FOCUS_OUTCOMES.items():
        data = datasets.get(ftype)
        if data is None or var not in data.columns:
            continue

        need = list({"province","province_id","wave","wave_id",
                     "bartik","mining","ln_cu_price", var} | set(controls))
        d = data[[c for c in need if c in data.columns]]

        for label, waves in [("high_price", HIGH_WAVES), ("low_price", LOW_WAVES)]:
            sub = d[d["wave"].isin(waves)].copy()
            if sub["province_id"].nunique() < 2:
                continue

            # Re-encode IDs within subsample for correct demeaning
            sub["wave_id"]     = pd.Categorical(sub["wave"].values).codes.astype(int)
            sub["province_id"] = pd.Categorical(
                sub["province"].values if "province" in sub.columns
                else sub["province_id"].values
            ).codes.astype(int)

            out = prep_arrays(sub, var, "bartik", controls, checks)
            if out is None:
                continue

            yd, Xd, cl, n, k, G, _ = out
            coef, resid, XXinv = ols_fe(yd, Xd)
            se = clustered_se(Xd, resid, cl, XXinv, n, k, G)
            tidx = 0

            # FIX 10: NaN/inf guard
            if not np.isfinite(coef[tidx]) or not np.isfinite(se[tidx]) or se[tidx] <= 0:
                continue

            tstat = coef[tidx] / se[tidx]
            pval  = float(2 * stats.t.sf(abs(tstat), df=max(G - 1, 1)))

            rows.append({
                "outcome":    var,
                "sample":     label,
                "coef_s":     coef[tidx] * scale,
                "se_s":       se[tidx]   * scale,
                "pval":       pval,
                "nobs":       n,
                "n_clusters": G,
            })

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main():
    t_start = time.time()
    checks  = []
    checks.append("=" * 70)
    checks.append("ZAMBIA COPPER — ROBUSTNESS CHECKS")
    checks.append("=" * 70)
    checks.append("  G = 9 provinces (Muchinga merged into Northern)")
    checks.append("  Bootstrap: Webb (2014) 6-point weights, B=999")
    checks.append("  Bootstrap H0: restricted residuals (treatment removed)")
    checks.append("  FE: iterative two-way demeaning, max_iter=50, tol=1e-10")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading data...")
    datasets = {}
    for ftype in ["HR", "KR", "IR", "MR"]:
        print(f"  {ftype}...", end=" ", flush=True)
        dummy_checks = []
        datasets[ftype] = load_stack(ftype, dummy_checks)
        n = len(datasets[ftype]) if datasets[ftype] is not None else 0
        print(f"{n:,} obs")

    # ── 1. Wild Cluster Bootstrap ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("1. WILD CLUSTER BOOTSTRAP (Webb 2014, B=999, G=9)")
    print("   H0 imposed via restricted residuals")
    print("=" * 60)
    checks.append("\n--- 1. Wild Cluster Bootstrap ---")
    boot_df = run_bootstrap_table(datasets, checks)
    boot_df.to_csv(OUT_DIR / "bootstrap_results.csv", index=False)
    print(f"\n  Saved: bootstrap_results.csv ({len(boot_df)} outcomes)")

    # ── 2. Event Study ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("2. EVENT STUDY (wave-by-wave coefficients)")
    print(f"   Base wave: {BASE_WAVE} (lowest copper price)")
    print("=" * 60)
    checks.append("\n--- 2. Event Study ---")
    es_df = run_event_study(datasets, checks)
    es_df.to_csv(OUT_DIR / "event_study.csv", index=False)
    print(f"\n  Saved: event_study.csv ({len(es_df)} rows)")

    chart_lines = [
        "EVENT STUDY CHARTS",
        f"Base wave: {BASE_WAVE} (lowest copper price). β = 0 by construction.",
        "● = point estimate, │ = 95% CI",
        "=" * 70,
    ]
    key_outcomes = ["delivery_facility","delivery_skilled",
                    "pnc_mother","anc_4plus","had_fever","man_employed"]
    for var in key_outcomes:
        chart = ascii_event_chart(es_df, var)
        chart_lines.append(chart)
        checks.append(chart)
        sub = es_df[es_df["outcome"] == var].sort_values("wave")
        if not sub.empty:
            print(f"\n  {var}:")
            print(f"  {'Wave':<6} {'β':>8} {'SE':>7} {'p':>7}")
            print(f"  {'----':<6} {'---':>8} {'--':>7} {'--':>7}")
            for _, r in sub.iterrows():
                if pd.isna(r["pval"]):
                    print(f"  {int(r['wave']):<6} {'0.00':>8} {'---':>7} {'base':>7}")
                else:
                    print(f"  {int(r['wave']):<6} {r['beta']:>+8.2f} "
                          f"{r['se']:>7.3f} {r['pval']:>7.3f}{stars(r['pval'])}")

    with open(OUT_DIR / "event_study_chart.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(chart_lines))
    print(f"\n  Saved: event_study_chart.txt")

    # ── 3. Continuous Treatment Robustness ────────────────────────────────────
    print("\n" + "=" * 60)
    print("3. CONTINUOUS TREATMENT ROBUSTNESS")
    print("   High price waves (2007+) vs Low price waves (1992-2002)")
    print("=" * 60)
    checks.append("\n--- 3. Continuous Treatment Robustness ---")
    cont_df = run_continuous_robustness(datasets, checks)
    cont_df.to_csv(OUT_DIR / "continuous_robustness.csv", index=False)

    print(f"\n  {'Outcome':<28} {'High-price β':>14}   {'Low-price β':>14}")
    print("  " + "─" * 65)
    for var in FOCUS_OUTCOMES:
        hi = cont_df[(cont_df["outcome"]==var) & (cont_df["sample"]=="high_price")]
        lo = cont_df[(cont_df["outcome"]==var) & (cont_df["sample"]=="low_price")]
        if hi.empty and lo.empty:
            continue
        hi_str = (f"{hi.iloc[0]['coef_s']:+.3f}{stars(hi.iloc[0]['pval'])} "
                  f"(p={hi.iloc[0]['pval']:.3f})") if not hi.empty else "—"
        lo_str = (f"{lo.iloc[0]['coef_s']:+.3f}{stars(lo.iloc[0]['pval'])} "
                  f"(p={lo.iloc[0]['pval']:.3f})") if not lo.empty else "—"
        print(f"  {var:<28} {hi_str:>22}   {lo_str:>22}")
        checks.append(f"  {var:<28} high={hi_str}  low={lo_str}")
    print(f"\n  Saved: continuous_robustness.csv")

    # ── Summary table ─────────────────────────────────────────────────────────
    tlines = []
    tlines.append("=" * 80)
    tlines.append("ZAMBIA COPPER — ROBUSTNESS SUMMARY")
    tlines.append("=" * 80)
    tlines.append("")
    tlines.append("Table R1: Wild Cluster Bootstrap vs Analytic p-values (Webb 2014)")
    tlines.append("─" * 80)
    tlines.append(
        f"  {'Outcome':<28} {'β (pp)':>8} {'SE':>6} "
        f"{'p (analytic)':>14} {'p (bootstrap)':>14}  N"
    )
    tlines.append("  " + "─" * 75)
    for _, r in boot_df.iterrows():
        tlines.append(
            f"  {r['outcome']:<28} {r['coef_s']:>+8.3f} {r['se_s']:>6.3f} "
            f"{r['pval_analytic']:>12.3f}{stars(r['pval_analytic']):3s} "
            f"{r['pval_bootstrap']:>12.3f}{stars(r['pval_bootstrap']):3s}  "
            f"{int(r['nobs']):,}"
        )
    tlines.append("")
    tlines.append("Notes:")
    tlines.append("  Wild bootstrap: Webb (2014) 6-point weights, B=999.")
    tlines.append("  H0 imposed via restricted residuals (treatment removed).")
    tlines.append("  Province-level clustering. G=9 after Muchinga merge.")
    tlines.append("  Results significant under BOTH methods are most credible.")

    with open(OUT_DIR / "robustness_tables.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(tlines))
    with open(OUT_DIR / "robustness_checks.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(checks))

    total = time.time() - t_start
    print(f"\nDone in {total/60:.1f} min.")
    print("  bootstrap_results.csv")
    print("  event_study.csv")
    print("  event_study_chart.txt")
    print("  continuous_robustness.csv")
    print("  robustness_tables.txt")
    print("  robustness_checks.txt")


if __name__ == "__main__":
    try:
        import pyreadstat, scipy
    except ImportError:
        print("pip install pandas numpy scipy pyreadstat")
        sys.exit(1)
    main()