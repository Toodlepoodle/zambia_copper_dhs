"""
mens_migration.py  —  LONG-RESIDENT SUBSAMPLE ROBUSTNESS FOR MEN
=================================================================
Mirrors the women's long-resident check in 05_mechanisms.py (Check 8),
but for the Men's Recode (MR) using mv104 (years of residence in province).

Logic:
  - Restricts MR sample to men who have lived in their current province
    for 5+ years (mv104 >= 5), or always (mv104 == 95).
  - Codes 96 (visitor) and 97 (not stated) → NaN (excluded).
  - Runs the same two-way province+wave FE regression as 03_regressions.py
    for all Table 6 men's outcomes, comparing full-sample vs long-resident
    subsample coefficients.
  - If results hold → in-migrants of working-age men are not driving
    the main Table 6 estimates.

DHS variable:
  mv104 — years lived in current place of residence (Men's Recode)
  Coding (standard across waves):
    0–94  = number of years (0 = less than 1 year)
    95    = always lived here
    96    = visitor (exclude)
    97    = not stated (exclude)

Outputs:
  OUTPUT/mens_longresident_robustness.csv   — machine-readable results
  OUTPUT/mens_longresident_tables.txt       — formatted comparison table
  OUTPUT/mens_longresident_checks.txt       — diagnostics and warnings

Usage:
  python mens_migration.py

Requirements:
  - config_00.py  in the same directory
  - 03_regressions.py in the same directory  (for load_stack and helpers)
  - DHS .dta files accessible via BASE_DIR / ZAMBIA_BASE_DIR env var

Does NOT modify any existing script.
"""

import sys
import warnings
import time
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
from scipy import stats

# =============================================================================
# CONFIG IMPORTS
# =============================================================================

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from config_00 import (
    BASE_DIR, OUT_DIR,
    COPPER_PRICES, PROVINCE_HARMONISE, MINING_PROVINCES, WAVE_FILES,
    DHS_MISSING_CODES,
    MIN_OBS, MIN_CLUSTERS,
)

# We borrow load_stack() and the build_mr() pipeline from 03_regressions.py
# so the data preparation is identical to the main paper.
try:
    from importlib import import_module
    reg        = import_module("03_regressions")
    load_stack = reg.load_stack
except Exception as e:
    print(f"ERROR: Could not import 03_regressions: {e}")
    print("Make sure 03_regressions.py is in the same directory as this script.")
    sys.exit(1)

# =============================================================================
# MEN'S TABLE 6 OUTCOMES
# (var, label, scale, controls)
# Matches OUTCOME_TABLES["Table 6: Men's Outcomes"] in 03_regressions.py
# =============================================================================

MEN_OUTCOMES = [
    ("man_edu_sec_p",  "Secondary+ education",    100, ["man_age", "man_urban"]),
    ("man_edu_level",  "Education level (0-3)",     1,  ["man_age", "man_urban"]),
    ("man_employed",   "Currently employed",       100, ["man_age", "man_urban"]),
    ("man_condom",     "Condom at last sex",       100, ["man_age", "man_urban"]),
    ("man_multi_part", "2+ partners last year",   100, ["man_age", "man_urban"]),
    ("man_dv_justify", "Justifies DV",            100, ["man_age", "man_urban"]),
]

# mv104 DHS variable name (Men's Recode equivalent of IR's v104)
MV104_COL = "mv104"

# Minimum years of residence to count as "long-resident"
LONGRES_YEARS = 5

# Code 95 = "always lived here" → counts as long-resident
ALWAYS_CODE = 95

# Codes that must be excluded entirely (not 0, not long-res)
EXCLUDE_CODES = {96, 97}  # visitor, not stated

# =============================================================================
# TWO-WAY FE HELPERS
# Copied here so this script is fully self-contained.
# Logic is identical to 05_mechanisms.py / 03_regressions.py.
# =============================================================================

def demean_by_fast(arr, group_ids):
    """Vectorised group-demeaning. Works for 1-D and 2-D arrays."""
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
    """Iterative two-way FE demeaning (province + wave)."""
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


def run_fe(data, y_var, treat_var, control_cols, scale, checks=None):
    """
    OLS with two-way province + wave FE and province-clustered SE (CR1S).
    Returns result dict or None if insufficient data.
    """
    if y_var not in data.columns:
        return None
    def _warn(msg):
        if checks is not None:
            checks.append(f"  WARN [{y_var}]: {msg}")

    needed = list(dict.fromkeys(
        [y_var, treat_var, "province", "province_id", "wave_id"]
        + [c for c in control_cols if c in data.columns]
    ))
    needed = [c for c in needed if c in data.columns]
    sub    = data[needed].dropna().copy()

    if len(sub) < MIN_OBS:
        _warn(f"n={len(sub)} < MIN_OBS={MIN_OBS} — skipped")
        return None

    prov = sub["province_id"].to_numpy(dtype=int)
    wave = sub["wave_id"].to_numpy(dtype=int)
    ctrl = [c for c in control_cols if c in sub.columns]

    Xraw = sub[[treat_var] + ctrl].to_numpy(dtype=float)
    yraw = sub[y_var].to_numpy(dtype=float)

    # Guard: zero-variance treatment
    if Xraw[:, 0].std() == 0:
        _warn("zero-variance treatment — skipped")
        return None

    n, k = Xraw.shape

    # Guard: n must exceed k for OLS to be identified
    if n <= k:
        _warn(f"n={n} <= k={k} — skipped")
        return None

    yd, _, cv_y = within_demean(yraw, prov, wave)
    Xd, _, cv_X = within_demean(Xraw, prov, wave)

    if not cv_y or not cv_X:
        _warn("demeaning did not converge in 50 iterations")

    try:
        XX    = Xd.T @ Xd
        XXinv = np.linalg.inv(XX)
    except np.linalg.LinAlgError:
        XXinv = np.linalg.pinv(Xd.T @ Xd)

    coef  = XXinv @ (Xd.T @ yd)
    resid = yd - Xd @ coef

    if np.any(np.isnan(coef)) or np.any(np.isinf(coef)):
        _warn("NaN/Inf in coefficients — skipped")
        return None

    cl       = pd.Categorical(sub["province"].values).codes.astype(int)
    unique_c = np.unique(cl)
    G        = len(unique_c)

    if G < MIN_CLUSTERS:
        _warn(f"G={G} < MIN_CLUSTERS={MIN_CLUSTERS} — skipped")
        return None

    meat = np.zeros((k, k))
    for c in unique_c:
        idx = cl == c
        sc  = Xd[idx].T @ resid[idx]
        meat += np.outer(sc, sc)

    correction = (G / (G - 1)) * ((n - 1) / (n - k))
    V    = correction * (XXinv @ meat @ XXinv)
    se   = np.sqrt(np.maximum(np.diag(V), 0.0))

    if np.any(np.isnan(se)) or np.any(np.isinf(se)):
        _warn("NaN/Inf in standard errors — skipped")
        return None

    tstat = np.where(se > 0, coef / se, np.nan)
    pval  = np.where(se > 0,
                     2 * stats.t.sf(np.abs(tstat), df=G - 1),
                     np.nan)

    tidx  = 0   # first column of Xd is treat_var (no intercept after demeaning)
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


def stars(p):
    if pd.isna(p): return ""
    if p < 0.01:   return "***"
    if p < 0.05:   return "**"
    if p < 0.10:   return "*"
    return ""


# =============================================================================
# MV104 EXTRACTION
# We need to pull mv104 from the raw MR files and merge it into the stacked
# MR dataset that load_stack() produces, because load_stack() does not retain
# mv104 by default.
# =============================================================================

def load_mv104(checks):
    """
    Load mv104 (years of residence) from each MR wave file.
    Returns a DataFrame with columns: [province, wave, mv104_raw]
    indexed the same way as the stacked MR data (individual rows).

    We re-use the same province harmonisation as 03_regressions.py so
    the merge on (province, wave) is consistent.
    """
    print("\nLoading mv104 from MR files...")

    from config_00 import WAVE_FILES as WF

    def _get_label(meta, col_name, code):
        labels = {}
        if meta and hasattr(meta, "variable_value_labels"):
            labels = meta.variable_value_labels.get(col_name, {})
        key = int(code) if isinstance(code, (float, np.floating)) else code
        return str(labels.get(key, labels.get(code, str(code)))).strip().lower()

    def _harmonise(raw):
        cleaned = " ".join(str(raw).strip().lower().split())
        result  = PROVINCE_HARMONISE.get(cleaned)
        if result is None:
            raise ValueError(f"Unknown province label: '{raw}'")
        return result

    frames = []
    for wave, files in WF.items():
        rel = files.get("MR")
        if not rel:
            checks.append(f"  mv104 {wave}: no MR file defined")
            continue
        path = BASE_DIR / rel
        if not path.exists():
            checks.append(f"  mv104 {wave}: MR file not found — {path.name}")
            continue

        try:
            df, meta = pyreadstat.read_dta(str(path))
            df.columns = df.columns.str.lower()
        except Exception as e:
            checks.append(f"  mv104 {wave}: load error — {e}")
            continue

        # Check mv104 exists
        if "mv104" not in df.columns:
            checks.append(f"  mv104 {wave}: mv104 column NOT FOUND in MR file")
            continue

        # Check province column
        if "mv024" not in df.columns:
            checks.append(f"  mv104 {wave}: mv024 (province) not found")
            continue

        try:
            df["province"] = df["mv024"].apply(
                lambda c: _harmonise(_get_label(meta, "mv024", c))
            )
        except Exception as e:
            checks.append(f"  mv104 {wave}: province harmonisation error — {e}")
            continue

        df["wave"]     = wave
        df["mv104_raw"] = pd.to_numeric(df["mv104"], errors="coerce")

        n_found = df["mv104_raw"].notna().sum()
        checks.append(f"  mv104 {wave}: n={len(df):,}  mv104 non-missing={n_found:,}")
        print(f"    {wave}: {len(df):,} men  mv104 non-missing={n_found:,}")

        frames.append(df[["province", "wave", "mv104_raw"]].copy())

    if not frames:
        return None

    return pd.concat(frames, ignore_index=True)


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_mens_longresident(mr_data, mv104_df, checks):
    """
    For each Table 6 outcome, run:
      (a) Full MR sample
      (b) Long-resident subsample (mv104 >= 5 or mv104 == 95)
    and compare the Bartik coefficients.
    """
    print("\n" + "=" * 60)
    print("MEN'S LONG-RESIDENT SUBSAMPLE (mv104 >= 5 years)")
    print("=" * 60)

    if mr_data is None:
        print("  SKIP — MR data not loaded")
        return pd.DataFrame()

    if mv104_df is None:
        print("  SKIP — mv104 data could not be loaded from any MR wave file")
        return pd.DataFrame()

    # ── Attach mv104_raw to the stacked MR data ───────────────────────────────
    # load_stack() produces one row per individual. We need to align mv104_raw.
    # Since load_stack() doesn't expose a row index back to the raw file, we
    # rebuild the variable from scratch by loading the raw files again.
    # We already did that in load_mv104(), so we concatenate and assign
    # row-by-row assuming the same sort order (province + wave within each file).
    #
    # Safer approach: we re-attach by using a sequential row index per
    # (province, wave) group — both the stacked MR and the mv104_df are built
    # from the same raw files in the same wave order, so the row counts match.

    mr = mr_data.copy()

    # Build a per-wave row counter for stacked MR
    mr["_row_in_wave"] = mr.groupby("wave").cumcount()
    mv104_df["_row_in_wave"] = mv104_df.groupby("wave").cumcount()

    merged = mr.merge(
        mv104_df[["wave", "_row_in_wave", "mv104_raw"]],
        on=["wave", "_row_in_wave"],
        how="left",
    )
    merged.drop(columns=["_row_in_wave"], inplace=True)

    n_attached = merged["mv104_raw"].notna().sum()
    print(f"  Rows in stacked MR:  {len(mr):,}")
    print(f"  mv104 attached:      {n_attached:,}")

    if n_attached == 0:
        print("  SKIP — mv104 could not be attached (no matches)")
        checks.append("  mens_migration: mv104 attachment produced 0 matches")
        return pd.DataFrame()

    # ── Clean mv104 ───────────────────────────────────────────────────────────
    v = merged["mv104_raw"].copy()

    # Visitor (96) and not-stated (97) → NaN
    v = v.where(~v.isin(EXCLUDE_CODES), np.nan)

    # Values > 95 that aren't in EXCLUDE_CODES → NaN (safety catch)
    v = v.where(v <= 95, np.nan)

    merged["mv104_clean"] = v

    # Long-resident flag: resident >= LONGRES_YEARS or always (== ALWAYS_CODE)
    merged["longres"] = (
        (merged["mv104_clean"] >= LONGRES_YEARS) |
        (merged["mv104_clean"] == ALWAYS_CODE)
    ).astype(float)
    merged.loc[merged["mv104_clean"].isna(), "longres"] = np.nan

    n_total   = merged["longres"].notna().sum()
    n_longres = (merged["longres"] == 1).sum()
    pct       = 100 * n_longres / n_total if n_total > 0 else 0.0

    print(f"  mv104 clean non-missing: {n_total:,}")
    print(f"  Long-residents (>={LONGRES_YEARS} yrs or always): "
          f"{n_longres:,} ({pct:.1f}%)")

    checks.append(f"  mv104 clean non-missing: {n_total:,}")
    checks.append(f"  Long-residents: {n_longres:,} ({pct:.1f}%)")

    if n_longres < 500:
        print("  SKIP — fewer than 500 long-resident men")
        checks.append("  SKIP — n_longres < 500")
        return pd.DataFrame()

    mr_full = merged.copy()
    mr_lr   = merged[merged["longres"] == 1].copy()

    print(f"\n  Running regressions on {len(MEN_OUTCOMES)} outcomes...")

    rows = []
    for var, label, scale, controls in MEN_OUTCOMES:
        if var not in mr_full.columns:
            checks.append(f"  {var}: NOT IN MR DATA — skip")
            print(f"    {var:<25} NOT IN DATA")
            continue

        n_valid = mr_full[var].notna().sum()
        if n_valid < MIN_OBS:
            checks.append(f"  {var}: only {n_valid} non-missing — skip")
            print(f"    {var:<25} only {n_valid} non-missing — skip")
            continue

        r_full = run_fe(mr_full, var, "bartik", controls, scale, checks=checks)
        r_lr   = run_fe(mr_lr,   var, "bartik", controls, scale, checks=checks)

        if r_full is None:
            print(f"    {var:<25} FULL SAMPLE FAILED")
            checks.append(f"  {var}: full-sample regression failed")
            continue

        b_full = f"{r_full['coef_s']:+.2f}{stars(r_full['pval'])}"
        b_lr   = f"{r_lr['coef_s']:+.2f}{stars(r_lr['pval'])}" if r_lr else "N/A"
        print(f"    {label:<25}  full={b_full:<8}  long-res={b_lr}")

        rows.append({
            "outcome":   var,
            "label":     label,
            "scale":     scale,
            # Full sample
            "beta_full": r_full["coef_s"],
            "se_full":   r_full["se_s"],
            "p_full":    r_full["pval"],
            "n_full":    r_full["nobs"],
            "g_full":    r_full["n_clusters"],
            "r2_full":   r_full["r2"],
            # Long-resident subsample
            "beta_lr":   r_lr["coef_s"]     if r_lr else np.nan,
            "se_lr":     r_lr["se_s"]       if r_lr else np.nan,
            "p_lr":      r_lr["pval"]       if r_lr else np.nan,
            "n_lr":      r_lr["nobs"]       if r_lr else np.nan,
            "g_lr":      r_lr["n_clusters"] if r_lr else np.nan,
            "r2_lr":     r_lr["r2"]         if r_lr else np.nan,
        })

    return pd.DataFrame(rows)


# =============================================================================
# FORMATTED TABLE OUTPUT
# =============================================================================

def build_table(df_results, tlines, checks):
    tlines.append("=" * 80)
    tlines.append("MEN'S LONG-RESIDENT SUBSAMPLE ROBUSTNESS")
    tlines.append("Y_igt = α + β(mining_g × ln_cu_price_t) + province_FE + wave_FE + controls + ε")
    tlines.append("SE clustered at province level (CR1S).  *** p<0.01  ** p<0.05  * p<0.10")
    tlines.append(f"Long-resident = mv104 >= {LONGRES_YEARS} years OR mv104 == {ALWAYS_CODE} (always lived here)")
    tlines.append("Excludes: mv104==96 (visitor), mv104==97 (not stated)")
    tlines.append("Interpretation: if coefficients hold in long-resident subsample,")
    tlines.append("  recent in-migrants of working-age men are not driving Table 6 estimates.")
    tlines.append("=" * 80)
    tlines.append("")

    if df_results.empty:
        tlines.append("NO RESULTS — check mens_longresident_checks.txt for diagnostics.")
        return

    hdr = (f"  {'Outcome':<25}  "
           f"{'Full β':>8}  {'SE':>6}  {'p':>6}  {'N':>8}  "
           f"{'LR β':>8}  {'SE':>6}  {'p':>6}  {'N (LR)':>8}")
    tlines.append(hdr)
    tlines.append("  " + "─" * 78)

    for _, row in df_results.iterrows():
        f_b  = f"{row['beta_full']:>+8.3f}"
        f_se = f"{row['se_full']:>6.3f}"
        f_p  = f"{row['p_full']:>6.3f}{stars(row['p_full']):<3}"
        f_n  = f"{int(row['n_full']):>8,}"

        if pd.notna(row["beta_lr"]):
            lr_b  = f"{row['beta_lr']:>+8.3f}"
            lr_se = f"{row['se_lr']:>6.3f}"
            lr_p  = f"{row['p_lr']:>6.3f}{stars(row['p_lr']):<3}"
            lr_n  = f"{int(row['n_lr']):>8,}"
        else:
            lr_b = lr_se = lr_p = lr_n = "     N/A"

        tlines.append(
            f"  {row['label']:<25}  "
            f"{f_b}  {f_se}  {f_p}  {f_n}  "
            f"{lr_b}  {lr_se}  {lr_p}  {lr_n}"
        )

    tlines.append("")
    tlines.append("─" * 80)
    tlines.append("SIGNIFICANT RESULTS (p < 0.10) — FULL SAMPLE vs LONG-RESIDENT")
    tlines.append("─" * 80)

    sig = df_results[df_results["p_full"] < 0.10]
    if sig.empty:
        tlines.append("  None at p < 0.10 in full sample.")
    else:
        for _, row in sig.iterrows():
            direction = (
                "CONSISTENT" if (pd.notna(row["p_lr"]) and row["p_lr"] < 0.10
                                 and np.sign(row["beta_full"]) == np.sign(row["beta_lr"]))
                else "WEAKENED/REVERSED" if pd.notna(row["p_lr"])
                else "LR N/A"
            )
            tlines.append(
                f"  {row['label']:<25}  "
                f"full β={row['beta_full']:+.3f} p={row['p_full']:.3f}  "
                f"LR β={row['beta_lr']:+.3f} p={row['p_lr']:.3f}  [{direction}]"
                if pd.notna(row["beta_lr"])
                else f"  {row['label']:<25}  full β={row['beta_full']:+.3f} p={row['p_full']:.3f}  LR=N/A"
            )

    tlines.append("")
    tlines.append("─" * 80)
    tlines.append("NULL RESULTS (p >= 0.10) — FULL SAMPLE")
    tlines.append("─" * 80)
    null = df_results[df_results["p_full"] >= 0.10]
    if null.empty:
        tlines.append("  All outcomes significant at p < 0.10.")
    else:
        for _, row in null.iterrows():
            tlines.append(
                f"  {row['label']:<25}  full β={row['beta_full']:+.3f} p={row['p_full']:.3f}"
            )


# =============================================================================
# MAIN
# =============================================================================

def main():
    t_start = time.time()
    checks  = []
    tlines  = []

    checks.append("=" * 65)
    checks.append("MENS_MIGRATION.PY — DIAGNOSTICS")
    checks.append("=" * 65)
    checks.append(f"  BASE_DIR       : {BASE_DIR}")
    checks.append(f"  MINING_PROVS   : {sorted(MINING_PROVINCES)}")
    checks.append(f"  DHS_MISSING    : {sorted(DHS_MISSING_CODES)}")
    checks.append(f"  MV104_COL      : {MV104_COL}")
    checks.append(f"  LONGRES_YEARS  : >= {LONGRES_YEARS} years OR == {ALWAYS_CODE} (always)")
    checks.append(f"  EXCLUDE_CODES  : {sorted(EXCLUDE_CODES)} (visitor / not stated)")
    checks.append(f"  MIN_OBS        : {MIN_OBS}  MIN_CLUSTERS: {MIN_CLUSTERS}")
    checks.append(f"  NOTE: Does NOT modify any existing pipeline script.")
    checks.append("")

    # ── Step 1: Load stacked MR data (same as 03_regressions.py Table 6) ─────
    print("Loading stacked MR data via 03_regressions.load_stack()...")
    checks.append("--- MR STACK LOAD ---")
    mr_data = load_stack("MR", checks)

    if mr_data is None:
        print("ERROR: MR data could not be loaded. Check checks file for details.")
        checks.append("FATAL: MR data is None after load_stack()")
    else:
        print(f"  MR stacked: {len(mr_data):,} obs  |  "
              f"{mr_data['province'].nunique()} provinces  |  "
              f"{mr_data['wave'].nunique()} waves")
        checks.append(f"  MR stacked: {len(mr_data):,} obs")

    # ── Step 2: Load mv104 separately from raw MR files ──────────────────────
    checks.append("\n--- MV104 LOAD ---")
    mv104_df = load_mv104(checks)

    if mv104_df is None:
        print("ERROR: mv104 could not be loaded from any MR wave. "
              "Check that mv104 exists in your DHS MR files.")
    else:
        print(f"  mv104 rows loaded: {len(mv104_df):,}")

    # ── Step 3: Run the long-resident robustness analysis ─────────────────────
    checks.append("\n--- REGRESSION ---")
    df_results = run_mens_longresident(mr_data, mv104_df, checks)

    # ── Step 4: Build formatted table ─────────────────────────────────────────
    build_table(df_results, tlines, checks)

    # ── Step 5: Save outputs ──────────────────────────────────────────────────
    csv_path    = OUT_DIR / "mens_longresident_robustness.csv"
    tables_path = OUT_DIR / "mens_longresident_tables.txt"
    checks_path = OUT_DIR / "mens_longresident_checks.txt"

    if not df_results.empty:
        df_results.to_csv(csv_path, index=False)
        print(f"\nResults saved:  {csv_path}  ({len(df_results)} rows)")
    else:
        print("\nNo results to save — see checks file.")

    tables_path.write_text("\n".join(tlines), encoding="utf-8")
    checks_path.write_text("\n".join(checks), encoding="utf-8")

    print(f"Table saved:    {tables_path}")
    print(f"Checks saved:   {checks_path}")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s")

    # ── Quick console summary ─────────────────────────────────────────────────
    if not df_results.empty:
        print("\n=== RESULTS SUMMARY ===")
        print(f"  {'Outcome':<25}  {'Full β':>8}  {'p':>6}  {'LR β':>8}  {'p (LR)':>6}")
        print("  " + "─" * 58)
        for _, row in df_results.iterrows():
            lr_b = f"{row['beta_lr']:>+8.3f}" if pd.notna(row["beta_lr"]) else "     N/A"
            lr_p = f"{row['p_lr']:>6.3f}"     if pd.notna(row["p_lr"])   else "     N/A"
            print(f"  {row['label']:<25}  "
                  f"{row['beta_full']:>+8.3f}  {row['p_full']:>6.3f}  "
                  f"{lr_b}  {lr_p}")

    return df_results


if __name__ == "__main__":
    try:
        import pyreadstat
        import scipy
    except ImportError:
        print("pip install pandas numpy scipy pyreadstat")
        sys.exit(1)
    main()
    
