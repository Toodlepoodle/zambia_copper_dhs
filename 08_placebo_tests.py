"""
04b_few_cluster_robust.py  —  FEW-CLUSTER ROBUSTNESS (2 TREATED PROVINCES)
============================================================================
Directly addresses the "2 treated clusters" inferential problem by running
two gold-standard tests that do NOT require asymptotic cluster theory:

  TEST A — Leave-One-Treated-Out (LOTO)
  ─────────────────────────────────────
  For each mining province in turn (Copperbelt, Northwestern), drop all
  observations from that province and re-run the main Bartik DiD.
  Logic: If the result is entirely driven by ONE treated province, the
  coefficient will collapse when that province is removed.  If both
  single-province sub-samples yield consistent, significant estimates,
  the result cannot be a province-specific artefact.

  TEST B — Placebo Treatment (Province Permutation)
  ─────────────────────────────────────────────────
  Assign "fake mining status" by randomly drawing 2 provinces from the
  7 non-mining provinces (never touching real mining provinces), recompute
  bartik = fake_mining × ln_cu_price, and re-run the regression.
  Repeat N_PLACEBO times.  The empirical distribution of placebo β gives
  the exact finite-sample null — far superior to t-table critical values
  with G=9.
  Inference: p-value = (count + 1) / (N + 1)  [Phipson & Smyth 2010].
  The +1 correction prevents p=0 with small N (minimum = 1/22 ≈ 0.045).
  A p-value < 0.05 means: "in 95%+ of random province assignments, the
  effect is smaller than what we observe for the actual mining provinces."

OUTPUTS
  OUTPUT/loto_results.csv          — leave-one-treated-out estimates
  OUTPUT/placebo_results.csv       — all N_PLACEBO placebo draws
  OUTPUT/placebo_pvalues.csv       — summary: real β vs placebo p-value
  OUTPUT/few_cluster_tables.txt    — formatted paper-ready tables
  OUTPUT/few_cluster_checks.txt    — diagnostics and warnings

DESIGN CHOICES
  • Inherits all data loading, FE demeaning, and SE machinery from
    03_regressions.py and 04_robustness.py — zero code duplication.
  • Non-mining provinces = all 9 minus {copperbelt, northwestern}.
    That gives C(7,2) = 21 unique combinations — every combination is
    run exhaustively in TEST B for exact permutation p-values, plus
    N_PLACEBO random draws if you want > 21 replications.
  • CR1S clustered SE still used throughout (same as main paper).
  • The FOCUS_OUTCOMES dict here mirrors 04_robustness.py exactly.
    Update both files together if the significant outcomes change.

USAGE
  python 04b_few_cluster_robust.py

NOTE: Both 03_regressions.py and 04_robustness.py must be in the same
      directory as this script.
"""

import sys, warnings, time, itertools
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ── Config and shared pipeline ────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from config_00 import BASE_DIR, OUT_DIR, COPPER_PRICES, MINING_PROVINCES
OUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    from importlib import import_module
    reg        = import_module("03_regressions")
    rob        = import_module("04_robustness")
    load_stack = reg.load_stack
except Exception as e:
    print(f"Could not import pipeline modules: {e}")
    print("Ensure 03_regressions.py and 04_robustness.py are in the same folder.")
    sys.exit(1)

# Borrow all numerical machinery from 04_robustness — no duplication
prep_arrays  = rob.prep_arrays
ols_fe       = rob.ols_fe
clustered_se = rob.clustered_se
within_demean = rob.within_demean
stars        = rob.stars

# =============================================================================
# CONFIGURATION
# =============================================================================

# Mirror FOCUS_OUTCOMES from 04_robustness.py exactly.
# Format: var → (file_type, controls, scale)
FOCUS_OUTCOMES = {
    "had_fever":         ("KR", ["child_age", "child_male", "birth_order", "urban"], 100),
    "floor_finished":    ("HR", ["urban", "hh_size"],                                100),
    "pnc_mother":        ("IR", ["age_woman", "urban", "married", "edu_level"],      100),
    "delivery_skilled":  ("IR", ["age_woman", "urban", "married", "edu_level"],      100),
    "delivery_facility": ("IR", ["age_woman", "urban", "married", "edu_level"],      100),
    "dv_choked":         ("IR", ["age_woman", "urban", "edu_secondary_p"],           100),
    "has_mobile":        ("IR", ["age_woman", "urban", "married"],                   100),
    "anc_4plus":         ("IR", ["age_woman", "urban", "married", "edu_level"],      100),
    "man_employed":      ("MR", ["man_age", "man_urban"],                            100),
    "owns_house":        ("IR", ["age_woman", "urban", "married"],                   100),
    "dv_justify_any":    ("IR", ["age_woman", "urban", "edu_secondary_p"],           100),
}

# All 9 harmonised provinces after Muchinga merge
ALL_PROVINCES = {
    "central", "copperbelt", "eastern", "luapula",
    "lusaka", "northern", "northwestern", "southern", "western",
}
NON_MINING_PROVINCES = ALL_PROVINCES - MINING_PROVINCES   # 7 provinces

# Number of random placebo draws ON TOP OF the exhaustive C(7,2)=21 combinations.
# Set to 0 to use only exact permutations (recommended for replicability).
N_PLACEBO_EXTRA = 0

# Random seed for any extra placebo draws
PLACEBO_SEED = 2024


# =============================================================================
# SHARED REGRESSION HELPER
# =============================================================================

def run_one(data_subset, var, ftype, controls, scale, checks, label=""):
    """
    Run a single Bartik DiD on data_subset.
    Returns dict with β, SE, t, p, N, G  — or None if skipped.
    All column presence checks done inside prep_arrays.
    """
    if data_subset is None or var not in data_subset.columns:
        return None

    need = list({"province", "province_id", "wave", "wave_id",
                 "bartik", "mining", var} | set(controls))
    d = data_subset[[c for c in need if c in data_subset.columns]].copy()

    # Re-encode province_id and wave_id within the subsample so demeaning
    # only absorbs the FEs that are actually present in this subset.
    d["province_id"] = pd.Categorical(d["province"].values).codes.astype(int)
    d["wave_id"]     = pd.Categorical(d["wave"].values).codes.astype(int)

    out = prep_arrays(d, var, "bartik", controls, checks)
    if out is None:
        return None

    yd, Xd, cl, n, k, G, _ = out
    coef, resid, XXinv = ols_fe(yd, Xd)
    se = clustered_se(Xd, resid, cl, XXinv, n, k, G)
    tidx = 0

    if not np.isfinite(coef[tidx]) or not np.isfinite(se[tidx]) or se[tidx] <= 0:
        checks.append(f"  {label} {var}: non-finite coef/SE — skipped")
        return None

    tstat = coef[tidx] / se[tidx]
    pval  = float(2 * stats.t.sf(abs(tstat), df=G - 1))

    return {
        "beta":   coef[tidx] * scale,
        "se":     se[tidx]   * scale,
        "tstat":  tstat,
        "pval":   pval,
        "nobs":   n,
        "G":      G,
    }


# =============================================================================
# TEST A — LEAVE-ONE-TREATED-OUT (LOTO)
# =============================================================================

def run_loto(datasets, checks):
    """
    For each mining province, remove it from the data and re-estimate.
    Also runs the full sample as the baseline for comparison.

    Returns DataFrame with columns:
      outcome | sample | beta | se | tstat | pval | nobs | G
    where sample ∈ {"full", "drop_copperbelt", "drop_northwestern"}
    """
    rows  = []
    total = len(FOCUS_OUTCOMES)
    mining_list = sorted(MINING_PROVINCES)   # ["copperbelt", "northwestern"]

    print(f"\n  Running LOTO for {total} outcomes × "
          f"{len(mining_list)} drop-province configurations...")

    for i, (var, (ftype, controls, scale)) in enumerate(FOCUS_OUTCOMES.items(), 1):
        data = datasets.get(ftype)
        if data is None or var not in data.columns:
            print(f"  [{i}/{total}] {var}: no data")
            continue

        # ── Baseline: full sample ──────────────────────────────────────────
        r_full = run_one(data, var, ftype, controls, scale, checks, "FULL")
        if r_full is None:
            print(f"  [{i}/{total}] {var}: baseline skipped")
            continue

        rows.append({"outcome": var, "sample": "full", **r_full})

        # ── Drop one treated province at a time ───────────────────────────
        for drop_prov in mining_list:
            subset = data[data["province"] != drop_prov].copy()
            label  = f"drop_{drop_prov}"
            r = run_one(subset, var, ftype, controls, scale, checks, label)
            if r is None:
                print(f"  [{i}/{total}] {var} [{label}]: skipped")
                continue
            rows.append({"outcome": var, "sample": label, **r})

        # Console summary
        full_b = r_full["beta"]
        parts  = [f"β_full={full_b:+.3f}{stars(r_full['pval'])}"]
        for drop_prov in mining_list:
            match = [r for r in rows
                     if r["outcome"] == var and r["sample"] == f"drop_{drop_prov}"]
            if match:
                r = match[-1]
                parts.append(f"drop_{drop_prov[:4]}={r['beta']:+.3f}{stars(r['pval'])}")
        print(f"  [{i}/{total}] {var:<28}  " + "   ".join(parts))

    return pd.DataFrame(rows)


# =============================================================================
# TEST B — PLACEBO TREATMENT (PROVINCE PERMUTATION)
# =============================================================================

def build_placebo_data(data, fake_provinces):
    """
    Reassign mining status using fake_provinces (a frozenset of 2 province names).
    Recomputes bartik = fake_mining × ln_cu_price.
    Returns a modified copy of data — original is untouched.
    """
    d = data.copy()
    d["mining"] = d["province"].apply(
        lambda p: 1.0 if p in fake_provinces else 0.0
    )
    d["bartik"] = d["mining"] * d["ln_cu_price"]
    return d


def run_placebo(datasets, checks):
    """
    Enumerate all C(7,2)=21 unique 2-province combinations from non-mining
    provinces, plus N_PLACEBO_EXTRA random draws (default 0).
    For each combination, re-run the Bartik DiD for every focus outcome.

    Returns (placebo_df, pvalue_df):
      placebo_df — one row per (outcome × placebo_combo)
      pvalue_df  — summary: real β, placebo mean/SD, exact permutation p-value
    """
    non_mining = sorted(NON_MINING_PROVINCES)
    exact_combos = list(itertools.combinations(non_mining, 2))
    # C(7,2) = 21 exact combinations — run all of them
    all_combos = [frozenset(c) for c in exact_combos]

    if N_PLACEBO_EXTRA > 0:
        rng = np.random.default_rng(PLACEBO_SEED)
        extras = set()
        while len(extras) < N_PLACEBO_EXTRA:
            pair = frozenset(rng.choice(non_mining, size=2, replace=False))
            if pair not in set(all_combos):
                extras.add(pair)
        all_combos.extend(extras)

    n_combos = len(all_combos)
    total    = len(FOCUS_OUTCOMES)

    print(f"\n  Running placebo: {n_combos} province combinations "
          f"× {total} outcomes...")

    placebo_rows = []

    for ci, fake_pair in enumerate(all_combos, 1):
        label = "+".join(sorted(fake_pair))
        if ci % 5 == 0 or ci == 1 or ci == n_combos:
            print(f"    combo {ci:>3}/{n_combos}: {label}")

        for var, (ftype, controls, scale) in FOCUS_OUTCOMES.items():
            data = datasets.get(ftype)
            if data is None or var not in data.columns:
                continue

            placebo_data = build_placebo_data(data, fake_pair)
            r = run_one(placebo_data, var, ftype, controls, scale, checks,
                        f"PLACEBO[{label}]")
            if r is None:
                continue

            placebo_rows.append({
                "outcome":      var,
                "placebo_pair": label,
                "beta":         r["beta"],
                "se":           r["se"],
                "pval":         r["pval"],
                "nobs":         r["nobs"],
                "G":            r["G"],
            })

    placebo_df = pd.DataFrame(placebo_rows)

    # ── Compute real β for each outcome (from the full dataset) ──────────────
    real_rows = []
    for var, (ftype, controls, scale) in FOCUS_OUTCOMES.items():
        data = datasets.get(ftype)
        if data is None or var not in data.columns:
            continue
        r = run_one(data, var, ftype, controls, scale, checks, "REAL")
        if r is None:
            continue
        real_rows.append({"outcome": var, "beta_real": r["beta"],
                           "pval_analytic": r["pval"]})
    real_df = pd.DataFrame(real_rows)

    # ── Permutation p-value: fraction of placebo |β| ≥ |β_real| ─────────────
    pvalue_rows = []
    for _, row in real_df.iterrows():
        var       = row["outcome"]
        beta_real = row["beta_real"]
        sub       = placebo_df[placebo_df["outcome"] == var]["beta"].values
        if len(sub) == 0:
            continue
        # Phipson & Smyth (2010) +1 correction: avoids p=0 with small B,
        # treats the observed statistic as one additional permutation draw.
        # Lower bound = 1/(N+1) ≈ 0.045 with N=21 — conservative and correct.
        count  = int(np.sum(np.abs(sub) >= abs(beta_real)))
        p_perm = (count + 1) / (len(sub) + 1)
        pvalue_rows.append({
            "outcome":       var,
            "beta_real":     beta_real,
            "pval_analytic": row["pval_analytic"],
            "n_placebos":    len(sub),
            "placebo_mean":  float(np.mean(sub)),
            "placebo_sd":    float(np.std(sub)),
            "placebo_p95":   float(np.percentile(np.abs(sub), 95)),
            "pval_permutation": p_perm,
        })

    pvalue_df = pd.DataFrame(pvalue_rows)
    return placebo_df, pvalue_df


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def fmt_loto_table(loto_df):
    """Format leave-one-treated-out table for the paper."""
    lines = []
    lines.append("=" * 90)
    lines.append("TABLE A: LEAVE-ONE-TREATED-OUT (LOTO) ROBUSTNESS")
    lines.append("=" * 90)
    lines.append(
        "Drops each mining province in turn and re-estimates the main Bartik DiD.\n"
        "If the coefficient survives both drops, the result cannot be driven by\n"
        "a single treated province."
    )
    lines.append("")
    lines.append(
        f"  {'Outcome':<28} "
        f"{'Full sample':>18} "
        f"{'Drop Copperbelt':>18} "
        f"{'Drop N-Western':>18}"
    )
    lines.append("  " + "─" * 84)

    for var in FOCUS_OUTCOMES:
        sub = loto_df[loto_df["outcome"] == var]
        if sub.empty:
            continue

        def fmt_cell(sample_name):
            r = sub[sub["sample"] == sample_name]
            if r.empty:
                return "—".rjust(18)
            b = r.iloc[0]["beta"]
            p = r.iloc[0]["pval"]
            se = r.iloc[0]["se"]
            return f"{b:+.3f}{stars(p)} ({se:.3f})".rjust(18)

        line = (f"  {var:<28} "
                f"{fmt_cell('full')} "
                f"{fmt_cell('drop_copperbelt')} "
                f"{fmt_cell('drop_northwestern')}")
        lines.append(line)

    lines.append("")
    lines.append(
        "Notes: β scaled to percentage points (×100). Standard errors in parentheses.\n"
        "  *** p<0.01  ** p<0.05  * p<0.10. Province- and wave-FE throughout.\n"
        "  CR1S clustered SE (clustered at province level, G = 8 after dropping one)."
    )
    return "\n".join(lines)


def fmt_placebo_table(pvalue_df):
    """Format placebo permutation table for the paper."""
    lines = []
    lines.append("")
    lines.append("=" * 90)
    lines.append("TABLE B: PLACEBO TREATMENT — PROVINCE PERMUTATION TEST")
    lines.append("=" * 90)
    lines.append(
        "Randomly assigns 'mining' status to 2 of the 7 non-mining provinces.\n"
        f"All C(7,2) = 21 unique combinations enumerated exhaustively.\n"
        "Permutation p-value = fraction of |β_placebo| ≥ |β_real|.\n"
        "Interpretation: a small p-value means the true mining provinces produce\n"
        "a larger effect than 95% of random province assignments."
    )
    lines.append("")
    lines.append(
        f"  {'Outcome':<28} "
        f"{'β real':>10} "
        f"{'β mean':>10} "
        f"{'β SD':>8} "
        f"{'|β| p95':>9} "
        f"{'p (perm)':>10} "
        f"{'p (analytic)':>13}"
    )
    lines.append("  " + "─" * 92)

    for _, row in pvalue_df.iterrows():
        sig_marker = stars(row["pval_permutation"])
        lines.append(
            f"  {row['outcome']:<28} "
            f"{row['beta_real']:>+10.3f} "
            f"{row['placebo_mean']:>+10.3f} "
            f"{row['placebo_sd']:>8.3f} "
            f"{row['placebo_p95']:>9.3f} "
            f"{row['pval_permutation']:>9.3f}{sig_marker:<2} "
            f"{row['pval_analytic']:>12.3f}{stars(row['pval_analytic'])}"
        )

    lines.append("")
    lines.append(
        "Notes: β scaled to percentage points (×100).\n"
        "  Placebo mining = 2 of 7 non-mining provinces, exhaustive C(7,2)=21 combos.\n"
        "  p (perm) = (count + 1)/(N + 1) where count = #{|β_placebo| ≥ |β_real|},\n"
        "  N = 21. Phipson & Smyth (2010) correction: avoids p=0 with small B;\n"
        "  minimum attainable p-value = 1/22 ≈ 0.045.\n"
        "  *** p<0.01  ** p<0.05  * p<0.10."
    )
    return "\n".join(lines)


def print_loto_console(loto_df):
    """Print LOTO results to stdout."""
    print(f"\n  {'Outcome':<28} {'Full':>12} {'−Copperbelt':>14} {'−N-Western':>13}")
    print("  " + "─" * 70)
    for var in FOCUS_OUTCOMES:
        sub = loto_df[loto_df["outcome"] == var]
        if sub.empty:
            continue

        def cell(sample_name):
            r = sub[sub["sample"] == sample_name]
            if r.empty: return "—".rjust(12)
            b = r.iloc[0]["beta"]; p = r.iloc[0]["pval"]
            return f"{b:+.3f}{stars(p)}".rjust(12)

        print(f"  {var:<28} {cell('full')} {cell('drop_copperbelt'):>14} "
              f"{cell('drop_northwestern'):>13}")


def print_placebo_console(pvalue_df):
    """Print placebo summary to stdout."""
    print(f"\n  {'Outcome':<28} {'β_real':>9} {'placebo_SD':>11} {'p_perm':>9}")
    print("  " + "─" * 62)
    for _, row in pvalue_df.iterrows():
        print(f"  {row['outcome']:<28} "
              f"{row['beta_real']:>+9.3f} "
              f"{row['placebo_sd']:>11.3f} "
              f"{row['pval_permutation']:>8.3f}{stars(row['pval_permutation'])}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    t_start = time.time()
    checks  = []
    checks.append("=" * 70)
    checks.append("FEW-CLUSTER ROBUSTNESS: LOTO + PLACEBO PERMUTATION")
    checks.append("=" * 70)
    checks.append(f"  Mining provinces : {sorted(MINING_PROVINCES)}")
    checks.append(f"  Non-mining       : {sorted(NON_MINING_PROVINCES)}")
    checks.append(f"  Placebo combos   : C(7,2) = 21 exhaustive + "
                  f"{N_PLACEBO_EXTRA} random extra")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading data...")
    datasets = {}
    needed_types = {ftype for _, (ftype, _, _) in FOCUS_OUTCOMES.items()}
    for ftype in sorted(needed_types):
        print(f"  {ftype}...", end=" ", flush=True)
        datasets[ftype] = load_stack(ftype, checks)
        n = len(datasets[ftype]) if datasets[ftype] is not None else 0
        print(f"{n:,} obs")

    # ── TEST A: Leave-One-Treated-Out ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST A — LEAVE-ONE-TREATED-OUT (LOTO)")
    print("  Drops Copperbelt, then Northwestern, and re-estimates.")
    print("  Credibility check: do results survive losing each treated province?")
    print("=" * 70)
    checks.append("\n--- TEST A: Leave-One-Treated-Out ---")

    loto_df = run_loto(datasets, checks)
    loto_df.to_csv(OUT_DIR / "loto_results.csv", index=False)

    print_loto_console(loto_df)
    print(f"\n  Saved: loto_results.csv ({len(loto_df)} rows)")

    # ── TEST B: Placebo Treatment ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST B — PLACEBO TREATMENT (PROVINCE PERMUTATION)")
    print("  Assigns fake mining status to 2 of 7 non-mining provinces.")
    print(f"  Exhaustive: C(7,2) = 21 combinations. Seed={PLACEBO_SEED}.")
    print("=" * 70)
    checks.append("\n--- TEST B: Placebo Treatment ---")

    placebo_df, pvalue_df = run_placebo(datasets, checks)
    placebo_df.to_csv(OUT_DIR / "placebo_results.csv", index=False)
    pvalue_df.to_csv(OUT_DIR / "placebo_pvalues.csv", index=False)

    print_placebo_console(pvalue_df)
    print(f"\n  Saved: placebo_results.csv ({len(placebo_df)} rows)")
    print(f"  Saved: placebo_pvalues.csv  ({len(pvalue_df)} outcomes)")

    # ── Write formatted tables ────────────────────────────────────────────────
    table_text = fmt_loto_table(loto_df) + "\n\n" + fmt_placebo_table(pvalue_df)

    with open(OUT_DIR / "few_cluster_tables.txt", "w", encoding="utf-8") as f:
        f.write(table_text)
    with open(OUT_DIR / "few_cluster_checks.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(checks))

    print(f"\n  Saved: few_cluster_tables.txt")
    print(f"  Saved: few_cluster_checks.txt")

    # ── Referee-ready summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("REFEREE-READY SUMMARY")
    print("=" * 70)

    # Count how many outcomes survive both LOTO drops
    loto_survive = 0
    for var in FOCUS_OUTCOMES:
        sub = loto_df[loto_df["outcome"] == var]
        drops = sub[sub["sample"].str.startswith("drop_")]
        if len(drops) == 2 and all(drops["pval"] < 0.10):
            loto_survive += 1
    print(f"  LOTO: {loto_survive}/{len(FOCUS_OUTCOMES)} outcomes significant "
          f"after dropping EITHER treated province (p<0.10).")

    # Count placebo p-values < 0.10
    if not pvalue_df.empty:
        n_sig_perm = int((pvalue_df["pval_permutation"] < 0.10).sum())
        print(f"  Placebo: {n_sig_perm}/{len(pvalue_df)} outcomes have "
              f"permutation p<0.10 (|β_real| exceeds 90% of placebo distribution).")
        n_both = 0
        for var in FOCUS_OUTCOMES:
            pv = pvalue_df[pvalue_df["outcome"] == var]
            if pv.empty:
                continue
            perm_sig = pv.iloc[0]["pval_permutation"] < 0.10
            loto_sub = loto_df[(loto_df["outcome"] == var) &
                                (loto_df["sample"].str.startswith("drop_"))]
            loto_sig = len(loto_sub) == 2 and all(loto_sub["pval"] < 0.10)
            if perm_sig and loto_sig:
                n_both += 1
        print(f"  Both tests pass: {n_both}/{len(FOCUS_OUTCOMES)} outcomes — "
              f"these are the most defensible results in the paper.")

    total_time = time.time() - t_start
    print(f"\nDone in {total_time/60:.1f} min.")
    print("\nOutputs written to:", OUT_DIR)
    for fname in ["loto_results.csv", "placebo_results.csv",
                  "placebo_pvalues.csv", "few_cluster_tables.txt",
                  "few_cluster_checks.txt"]:
        print(f"  {fname}")


if __name__ == "__main__":
    try:
        import pyreadstat, scipy
    except ImportError:
        print("Required: pip install pandas numpy scipy pyreadstat")
        sys.exit(1)
    main()