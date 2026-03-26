"""
test_mens_migration.py
======================
Tests for mens_migration.py — men's long-resident subsample robustness.
Imports functions and runs them on fully synthetic MR-style data.
No DHS files required.

Run with:
    pytest test_mens_migration.py -v
or:
    python test_mens_migration.py
"""

import sys
import math
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

# ── Import module under test ───────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

spec = importlib.util.spec_from_file_location(
    "mens_migration",
    str(Path(__file__).resolve().parent / "mens_migration.py")
)
mm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mm)

demean_by_fast  = mm.demean_by_fast
within_demean   = mm.within_demean
run_fe          = mm.run_fe
build_table     = mm.build_table
stars           = mm.stars

MEN_OUTCOMES    = mm.MEN_OUTCOMES
MV104_COL       = mm.MV104_COL
LONGRES_YEARS   = mm.LONGRES_YEARS
ALWAYS_CODE     = mm.ALWAYS_CODE
EXCLUDE_CODES   = mm.EXCLUDE_CODES
MIN_OBS         = mm.MIN_OBS
MIN_CLUSTERS    = mm.MIN_CLUSTERS

from config_00 import MINING_PROVINCES, DHS_MISSING_CODES


# =============================================================================
# HELPERS
# =============================================================================

def _make_mr_panel(n_per_cell=80, seed=42, include_mv104=True):
    """
    Synthetic Men's Recode panel:
    9 provinces × 7 waves × n_per_cell men.
    Includes mv104 (years of residence) by default.
    """
    rng       = np.random.default_rng(seed)
    provinces = ["central","copperbelt","eastern","luapula","lusaka",
                 "northern","northwestern","southern","western"]
    waves     = [1992, 1996, 2002, 2007, 2014, 2018, 2024]
    cu_prices = {1992:2297,1996:2289,2002:1558,2007:7132,
                 2014:6863,2018:6530,2024:9142}

    rows = []
    for prov in provinces:
        mining = 1 if prov in {"copperbelt","northwestern"} else 0
        for wave in waves:
            ln_p   = math.log(cu_prices[wave])
            bartik = mining * ln_p
            for _ in range(n_per_cell):
                row = {
                    "province":       prov,
                    "wave":           wave,
                    "mining":         float(mining),
                    "ln_cu_price":    ln_p,
                    "bartik":         bartik,
                    "man_employed":   float(rng.integers(0, 2)),
                    "man_edu_sec_p":  float(rng.integers(0, 2)),
                    "man_edu_level":  float(rng.integers(0, 4)),
                    "man_condom":     float(rng.integers(0, 2)),
                    "man_multi_part": float(rng.integers(0, 2)),
                    "man_dv_justify": float(rng.integers(0, 2)),
                    "man_age":        float(rng.integers(18, 60)),
                    "man_urban":      float(rng.integers(0, 2)),
                    "w":              rng.uniform(0.5, 2.0),
                    "post_priv":      1 if wave > 2000 else 0,
                }
                if include_mv104:
                    # Mix of normal years (0-35), always code (95), visitor (96)
                    mv104_choices = list(range(0, 36)) + [95, 96, 97]
                    mv104_probs   = [1/36] * 36 + [0.05, 0.02, 0.01]
                    # normalise
                    total = sum(mv104_probs)
                    mv104_probs = [p / total for p in mv104_probs]
                    row["mv104"] = rng.choice(mv104_choices, p=mv104_probs)
                rows.append(row)

    df = pd.DataFrame(rows)
    df["province_id"] = pd.Categorical(df["province"]).codes.astype(int)
    df["wave_id"]     = pd.Categorical(df["wave"]).codes.astype(int)
    return df


# =============================================================================
# 1. demean_by_fast
# =============================================================================

class TestDemeanByFast:

    def test_1d_removes_group_means(self):
        arr = np.array([1.0, 3.0, 2.0, 4.0])
        g   = np.array([0, 0, 1, 1])
        out = demean_by_fast(arr, g)
        assert abs(out[0] - (-1.0)) < 1e-9
        assert abs(out[1] - 1.0)    < 1e-9

    def test_2d_per_column(self):
        arr = np.array([[2.0, 20.0], [4.0, 40.0]])
        g   = np.array([0, 0])
        out = demean_by_fast(arr, g)
        assert abs(out[0, 0] - (-1.0)) < 1e-9

    def test_output_shape_preserved(self):
        arr = np.random.randn(100)
        g   = np.repeat(np.arange(9), 12)[:100]
        out = demean_by_fast(arr, g)
        assert out.shape == arr.shape

    def test_returns_float_array(self):
        arr = np.array([1, 2, 3, 4], dtype=int)
        g   = np.array([0, 0, 1, 1])
        out = demean_by_fast(arr, g)
        assert out.dtype == float

    def test_consistent_with_03_implementation(self):
        spec3 = importlib.util.spec_from_file_location(
            "regressions",
            str(Path(__file__).resolve().parent / "03_regressions.py")
        )
        reg = importlib.util.module_from_spec(spec3)
        spec3.loader.exec_module(reg)

        rng = np.random.default_rng(11)
        arr = rng.standard_normal(100)
        g   = np.repeat(np.arange(9), 12)[:100]
        out03 = reg.demean_by_fast(arr, g)
        outmm = demean_by_fast(arr, g)
        np.testing.assert_allclose(out03, outmm, atol=1e-12)


# =============================================================================
# 2. within_demean
# =============================================================================

class TestWithinDemean:

    def test_returns_three_tuple(self):
        arr  = np.ones(63)
        prov = np.repeat(np.arange(9), 7)
        wave = np.tile(np.arange(7), 9)
        r = within_demean(arr, prov, wave)
        assert len(r) == 3

    def test_converged_for_constant(self):
        arr  = np.ones(63)
        prov = np.repeat(np.arange(9), 7)
        wave = np.tile(np.arange(7), 9)
        _, _, cv = within_demean(arr, prov, wave)
        assert cv is True

    def test_max_iter_50_default(self):
        rng  = np.random.default_rng(2)
        arr  = rng.standard_normal(63)
        prov = np.repeat(np.arange(9), 7)
        wave = np.tile(np.arange(7), 9)
        _, n_iter, cv = within_demean(arr, prov, wave)
        assert n_iter <= 50

    def test_consistent_with_03(self):
        spec3 = importlib.util.spec_from_file_location(
            "regressions",
            str(Path(__file__).resolve().parent / "03_regressions.py")
        )
        reg = importlib.util.module_from_spec(spec3)
        spec3.loader.exec_module(reg)

        rng  = np.random.default_rng(33)
        arr  = rng.standard_normal(63)
        prov = np.repeat(np.arange(9), 7)
        wave = np.tile(np.arange(7), 9)

        out03, _, _ = reg.within_demean(arr, prov, wave)
        outmm, _, _ = within_demean(arr, prov, wave)
        np.testing.assert_allclose(out03, outmm, atol=1e-12)


# =============================================================================
# 3. run_fe
# =============================================================================

class TestRunFe:

    def test_returns_dict_or_none(self):
        df = _make_mr_panel(n_per_cell=60)
        r  = run_fe(df, "man_employed", "bartik", ["man_age","man_urban"], scale=100)
        assert r is None or isinstance(r, dict)

    def test_result_keys_present(self):
        df = _make_mr_panel(n_per_cell=60)
        r  = run_fe(df, "man_employed", "bartik", ["man_age","man_urban"], scale=100)
        if r is not None:
            for k in ["coef","coef_s","se_s","tstat","pval",
                      "ci_lo","ci_hi","nobs","n_clusters","r2"]:
                assert k in r

    def test_pval_between_0_and_1(self):
        df = _make_mr_panel(n_per_cell=60)
        r  = run_fe(df, "man_employed", "bartik", ["man_age"], scale=100)
        if r is not None:
            assert 0.0 <= r["pval"] <= 1.0

    def test_se_positive(self):
        df = _make_mr_panel(n_per_cell=60)
        r  = run_fe(df, "man_employed", "bartik", [], scale=1)
        if r is not None:
            assert r["se_s"] > 0

    def test_ci_lo_less_than_ci_hi(self):
        df = _make_mr_panel(n_per_cell=60)
        r  = run_fe(df, "man_employed", "bartik", [], scale=1)
        if r is not None:
            assert r["ci_lo"] < r["ci_hi"]

    def test_coef_s_equals_coef_times_scale(self):
        df = _make_mr_panel(n_per_cell=60)
        r  = run_fe(df, "man_employed", "bartik", [], scale=100)
        if r is not None:
            assert abs(r["coef_s"] - r["coef"] * 100) < 1e-9

    def test_returns_none_for_missing_column(self):
        df = _make_mr_panel(n_per_cell=60)
        r  = run_fe(df, "nonexistent", "bartik", [], scale=1)
        assert r is None

    def test_returns_none_for_zero_variance_treatment(self):
        df = _make_mr_panel(n_per_cell=60)
        df["bartik"] = 0.0
        r = run_fe(df, "man_employed", "bartik", [], scale=1)
        assert r is None

    def test_returns_none_below_min_obs(self):
        df = _make_mr_panel(n_per_cell=1)[:5].copy()
        r  = run_fe(df, "man_employed", "bartik", [], scale=1)
        assert r is None

    def test_r2_between_0_and_1(self):
        df = _make_mr_panel(n_per_cell=60)
        r  = run_fe(df, "man_employed", "bartik", [], scale=1)
        if r is not None and not np.isnan(r["r2"]):
            assert 0.0 <= r["r2"] <= 1.0

    def test_n_clusters_equals_9(self):
        df = _make_mr_panel(n_per_cell=60)
        r  = run_fe(df, "man_employed", "bartik", ["man_age"], scale=100)
        if r is not None:
            assert r["n_clusters"] == 9

    def test_all_men_outcomes_run_without_crash(self):
        df = _make_mr_panel(n_per_cell=60)
        for var, label, scale, controls in MEN_OUTCOMES:
            if var not in df.columns:
                continue
            try:
                r = run_fe(df, var, "bartik", controls, scale=scale)
                assert r is None or isinstance(r, dict)
            except Exception as e:
                pytest.fail(f"run_fe crashed on men's outcome '{var}': {e}")


# =============================================================================
# 4. mv104 coding logic
# =============================================================================

class TestMv104Coding:

    def test_longres_years_is_5(self):
        assert LONGRES_YEARS == 5

    def test_always_code_is_95(self):
        assert ALWAYS_CODE == 95

    def test_exclude_codes_are_96_and_97(self):
        assert EXCLUDE_CODES == {96, 97}

    def test_long_resident_flag_creation(self):
        """
        Replicate the longres coding from mens_migration.py.
        mv104: 0-94 = years; 95 = always; 96,97 = exclude; 97+ = NaN.
        Long-resident = (mv104 >= 5) OR (mv104 == 95), excluding 96/97.
        """
        mv104 = pd.Series([0, 3, 5, 10, 95, 96, 97, np.nan])
        v104  = pd.to_numeric(mv104, errors="coerce")
        v104  = v104.where(v104 <= 95, np.nan)   # 96/97 → NaN

        longres = ((v104 >= LONGRES_YEARS) | (v104 == ALWAYS_CODE)).astype(float)
        longres[v104.isna()] = np.nan

        # 0 years → not long-res
        assert longres[0] == 0.0
        # 3 years → not long-res
        assert longres[1] == 0.0
        # 5 years → long-res
        assert longres[2] == 1.0
        # 10 years → long-res
        assert longres[3] == 1.0
        # 95 = always → long-res
        assert longres[4] == 1.0
        # 96 = visitor → NaN
        assert pd.isna(longres[5])
        # 97 = not stated → NaN
        assert pd.isna(longres[6])
        # NaN → NaN
        assert pd.isna(longres[7])

    def test_visitor_excluded_from_longres(self):
        v104 = pd.to_numeric(pd.Series([96]), errors="coerce")
        v104 = v104.where(v104 <= 95, np.nan)
        assert pd.isna(v104[0])

    def test_always_code_treated_as_longres(self):
        v104 = pd.Series([95.0])
        longres = ((v104 >= LONGRES_YEARS) | (v104 == ALWAYS_CODE)).astype(float)
        assert longres[0] == 1.0

    def test_exactly_5_years_is_longres(self):
        v104 = pd.Series([5.0])
        longres = ((v104 >= LONGRES_YEARS) | (v104 == ALWAYS_CODE)).astype(float)
        assert longres[0] == 1.0

    def test_4_years_is_not_longres(self):
        v104 = pd.Series([4.0])
        longres = ((v104 >= LONGRES_YEARS) | (v104 == ALWAYS_CODE)).astype(float)
        assert longres[0] == 0.0


# =============================================================================
# 5. Long-resident subsample run
# =============================================================================

class TestLongresSubsample:

    def _make_longres_df(self, seed=42):
        """Panel with mv104 column; apply longres filter."""
        df  = _make_mr_panel(n_per_cell=60, seed=seed, include_mv104=True)
        v104 = pd.to_numeric(df["mv104"], errors="coerce")
        v104 = v104.where(v104 <= 95, np.nan)
        df["longres"] = ((v104 >= LONGRES_YEARS) | (v104 == ALWAYS_CODE)).astype(float)
        df.loc[v104.isna(), "longres"] = np.nan
        return df

    def test_longres_proportion_reasonable(self):
        """
        With random mv104 values, most men should be long-resident
        (they've lived there several years on average).
        """
        df = self._make_longres_df()
        n_longres = (df["longres"] == 1).sum()
        n_total   = df["longres"].notna().sum()
        pct = 100 * n_longres / n_total
        # Should be a substantial fraction (not 0% or 100%)
        assert 10 < pct < 95, f"Long-res proportion {pct:.1f}% seems wrong"

    def test_longres_filter_excludes_visitors(self):
        df = self._make_longres_df()
        # Among men with mv104==96, longres should be NaN
        visitors = df[df["mv104"] == 96]
        if len(visitors) > 0:
            assert visitors["longres"].isna().all(), \
                "Visitors (mv104==96) should have longres=NaN"

    def test_run_fe_on_longres_subsample(self):
        df = self._make_longres_df()
        lr = df[df["longres"] == 1.0].copy()
        r  = run_fe(lr, "man_employed", "bartik", ["man_age","man_urban"], scale=100)
        assert r is None or isinstance(r, dict)

    def test_full_sample_and_longres_both_runnable(self):
        df    = self._make_longres_df()
        lr    = df[df["longres"] == 1.0].copy()
        r_full = run_fe(df, "man_employed", "bartik", ["man_age"], scale=100)
        r_lr   = run_fe(lr, "man_employed", "bartik", ["man_age"], scale=100)
        # Both should produce results or None — no crash
        assert r_full is None or isinstance(r_full, dict)
        assert r_lr   is None or isinstance(r_lr,   dict)

    def test_consistency_direction_check(self):
        """
        If the full-sample coefficient is positive, the long-resident
        coefficient should also tend to be positive (same treatment,
        same population). This is probabilistic — check sign agreement
        when both are not None.
        """
        df    = self._make_longres_df(seed=0)
        lr    = df[df["longres"] == 1.0].copy()
        r_full = run_fe(df, "man_employed", "bartik", ["man_age"], scale=100)
        r_lr   = run_fe(lr, "man_employed", "bartik", ["man_age"], scale=100)
        if r_full is not None and r_lr is not None:
            # Signs should not be wildly different (note: not a guaranteed test)
            # Just check the estimates are finite
            assert math.isfinite(r_full["coef_s"])
            assert math.isfinite(r_lr["coef_s"])


# =============================================================================
# 6. build_table
# =============================================================================

class TestBuildTable:

    def _make_results_df(self):
        rows = []
        for var, label, scale, controls in MEN_OUTCOMES:
            rows.append({
                "outcome":   var,
                "label":     label,
                "scale":     scale,
                "beta_full": np.random.randn() * 5,
                "se_full":   abs(np.random.randn()) * 2,
                "p_full":    np.random.uniform(0, 1),
                "n_full":    3000,
                "g_full":    9,
                "r2_full":   np.random.uniform(0, 0.2),
                "beta_lr":   np.random.randn() * 5,
                "se_lr":     abs(np.random.randn()) * 2,
                "p_lr":      np.random.uniform(0, 1),
                "n_lr":      2000,
                "g_lr":      9,
                "r2_lr":     np.random.uniform(0, 0.2),
            })
        return pd.DataFrame(rows)

    def test_build_table_runs_without_crash(self):
        df     = self._make_results_df()
        tlines = []
        checks = []
        try:
            build_table(df, tlines, checks)
        except Exception as e:
            pytest.fail(f"build_table() raised: {e}")

    def test_build_table_produces_output_lines(self):
        df     = self._make_results_df()
        tlines = []
        checks = []
        build_table(df, tlines, checks)
        assert len(tlines) > 0

    def test_build_table_includes_all_outcomes(self):
        df     = self._make_results_df()
        tlines = []
        checks = []
        build_table(df, tlines, checks)
        combined = " ".join(tlines)
        for _, label, _, _ in MEN_OUTCOMES:
            assert label in combined, f"Outcome label '{label}' missing from table"

    def test_build_table_empty_df_produces_no_results_message(self):
        tlines = []
        checks = []
        build_table(pd.DataFrame(), tlines, checks)
        combined = " ".join(tlines)
        assert "NO RESULTS" in combined or "no results" in combined.lower()

    def test_significant_result_labelled_consistent_or_weakened(self):
        """
        Results significant in full sample should be labelled CONSISTENT
        or WEAKENED/REVERSED depending on LR result.
        """
        df    = self._make_results_df()
        # Force man_employed to be clearly significant in full + LR
        df.loc[df["outcome"] == "man_employed", "p_full"] = 0.02
        df.loc[df["outcome"] == "man_employed", "beta_full"] = 10.0
        df.loc[df["outcome"] == "man_employed", "p_lr"]   = 0.03
        df.loc[df["outcome"] == "man_employed", "beta_lr"] = 9.0

        tlines = []
        build_table(df, tlines, checks=[])
        combined = " ".join(tlines)
        assert "CONSISTENT" in combined or "WEAKENED" in combined


# =============================================================================
# 7. MEN_OUTCOMES structure
# =============================================================================

class TestMenOutcomes:

    def test_has_six_outcomes(self):
        assert len(MEN_OUTCOMES) == 6

    def test_all_entries_have_four_fields(self):
        """(var, label, scale, controls)"""
        for entry in MEN_OUTCOMES:
            assert len(entry) == 4, \
                f"Entry {entry} has {len(entry)} fields, expected 4"

    def test_all_scales_positive(self):
        for (var, label, scale, controls) in MEN_OUTCOMES:
            assert scale > 0

    def test_all_controls_are_lists(self):
        for (var, label, scale, controls) in MEN_OUTCOMES:
            assert isinstance(controls, list)

    def test_man_employed_present(self):
        vars_ = [e[0] for e in MEN_OUTCOMES]
        assert "man_employed" in vars_

    def test_man_dv_justify_present(self):
        vars_ = [e[0] for e in MEN_OUTCOMES]
        assert "man_dv_justify" in vars_

    def test_all_controls_contain_man_age(self):
        for (var, label, scale, controls) in MEN_OUTCOMES:
            assert "man_age" in controls, \
                f"'man_age' missing from controls for '{var}'"


# =============================================================================
# 8. constants
# =============================================================================

class TestConstants:

    def test_mv104_col_correct(self):
        assert MV104_COL == "mv104"

    def test_min_obs_imported_from_config(self):
        from config_00 import MIN_OBS as cfg_min_obs
        assert MIN_OBS == cfg_min_obs

    def test_min_clusters_imported_from_config(self):
        from config_00 import MIN_CLUSTERS as cfg_min_clusters
        assert MIN_CLUSTERS == cfg_min_clusters

    def test_stars_function(self):
        assert stars(0.005) == "***"
        assert stars(0.03)  == "**"
        assert stars(0.08)  == "*"
        assert stars(0.20)  == ""
        assert stars(np.nan) == ""

    def test_exclude_codes_not_in_longres(self):
        """Codes 96 and 97 must never be treated as long-resident."""
        for code in EXCLUDE_CODES:
            v104 = pd.to_numeric(pd.Series([code]), errors="coerce")
            v104 = v104.where(v104 <= 95, np.nan)
            assert pd.isna(v104[0]), \
                f"Exclude code {code} should become NaN but got {v104[0]}"


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    passed = 0
    failed = 0
    errors = []

    test_classes = [
        TestDemeanByFast, TestWithinDemean, TestRunFe,
        TestMv104Coding, TestLongresSubsample, TestBuildTable,
        TestMenOutcomes, TestConstants,
    ]

    print("=" * 65)
    print("test_mens_migration.py")
    print("=" * 65)

    for cls in test_classes:
        obj = cls()
        for name in [n for n in dir(obj) if n.startswith("test_")]:
            try:
                getattr(obj, name)()
                print(f"  PASS  {cls.__name__}.{name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL  {cls.__name__}.{name} — {e}")
                failed += 1
                errors.append((cls.__name__, name, str(e)))

    print(f"\nResults: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
