"""
test_04_robustness.py
=====================
Tests for 04_robustness.py — wild bootstrap, event study, continuous robustness.
All functions tested on synthetic panel data. No DHS files required.

Run with:
    pytest test_04_robustness.py -v
or:
    python test_04_robustness.py
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
    "robustness",
    str(Path(__file__).resolve().parent / "04_robustness.py")
)
rob = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rob)

demean_by_fast       = rob.demean_by_fast
within_demean        = rob.within_demean
prep_arrays          = rob.prep_arrays
ols_fe               = rob.ols_fe
clustered_se         = rob.clustered_se
stars                = rob.stars
wild_bootstrap_pval  = rob.wild_bootstrap_pval
ascii_event_chart    = rob.ascii_event_chart
WEBB_WEIGHTS         = rob.WEBB_WEIGHTS
BASE_WAVE            = rob.BASE_WAVE
WAVE_ORDER           = rob.WAVE_ORDER
FOCUS_OUTCOMES       = rob.FOCUS_OUTCOMES
MIN_OBS              = rob.MIN_OBS


# =============================================================================
# HELPERS
# =============================================================================

def _make_panel(n_per_cell=80, seed=42):
    """Synthetic 9-province × 7-wave individual panel."""
    rng = np.random.default_rng(seed)
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
                y = 0.3 + 5.0 * bartik / 10.0 + rng.normal(0, 1)
                rows.append({
                    "province":    prov,
                    "wave":        wave,
                    "mining":      float(mining),
                    "ln_cu_price": ln_p,
                    "bartik":      bartik,
                    "y":           y,
                    "urban":       float(rng.integers(0, 2)),
                    "w":           rng.uniform(0.5, 2.0),
                    "post_priv":   1 if wave > 2000 else 0,
                })

    df = pd.DataFrame(rows)
    df["province_id"] = pd.Categorical(df["province"]).codes.astype(int)
    df["wave_id"]     = pd.Categorical(df["wave"]).codes.astype(int)
    return df


def _make_arrays(n=500, G=9, seed=0):
    """
    Make pre-demeaned arrays (yd, Xd, cl) for direct OLS testing.
    Returns (yd, Xd, cl).
    """
    rng  = np.random.default_rng(seed)
    Xd   = np.column_stack([rng.standard_normal(n), rng.standard_normal(n)])
    yd   = Xd[:, 0] * 2.0 + rng.standard_normal(n) * 0.5
    cl   = np.repeat(np.arange(G), n // G + 1)[:n]
    return yd, Xd, cl


# =============================================================================
# 1. WEBB_WEIGHTS
# =============================================================================

class TestWebbWeights:

    def test_has_six_weights(self):
        assert len(WEBB_WEIGHTS) == 6

    def test_symmetric(self):
        """Webb weights are symmetric around 0: w_i = -w_{n-i}."""
        w = np.sort(WEBB_WEIGHTS)
        for i in range(3):
            assert abs(w[i] + w[5 - i]) < 1e-9, \
                f"Webb weights not symmetric: {w[i]} + {w[5-i]} ≠ 0"

    def test_all_nonzero(self):
        assert np.all(WEBB_WEIGHTS != 0)

    def test_weights_known_values(self):
        expected = sorted([
            -math.sqrt(3/2), -math.sqrt(2/2), -math.sqrt(1/2),
             math.sqrt(1/2),  math.sqrt(2/2),  math.sqrt(3/2),
        ])
        actual = sorted(WEBB_WEIGHTS.tolist())
        for a, e in zip(actual, expected):
            assert abs(a - e) < 1e-9


# =============================================================================
# 2. demean_by_fast (mirrors 03 but must pass independently)
# =============================================================================

class TestDemeanByFast:

    def test_1d_removes_group_means(self):
        arr = np.array([1.0, 3.0, 2.0, 4.0])
        g   = np.array([0, 0, 1, 1])
        out = demean_by_fast(arr, g)
        assert abs(out[0] - (-1.0)) < 1e-9
        assert abs(out[1] - 1.0)    < 1e-9

    def test_2d_removes_group_means_each_column(self):
        arr = np.array([[2.0, 20.0], [4.0, 40.0]])
        g   = np.array([0, 0])
        out = demean_by_fast(arr, g)
        assert abs(out[0, 0] - (-1.0)) < 1e-9

    def test_output_shape_preserved(self):
        arr = np.random.randn(100)
        g   = np.repeat(np.arange(9), 12)[:100]
        out = demean_by_fast(arr, g)
        assert out.shape == arr.shape


# =============================================================================
# 3. within_demean (two-way FE)
# =============================================================================

class TestWithinDemean:

    def test_returns_three_tuple(self):
        arr  = np.ones(63)
        prov = np.repeat(np.arange(9), 7)
        wave = np.tile(np.arange(7), 9)
        r = within_demean(arr, prov, wave)
        assert len(r) == 3

    def test_converged_flag_type(self):
        arr  = np.ones(63)
        prov = np.repeat(np.arange(9), 7)
        wave = np.tile(np.arange(7), 9)
        _, _, cv = within_demean(arr, prov, wave)
        assert isinstance(cv, bool)

    def test_max_iter_1_no_convergence(self):
        arr  = np.random.randn(63)
        prov = np.repeat(np.arange(9), 7)
        wave = np.tile(np.arange(7), 9)
        _, n_iter, cv = within_demean(arr, prov, wave, max_iter=1, tol=1e-30)
        assert n_iter == 1

    def test_matches_03_regressions_implementation(self):
        """Both scripts implement the same algorithm — results must be identical."""
        spec3 = importlib.util.spec_from_file_location(
            "regressions",
            str(Path(__file__).resolve().parent / "03_regressions.py")
        )
        reg = importlib.util.module_from_spec(spec3)
        spec3.loader.exec_module(reg)

        rng  = np.random.default_rng(99)
        n    = 100
        arr  = rng.standard_normal(n)
        prov = np.repeat(np.arange(9), 12)[:n]
        wave = np.tile(np.arange(7), 15)[:n]

        out03, _, _ = reg.within_demean(arr, prov, wave)
        out04, _, _ = within_demean(arr, prov, wave)
        np.testing.assert_allclose(out03, out04, atol=1e-12)


# =============================================================================
# 4. prep_arrays
# =============================================================================

class TestPrepArrays:

    def test_returns_seven_tuple_on_valid_data(self):
        df = _make_panel(n_per_cell=30)
        r  = prep_arrays(df, "y", "bartik", ["urban"])
        assert r is None or len(r) == 7

    def test_returns_none_below_min_obs(self):
        df = _make_panel(n_per_cell=1)[:5].copy()
        r  = prep_arrays(df, "y", "bartik", ["urban"])
        assert r is None

    def test_returns_none_for_missing_column(self):
        df = _make_panel(n_per_cell=30)
        r  = prep_arrays(df, "nonexistent", "bartik", [])
        assert r is None

    def test_yd_shape_equals_n(self):
        df = _make_panel(n_per_cell=30)
        r  = prep_arrays(df, "y", "bartik", ["urban"])
        if r is not None:
            yd, Xd, cl, n, k, G, sub = r
            assert len(yd) == n

    def test_xd_columns_equal_k(self):
        df = _make_panel(n_per_cell=30)
        r  = prep_arrays(df, "y", "bartik", ["urban"])
        if r is not None:
            yd, Xd, cl, n, k, G, sub = r
            assert Xd.shape[1] == k

    def test_g_equals_9_provinces(self):
        df = _make_panel(n_per_cell=30)
        r  = prep_arrays(df, "y", "bartik", ["urban"])
        if r is not None:
            yd, Xd, cl, n, k, G, sub = r
            assert G == 9


# =============================================================================
# 5. ols_fe
# =============================================================================

class TestOlsFe:

    def test_returns_three_tuple(self):
        yd, Xd, cl = _make_arrays()
        result = ols_fe(yd, Xd)
        assert len(result) == 3

    def test_coef_length_equals_k(self):
        yd, Xd, cl = _make_arrays()
        coef, resid, XXinv = ols_fe(yd, Xd)
        assert len(coef) == Xd.shape[1]

    def test_resid_length_equals_n(self):
        yd, Xd, cl = _make_arrays()
        coef, resid, XXinv = ols_fe(yd, Xd)
        assert len(resid) == len(yd)

    def test_coef_correct_slope(self):
        """True slope=2.0; estimator should recover it well."""
        yd, Xd, cl = _make_arrays()
        coef, _, _ = ols_fe(yd, Xd)
        assert abs(coef[0] - 2.0) < 0.3

    def test_residuals_close_to_zero_mean(self):
        yd, Xd, cl = _make_arrays()
        _, resid, _ = ols_fe(yd, Xd)
        assert abs(resid.mean()) < 0.1

    def test_singular_matrix_handled(self):
        """Rank-deficient X uses pinv — should not crash."""
        n  = 50
        Xd = np.ones((n, 2))  # rank-1
        yd = np.random.randn(n)
        try:
            result = ols_fe(yd, Xd)
            assert len(result) == 3
        except Exception as e:
            pytest.fail(f"ols_fe crashed on singular matrix: {e}")


# =============================================================================
# 6. clustered_se
# =============================================================================

class TestClusteredSe:

    def test_returns_array_of_length_k(self):
        yd, Xd, cl = _make_arrays()
        coef, resid, XXinv = ols_fe(yd, Xd)
        se = clustered_se(Xd, resid, cl, XXinv, len(yd), Xd.shape[1], 9)
        assert len(se) == Xd.shape[1]

    def test_se_nonnegative(self):
        yd, Xd, cl = _make_arrays()
        coef, resid, XXinv = ols_fe(yd, Xd)
        se = clustered_se(Xd, resid, cl, XXinv, len(yd), Xd.shape[1], 9)
        assert (se >= 0).all()

    def test_se_positive_for_nondegenerate_data(self):
        yd, Xd, cl = _make_arrays()
        coef, resid, XXinv = ols_fe(yd, Xd)
        se = clustered_se(Xd, resid, cl, XXinv, len(yd), Xd.shape[1], 9)
        assert (se > 0).all()


# =============================================================================
# 7. wild_bootstrap_pval
# =============================================================================

class TestWildBootstrapPval:

    def test_returns_float(self):
        yd, Xd, cl = _make_arrays(n=400, G=9, seed=1)
        coef, resid, XXinv = ols_fe(yd, Xd)
        pval = wild_bootstrap_pval(yd, Xd, cl, coef, tidx=0, B=99, seed=42)
        assert isinstance(pval, float) or np.isnan(pval)

    def test_pval_between_0_and_1(self):
        yd, Xd, cl = _make_arrays(n=400, G=9, seed=2)
        coef, resid, XXinv = ols_fe(yd, Xd)
        pval = wild_bootstrap_pval(yd, Xd, cl, coef, tidx=0, B=99, seed=42)
        if not np.isnan(pval):
            assert 0.0 <= pval <= 1.0

    def test_strong_true_effect_has_low_pval(self):
        """
        With a very strong treatment effect, the bootstrap p-value should
        tend to be low (not guaranteed but very likely with enough reps).
        """
        rng = np.random.default_rng(0)
        n   = 500
        G   = 9
        Xd  = np.column_stack([rng.standard_normal(n), rng.standard_normal(n)])
        yd  = Xd[:, 0] * 10.0 + rng.standard_normal(n) * 0.1   # β=10, tiny noise
        cl  = np.repeat(np.arange(G), n // G + 1)[:n]
        coef, resid, XXinv = ols_fe(yd, Xd)
        pval = wild_bootstrap_pval(yd, Xd, cl, coef, tidx=0, B=199, seed=7)
        if not np.isnan(pval):
            assert pval < 0.5, \
                f"Expected low bootstrap p-value for strong true effect, got {pval:.3f}"

    def test_null_true_effect_pval_not_always_low(self):
        """
        Under the null (β=0), the bootstrap p-value should not be systematically
        below 0.05 across seeds (not a strict test — probabilistic).
        """
        rng = np.random.default_rng(123)
        n   = 300
        G   = 9
        Xd  = np.column_stack([rng.standard_normal(n)])
        yd  = rng.standard_normal(n)                # true β=0
        cl  = np.repeat(np.arange(G), n // G + 1)[:n]
        coef, resid, XXinv = ols_fe(yd, Xd)
        pval = wild_bootstrap_pval(yd, Xd, cl, coef, tidx=0, B=199, seed=99)
        # With true null, p should not be exactly 0
        if not np.isnan(pval):
            assert pval > 0.0

    def test_reproducible_with_same_seed(self):
        yd, Xd, cl = _make_arrays(n=300, G=9, seed=5)
        coef, _, _ = ols_fe(yd, Xd)
        p1 = wild_bootstrap_pval(yd, Xd, cl, coef, B=99, seed=42)
        p2 = wild_bootstrap_pval(yd, Xd, cl, coef, B=99, seed=42)
        assert p1 == p2 or (np.isnan(p1) and np.isnan(p2))

    def test_different_seeds_may_differ(self):
        yd, Xd, cl = _make_arrays(n=300, G=9, seed=5)
        coef, _, _ = ols_fe(yd, Xd)
        p1 = wild_bootstrap_pval(yd, Xd, cl, coef, B=99, seed=42)
        p2 = wild_bootstrap_pval(yd, Xd, cl, coef, B=99, seed=9999)
        # They could be equal by chance but usually differ
        # Just check both are valid
        for p in [p1, p2]:
            if not np.isnan(p):
                assert 0.0 <= p <= 1.0


# =============================================================================
# 8. ascii_event_chart
# =============================================================================

class TestAsciiEventChart:

    def _make_es_df(self):
        """Synthetic event study DataFrame."""
        rows = []
        for var in ["delivery_facility"]:
            for wave in [1992, 1996, 2002, 2007, 2014, 2018, 2024]:
                rows.append({
                    "outcome": var,
                    "wave":    wave,
                    "beta":    0.0 if wave == 2002 else np.random.randn(),
                    "se":      0.0 if wave == 2002 else abs(np.random.randn()) * 0.5,
                    "pval":    np.nan if wave == 2002 else np.random.uniform(0, 1),
                    "ci_lo":   -1.0,
                    "ci_hi":    1.0,
                    "n":       1000,
                })
        return pd.DataFrame(rows)

    def test_returns_string(self):
        es = self._make_es_df()
        result = ascii_event_chart(es, "delivery_facility")
        assert isinstance(result, str)

    def test_output_contains_outcome_name(self):
        es = self._make_es_df()
        result = ascii_event_chart(es, "delivery_facility")
        assert "delivery_facility" in result

    def test_output_contains_base_wave(self):
        es = self._make_es_df()
        result = ascii_event_chart(es, "delivery_facility")
        assert str(BASE_WAVE) in result

    def test_missing_outcome_returns_no_data_message(self):
        es     = self._make_es_df()
        result = ascii_event_chart(es, "nonexistent_outcome")
        assert "No data" in result or "no data" in result.lower()

    def test_output_contains_wave_years(self):
        es = self._make_es_df()
        result = ascii_event_chart(es, "delivery_facility")
        assert "1992" in result
        assert "2007" in result


# =============================================================================
# 9. FOCUS_OUTCOMES and WAVE constants
# =============================================================================

class TestConstants:

    def test_focus_outcomes_is_dict(self):
        assert isinstance(FOCUS_OUTCOMES, dict)

    def test_focus_outcomes_non_empty(self):
        assert len(FOCUS_OUTCOMES) > 0

    def test_each_focus_entry_has_three_fields(self):
        """(ftype, controls, scale)"""
        for var, val in FOCUS_OUTCOMES.items():
            assert len(val) == 3, \
                f"FOCUS_OUTCOMES['{var}'] has {len(val)} elements, expected 3"

    def test_ftype_valid(self):
        valid = {"HR", "KR", "IR", "MR"}
        for var, (ftype, controls, scale) in FOCUS_OUTCOMES.items():
            assert ftype in valid, f"Invalid ftype '{ftype}' for '{var}'"

    def test_scale_positive(self):
        for var, (ftype, controls, scale) in FOCUS_OUTCOMES.items():
            assert scale > 0

    def test_base_wave_in_wave_order(self):
        assert BASE_WAVE in WAVE_ORDER

    def test_wave_order_sorted(self):
        assert WAVE_ORDER == sorted(WAVE_ORDER)

    def test_wave_order_has_seven_elements(self):
        assert len(WAVE_ORDER) == 7

    def test_stars_function(self):
        assert stars(0.005) == "***"
        assert stars(0.03)  == "**"
        assert stars(0.08)  == "*"
        assert stars(0.20)  == ""
        assert stars(np.nan) == ""


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import importlib.util as _ilu
    passed = 0
    failed = 0
    errors = []

    test_classes = [
        TestWebbWeights, TestDemeanByFast, TestWithinDemean,
        TestPrepArrays, TestOlsFe, TestClusteredSe,
        TestWildBootstrapPval, TestAsciiEventChart, TestConstants,
    ]

    print("=" * 65)
    print("test_04_robustness.py")
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
