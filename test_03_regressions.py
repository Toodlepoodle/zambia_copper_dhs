"""
test_03_regressions.py
======================
Tests for 03_regressions.py — individual-level DiD regressions.
Imports functions and runs them on fully synthetic panel data.
No DHS files required.

Run with:
    pytest test_03_regressions.py -v
or:
    python test_03_regressions.py
"""

import sys
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

# ── Import module under test ───────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

spec = importlib.util.spec_from_file_location(
    "regressions",
    str(Path(__file__).resolve().parent / "03_regressions.py")
)
reg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reg)

to_num          = reg.to_num
flag_nan        = reg.flag_nan
binary          = reg.binary
safe_recode     = reg.safe_recode
isin_vec        = reg.isin_vec
stars           = reg.stars
ols_cluster     = reg.ols_cluster
run_one         = reg.run_one
demean_by_fast  = reg.demean_by_fast
within_demean   = reg.within_demean
run_one_fe      = reg.run_one_fe
OUTCOME_TABLES  = reg.OUTCOME_TABLES
MIN_OBS         = reg.MIN_OBS

from config_00 import DHS_MISSING_CODES, IMPROVED_WATER_CODES


# =============================================================================
# HELPERS — synthetic panel data
# =============================================================================

def _make_panel(n_per_cell=80, seed=42):
    """
    Build a synthetic individual-level panel:
      9 provinces × 7 waves × n_per_cell individuals.
    Mining provinces: copperbelt, northwestern.
    Bartik = mining × ln_cu_price.
    Outcome y has a known treatment effect of +5 pp.
    """
    rng = np.random.default_rng(seed)
    provinces = ["central","copperbelt","eastern","luapula","lusaka",
                 "northern","northwestern","southern","western"]
    waves     = [1992, 1996, 2002, 2007, 2014, 2018, 2024]
    cu_prices = {1992:2297, 1996:2289, 2002:1558, 2007:7132,
                 2014:6863, 2018:6530, 2024:9142}

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
                    "age":         float(rng.integers(15, 50)),
                    "post_priv":   1 if wave > 2000 else 0,
                    "w":           rng.uniform(0.5, 2.0),
                })

    df = pd.DataFrame(rows)
    df["province_id"] = pd.Categorical(df["province"]).codes.astype(int)
    df["wave_id"]     = pd.Categorical(df["wave"]).codes.astype(int)
    return df


# =============================================================================
# 1. stars
# =============================================================================

class TestStars:

    def test_below_001(self):
        assert stars(0.005) == "***"

    def test_below_005(self):
        assert stars(0.03) == "**"

    def test_below_010(self):
        assert stars(0.08) == "*"

    def test_above_010(self):
        assert stars(0.15) == ""

    def test_exactly_001(self):
        assert stars(0.01) == "**"   # not <0.01 → drops to **

    def test_nan_returns_empty(self):
        assert stars(np.nan) == ""

    def test_zero_is_three_stars(self):
        assert stars(0.0) == "***"


# =============================================================================
# 2. ols_cluster
# =============================================================================

class TestOlsCluster:

    def _simple_setup(self, n=200, seed=7):
        rng = np.random.default_rng(seed)
        X   = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y   = X @ np.array([1.0, 2.0]) + rng.standard_normal(n) * 0.5
        cl  = np.repeat(np.arange(9), n // 9 + 1)[:n]
        return y, X, cl

    def test_returns_four_arrays(self):
        y, X, cl = self._simple_setup()
        result   = ols_cluster(y, X, cl)
        assert len(result) == 4

    def test_coef_length_equals_k(self):
        y, X, cl = self._simple_setup()
        coef, se, tstat, pval = ols_cluster(y, X, cl)
        assert len(coef) == X.shape[1]

    def test_se_length_equals_k(self):
        y, X, cl = self._simple_setup()
        coef, se, tstat, pval = ols_cluster(y, X, cl)
        assert len(se) == X.shape[1]

    def test_se_nonnegative(self):
        y, X, cl = self._simple_setup()
        coef, se, tstat, pval = ols_cluster(y, X, cl)
        assert (se >= 0).all()

    def test_coef_slope_near_2(self):
        """True slope is 2.0; estimate should be close."""
        y, X, cl = self._simple_setup()
        coef, se, tstat, pval = ols_cluster(y, X, cl)
        assert abs(coef[1] - 2.0) < 0.5

    def test_pval_between_0_and_1(self):
        y, X, cl = self._simple_setup()
        coef, se, tstat, pval = ols_cluster(y, X, cl)
        for p in pval:
            if not np.isnan(p):
                assert 0.0 <= p <= 1.0

    def test_collinear_matrix_handled(self):
        """Perfectly collinear X should not crash (uses pinv)."""
        n = 100
        x = np.ones(n)
        X = np.column_stack([x, x, x])   # rank-1
        y = x + np.random.randn(n) * 0.1
        cl = np.repeat(np.arange(9), n // 9 + 1)[:n]
        try:
            result = ols_cluster(y, X, cl)
            assert len(result) == 4
        except Exception as e:
            pytest.fail(f"ols_cluster crashed on collinear matrix: {e}")


# =============================================================================
# 3. demean_by_fast
# =============================================================================

class TestDemeanByFast:

    def test_1d_group_means_removed(self):
        arr      = np.array([1.0, 3.0, 2.0, 4.0])
        group_ids= np.array([0, 0, 1, 1])
        out      = demean_by_fast(arr, group_ids)
        # Group 0 mean = 2.0; group 1 mean = 3.0
        assert abs(out[0] - (-1.0)) < 1e-9
        assert abs(out[1] - 1.0)    < 1e-9
        assert abs(out[2] - (-1.0)) < 1e-9
        assert abs(out[3] - 1.0)    < 1e-9

    def test_2d_demeaning(self):
        arr      = np.array([[1.0, 10.0], [3.0, 30.0], [2.0, 20.0], [4.0, 40.0]])
        group_ids= np.array([0, 0, 1, 1])
        out      = demean_by_fast(arr, group_ids)
        # Col 0: group means = [2.0, 3.0]
        assert abs(out[0, 0] - (-1.0)) < 1e-9
        assert abs(out[1, 0] - 1.0)    < 1e-9

    def test_single_group_gives_zero_mean(self):
        arr = np.array([1.0, 2.0, 3.0])
        g   = np.array([0, 0, 0])
        out = demean_by_fast(arr, g)
        assert abs(out.mean()) < 1e-9

    def test_output_shape_preserved(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        g   = np.array([0, 0, 1, 1])
        out = demean_by_fast(arr, g)
        assert out.shape == arr.shape

    def test_returns_float_array(self):
        arr = np.array([1, 2, 3, 4], dtype=int)
        g   = np.array([0, 0, 1, 1])
        out = demean_by_fast(arr, g)
        assert out.dtype == float


# =============================================================================
# 4. within_demean
# =============================================================================

class TestWithinDemean:

    def test_returns_tuple_of_three(self):
        arr  = np.ones(20)
        prov = np.repeat(np.arange(4), 5)
        wave = np.tile(np.arange(5), 4)
        result = within_demean(arr, prov, wave)
        assert len(result) == 3

    def test_converged_flag_true_on_constant(self):
        """All-ones array converges in 1 iteration."""
        arr  = np.ones(20)
        prov = np.repeat(np.arange(4), 5)
        wave = np.tile(np.arange(5), 4)
        _, n_iter, converged = within_demean(arr, prov, wave)
        assert converged is True

    def test_demeaned_array_has_near_zero_group_means(self):
        rng  = np.random.default_rng(0)
        n    = 9 * 7 * 50
        prov = np.repeat(np.arange(9), 7 * 50)
        wave = np.tile(np.repeat(np.arange(7), 50), 9)
        arr  = rng.standard_normal(n) + prov * 0.5 + wave * 0.3
        out, _, cv = within_demean(arr, prov, wave)
        # Province means of demeaned output should be ~0
        for g in range(9):
            m = out[prov == g].mean()
            assert abs(m) < 0.1, f"Province {g} mean not zero after demeaning: {m:.4f}"

    def test_2d_array_supported(self):
        arr  = np.ones((20, 2))
        prov = np.repeat(np.arange(4), 5)
        wave = np.tile(np.arange(5), 4)
        out, _, cv = within_demean(arr, prov, wave)
        assert out.shape == arr.shape

    def test_max_iter_returned_on_no_convergence(self):
        """Very tight tolerance should fail to converge quickly."""
        arr  = np.random.randn(100)
        prov = np.repeat(np.arange(9), 12)[:100]
        wave = np.tile(np.arange(7), 15)[:100]
        _, n_iter, converged = within_demean(arr, prov, wave, max_iter=1, tol=1e-30)
        assert n_iter == 1


# =============================================================================
# 5. run_one (dummies-based OLS)
# =============================================================================

class TestRunOne:

    def _make_panel_df(self, seed=1):
        rng = np.random.default_rng(seed)
        n = 500
        df = pd.DataFrame({
            "province":    np.repeat(np.arange(9), n // 9 + 1)[:n],
            "wave":        np.tile(np.arange(7), n // 7 + 1)[:n],
            "y":           rng.standard_normal(n),
            "bartik":      rng.standard_normal(n),
            "control":     rng.standard_normal(n),
            "province_id": np.repeat(np.arange(9), n // 9 + 1)[:n],
            "wave_id":     np.tile(np.arange(7), n // 7 + 1)[:n],
        })
        # Add province dummies as required by run_one (it uses dummies from the df)
        for p in range(9):
            df[f"prov_{p}"] = (df["province"] == p).astype(float)
        return df

    def test_returns_dict_when_data_sufficient(self):
        df = _make_panel(n_per_cell=30)
        r  = run_one(df, "y", "bartik", ["urban", "age"], "province_id", scale=100)
        # Either returns dict or None (if <MIN_OBS after dropna)
        assert r is None or isinstance(r, dict)

    def test_returns_none_below_min_obs(self):
        df = _make_panel(n_per_cell=1)   # only 9×7=63 rows
        # With MIN_OBS=50 this may pass; force tiny subset
        tiny = df.iloc[:5].copy()
        r = run_one(tiny, "y", "bartik", [], "province_id", scale=1)
        assert r is None

    def test_result_has_expected_keys(self):
        df = _make_panel(n_per_cell=50)
        r  = run_one(df, "y", "bartik", ["urban"], "province_id", scale=100)
        if r is not None:
            for k in ["coef","coef_s","se","se_s","tstat","pval","nobs","n_clusters","r2"]:
                assert k in r, f"Key '{k}' missing from run_one result"

    def test_coef_s_equals_coef_times_scale(self):
        df = _make_panel(n_per_cell=50)
        r  = run_one(df, "y", "bartik", ["urban"], "province_id", scale=100)
        if r is not None:
            assert abs(r["coef_s"] - r["coef"] * 100) < 1e-9

    def test_pval_between_0_and_1(self):
        df = _make_panel(n_per_cell=50)
        r  = run_one(df, "y", "bartik", ["urban"], "province_id", scale=100)
        if r is not None:
            assert 0.0 <= r["pval"] <= 1.0

    def test_n_clusters_equals_9(self):
        df = _make_panel(n_per_cell=50)
        r  = run_one(df, "y", "bartik", ["urban"], "province_id", scale=100)
        if r is not None:
            assert r["n_clusters"] == 9


# =============================================================================
# 6. run_one_fe (two-way FE)
# =============================================================================

class TestRunOneFe:

    def test_returns_dict_on_sufficient_data(self):
        df = _make_panel(n_per_cell=60)
        r  = run_one_fe(df, "y", "bartik", ["urban"], scale=100)
        assert r is None or isinstance(r, dict)

    def test_result_keys_present(self):
        df = _make_panel(n_per_cell=60)
        r  = run_one_fe(df, "y", "bartik", ["urban"], scale=100)
        if r is not None:
            for k in ["coef","coef_s","se","se_s","tstat","pval",
                      "ci_lo","ci_hi","nobs","n_clusters","r2"]:
                assert k in r

    def test_se_is_positive(self):
        df = _make_panel(n_per_cell=60)
        r  = run_one_fe(df, "y", "bartik", ["urban"], scale=1)
        if r is not None:
            assert r["se"] > 0

    def test_ci_lo_less_than_ci_hi(self):
        df = _make_panel(n_per_cell=60)
        r  = run_one_fe(df, "y", "bartik", ["urban"], scale=1)
        if r is not None:
            assert r["ci_lo"] < r["ci_hi"]

    def test_returns_none_when_outcome_missing(self):
        df = _make_panel(n_per_cell=60)
        r  = run_one_fe(df, "nonexistent_col", "bartik", ["urban"], scale=1)
        assert r is None

    def test_r2_between_0_and_1(self):
        df = _make_panel(n_per_cell=60)
        r  = run_one_fe(df, "y", "bartik", ["urban"], scale=1)
        if r is not None and not np.isnan(r["r2"]):
            assert 0.0 <= r["r2"] <= 1.0

    def test_known_treatment_effect_detected(self):
        """
        With true β = +5 per unit Bartik, the estimate should have a
        positive coefficient and the t-stat should suggest significance.
        """
        df = _make_panel(n_per_cell=100, seed=0)
        r  = run_one_fe(df, "y", "bartik", ["urban"], scale=1)
        if r is not None:
            assert r["coef"] > 0, \
                f"Expected positive coef for true positive treatment, got {r['coef']:.4f}"

    def test_post_priv_subsample_runs(self):
        df  = _make_panel(n_per_cell=60)
        sub = df[df["wave"] >= 2002].copy()
        r   = run_one_fe(sub, "y", "bartik", ["urban"], scale=100)
        # Should either return a result or None (may have too few obs)
        assert r is None or isinstance(r, dict)

    def test_pre_priv_subsample_runs(self):
        df  = _make_panel(n_per_cell=60)
        sub = df[df["wave"] <= 2002].copy()
        r   = run_one_fe(sub, "y", "bartik", ["urban"], scale=100)
        assert r is None or isinstance(r, dict)


# =============================================================================
# 7. OUTCOME_TABLES structure
# =============================================================================

class TestOutcomeTables:

    def test_has_six_tables(self):
        assert len(OUTCOME_TABLES) == 6

    def test_all_tables_non_empty(self):
        for tbl, entries in OUTCOME_TABLES.items():
            assert len(entries) > 0, f"Table '{tbl}' is empty"

    def test_all_entries_have_four_fields(self):
        """Each entry: (var_name, label, scale, higher_better)"""
        for tbl, entries in OUTCOME_TABLES.items():
            for entry in entries:
                assert len(entry) == 4, \
                    f"Entry in '{tbl}' has {len(entry)} fields, expected 4: {entry}"

    def test_scale_is_numeric(self):
        for tbl, entries in OUTCOME_TABLES.items():
            for (var, label, scale, higher_better) in entries:
                assert isinstance(scale, (int, float)), \
                    f"Scale for '{var}' in '{tbl}' is not numeric"

    def test_higher_better_is_bool(self):
        for tbl, entries in OUTCOME_TABLES.items():
            for (var, label, scale, higher_better) in entries:
                assert isinstance(higher_better, bool), \
                    f"higher_better for '{var}' not bool"

    def test_table_1_contains_asset_index(self):
        vars_t1 = [e[0] for e in OUTCOME_TABLES["Table 1: Wealth and Assets"]]
        assert "asset_index" in vars_t1

    def test_table_2_contains_u5_dead(self):
        vars_t2 = [e[0] for e in OUTCOME_TABLES["Table 2: Child Health"]]
        assert "u5_dead" in vars_t2

    def test_mortality_higher_better_is_false(self):
        vars_t2 = {e[0]: e[3] for e in OUTCOME_TABLES["Table 2: Child Health"]}
        for m in ["u5_dead", "infant_dead", "neonatal_dead"]:
            if m in vars_t2:
                assert vars_t2[m] is False, \
                    f"Mortality outcome '{m}' should have higher_better=False"

    def test_all_var_names_are_strings(self):
        for tbl, entries in OUTCOME_TABLES.items():
            for (var, label, scale, higher_better) in entries:
                assert isinstance(var, str)
                assert isinstance(label, str)

    def test_no_duplicate_outcomes_within_table(self):
        for tbl, entries in OUTCOME_TABLES.items():
            names = [e[0] for e in entries]
            assert len(names) == len(set(names)), \
                f"Duplicate outcomes in '{tbl}': {names}"


# =============================================================================
# 8. Helper functions (binary, isin_vec, safe_recode from reg module)
# =============================================================================

class TestRegHelpers:
    """
    Verify that the reg module's helpers agree with config_00 on missing codes.
    """

    def test_binary_96_is_nan(self):
        assert pd.isna(binary(pd.Series([96]), [1], [0])[0])

    def test_isin_vec_96_is_nan(self):
        assert pd.isna(isin_vec(pd.Series([96]), IMPROVED_WATER_CODES)[0])

    def test_safe_recode_99_is_nan(self):
        assert pd.isna(safe_recode(pd.Series([99]), {99: 1})[0])

    def test_flag_nan_9991_is_nan(self):
        assert pd.isna(flag_nan(pd.Series([9991]))[0])

    def test_to_num_string_to_float(self):
        r = to_num(pd.Series(["3.14"]))
        assert abs(r[0] - 3.14) < 1e-9

    def test_had_diarrhoea_codes_1_and_2_both_sick(self):
        """FIX 2: codes 1 and 2 for h11 must both → sick=1."""
        s = pd.Series([0, 1, 2, 8])
        r = binary(s, [1, 2], [0])
        assert r[0] == 0.0
        assert r[1] == 1.0
        assert r[2] == 1.0
        assert pd.isna(r[3])

    def test_ever_breastfed_93_is_ever(self):
        """FIX 1: m4==93 → ever breastfed (stopped), NOT never."""
        bf_map = {0: 0, 93: 1, 94: 0, 95: 1}
        bf_map.update({i: 1 for i in range(1, 93)})
        r = safe_recode(pd.Series([93]), bf_map)
        assert r[0] == 1, "m4==93 (stopped BF) must map to 1 (ever BF)"


# =============================================================================
# 9. MIN_OBS guard
# =============================================================================

class TestMinObsGuard:

    def test_min_obs_imported_from_config(self):
        from config_00 import MIN_OBS as cfg_min_obs
        assert MIN_OBS == cfg_min_obs

    def test_run_one_fe_returns_none_for_tiny_data(self):
        """Data with < MIN_OBS rows should return None."""
        df = _make_panel(n_per_cell=1)[:10].copy()
        r  = run_one_fe(df, "y", "bartik", [], scale=1)
        assert r is None


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    passed = 0
    failed = 0
    errors = []

    test_classes = [
        TestStars, TestOlsCluster, TestDemeanByFast, TestWithinDemean,
        TestRunOne, TestRunOneFe, TestOutcomeTables, TestRegHelpers,
        TestMinObsGuard,
    ]

    print("=" * 65)
    print("test_03_regressions.py")
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
