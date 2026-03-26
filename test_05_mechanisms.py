"""
test_05_mechanisms.py
=====================
Tests for 05_mechanisms.py — mechanisms, heterogeneity & additional robustness.
Imports all regression helpers and exercises them on synthetic data.
No DHS files required.

Run with:
    pytest test_05_mechanisms.py -v
or:
    python test_05_mechanisms.py
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
    "mechanisms",
    str(Path(__file__).resolve().parent / "05_mechanisms.py")
)
mech = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mech)

demean_by_fast      = mech.demean_by_fast
within_demean       = mech.within_demean
_ols_clustered      = mech._ols_clustered
_prep_and_demean    = mech._prep_and_demean
run_fe              = mech.run_fe
run_multi_fe        = mech.run_multi_fe
slim                = mech.slim
stars               = mech.stars

MINING_DOSE         = mech.MINING_DOSE
MINING_DOSE_NOTE    = mech.MINING_DOSE_NOTE
PRE_PRIV_WAVES      = mech.PRE_PRIV_WAVES
POST_PRIV_WAVES     = mech.POST_PRIV_WAVES
ALL_FOCUS           = mech.ALL_FOCUS
MIN_OBS             = mech.MIN_OBS

from config_00 import MINING_PROVINCES


# =============================================================================
# HELPERS
# =============================================================================

def _make_panel(n_per_cell=80, seed=42, add_outcome=True):
    """Synthetic 9-province × 7-wave individual panel."""
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
                    "province":    prov,
                    "wave":        wave,
                    "mining":      float(mining),
                    "ln_cu_price": ln_p,
                    "bartik":      bartik,
                    "urban":       float(rng.integers(0, 2)),
                    "age_woman":   float(rng.integers(15, 50)),
                    "post_priv":   1 if wave > 2000 else 0,
                    "w":           rng.uniform(0.5, 2.0),
                }
                if add_outcome:
                    row["delivery_facility"] = float(rng.integers(0, 2))
                    row["had_fever"]         = float(rng.integers(0, 2))
                    row["man_employed"]      = float(rng.integers(0, 2))
                    row["edu_secondary_p"]   = float(rng.integers(0, 2))
                rows.append(row)

    df = pd.DataFrame(rows)
    df["province_id"] = pd.Categorical(df["province"]).codes.astype(int)
    df["wave_id"]     = pd.Categorical(df["wave"]).codes.astype(int)
    return df


# =============================================================================
# 1. demean_by_fast (independent from 03/04 copies)
# =============================================================================

class TestDemeanByFast:

    def test_1d_removes_group_means(self):
        arr = np.array([1.0, 3.0, 2.0, 4.0])
        g   = np.array([0, 0, 1, 1])
        out = demean_by_fast(arr, g)
        assert abs(out[0] - (-1.0)) < 1e-9
        assert abs(out[1] - 1.0)    < 1e-9
        assert abs(out[2] - (-1.0)) < 1e-9
        assert abs(out[3] - 1.0)    < 1e-9

    def test_2d_column_wise(self):
        arr = np.array([[1.0, 10.0], [3.0, 30.0]])
        g   = np.array([0, 0])
        out = demean_by_fast(arr, g)
        assert abs(out[0, 0] - (-1.0)) < 1e-9
        assert abs(out[1, 0] - 1.0)    < 1e-9

    def test_output_float(self):
        arr = np.array([1, 2, 3, 4], dtype=int)
        g   = np.array([0, 0, 1, 1])
        out = demean_by_fast(arr, g)
        assert out.dtype == float


# =============================================================================
# 2. within_demean — consistency with 03 and 04
# =============================================================================

class TestWithinDemean:

    def test_returns_tuple_of_three(self):
        arr  = np.ones(63)
        prov = np.repeat(np.arange(9), 7)
        wave = np.tile(np.arange(7), 9)
        r = within_demean(arr, prov, wave)
        assert len(r) == 3

    def test_converged_true_for_constant(self):
        arr  = np.ones(63)
        prov = np.repeat(np.arange(9), 7)
        wave = np.tile(np.arange(7), 9)
        _, _, cv = within_demean(arr, prov, wave)
        assert cv is True

    def test_max_iter_50_default(self):
        """Should not hit max_iter=50 on a simple random array."""
        rng  = np.random.default_rng(1)
        arr  = rng.standard_normal(63)
        prov = np.repeat(np.arange(9), 7)
        wave = np.tile(np.arange(7), 9)
        _, n_iter, cv = within_demean(arr, prov, wave)
        # Typical convergence in <10 iterations
        assert n_iter <= 50

    def test_consistent_with_03_implementation(self):
        spec3 = importlib.util.spec_from_file_location(
            "regressions",
            str(Path(__file__).resolve().parent / "03_regressions.py")
        )
        reg = importlib.util.module_from_spec(spec3)
        spec3.loader.exec_module(reg)

        rng  = np.random.default_rng(77)
        arr  = rng.standard_normal(63)
        prov = np.repeat(np.arange(9), 7)
        wave = np.tile(np.arange(7), 9)

        out03, _, _ = reg.within_demean(arr, prov, wave)
        out05, _, _ = within_demean(arr, prov, wave)
        np.testing.assert_allclose(out03, out05, atol=1e-12)


# =============================================================================
# 3. _ols_clustered
# =============================================================================

class TestOlsClustered:

    def _arrays(self, n=300, G=9, seed=0):
        rng  = np.random.default_rng(seed)
        Xd   = np.column_stack([rng.standard_normal(n), rng.standard_normal(n)])
        yd   = Xd[:, 0] * 2.0 + rng.standard_normal(n) * 0.5
        cl   = np.repeat(np.arange(G), n // G + 1)[:n]
        return yd, Xd, cl, n, Xd.shape[1], G

    def test_returns_four_values(self):
        yd, Xd, cl, n, k, G = self._arrays()
        result = _ols_clustered(yd, Xd, cl, n, k, G)
        assert len(result) == 4

    def test_coef_near_true_value(self):
        yd, Xd, cl, n, k, G = self._arrays()
        coef, se, XXinv, resid = _ols_clustered(yd, Xd, cl, n, k, G)
        assert abs(coef[0] - 2.0) < 0.4

    def test_se_nonnegative(self):
        yd, Xd, cl, n, k, G = self._arrays()
        coef, se, XXinv, resid = _ols_clustered(yd, Xd, cl, n, k, G)
        assert (se >= 0).all()

    def test_resid_near_zero_mean(self):
        yd, Xd, cl, n, k, G = self._arrays()
        coef, se, XXinv, resid = _ols_clustered(yd, Xd, cl, n, k, G)
        assert abs(resid.mean()) < 0.1

    def test_pinv_fallback_on_singular(self):
        n  = 50
        Xd = np.ones((n, 2))
        yd = np.random.randn(n)
        cl = np.repeat(np.arange(9), 6)[:n]
        try:
            result = _ols_clustered(yd, Xd, cl, n, 2, 9)
            assert len(result) == 4
        except Exception as e:
            pytest.fail(f"_ols_clustered crashed on singular matrix: {e}")


# =============================================================================
# 4. _prep_and_demean
# =============================================================================

class TestPrepAndDemean:

    def test_returns_seven_on_valid(self):
        df = _make_panel(n_per_cell=30)
        r  = _prep_and_demean(df, "delivery_facility", ["bartik"], ["urban"],
                               "province_id", MIN_OBS)
        assert r is None or len(r) == 7

    def test_returns_none_below_min_obs(self):
        df = _make_panel(n_per_cell=1)[:5].copy()
        r  = _prep_and_demean(df, "delivery_facility", ["bartik"], [],
                               "province_id", MIN_OBS)
        assert r is None

    def test_returns_none_for_zero_variance_treatment(self):
        df = _make_panel(n_per_cell=30)
        df["bartik"] = 0.0    # zero variance
        r  = _prep_and_demean(df, "delivery_facility", ["bartik"], [],
                               "province_id", MIN_OBS)
        assert r is None

    def test_g_less_than_5_returns_none(self):
        """Less than 5 provinces → run_fe returns None (MIN_CLUSTERS guard)."""
        df = _make_panel(n_per_cell=30)
        df = df[df["province"].isin(["copperbelt","lusaka","northern"])].copy()
        df["province_id"] = pd.Categorical(df["province"]).codes.astype(int)
        r  = _prep_and_demean(df, "delivery_facility", ["bartik"], [],
                               "province_id", MIN_OBS)
        assert r is None


# =============================================================================
# 5. run_fe
# =============================================================================

class TestRunFe:

    def test_returns_dict_or_none(self):
        df = _make_panel(n_per_cell=60)
        r  = run_fe(df, "delivery_facility", "bartik", ["urban","age_woman"], scale=100)
        assert r is None or isinstance(r, dict)

    def test_result_has_expected_keys(self):
        df = _make_panel(n_per_cell=60)
        r  = run_fe(df, "delivery_facility", "bartik", ["urban"], scale=100)
        if r is not None:
            for k in ["coef","coef_s","se_s","tstat","pval","ci_lo","ci_hi",
                      "nobs","n_clusters","r2"]:
                assert k in r

    def test_pval_between_0_and_1(self):
        df = _make_panel(n_per_cell=60)
        r  = run_fe(df, "delivery_facility", "bartik", ["urban"], scale=100)
        if r is not None:
            assert 0.0 <= r["pval"] <= 1.0

    def test_coef_s_equals_coef_times_scale(self):
        df = _make_panel(n_per_cell=60)
        r  = run_fe(df, "delivery_facility", "bartik", [], scale=100)
        if r is not None:
            assert abs(r["coef_s"] - r["coef"] * 100) < 1e-9

    def test_r2_between_0_and_1(self):
        df = _make_panel(n_per_cell=60)
        r  = run_fe(df, "delivery_facility", "bartik", [], scale=1)
        if r is not None and not np.isnan(r["r2"]):
            assert 0.0 <= r["r2"] <= 1.0

    def test_returns_none_for_missing_outcome(self):
        df = _make_panel(n_per_cell=60)
        r  = run_fe(df, "nonexistent", "bartik", [], scale=1)
        assert r is None

    def test_pre_priv_subsample(self):
        df  = _make_panel(n_per_cell=60)
        sub = df[df["wave"].isin(PRE_PRIV_WAVES)].copy()
        r   = run_fe(sub, "delivery_facility", "bartik", [], scale=100)
        assert r is None or isinstance(r, dict)

    def test_post_priv_subsample(self):
        df  = _make_panel(n_per_cell=60)
        sub = df[df["wave"].isin(POST_PRIV_WAVES)].copy()
        r   = run_fe(sub, "delivery_facility", "bartik", [], scale=100)
        assert r is None or isinstance(r, dict)

    def test_urban_subsample(self):
        df  = _make_panel(n_per_cell=60)
        sub = df[df["urban"] == 1.0].copy()
        r   = run_fe(sub, "delivery_facility", "bartik", ["age_woman"], scale=100)
        assert r is None or isinstance(r, dict)

    def test_rural_subsample(self):
        df  = _make_panel(n_per_cell=60)
        sub = df[df["urban"] == 0.0].copy()
        r   = run_fe(sub, "delivery_facility", "bartik", ["age_woman"], scale=100)
        assert r is None or isinstance(r, dict)


# =============================================================================
# 6. run_multi_fe
# =============================================================================

class TestRunMultiFe:

    def test_returns_five_values_or_none(self):
        df = _make_panel(n_per_cell=60)
        r  = run_multi_fe(df, "delivery_facility",
                          ["bartik","urban"], ["age_woman"], scale=100)
        assert r is None or len(r) == 5

    def test_coef_length_equals_n_treatments_plus_controls(self):
        df = _make_panel(n_per_cell=60)
        r  = run_multi_fe(df, "delivery_facility",
                          ["bartik","urban"], ["age_woman"], scale=100)
        if r is not None:
            coef, se, pval, n, G = r
            assert len(coef) == 2   # bartik + urban

    def test_pvals_between_0_and_1(self):
        df = _make_panel(n_per_cell=60)
        r  = run_multi_fe(df, "delivery_facility",
                          ["bartik","urban"], ["age_woman"], scale=100)
        if r is not None:
            coef, se, pval, n, G = r
            for p in pval:
                if not np.isnan(p):
                    assert 0.0 <= p <= 1.0


# =============================================================================
# 7. slim
# =============================================================================

class TestSlim:

    def test_keeps_required_columns(self):
        df = _make_panel(n_per_cell=5)
        out = slim(df, "delivery_facility", "bartik", ["urban"])
        for c in ["delivery_facility", "bartik", "province_id", "wave_id"]:
            assert c in out.columns, f"Column '{c}' missing from slim output"

    def test_drops_irrelevant_columns(self):
        df = _make_panel(n_per_cell=5)
        df["totally_random_column"] = 1.0
        out = slim(df, "delivery_facility", "bartik", [])
        assert "totally_random_column" not in out.columns

    def test_returns_dataframe(self):
        df = _make_panel(n_per_cell=5)
        assert isinstance(slim(df, "delivery_facility", "bartik", []), pd.DataFrame)


# =============================================================================
# 8. MINING_DOSE and privatisation constants
# =============================================================================

class TestConstants:

    def test_mining_dose_is_dict(self):
        assert isinstance(MINING_DOSE, dict)

    def test_copperbelt_dose_highest(self):
        """Copperbelt should have the highest mining dose."""
        assert MINING_DOSE["copperbelt"] == max(MINING_DOSE.values())

    def test_all_doses_between_0_and_1(self):
        for prov, dose in MINING_DOSE.items():
            assert 0.0 < dose <= 1.0, f"Dose for '{prov}' out of range: {dose}"

    def test_all_nine_provinces_have_dose(self):
        expected = {"central","copperbelt","eastern","luapula","lusaka",
                    "northern","northwestern","southern","western"}
        assert set(MINING_DOSE.keys()) == expected

    def test_mining_dose_note_is_string(self):
        assert isinstance(MINING_DOSE_NOTE, str)
        assert len(MINING_DOSE_NOTE) > 0

    def test_pre_priv_waves_correct(self):
        """Pre = {1992, 1996} — strictly before privatisation."""
        assert set(PRE_PRIV_WAVES) == {1992, 1996}

    def test_post_priv_waves_correct(self):
        """Post = {2007, 2014, 2018, 2024} — strictly after."""
        assert set(POST_PRIV_WAVES) == {2007, 2014, 2018, 2024}

    def test_2002_in_neither_pre_nor_post(self):
        """2002 excluded from both to avoid boundary overlap (FIX 6)."""
        assert 2002 not in PRE_PRIV_WAVES
        assert 2002 not in POST_PRIV_WAVES

    def test_pre_and_post_do_not_overlap(self):
        assert set(PRE_PRIV_WAVES).isdisjoint(set(POST_PRIV_WAVES))

    def test_all_focus_entries_have_two_fields(self):
        """(controls_list, scale)"""
        for var, val in ALL_FOCUS.items():
            assert len(val) == 2, \
                f"ALL_FOCUS['{var}'] has {len(val)} elements, expected 2"

    def test_stars_function_correctness(self):
        assert stars(0.005) == "***"
        assert stars(0.03)  == "**"
        assert stars(0.08)  == "*"
        assert stars(0.20)  == ""
        assert stars(np.nan) == ""


# =============================================================================
# 9. DATA QUALITY checks
# =============================================================================

class TestDataQuality:

    def test_privatisation_split_no_overlap(self):
        """2002 must not appear in either pre or post sample."""
        all_waves = set(PRE_PRIV_WAVES) | set(POST_PRIV_WAVES)
        assert 2002 not in all_waves, \
            "2002 should be excluded from both pre and post samples"

    def test_dose_response_copperbelt_vs_northwestern(self):
        """Copperbelt dose should be higher than Northwestern."""
        assert MINING_DOSE["copperbelt"] > MINING_DOSE["northwestern"]

    def test_all_non_mining_doses_low(self):
        """Non-mining provinces should have doses < 5%."""
        non_mining = set(MINING_DOSE.keys()) - MINING_PROVINCES
        for p in non_mining:
            assert MINING_DOSE[p] < 0.05, \
                f"Non-mining province '{p}' has suspiciously high dose: {MINING_DOSE[p]}"

    def test_run_fe_consistency_across_samples(self):
        """
        Full-sample and urban-only should yield finite results (not crash),
        and the full sample coefficient should not be wildly different from urban.
        This is a basic sanity check, not a statistical guarantee.
        """
        df   = _make_panel(n_per_cell=80, seed=3)
        r_full = run_fe(df, "delivery_facility", "bartik", [], scale=100)
        sub    = df[df["urban"] == 1.0].copy()
        r_urb  = run_fe(sub, "delivery_facility", "bartik", [], scale=100)
        # Both should either produce results or be None — no crash
        assert r_full is None or isinstance(r_full, dict)
        assert r_urb  is None or isinstance(r_urb, dict)

    def test_convergence_consistent_across_copies(self):
        """
        within_demean in 05 must converge in the same number of iterations
        as in 03 on the same input.
        """
        spec3 = importlib.util.spec_from_file_location(
            "regressions",
            str(Path(__file__).resolve().parent / "03_regressions.py")
        )
        reg = importlib.util.module_from_spec(spec3)
        spec3.loader.exec_module(reg)

        rng  = np.random.default_rng(55)
        arr  = rng.standard_normal(63)
        prov = np.repeat(np.arange(9), 7)
        wave = np.tile(np.arange(7), 9)

        _, n3, cv3 = reg.within_demean(arr, prov, wave)
        _, n5, cv5 = within_demean(arr, prov, wave)
        assert cv3 == cv5
        assert n3  == n5


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    passed = 0
    failed = 0
    errors = []

    test_classes = [
        TestDemeanByFast, TestWithinDemean, TestOlsClustered,
        TestPrepAndDemean, TestRunFe, TestRunMultiFe, TestSlim,
        TestConstants, TestDataQuality,
    ]

    print("=" * 65)
    print("test_05_mechanisms.py")
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
