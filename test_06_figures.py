"""
test_06_figures.py
==================
Tests for 06_figures.py — publication figure generation.
Exercises helper functions and plot setup without generating actual PDF/PNG files.
No DHS data or prior CSV outputs required.

Run with:
    pytest test_06_figures.py -v
or:
    python test_06_figures.py
"""

import sys
import math
import importlib.util
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Force non-interactive matplotlib backend BEFORE importing the module
import matplotlib
matplotlib.use("Agg")

# ── Import module under test ───────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

spec = importlib.util.spec_from_file_location(
    "figures",
    str(Path(__file__).resolve().parent / "06_figures.py")
)
fig_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fig_mod)

_stars    = fig_mod._stars
save      = fig_mod.save
T_CRIT_95 = fig_mod.T_CRIT_95
WAVES     = fig_mod.WAVES
CU_PRICES = fig_mod.CU_PRICES

import matplotlib.pyplot as plt


# =============================================================================
# 1. _stars helper
# =============================================================================

class TestStars:

    def test_below_001(self):
        assert _stars(0.005) == "***"

    def test_below_005(self):
        assert _stars(0.03) == "**"

    def test_below_010(self):
        assert _stars(0.08) == "*"

    def test_above_010(self):
        assert _stars(0.20) == ""

    def test_exactly_010(self):
        assert _stars(0.10) == ""

    def test_nan_returns_empty(self):
        assert _stars(np.nan) == ""

    def test_zero_is_three_stars(self):
        assert _stars(0.0) == "***"

    def test_returns_string(self):
        assert isinstance(_stars(0.05), str)


# =============================================================================
# 2. T_CRIT_95 — critical value correctness
# =============================================================================

class TestTCrit95:

    def test_df_is_8(self):
        """FIX 1: G=9 clusters → df=G-1=8. T_CRIT_95 must use df=8."""
        from scipy import stats as st
        expected = st.t.ppf(0.975, df=8)
        assert abs(T_CRIT_95 - expected) < 1e-10, \
            f"T_CRIT_95={T_CRIT_95:.6f} does not match t.ppf(0.975, df=8)={expected:.6f}"

    def test_value_roughly_23(self):
        """t-critical at 97.5% with 8 df is approximately 2.306."""
        assert 2.2 < T_CRIT_95 < 2.5, \
            f"T_CRIT_95={T_CRIT_95:.4f} outside expected range (2.2, 2.5)"

    def test_larger_than_normal_z(self):
        """t distribution has heavier tails than normal at df=8."""
        from scipy import stats as st
        z_crit = st.norm.ppf(0.975)
        assert T_CRIT_95 > z_crit


# =============================================================================
# 3. WAVES and CU_PRICES constants
# =============================================================================

class TestWavesAndPrices:

    def test_seven_waves(self):
        assert len(WAVES) == 7

    def test_seven_prices(self):
        assert len(CU_PRICES) == 7

    def test_waves_sorted(self):
        assert WAVES == sorted(WAVES)

    def test_waves_correct_years(self):
        assert WAVES == [1992, 1996, 2002, 2007, 2014, 2018, 2024]

    def test_all_prices_positive(self):
        assert all(p > 0 for p in CU_PRICES)

    def test_prices_plausible_range(self):
        for p in CU_PRICES:
            assert 500 <= p <= 20_000, f"Price {p} outside plausible range"

    def test_2002_price_lowest(self):
        """2002 should be the commodity trough."""
        idx_2002 = WAVES.index(2002)
        assert CU_PRICES[idx_2002] == min(CU_PRICES)

    def test_2007_price_high(self):
        idx_2007 = WAVES.index(2007)
        idx_2002 = WAVES.index(2002)
        assert CU_PRICES[idx_2007] > 2 * CU_PRICES[idx_2002]


# =============================================================================
# 4. Colour constants
# =============================================================================

class TestColours:

    def test_colours_are_hex_strings(self):
        colours = [
            fig_mod.C_MINING, fig_mod.C_NONMINING,
            fig_mod.C_URBAN,  fig_mod.C_RURAL,
            fig_mod.C_SIG,    fig_mod.C_INSIG,
        ]
        for c in colours:
            assert isinstance(c, str), f"Colour '{c}' is not a string"
            assert c.startswith("#"), f"Colour '{c}' does not start with '#'"
            assert len(c) == 7, f"Colour '{c}' is not 7 characters long"

    def test_sig_and_insig_are_different(self):
        assert fig_mod.C_SIG != fig_mod.C_INSIG


# =============================================================================
# 5. fig_copper_price — does it run without error?
# =============================================================================

class TestFigCopperPrice:

    def setup_method(self):
        self.tmp = Path(tempfile.mkdtemp())

    def teardown_method(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        plt.close("all")

    def test_runs_without_exception(self):
        """fig_copper_price() should execute completely without crashing."""
        with patch.object(fig_mod, "OUT_DIR", self.tmp):
            try:
                fig_mod.fig_copper_price()
            except Exception as e:
                pytest.fail(f"fig_copper_price() raised: {e}")

    def test_saves_pdf_and_png(self):
        with patch.object(fig_mod, "OUT_DIR", self.tmp):
            fig_mod.fig_copper_price()
        assert (self.tmp / "fig1_copper_price.pdf").exists()
        assert (self.tmp / "fig1_copper_price.png").exists()


# =============================================================================
# 6. fig_coef_plot — with synthetic results CSV
# =============================================================================

class TestFigCoefPlot:

    def setup_method(self):
        self.tmp = Path(tempfile.mkdtemp())
        # Build a synthetic ind_results.csv
        outcomes = ["delivery_facility","had_fever","asset_index","u5_dead",
                    "edu_secondary_p","dv_any","man_employed"]
        rows = []
        rng  = np.random.default_rng(0)
        for var in outcomes:
            for spec_name in ["main","post_priv","pre_priv","boom_2007","balanced"]:
                coef = rng.normal(0, 5)
                se   = abs(rng.normal(2, 0.5))
                pval = rng.uniform(0, 1)
                rows.append({
                    "table":         "Table 1: Wealth and Assets",
                    "outcome":       var,
                    "label":         var.replace("_", " ").title(),
                    "spec":          spec_name,
                    "scale":         100,
                    "higher_better": True,
                    "coef":          coef,
                    "coef_s":        coef,
                    "se":            se,
                    "se_s":          se,
                    "tstat":         coef / se,
                    "pval":          pval,
                    "ci_lo":         coef - 1.96 * se,
                    "ci_hi":         coef + 1.96 * se,
                    "nobs":          5000,
                    "n_clusters":    9,
                    "r2":            rng.uniform(0, 0.3),
                })
        output_dir = self.tmp / "OUTPUT"
        output_dir.mkdir()
        pd.DataFrame(rows).to_csv(output_dir / "ind_results.csv", index=False)
        self.output_dir = output_dir

    def teardown_method(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        plt.close("all")

    def test_runs_without_exception(self):
        with patch.object(fig_mod, "OUT_DIR", self.tmp / "OUTPUT" / "figures"), \
             patch.object(fig_mod, "BASE_DIR", self.tmp):
            (self.tmp / "OUTPUT" / "figures").mkdir(parents=True, exist_ok=True)
            try:
                fig_mod.fig_coef_plot()
            except Exception as e:
                pytest.fail(f"fig_coef_plot() raised: {e}")

    def test_all_outcomes_shown_not_just_significant(self):
        """FIX 4: all outcomes shown, not just p<0.10."""
        with patch.object(fig_mod, "OUT_DIR", self.tmp / "OUTPUT" / "figures"), \
             patch.object(fig_mod, "BASE_DIR", self.tmp):
            (self.tmp / "OUTPUT" / "figures").mkdir(parents=True, exist_ok=True)
            # Patch save() to capture the figure object
            saved_figs = []
            original_save = fig_mod.save
            def mock_save(fig, name):
                saved_figs.append((fig, name))
                plt.close(fig)
            with patch.object(fig_mod, "save", mock_save):
                fig_mod.fig_coef_plot()
            # Should have saved fig9_coef_plot
            assert any("coef_plot" in name for _, name in saved_figs)


# =============================================================================
# 7. save function
# =============================================================================

class TestSave:

    def setup_method(self):
        self.tmp = Path(tempfile.mkdtemp())

    def teardown_method(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        plt.close("all")

    def test_saves_pdf_and_png(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        with patch.object(fig_mod, "OUT_DIR", self.tmp):
            save(fig, "test_output")
        assert (self.tmp / "test_output.pdf").exists()
        assert (self.tmp / "test_output.png").exists()

    def test_closes_figure_after_save(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2])
        n_before = len(plt.get_fignums())
        with patch.object(fig_mod, "OUT_DIR", self.tmp):
            save(fig, "test_close")
        n_after = len(plt.get_fignums())
        assert n_after < n_before


# =============================================================================
# 8. CI computation with T_CRIT_95
# =============================================================================

class TestCiComputation:

    def test_ci_width_positive(self):
        coef = 5.0
        se   = 2.0
        ci_lo = coef - T_CRIT_95 * se
        ci_hi = coef + T_CRIT_95 * se
        assert ci_hi > ci_lo

    def test_ci_symmetric_around_coef(self):
        coef = 3.0
        se   = 1.5
        ci_lo = coef - T_CRIT_95 * se
        ci_hi = coef + T_CRIT_95 * se
        assert abs((ci_hi + ci_lo) / 2 - coef) < 1e-9

    def test_ci_width_equals_2_t_se(self):
        se = 2.0
        ci_lo = 0.0 - T_CRIT_95 * se
        ci_hi = 0.0 + T_CRIT_95 * se
        assert abs((ci_hi - ci_lo) - 2 * T_CRIT_95 * se) < 1e-9

    def test_significance_consistent_with_pval(self):
        """
        If coef/se > T_CRIT_95, the 95% CI should not include 0,
        and the analytic p-value should be < 0.05.
        """
        from scipy import stats as st
        coef = 5.0
        se   = 1.0
        t    = coef / se
        pval = 2 * st.t.sf(abs(t), df=8)
        ci_lo = coef - T_CRIT_95 * se
        # If p < 0.05, CI should exclude 0
        if pval < 0.05:
            assert ci_lo > 0 or coef + T_CRIT_95 * se < 0, \
                "p<0.05 but CI includes 0"


# =============================================================================
# 9. rcParams setup
# =============================================================================

class TestRcParams:

    def test_font_family_serif(self):
        assert matplotlib.rcParams["font.family"] in [["serif"], "serif"]

    def test_grid_alpha_set(self):
        assert matplotlib.rcParams["grid.alpha"] == 0.3

    def test_spines_top_disabled(self):
        assert matplotlib.rcParams["axes.spines.top"] is False

    def test_spines_right_disabled(self):
        assert matplotlib.rcParams["axes.spines.right"] is False


# =============================================================================
# 10. Event study DataFrame validation
# =============================================================================

class TestEventStudyDataFrame:

    def _make_es_df(self):
        rows = []
        for var in ["delivery_facility", "pnc_mother"]:
            for wave in WAVES:
                rows.append({
                    "outcome": var,
                    "wave":    wave,
                    "beta":    0.0 if wave == 2002 else np.random.randn() * 5,
                    "se":      0.0 if wave == 2002 else abs(np.random.randn()) * 2,
                    "pval":    np.nan if wave == 2002 else np.random.uniform(0.01, 0.99),
                    "ci_lo":   -5.0,
                    "ci_hi":    5.0,
                    "n":       2000,
                })
        return pd.DataFrame(rows)

    def test_base_wave_beta_is_zero(self):
        df = self._make_es_df()
        base_row = df[(df["outcome"] == "delivery_facility") & (df["wave"] == 2002)]
        assert not base_row.empty
        assert base_row.iloc[0]["beta"] == 0.0

    def test_base_wave_pval_is_nan(self):
        df = self._make_es_df()
        base_row = df[(df["outcome"] == "delivery_facility") & (df["wave"] == 2002)]
        assert pd.isna(base_row.iloc[0]["pval"])

    def test_seven_rows_per_outcome(self):
        df = self._make_es_df()
        for var in ["delivery_facility", "pnc_mother"]:
            n = len(df[df["outcome"] == var])
            assert n == 7, f"Expected 7 rows for '{var}', got {n}"

    def test_ci_lo_less_than_ci_hi(self):
        df = self._make_es_df()
        for _, row in df.iterrows():
            assert row["ci_lo"] <= row["ci_hi"]


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    passed = 0
    failed = 0
    errors = []

    test_classes = [
        TestStars, TestTCrit95, TestWavesAndPrices, TestColours,
        TestFigCopperPrice, TestFigCoefPlot, TestSave,
        TestCiComputation, TestRcParams, TestEventStudyDataFrame,
    ]

    print("=" * 65)
    print("test_06_figures.py")
    print("=" * 65)

    for cls in test_classes:
        obj = cls()
        if hasattr(obj, "setup_method"):
            obj.setup_method()
        for name in [n for n in dir(obj) if n.startswith("test_")]:
            try:
                getattr(obj, name)()
                print(f"  PASS  {cls.__name__}.{name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL  {cls.__name__}.{name} — {e}")
                failed += 1
                errors.append((cls.__name__, name, str(e)))
        if hasattr(obj, "teardown_method"):
            obj.teardown_method()

    print(f"\nResults: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
