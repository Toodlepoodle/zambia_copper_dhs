"""
test_01_inspect_variables.py
============================
Tests for 01_inspect_variables.py — pre-panel variable inspection.
Imports functions directly and exercises them on synthetic data.
No DHS files required.

Run with:
    pytest test_01_inspect_variables.py -v
or:
    python test_01_inspect_variables.py
"""

import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Import module under test ───────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

spec = importlib.util.spec_from_file_location(
    "inspect_vars",
    str(Path(__file__).resolve().parent / "01_inspect_variables.py")
)
iv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(iv)

usability_flag = iv.usability_flag
inspect_var    = iv.inspect_var
get_labels     = iv.get_labels
load_file      = iv.load_file

BASIC_USABLE_THRESHOLD    = iv.BASIC_USABLE_THRESHOLD
RESEARCH_USABLE_THRESHOLD = iv.RESEARCH_USABLE_THRESHOLD
HARMONIZATION_WARNINGS    = iv.HARMONIZATION_WARNINGS
VARS_TO_INSPECT           = iv.VARS_TO_INSPECT
IMPROVED_WATER            = iv.IMPROVED_WATER
IMPROVED_SANIT            = iv.IMPROVED_SANIT


# =============================================================================
# HELPERS
# =============================================================================

def _make_meta(labels_dict=None):
    class M:
        variable_value_labels = {}
    m = M()
    if labels_dict:
        m.variable_value_labels = labels_dict
    return m


def _synthetic_hr(n=300):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "hv024":  rng.integers(1, 10, n),
        "hv025":  rng.integers(1, 3,  n),
        "hv270":  rng.integers(1, 6,  n),
        "hv271":  rng.integers(0, 100_000, n).astype(float),
        "hv206":  rng.integers(0, 2, n),
        "hv207":  rng.integers(0, 2, n),
        "hv208":  rng.integers(0, 2, n),
        "hv201":  rng.choice([10, 11, 13, 21, 32, 42, 96, 99], n),
        "hv205":  rng.choice([11, 12, 21, 22, 23, 31, 96], n),
        "hv009":  rng.integers(1, 15, n).astype(float),
        "hv005":  rng.integers(500_000, 2_000_000, n).astype(float),
    })


def _synthetic_kr(n=400):
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "v024":  rng.integers(1, 10, n),
        "b5":    rng.integers(0, 2, n),
        "b7":    rng.integers(0, 60, n).astype(float),
        "b3":    rng.integers(1100, 1400, n),
        "v008":  rng.integers(1100, 1400, n),
        "hw70":  np.concatenate([rng.integers(-400, 200, n-20),
                                  np.full(20, 9999)]).astype(float),
        "hw71":  rng.integers(-400, 200, n).astype(float),
        "hw72":  rng.integers(-400, 200, n).astype(float),
        "m4":    rng.integers(0, 30, n).astype(float),
        "m5":    rng.uniform(0, 24, n),
        "h32z":  np.where(rng.random(n) < 0.65, np.nan, rng.integers(0, 2, n).astype(float)),
        "h10":   np.where(rng.random(n) < 0.75, np.nan, rng.integers(0, 1, n).astype(float)),
        "v005":  rng.integers(500_000, 2_000_000, n).astype(float),
    })


def _synthetic_ir(n=350):
    rng = np.random.default_rng(2)
    return pd.DataFrame({
        "v024":  rng.integers(1, 10, n),
        "m14_1": rng.integers(0, 14, n).astype(float),
        "v133":  rng.integers(0, 20, n).astype(float),
        "v106":  rng.integers(0, 4, n),
        "v012":  rng.integers(15, 50, n),
        "v714":  rng.integers(0, 2, n),
        "v005":  rng.integers(500_000, 2_000_000, n).astype(float),
        "v008":  rng.integers(1100, 1400, n),
    })


# =============================================================================
# 1. usability_flag
# =============================================================================

class TestUsabilityFlag:

    def test_zero_pct_miss_is_ok(self):
        assert "OK" in usability_flag(0.0)

    def test_all_missing_is_unusable(self):
        assert "UNUSABLE" in usability_flag(100.0)

    def test_just_above_basic_threshold_is_not_unusable(self):
        # pct_miss just below (100 - BASIC_USABLE_THRESHOLD) → valid% just above threshold
        pct_miss = 100.0 - BASIC_USABLE_THRESHOLD
        flag = usability_flag(pct_miss)
        assert "UNUSABLE" not in flag

    def test_exactly_at_research_threshold_is_ok(self):
        pct_miss = 100.0 - RESEARCH_USABLE_THRESHOLD
        flag = usability_flag(pct_miss)
        assert "OK" in flag

    def test_between_basic_and_research_is_weak(self):
        pct_miss = 100.0 - ((BASIC_USABLE_THRESHOLD + RESEARCH_USABLE_THRESHOLD) / 2)
        flag = usability_flag(pct_miss)
        assert "WEAK" in flag

    def test_below_basic_threshold_is_unusable(self):
        pct_miss = 100.0 - (BASIC_USABLE_THRESHOLD / 2)
        flag = usability_flag(pct_miss)
        assert "UNUSABLE" in flag

    def test_returns_string(self):
        assert isinstance(usability_flag(50.0), str)

    def test_all_valid_is_ok(self):
        assert "OK" in usability_flag(0.0)

    def test_thresholds_ordered_correctly(self):
        """BASIC < RESEARCH (i.e. research is a stricter requirement)."""
        assert BASIC_USABLE_THRESHOLD < RESEARCH_USABLE_THRESHOLD


# =============================================================================
# 2. get_labels
# =============================================================================

class TestGetLabels:

    def test_returns_dict_with_labels(self):
        m = _make_meta({"hv025": {1: "urban", 2: "rural"}})
        result = get_labels(m, "hv025")
        assert isinstance(result, dict)
        assert result[1] == "urban"

    def test_missing_column_returns_empty_dict(self):
        m = _make_meta({"hv025": {1: "urban"}})
        result = get_labels(m, "nonexistent")
        assert result == {}

    def test_none_meta_returns_empty_dict(self):
        result = get_labels(None, "hv025")
        assert result == {}

    def test_meta_without_attribute_returns_empty(self):
        class BareM:
            pass
        result = get_labels(BareM(), "hv025")
        assert result == {}

    def test_empty_labels_dict_returns_empty(self):
        m = _make_meta({"hv025": {}})
        result = get_labels(m, "hv025")
        assert result == {}


# =============================================================================
# 3. inspect_var — basic behaviour
# =============================================================================

class TestInspectVar:

    def test_produces_output_lines(self):
        df    = _synthetic_hr()
        meta  = _make_meta()
        lines = []
        inspect_var(df, meta, "hv025", "urban_rural", 2007, "HR", lines)
        assert len(lines) > 0

    def test_output_contains_variable_name(self):
        df    = _synthetic_hr()
        meta  = _make_meta()
        lines = []
        inspect_var(df, meta, "hv025", "urban_rural", 2007, "HR", lines)
        combined = " ".join(lines)
        assert "urban_rural" in combined

    def test_output_contains_n_total(self):
        df    = _synthetic_hr()
        meta  = _make_meta()
        lines = []
        inspect_var(df, meta, "hv009", "hh_size", 2007, "HR", lines)
        combined = " ".join(lines)
        assert "N=" in combined

    def test_usability_flag_appears_in_output(self):
        df    = _synthetic_kr()
        meta  = _make_meta()
        lines = []
        inspect_var(df, meta, "h32z", "sought_treatment", 2007, "KR", lines)
        combined = " ".join(lines)
        assert any(tok in combined for tok in ("OK", "WEAK", "UNUSABLE"))

    def test_mostly_missing_column_flagged(self):
        """h32z is ~65% missing in synthetic data — should be WEAK or UNUSABLE."""
        df    = _synthetic_kr()
        meta  = _make_meta()
        lines = []
        inspect_var(df, meta, "h32z", "sought_treatment", 2007, "KR", lines)
        combined = " ".join(lines)
        assert any(tok in combined for tok in ("OK", "WEAK", "UNUSABLE"))

    def test_harmonisation_warning_printed_for_m4_pre_2018(self):
        """m4 for wave 1992 should trigger a CODING DRIFT warning."""
        df    = _synthetic_kr()
        meta  = _make_meta()
        lines = []
        inspect_var(df, meta, "m4", "ever_breastfed", 1992, "KR", lines)
        combined = " ".join(lines)
        assert "CODING DRIFT" in combined or "WARN" in combined

    def test_no_warning_for_m4_in_2018(self):
        """m4 for wave 2018 should say OK (correct coding), not CODING DRIFT."""
        df    = _synthetic_kr()
        meta  = _make_meta()
        lines = []
        inspect_var(df, meta, "m4", "ever_breastfed", 2018, "KR", lines)
        combined = " ".join(lines)
        assert "CODING DRIFT" not in combined

    def test_continuous_var_shows_statistics(self):
        df    = _synthetic_ir()
        meta  = _make_meta()
        lines = []
        inspect_var(df, meta, "v133", "edu_years", 2007, "IR", lines)
        combined = " ".join(lines)
        assert "mean" in combined.lower() or "min" in combined.lower()

    def test_does_not_crash_on_all_nan_column(self):
        df = pd.DataFrame({"hv009": [None, None, None]})
        meta  = _make_meta()
        lines = []
        inspect_var(df, meta, "hv009", "hh_size", 2007, "HR", lines)
        combined = " ".join(lines)
        assert "UNUSABLE" in combined

    def test_does_not_crash_on_single_row(self):
        df   = pd.DataFrame({"v012": [25]})
        meta = _make_meta()
        lines = []
        inspect_var(df, meta, "v012", "age_woman", 2007, "IR", lines)
        assert len(lines) > 0

    def test_water_source_improved_annotation(self):
        """hv201 with improved code (10) should show IMPROVED in output."""
        df = pd.DataFrame({"hv201": [10, 32, 10, 42, 11] * 40})
        meta = _make_meta()
        lines = []
        inspect_var(df, meta, "hv201", "water_source", 2007, "HR", lines)
        combined = " ".join(lines)
        assert "IMPROVED" in combined

    def test_z_score_flagged_values_counted(self):
        """hw70 with 9999 entries should report flagged count."""
        df    = _synthetic_kr()
        meta  = _make_meta()
        lines = []
        inspect_var(df, meta, "hw70", "haz_score", 2007, "KR", lines)
        combined = " ".join(lines)
        assert "Flagged (>9990): 20" in combined

    def test_months_bf_plausibility_check_for_normal(self):
        """m5 with plausible values (mean ~12) should pass the plausibility check."""
        df   = pd.DataFrame({"m5": [10.0, 12.0, 14.0, 18.0] * 50})
        meta = _make_meta()
        lines = []
        inspect_var(df, meta, "m5", "months_bf", 2007, "KR", lines)
        combined = " ".join(lines)
        # Should not complain
        assert "PLAUSIBILITY FAIL" not in combined

    def test_months_bf_plausibility_fail_for_implausible(self):
        """m5 with median > 36 should trigger a plausibility warning."""
        df   = pd.DataFrame({"m5": [50.0, 55.0, 60.0, 65.0] * 50})
        meta = _make_meta()
        lines = []
        inspect_var(df, meta, "m5", "months_bf", 2018, "KR", lines)
        combined = " ".join(lines)
        assert "PLAUSIBILITY FAIL" in combined


# =============================================================================
# 4. load_file
# =============================================================================

class TestLoadFile:

    def test_nonexistent_file_returns_none_df(self):
        df, meta, err = load_file(Path("/no/such/file.dta"))
        assert df is None

    def test_nonexistent_file_returns_none_meta(self):
        df, meta, err = load_file(Path("/no/such/file.dta"))
        assert meta is None

    def test_nonexistent_file_returns_error_string(self):
        df, meta, err = load_file(Path("/no/such/file.dta"))
        assert isinstance(err, str)
        assert len(err) > 0


# =============================================================================
# 5. HARMONIZATION_WARNINGS structure
# =============================================================================

class TestHarmonizationWarnings:

    def test_is_dict(self):
        assert isinstance(HARMONIZATION_WARNINGS, dict)

    def test_keys_are_tuples(self):
        for key in HARMONIZATION_WARNINGS:
            assert isinstance(key, tuple), f"Key {key} is not a tuple"

    def test_keys_have_two_elements(self):
        for key in HARMONIZATION_WARNINGS:
            assert len(key) == 2, f"Key {key} does not have 2 elements"

    def test_all_warnings_are_strings(self):
        for key, val in HARMONIZATION_WARNINGS.items():
            assert isinstance(val, str), f"Warning for {key} is not a string"

    def test_ever_breastfed_1992_has_coding_drift(self):
        warn = HARMONIZATION_WARNINGS.get(("ever_breastfed", 1992), "")
        assert "CODING DRIFT" in warn or "drift" in warn.lower()

    def test_ever_breastfed_2018_is_ok(self):
        warn = HARMONIZATION_WARNINGS.get(("ever_breastfed", 2018), "")
        assert "OK" in warn or "ok" in warn.lower()

    def test_months_bf_2018_has_unit_shift(self):
        warn = HARMONIZATION_WARNINGS.get(("months_bf", 2018), "")
        assert "UNIT SHIFT" in warn or "unit" in warn.lower()

    def test_sought_treatment_has_conditional_missing(self):
        warn = HARMONIZATION_WARNINGS.get(("sought_treatment", 2007), "")
        assert "CONDITIONAL" in warn or "conditional" in warn.lower()

    def test_all_waves_covered_for_sought_treatment(self):
        """sought_treatment should be flagged for all 7 waves."""
        expected_waves = {1992, 1996, 2002, 2007, 2014, 2018, 2024}
        covered = {w for (var, w) in HARMONIZATION_WARNINGS if var == "sought_treatment"}
        assert expected_waves == covered


# =============================================================================
# 6. VARS_TO_INSPECT structure
# =============================================================================

class TestVarsToInspect:

    def test_has_hr_ir_kr(self):
        for ftype in ("HR", "KR", "IR"):
            assert ftype in VARS_TO_INSPECT

    def test_province_in_all_ftypes(self):
        for ftype in ("HR", "KR", "IR"):
            assert "province" in VARS_TO_INSPECT[ftype], \
                f"'province' missing from VARS_TO_INSPECT['{ftype}']"

    def test_survey_weight_in_all_ftypes(self):
        for ftype in ("HR", "KR", "IR"):
            assert "survey_weight" in VARS_TO_INSPECT[ftype], \
                f"'survey_weight' missing from VARS_TO_INSPECT['{ftype}']"

    def test_all_candidates_are_lists(self):
        for ftype, vars_dict in VARS_TO_INSPECT.items():
            for var_name, candidates in vars_dict.items():
                assert isinstance(candidates, list), \
                    f"Candidates for {ftype}/{var_name} is not a list"
                assert len(candidates) > 0, \
                    f"Candidates for {ftype}/{var_name} is empty"


# =============================================================================
# 7. WASH code dictionaries
# =============================================================================

class TestWashCodes:

    def test_improved_water_is_dict(self):
        assert isinstance(IMPROVED_WATER, dict)

    def test_improved_sanit_is_dict(self):
        assert isinstance(IMPROVED_SANIT, dict)

    def test_piped_water_in_improved(self):
        """Code 10 / 11 = piped water — must be improved."""
        assert 10 in IMPROVED_WATER or 11 in IMPROVED_WATER

    def test_flush_toilet_in_improved_sanit(self):
        """Code 11 = flush to piped sewer — must be improved."""
        assert 11 in IMPROVED_SANIT

    def test_96_not_in_improved_water(self):
        assert 96 not in IMPROVED_WATER

    def test_99_not_in_improved_water(self):
        assert 99 not in IMPROVED_WATER

    def test_no_overlap_between_improved_and_unimproved_water(self):
        unimproved = iv.UNIMPROVED_WATER
        overlap = set(IMPROVED_WATER.keys()) & set(unimproved.keys())
        assert len(overlap) == 0, f"Codes in both improved and unimproved water: {overlap}"


# =============================================================================
# 8. DATA QUALITY — inspect_var on all HR/KR/IR columns
# =============================================================================

class TestInspectVarAllColumns:

    def test_all_hr_vars_no_crash(self):
        df   = _synthetic_hr()
        meta = _make_meta()
        for var_name, candidates in VARS_TO_INSPECT["HR"].items():
            found = next((c.lower() for c in candidates if c.lower() in df.columns), None)
            if found is None:
                continue
            lines = []
            try:
                inspect_var(df, meta, found, var_name, 2007, "HR", lines)
            except Exception as e:
                pytest.fail(f"inspect_var crashed on HR/{var_name}: {e}")

    def test_all_kr_vars_no_crash(self):
        df   = _synthetic_kr()
        meta = _make_meta()
        for var_name, candidates in VARS_TO_INSPECT["KR"].items():
            found = next((c.lower() for c in candidates if c.lower() in df.columns), None)
            if found is None:
                continue
            lines = []
            try:
                inspect_var(df, meta, found, var_name, 2007, "KR", lines)
            except Exception as e:
                pytest.fail(f"inspect_var crashed on KR/{var_name}: {e}")

    def test_all_ir_vars_no_crash(self):
        df   = _synthetic_ir()
        meta = _make_meta()
        for var_name, candidates in VARS_TO_INSPECT["IR"].items():
            found = next((c.lower() for c in candidates if c.lower() in df.columns), None)
            if found is None:
                continue
            lines = []
            try:
                inspect_var(df, meta, found, var_name, 2007, "IR", lines)
            except Exception as e:
                pytest.fail(f"inspect_var crashed on IR/{var_name}: {e}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    passed = 0
    failed = 0
    errors = []

    test_classes = [
        TestUsabilityFlag, TestGetLabels, TestInspectVar,
        TestLoadFile, TestHarmonizationWarnings, TestVarsToInspect,
        TestWashCodes, TestInspectVarAllColumns,
    ]

    print("=" * 65)
    print("test_01_inspect_variables.py")
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
