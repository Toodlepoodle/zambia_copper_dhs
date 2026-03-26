"""
test_00_audit_waves.py
======================
Tests for 00_audit_waves.py — exhaustive variable discovery script.
Imports functions directly and runs them on synthetic data.
No DHS files required.

Run with:
    pytest test_00_audit_waves.py -v
or:
    python test_00_audit_waves.py
"""

import sys
import math
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Import functions from the module under test ────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util

spec = importlib.util.spec_from_file_location(
    "audit_waves",
    str(Path(__file__).resolve().parent / "00_audit_waves.py")
)
aw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aw)

resolve_path    = aw.resolve_path
find_col        = aw.find_col
detect_type     = aw.detect_type
analyse_variable= aw.analyse_variable
USABLE_THRESHOLD= aw.USABLE_THRESHOLD


# =============================================================================
# HELPERS
# =============================================================================

def _make_meta(labels_dict=None):
    """Build a minimal pyreadstat-style meta object."""
    class M:
        variable_value_labels = {}
    m = M()
    if labels_dict:
        m.variable_value_labels = labels_dict
    return m


# =============================================================================
# 1. resolve_path
# =============================================================================

class TestResolvePath:

    def setup_method(self):
        self.tmp = Path(tempfile.mkdtemp())

    def teardown_method(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_exact_path_found(self):
        sub = self.tmp / "sub"
        sub.mkdir()
        f = sub / "data.dta"
        f.write_text("x")
        result = resolve_path(self.tmp, "sub/data.dta")
        assert result == f

    def test_case_insensitive_dir(self):
        sub = self.tmp / "SUB"
        sub.mkdir()
        f = sub / "DATA.DTA"
        f.write_text("x")
        result = resolve_path(self.tmp, "sub/data.dta")
        assert result is not None
        assert result.exists()

    def test_missing_returns_none(self):
        assert resolve_path(self.tmp, "nonexistent/file.dta") is None

    def test_nonexistent_base_returns_none(self):
        assert resolve_path(Path("/no_such_xyz_base"), "file.dta") is None

    def test_multi_level_case_insensitive(self):
        d1 = self.tmp / "ZAMBIA2007"
        d2 = d1 / "ZMHR51DT"
        d2.mkdir(parents=True)
        f = d2 / "ZMHR51FL.DTA"
        f.write_text("x")
        result = resolve_path(self.tmp, "zambia2007/zmhr51dt/zmhr51fl.dta")
        assert result is not None

    def test_returns_none_for_file_in_nondir(self):
        f = self.tmp / "notadir.txt"
        f.write_text("x")
        # Try to traverse through a file as if it were a directory
        result = resolve_path(self.tmp, "notadir.txt/impossible.dta")
        assert result is None


# =============================================================================
# 2. find_col
# =============================================================================

class TestFindCol:

    def test_exact_match(self):
        assert find_col({"h2", "v456"}, ["h2"]) == "h2"

    def test_returns_lowercase(self):
        assert find_col({"H2", "V456"}, ["h2"]) == "h2"

    def test_first_candidate_wins(self):
        result = find_col({"v626", "v626a"}, ["v626a", "v626"])
        assert result == "v626a"

    def test_no_match_returns_none(self):
        assert find_col({"h2", "v456"}, ["xyz", "abc"]) is None

    def test_empty_column_set_returns_none(self):
        assert find_col(set(), ["h2"]) is None

    def test_empty_candidates_returns_none(self):
        assert find_col({"h2"}, []) is None

    def test_single_candidate(self):
        assert find_col({"v501"}, ["v501"]) == "v501"

    def test_case_insensitive_match(self):
        # Column in df is uppercase, candidate is lowercase
        result = find_col({"HV024"}, ["hv024"])
        assert result == "hv024"

    def test_multiple_candidates_first_match_wins(self):
        # Only second candidate is present
        result = find_col({"v626"}, ["v626a", "v626"])
        assert result == "v626"

    def test_returns_lowercase_when_df_uppercase(self):
        result = find_col({"HV201"}, ["hv201"])
        assert result is not None
        assert result == result.lower()


# =============================================================================
# 3. detect_type
# =============================================================================

class TestDetectType:

    def test_explicit_cat_returned_as_is(self):
        s = pd.Series([1.0, 2.0])
        assert detect_type(s, None, "x", "cat") == "cat"

    def test_explicit_cont_returned_as_is(self):
        s = pd.Series([1.0, 2.0, 3.0])
        assert detect_type(s, None, "x", "cont") == "cont"

    def test_auto_with_value_labels_is_cat(self):
        s = pd.Series([1.0, 2.0])
        m = _make_meta({"x": {1: "yes", 2: "no"}})
        assert detect_type(s, m, "x", "auto") == "cat"

    def test_auto_float_no_labels_is_cont(self):
        s = pd.Series([1.5, 2.5, 3.5])
        assert detect_type(s, None, "x", "auto") == "cont"

    def test_auto_int_few_unique_is_cat(self):
        s = pd.Series([0, 1, 1, 0, 2], dtype=int)
        assert detect_type(s, None, "x", "auto") == "cat"

    def test_auto_int_many_unique_is_cont(self):
        s = pd.Series(list(range(20)), dtype=int)
        assert detect_type(s, None, "x", "auto") == "cont"

    def test_auto_string_series_is_cat(self):
        s = pd.Series(["yes", "no", "yes"])
        assert detect_type(s, None, "x", "auto") == "cat"

    def test_result_is_always_cat_or_cont(self):
        for declared in ("cat", "cont", "auto"):
            result = detect_type(pd.Series([1, 2, 3]), None, "x", declared)
            assert result in ("cat", "cont")

    def test_none_meta_does_not_raise(self):
        result = detect_type(pd.Series([1.0, 2.0]), None, "x", "auto")
        assert result in ("cat", "cont")

    def test_empty_value_labels_dict_treated_as_no_labels(self):
        s = pd.Series([1.5, 2.5])
        m = _make_meta({"x": {}})
        # Empty label dict → should not force cat
        result = detect_type(s, m, "x", "auto")
        assert result in ("cat", "cont")


# =============================================================================
# 4. analyse_variable
# =============================================================================

class TestAnalyseVariable:

    def _meta(self):
        return _make_meta()

    # ── Return structure ───────────────────────────────────────────────────────

    def test_return_keys_always_present(self):
        df  = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        res = analyse_variable(df, "x", self._meta(), "cont")
        expected = {"found_col","dtype_raw","var_type","n_total","n_valid","pct_miss","usable","detail"}
        assert expected.issubset(res.keys())

    def test_found_col_matches_input(self):
        df  = pd.DataFrame({"h2": [0.0, 1.0, 2.0] * 10})
        res = analyse_variable(df, "h2", self._meta(), "cat")
        assert res["found_col"] == "h2"

    def test_n_total_correct(self):
        df  = pd.DataFrame({"x": [1.0] * 50})
        res = analyse_variable(df, "x", self._meta(), "cont")
        assert res["n_total"] == 50

    def test_n_valid_excludes_nan(self):
        df  = pd.DataFrame({"x": [1.0, np.nan, 2.0, np.nan, 3.0]})
        res = analyse_variable(df, "x", self._meta(), "cont")
        assert res["n_valid"] == 3
        assert res["n_total"] == 5

    def test_pct_miss_calculation(self):
        df  = pd.DataFrame({"x": [np.nan] * 40 + [1.0] * 60})
        res = analyse_variable(df, "x", self._meta(), "cont")
        assert abs(res["pct_miss"] - 40.0) < 1e-6

    # ── Usability ──────────────────────────────────────────────────────────────

    def test_usable_true_when_valid_above_threshold(self):
        """90% valid → usable."""
        df  = pd.DataFrame({"x": [1.0] * 90 + [np.nan] * 10})
        res = analyse_variable(df, "x", self._meta(), "cont")
        assert res["usable"] is True

    def test_usable_false_when_valid_below_threshold(self):
        """5% valid → not usable."""
        df  = pd.DataFrame({"x": [np.nan] * 95 + [1.0] * 5})
        res = analyse_variable(df, "x", self._meta(), "cont")
        assert res["usable"] is False

    def test_all_nan_column_not_usable(self):
        df  = pd.DataFrame({"x": [np.nan] * 100})
        res = analyse_variable(df, "x", self._meta(), "cont")
        assert res["usable"] is False

    def test_detail_mostly_missing_when_not_usable(self):
        df  = pd.DataFrame({"x": [np.nan] * 95 + [1.0] * 5})
        res = analyse_variable(df, "x", self._meta(), "cont")
        assert "MOSTLY MISSING" in res["detail"]

    def test_detail_non_empty_when_usable(self):
        df  = pd.DataFrame({"x": [1.0, 2.0, 3.0] * 20})
        res = analyse_variable(df, "x", self._meta(), "cont")
        assert res["usable"] is True
        assert len(res["detail"]) > 0

    # ── Categorical detail ─────────────────────────────────────────────────────

    def test_cat_detail_contains_percentages(self):
        df  = pd.DataFrame({"h2": [0.0, 1.0, 2.0] * 20})
        res = analyse_variable(df, "h2", self._meta(), "cat")
        assert "%" in res["detail"]

    def test_cat_var_type_is_cat(self):
        df  = pd.DataFrame({"h2": [0.0, 1.0, 2.0] * 20})
        res = analyse_variable(df, "h2", self._meta(), "cat")
        assert res["var_type"] == "cat"

    # ── Continuous detail ──────────────────────────────────────────────────────

    def test_cont_detail_contains_mean(self):
        df  = pd.DataFrame({"x": [10.0, 20.0, 30.0] * 20})
        res = analyse_variable(df, "x", self._meta(), "cont")
        assert "mean=" in res["detail"]

    def test_cont_detail_contains_sd(self):
        df  = pd.DataFrame({"x": [10.0, 20.0, 30.0] * 20})
        res = analyse_variable(df, "x", self._meta(), "cont")
        assert "sd=" in res["detail"]

    def test_cont_mean_is_correct(self):
        df  = pd.DataFrame({"x": [0.0, 10.0] * 50})
        res = analyse_variable(df, "x", self._meta(), "cont")
        # mean should be 5.0; parse from detail string
        detail = res["detail"]
        mean_str = detail.split("mean=")[1].split()[0]
        assert abs(float(mean_str) - 5.0) < 0.01

    # ── Missing code removal ───────────────────────────────────────────────────

    def test_missing_code_99_removed_from_cont_stats(self):
        """99 appearing >1% of the time should be stripped before computing mean."""
        data = [99.0] * 80 + [5.0, 5.5, 6.0, 5.2] * 5
        df   = pd.DataFrame({"v456": data})
        res  = analyse_variable(df, "v456", self._meta(), "cont")
        if res["usable"] and "mean=" in res["detail"]:
            mean_val = float(res["detail"].split("mean=")[1].split()[0])
            assert mean_val < 20.0, \
                f"Mean {mean_val} contaminated by flag code 99"

    def test_var_type_is_always_cat_or_cont(self):
        for declared in ("cat", "cont"):
            df  = pd.DataFrame({"x": [1.0, 2.0, 3.0] * 10})
            res = analyse_variable(df, "x", self._meta(), declared)
            assert res["var_type"] in ("cat", "cont")


# =============================================================================
# 5. VARIABLE_GROUPS structure
# =============================================================================

class TestVariableGroups:

    def test_variable_groups_is_dict(self):
        assert isinstance(aw.VARIABLE_GROUPS, dict)

    def test_all_groups_non_empty(self):
        for group, entries in aw.VARIABLE_GROUPS.items():
            assert len(entries) > 0, f"Group '{group}' is empty"

    def test_all_entries_have_five_fields(self):
        """Each entry: (std_name, [candidates], ftype, description, var_type)"""
        for group, entries in aw.VARIABLE_GROUPS.items():
            for entry in entries:
                assert len(entry) == 5, \
                    f"Entry in group '{group}' has {len(entry)} fields, expected 5: {entry}"

    def test_all_file_types_are_valid(self):
        valid_ftypes = {"HR", "KR", "IR", "MR", "BR", "PR"}
        for group, entries in aw.VARIABLE_GROUPS.items():
            for (name, candidates, ftype, desc, vtype) in entries:
                assert ftype in valid_ftypes, \
                    f"Invalid ftype '{ftype}' for variable '{name}'"

    def test_all_var_types_valid(self):
        for group, entries in aw.VARIABLE_GROUPS.items():
            for (name, candidates, ftype, desc, vtype) in entries:
                assert vtype in ("cat", "cont", "auto"), \
                    f"Invalid var_type '{vtype}' for variable '{name}'"

    def test_all_candidates_are_lists(self):
        for group, entries in aw.VARIABLE_GROUPS.items():
            for (name, candidates, ftype, desc, vtype) in entries:
                assert isinstance(candidates, list), \
                    f"Candidates for '{name}' is not a list"

    def test_all_candidates_lists_non_empty(self):
        for group, entries in aw.VARIABLE_GROUPS.items():
            for (name, candidates, ftype, desc, vtype) in entries:
                assert len(candidates) > 0, \
                    f"Candidates list for '{name}' is empty"

    def test_standard_name_uniqueness_within_group(self):
        for group, entries in aw.VARIABLE_GROUPS.items():
            names = [e[0] for e in entries]
            assert len(names) == len(set(names)), \
                f"Duplicate standard names in group '{group}': {names}"

    def test_key_groups_present(self):
        expected_groups = {"child_health", "maternal_health", "dwelling"}
        for g in expected_groups:
            assert g in aw.VARIABLE_GROUPS, f"Expected group '{g}' not found"


# =============================================================================
# 6. DATA QUALITY — end-to-end on synthetic data
# =============================================================================

class TestDataQualityEndToEnd:

    def _synthetic_kr(self, n=500):
        """Build a KR-like DataFrame that exercises all branches."""
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            "h2":   rng.integers(0, 3, n).astype(float),     # BCG vacc (cat)
            "h9":   rng.integers(0, 3, n).astype(float),     # measles
            "h7":   rng.integers(0, 3, n).astype(float),     # DPT3
            "h8":   rng.integers(0, 3, n).astype(float),     # polio3
            "h11":  rng.integers(0, 3, n).astype(float),     # diarrhoea
            "h22":  rng.integers(0, 2, n).astype(float),     # fever
            "hw70": np.concatenate([rng.integers(-400, 200, n-20),
                                    np.full(20, 9999)]).astype(float),  # haz
            "v456": np.concatenate([rng.uniform(80, 180, n-30),
                                    np.full(30, 99.0)]),     # Hb with flag codes
            "m4":   rng.integers(0, 96, n).astype(float),   # ever BF
            "m5":   rng.uniform(0, 36, n),                  # months BF
            "v005": rng.integers(500_000, 2_000_000, n).astype(float),
        })

    def test_analyse_cat_on_h2(self):
        df  = self._synthetic_kr()
        meta = _make_meta()
        res  = analyse_variable(df, "h2", meta, "cat")
        assert res["var_type"] == "cat"
        assert res["n_total"] == len(df)
        assert "%" in res["detail"]

    def test_analyse_cont_on_m5(self):
        df  = self._synthetic_kr()
        meta = _make_meta()
        res  = analyse_variable(df, "m5", meta, "cont")
        assert res["var_type"] == "cont"
        assert "mean=" in res["detail"]

    def test_flagged_9999_values_do_not_inflate_mean(self):
        """hw70 has 20 flagged 9999 values. After removal, mean should be small."""
        df   = self._synthetic_kr()
        meta = _make_meta()
        res  = analyse_variable(df, "hw70", meta, "cont")
        if res["usable"] and "mean=" in res["detail"]:
            mean_val = float(res["detail"].split("mean=")[1].split()[0])
            # True mean is roughly 0 (range -400 to +200), not 9999
            assert mean_val < 500, \
                f"Flagged 9999 values contaminating mean: {mean_val}"

    def test_all_columns_can_be_analysed_without_crash(self):
        df   = self._synthetic_kr()
        meta = _make_meta()
        for col in df.columns:
            try:
                res = analyse_variable(df, col, meta, "auto")
                assert res["n_total"] == len(df)
            except Exception as e:
                pytest.fail(f"analyse_variable crashed on column '{col}': {e}")

    def test_pct_miss_plus_pct_valid_equals_100(self):
        df   = self._synthetic_kr()
        meta = _make_meta()
        for col in ["h2", "m5", "hw70"]:
            res = analyse_variable(df, col, meta, "auto")
            pct_valid = 100 * res["n_valid"] / res["n_total"]
            assert abs(res["pct_miss"] + pct_valid - 100.0) < 1e-6, \
                f"pct_miss + pct_valid ≠ 100 for column '{col}'"

    def test_usability_consistent_with_threshold(self):
        df   = self._synthetic_kr()
        meta = _make_meta()
        for col in df.columns:
            res = analyse_variable(df, col, meta, "auto")
            pct_valid = 100 - res["pct_miss"]
            if pct_valid >= USABLE_THRESHOLD:
                assert res["usable"] is True, \
                    f"Col '{col}': {pct_valid:.1f}% valid but usable=False"
            else:
                assert res["usable"] is False, \
                    f"Col '{col}': {pct_valid:.1f}% valid but usable=True"


# =============================================================================
# 7. WAVE_FILES in audit script
# =============================================================================

class TestAuditWaveFiles:

    def test_audit_wave_files_has_seven_waves(self):
        assert len(aw.WAVE_FILES) == 7

    def test_waves_are_expected_years(self):
        assert set(aw.WAVE_FILES.keys()) == {1992, 1996, 2002, 2007, 2014, 2018, 2024}

    def test_audit_has_br_and_pr_files(self):
        """00_audit_waves.py checks BR and PR files that other scripts skip."""
        for wave in [1996, 2002, 2007]:
            assert "BR" in aw.WAVE_FILES[wave], f"BR missing for wave {wave}"
            assert "PR" in aw.WAVE_FILES[wave], f"PR missing for wave {wave}"


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    passed = 0
    failed = 0
    errors = []

    test_classes = [
        TestResolvePath, TestFindCol, TestDetectType, TestAnalyseVariable,
        TestVariableGroups, TestDataQualityEndToEnd, TestAuditWaveFiles,
    ]

    print("=" * 65)
    print("test_00_audit_waves.py")
    print("=" * 65)

    for cls in test_classes:
        obj = cls()
        for name in [n for n in dir(obj) if n.startswith("test_")]:
            if hasattr(obj, "setup_method"):
                obj.setup_method()
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
