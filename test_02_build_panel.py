"""
test_02_build_panel.py
======================
Tests for 02_build_panel.py — province-level panel builder.
Imports every helper function and runs them on synthetic data.
No DHS files required.

Run with:
    pytest test_02_build_panel.py -v
or:
    python test_02_build_panel.py
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
    "build_panel",
    str(Path(__file__).resolve().parent / "02_build_panel.py")
)
bp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bp)

to_num      = bp.to_num
flag_nan    = bp.flag_nan
harmonise   = bp.harmonise
is_mining   = bp.is_mining
binary      = bp.binary
safe_recode = bp.safe_recode
isin_vec    = bp.isin_vec
wt_mean     = bp.wt_mean
collapse    = bp.collapse

from config_00 import (
    DHS_MISSING_CODES, IMPROVED_WATER_CODES, IMPROVED_SANIT_CODES,
    IMPROVED_FLOOR_CODES, PROVINCE_HARMONISE, MINING_PROVINCES,
    COPPER_PRICES,
)


# =============================================================================
# 1. to_num
# =============================================================================

class TestToNum:

    def test_numeric_string_converts(self):
        r = to_num(pd.Series(["1", "2", "3"]))
        assert list(r) == [1.0, 2.0, 3.0]

    def test_non_numeric_becomes_nan(self):
        r = to_num(pd.Series(["abc", "xyz"]))
        assert r.isna().all()

    def test_none_becomes_nan(self):
        r = to_num(pd.Series([None]))
        assert pd.isna(r[0])

    def test_mixed_series(self):
        r = to_num(pd.Series([1, "x", 3.5]))
        assert r[0] == 1.0
        assert pd.isna(r[1])
        assert r[2] == 3.5

    def test_float_passthrough(self):
        r = to_num(pd.Series([1.5, 2.5]))
        assert list(r) == [1.5, 2.5]

    def test_integer_passthrough(self):
        r = to_num(pd.Series([1, 2, 3]))
        assert list(r) == [1.0, 2.0, 3.0]


# =============================================================================
# 2. flag_nan
# =============================================================================

class TestFlagNan:

    def test_at_threshold_kept(self):
        r = flag_nan(pd.Series([9990]))
        assert r[0] == 9990.0

    def test_above_threshold_is_nan(self):
        r = flag_nan(pd.Series([9991]))
        assert pd.isna(r[0])

    def test_normal_values_unchanged(self):
        assert list(flag_nan(pd.Series([0, 100, 500]))) == [0.0, 100.0, 500.0]

    def test_custom_threshold(self):
        r = flag_nan(pd.Series([5, 10, 15]), threshold=10)
        assert r[0] == 5.0
        assert r[1] == 10.0
        assert pd.isna(r[2])

    def test_nan_input_stays_nan(self):
        r = flag_nan(pd.Series([np.nan]))
        assert pd.isna(r[0])

    def test_hb_flag_code(self):
        r = flag_nan(pd.Series([120, 145, 9991]))
        assert r[0] == 120.0
        assert r[1] == 145.0
        assert pd.isna(r[2])


# =============================================================================
# 3. harmonise
# =============================================================================

class TestHarmonise:

    def test_muchinga_to_northern(self):
        assert harmonise("muchinga") == "northern"

    def test_muchinga_titlecase(self):
        assert harmonise("Muchinga") == "northern"

    def test_northern_stays_northern(self):
        assert harmonise("Northern") == "northern"

    def test_north_western_hyphen(self):
        assert harmonise("north-western") == "northwestern"

    def test_north_western_space(self):
        assert harmonise("north western") == "northwestern"

    def test_northwestern_no_space(self):
        assert harmonise("northwestern") == "northwestern"

    def test_copperbelt(self):
        assert harmonise("copperbelt") == "copperbelt"

    def test_lusaka(self):
        assert harmonise("lusaka") == "lusaka"

    def test_strips_leading_trailing_whitespace(self):
        assert harmonise("  central  ") == "central"

    def test_case_insensitive(self):
        assert harmonise("LUSAKA") == "lusaka"

    def test_unknown_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown province label"):
            harmonise("atlantis")

    def test_exactly_nine_unique_outputs(self):
        """After all mappings, there are exactly 9 harmonised provinces."""
        outputs = set(PROVINCE_HARMONISE[k] for k in PROVINCE_HARMONISE)
        assert len(outputs) == 9

    def test_all_harmonised_values_reachable(self):
        for raw in PROVINCE_HARMONISE:
            result = harmonise(raw)
            assert isinstance(result, str) and len(result) > 0


# =============================================================================
# 4. is_mining
# =============================================================================

class TestIsMining:

    def test_copperbelt_is_1(self):
        assert is_mining("copperbelt") == 1

    def test_northwestern_is_1(self):
        assert is_mining("northwestern") == 1

    def test_lusaka_is_0(self):
        assert is_mining("lusaka") == 0

    def test_northern_is_0(self):
        assert is_mining("northern") == 0

    def test_returns_int(self):
        assert isinstance(is_mining("copperbelt"), int)

    def test_all_non_mining_provinces_return_0(self):
        non_mining = set(PROVINCE_HARMONISE.values()) - MINING_PROVINCES
        for p in non_mining:
            assert is_mining(p) == 0, f"Expected 0 for '{p}'"


# =============================================================================
# 5. binary
# =============================================================================

class TestBinary:

    def test_true_val_gives_1(self):
        r = binary(pd.Series([1]), [1], [0])
        assert r[0] == 1.0

    def test_false_val_gives_0(self):
        r = binary(pd.Series([0]), [1], [0])
        assert r[0] == 0.0

    def test_dhs_code_8_is_nan(self):
        assert pd.isna(binary(pd.Series([8]), [1], [0])[0])

    def test_dhs_code_9_is_nan(self):
        assert pd.isna(binary(pd.Series([9]), [1], [0])[0])

    def test_dhs_code_96_is_nan(self):
        assert pd.isna(binary(pd.Series([96]), [1], [0])[0])

    def test_dhs_code_99_is_nan(self):
        assert pd.isna(binary(pd.Series([99]), [1], [0])[0])

    def test_unlisted_code_is_nan(self):
        assert pd.isna(binary(pd.Series([5]), [1], [0])[0])

    def test_nan_input_stays_nan(self):
        assert pd.isna(binary(pd.Series([np.nan]), [1], [0])[0])

    def test_vaccination_card_and_recall(self):
        r = binary(pd.Series([0, 1, 2, 8]), [1, 2], [0])
        assert r[0] == 0.0
        assert r[1] == 1.0
        assert r[2] == 1.0
        assert pd.isna(r[3])

    def test_urban_rural(self):
        r = binary(pd.Series([1, 2]), [1], [2])
        assert r[0] == 1.0
        assert r[1] == 0.0

    def test_all_dhs_missing_codes_are_nan(self):
        for code in DHS_MISSING_CODES:
            r = binary(pd.Series([code]), [1], [0])
            assert pd.isna(r[0]), f"DHS missing code {code} must produce NaN"

    def test_output_dtype_is_float(self):
        r = binary(pd.Series([0, 1]), [1], [0])
        assert r.dtype == float


# =============================================================================
# 6. safe_recode
# =============================================================================

class TestSafeRecode:

    def test_known_code_maps_correctly(self):
        r = safe_recode(pd.Series([1]), {1: 99})
        assert r[0] == 99

    def test_unknown_code_is_nan(self):
        assert pd.isna(safe_recode(pd.Series([5]), {1: 1})[0])

    def test_dhs_code_8_is_nan(self):
        assert pd.isna(safe_recode(pd.Series([8]), {8: 1})[0])

    def test_dhs_code_99_is_nan(self):
        assert pd.isna(safe_recode(pd.Series([99]), {99: 1})[0])

    def test_above_threshold_is_nan(self):
        assert pd.isna(safe_recode(pd.Series([9991]), {9991: 1})[0])

    def test_facility_delivery_coding(self):
        fac_map = {**{i: 0 for i in range(10, 20)}, **{i: 1 for i in range(20, 40)}}
        r = safe_recode(pd.Series([10, 20, 99]), fac_map)
        assert r[0] == 0
        assert r[1] == 1
        assert pd.isna(r[2])

    def test_ever_breastfed_pre_2018_era(self):
        """1-92 → 1 (ever BF); 0 → 0 (never)."""
        bf_map = {0: 0, 93: 1, 94: 0, 95: 1}
        bf_map.update({i: 1 for i in range(1, 93)})
        r = safe_recode(pd.Series([0, 12, 94]), bf_map)
        assert r[0] == 0   # never
        assert r[1] == 1   # breastfed for 12 months = ever
        assert r[2] == 0   # 94 = never breastfed (2018+ code)

    def test_ever_breastfed_2018_era(self):
        """93→1, 94→0, 95→1."""
        bf_map = {0: 0, 93: 1, 94: 0, 95: 1}
        bf_map.update({i: 1 for i in range(1, 93)})
        r = safe_recode(pd.Series([93, 94, 95]), bf_map)
        assert r[0] == 1
        assert r[1] == 0
        assert r[2] == 1

    def test_all_dhs_missing_codes_are_nan(self):
        for code in DHS_MISSING_CODES:
            r = safe_recode(pd.Series([code]), {code: 1})
            assert pd.isna(r[0]), f"DHS missing code {code} must produce NaN"


# =============================================================================
# 7. isin_vec
# =============================================================================

class TestIsinVec:

    def test_in_set_is_1(self):
        assert isin_vec(pd.Series([11]), {11, 12})[0] == 1.0

    def test_not_in_set_is_0(self):
        assert isin_vec(pd.Series([32]), {11, 12})[0] == 0.0

    def test_96_is_nan_not_0(self):
        """96 = other/unknown — must never be coded as 0 (unimproved)."""
        assert pd.isna(isin_vec(pd.Series([96]), IMPROVED_WATER_CODES)[0])

    def test_99_is_nan_not_0(self):
        assert pd.isna(isin_vec(pd.Series([99]), IMPROVED_WATER_CODES)[0])

    def test_above_threshold_is_nan(self):
        assert pd.isna(isin_vec(pd.Series([9991]), {9991})[0])

    def test_nan_input_stays_nan(self):
        assert pd.isna(isin_vec(pd.Series([np.nan]), {1})[0])

    def test_all_improved_water_codes_give_1(self):
        r = isin_vec(pd.Series(list(IMPROVED_WATER_CODES)), IMPROVED_WATER_CODES)
        assert (r == 1.0).all()

    def test_unimproved_water_codes_give_0(self):
        unimproved = [32, 42, 43, 44, 61, 62]
        r = isin_vec(pd.Series(unimproved), IMPROVED_WATER_CODES)
        assert (r == 0.0).all()

    def test_no_dhs_missing_code_gives_0(self):
        for code in DHS_MISSING_CODES:
            r = isin_vec(pd.Series([code]), IMPROVED_WATER_CODES)
            assert pd.isna(r[0]), f"DHS missing code {code} should be NaN, got {r[0]}"


# =============================================================================
# 8. wt_mean
# =============================================================================

class TestWtMean:

    def test_equal_weights(self):
        r = wt_mean(pd.Series([1.0, 2.0, 3.0]), pd.Series([1.0, 1.0, 1.0]))
        assert abs(r - 2.0) < 1e-9

    def test_unequal_weights(self):
        r = wt_mean(pd.Series([0.0, 1.0]), pd.Series([1.0, 3.0]))
        assert abs(r - 0.75) < 1e-9

    def test_all_nan_values_returns_nan(self):
        r = wt_mean(pd.Series([np.nan, np.nan]), pd.Series([1.0, 1.0]))
        assert pd.isna(r)

    def test_zero_weight_excluded(self):
        r = wt_mean(pd.Series([0.0, 1.0]), pd.Series([0.0, 1.0]))
        assert abs(r - 1.0) < 1e-9

    def test_nan_weight_excluded(self):
        r = wt_mean(pd.Series([0.0, 1.0]), pd.Series([np.nan, 1.0]))
        assert abs(r - 1.0) < 1e-9

    def test_no_valid_obs_returns_nan(self):
        r = wt_mean(pd.Series([1.0]), pd.Series([0.0]))
        assert pd.isna(r)

    def test_all_same_value_returns_that_value(self):
        r = wt_mean(pd.Series([5.0, 5.0, 5.0]), pd.Series([1.0, 2.0, 3.0]))
        assert abs(r - 5.0) < 1e-9


# =============================================================================
# 9. collapse
# =============================================================================

class TestCollapse:

    def _make_df(self):
        return pd.DataFrame({
            "province_name": ["copperbelt", "copperbelt", "lusaka", "lusaka"],
            "x": [1.0, 3.0, 0.0, 2.0],
            "w": [1.0, 1.0, 1.0, 1.0],
        })

    def test_correct_provinces_in_output(self):
        out = collapse(self._make_df(), ["x"])
        assert set(out["province"]) == {"copperbelt", "lusaka"}

    def test_weighted_mean_copperbelt(self):
        out = collapse(self._make_df(), ["x"]).set_index("province")
        assert abs(out.loc["copperbelt", "x"] - 2.0) < 1e-9

    def test_weighted_mean_lusaka(self):
        out = collapse(self._make_df(), ["x"]).set_index("province")
        assert abs(out.loc["lusaka", "x"] - 1.0) < 1e-9

    def test_n_counts_rows(self):
        out = collapse(self._make_df(), ["x"]).set_index("province")
        assert out.loc["copperbelt", "n"] == 2
        assert out.loc["lusaka", "n"] == 2

    def test_missing_outcome_column_gives_nan(self):
        df  = pd.DataFrame({"province_name": ["lusaka"], "w": [1.0]})
        out = collapse(df, ["nonexistent"])
        assert out["nonexistent"].isna().all()

    def test_nine_provinces_produce_nine_rows(self):
        """Simulate 9 provinces × 2 obs each."""
        provinces = ["central","copperbelt","eastern","luapula","lusaka",
                     "northern","northwestern","southern","western"]
        df = pd.DataFrame({
            "province_name": provinces * 2,
            "x": np.random.uniform(0, 1, 18),
            "w": np.ones(18),
        })
        out = collapse(df, ["x"])
        assert len(out) == 9


# =============================================================================
# 10. DATA QUALITY — panel-level checks
# =============================================================================

class TestDataQuality:

    def test_96_99_never_zero_in_isin_vec(self):
        for code in [96, 99]:
            r = isin_vec(pd.Series([code]), IMPROVED_WATER_CODES)
            assert pd.isna(r[0]), f"Code {code} must be NaN, not 0"

    def test_muchinga_northern_same_harmonised_label(self):
        assert harmonise("Muchinga") == harmonise("Northern")

    def test_exactly_9_provinces_after_merge(self):
        assert len(set(PROVINCE_HARMONISE.values())) == 9

    def test_only_two_mining_provinces(self):
        all_provs = set(PROVINCE_HARMONISE.values())
        for p in all_provs:
            expected = 1 if p in {"copperbelt", "northwestern"} else 0
            assert is_mining(p) == expected

    def test_vacc_full_requires_all_four(self):
        """vacc_full = 1 only if all 4 component vaccines = 1."""
        full_cols = ["vacc_bcg", "vacc_measles", "vacc_dpt3", "vacc_polio3"]
        df = pd.DataFrame({
            "vacc_bcg":     [1, 1, 0, 1],
            "vacc_measles": [1, 1, 1, 1],
            "vacc_dpt3":    [1, 1, 1, 1],
            "vacc_polio3":  [1, 0, 1, np.nan],
        })
        valid = df[full_cols].notna().all(axis=1)
        vf    = df[full_cols].min(axis=1)
        vf[~valid] = np.nan
        assert vf[0] == 1.0    # all 4 present
        assert vf[1] == 0.0    # polio3 = 0
        assert vf[2] == 0.0    # bcg = 0
        assert pd.isna(vf[3])  # polio3 = NaN

    def test_sought_treatment_nan_for_healthy_children(self):
        fever  = pd.Series([1, 0, np.nan])
        cough  = pd.Series([0, 0, 0])
        h32z   = binary(pd.Series([1, 1, 1]), [1], [0])
        is_sick = (fever == 1) | (cough.isin([1, 2]))
        sought  = h32z.where(is_sick, np.nan)
        assert sought[0] == 1.0    # sick, sought treatment
        assert pd.isna(sought[1])  # healthy → NaN
        assert pd.isna(sought[2])  # NaN fever → NaN

    def test_months_bf_capped_at_36(self):
        m5  = pd.Series([0, 18, 36, 37, 93])
        mbf = m5.where((m5 >= 0) & (m5 <= 36), np.nan)
        assert mbf[0] == 0.0
        assert mbf[2] == 36.0
        assert pd.isna(mbf[3])
        assert pd.isna(mbf[4])

    def test_ever_breastfed_both_eras_via_safe_recode(self):
        bf_map = {0: 0, 93: 1, 94: 0, 95: 1}
        bf_map.update({i: 1 for i in range(1, 93)})
        assert safe_recode(pd.Series([0]),  bf_map)[0] == 0
        assert safe_recode(pd.Series([12]), bf_map)[0] == 1
        assert safe_recode(pd.Series([93]), bf_map)[0] == 1
        assert safe_recode(pd.Series([94]), bf_map)[0] == 0
        assert safe_recode(pd.Series([95]), bf_map)[0] == 1

    def test_bmi_plausible_range(self):
        bmi_raw = pd.Series([1199, 1200, 6000, 6001])
        bmi     = bmi_raw.where(bmi_raw.between(1200, 6000), np.nan) / 100
        assert pd.isna(bmi[0])
        assert bmi[1] == 12.0
        assert bmi[2] == 60.0
        assert pd.isna(bmi[3])

    def test_wealth_quintile_1_to_5(self):
        wq = to_num(pd.Series([0, 1, 5, 6]))
        wq = wq.where(wq.between(1, 5), np.nan)
        assert pd.isna(wq[0])
        assert wq[1] == 1.0
        assert wq[2] == 5.0
        assert pd.isna(wq[3])

    def test_haemoglobin_divided_by_10(self):
        hb = flag_nan(pd.Series([120, 145, 9991])) / 10
        assert hb[0] == 12.0
        assert hb[1] == 14.5
        assert pd.isna(hb[2])

    def test_anc_98_99_are_nan(self):
        anc = to_num(pd.Series([4, 98, 99])).where(lambda x: x < 98, np.nan)
        assert anc[0] == 4.0
        assert pd.isna(anc[1])
        assert pd.isna(anc[2])

    def test_crowding_zero_rooms_is_nan(self):
        crowding = pd.Series([5.0]) / pd.Series([0.0]).replace(0, np.nan)
        assert pd.isna(crowding[0])

    def test_copper_prices_present_for_all_waves(self):
        assert set(COPPER_PRICES.keys()) == {1992, 1996, 2002, 2007, 2014, 2018, 2024}

    def test_bartik_is_computable(self):
        """Bartik = mining × ln(cu_price) must be finite for all waves."""
        import math
        for wave, price in COPPER_PRICES.items():
            for m in [0, 1]:
                b = m * math.log(price)
                assert math.isfinite(b)

    def test_post_priv_coding(self):
        """post_priv = 1 for waves > 2000."""
        waves = pd.Series([1992, 1996, 2002, 2007, 2014, 2018, 2024])
        post  = (waves > 2000).astype(int)
        assert list(post) == [0, 0, 0, 1, 1, 1, 1]

    def test_period_labelling(self):
        """Period labels match the defined thresholds."""
        def period(w):
            if w <= 2002: return "pre_boom"
            if w == 2007: return "boom"
            if w == 2014: return "bust"
            return "recovery"

        assert period(1992) == "pre_boom"
        assert period(2002) == "pre_boom"
        assert period(2007) == "boom"
        assert period(2014) == "bust"
        assert period(2018) == "recovery"
        assert period(2024) == "recovery"


# =============================================================================
# 11. MINING_PROVINCES assertion guard
# =============================================================================

class TestMiningAssertionGuard:

    def test_mining_provinces_constant_value(self):
        """The module-level assert must pass — MINING_PROVINCES must be exactly these two."""
        assert MINING_PROVINCES == {"copperbelt", "northwestern"}


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    passed = 0
    failed = 0
    errors = []

    test_classes = [
        TestToNum, TestFlagNan, TestHarmonise, TestIsMining, TestBinary,
        TestSafeRecode, TestIsinVec, TestWtMean, TestCollapse,
        TestDataQuality, TestMiningAssertionGuard,
    ]

    print("=" * 65)
    print("test_02_build_panel.py")
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
