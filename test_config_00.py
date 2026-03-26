"""
test_config_00.py
=================
Tests for config_00.py — the single source of truth for all shared constants.
No DHS data files required. Runs entirely on imported constants.

Run with:
    pytest test_config_00.py -v
or:
    python test_config_00.py
"""

import sys
import math
from pathlib import Path

import pytest

# ── Import the module under test ───────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_00 as cfg


# =============================================================================
# 1. PATHS
# =============================================================================

class TestPaths:

    def test_base_dir_is_path(self):
        """BASE_DIR must be a pathlib.Path, not a raw string."""
        assert isinstance(cfg.BASE_DIR, Path)

    def test_out_dir_is_path(self):
        assert isinstance(cfg.OUT_DIR, Path)

    def test_out_dir_is_subdir_of_base(self):
        """OUT_DIR should live inside BASE_DIR."""
        # Parts of OUT_DIR should start with BASE_DIR parts
        assert str(cfg.OUT_DIR).startswith(str(cfg.BASE_DIR))

    def test_out_dir_name(self):
        """OUTPUT folder is called 'OUTPUT' by convention."""
        assert cfg.OUT_DIR.name == "OUTPUT"


# =============================================================================
# 2. COPPER PRICES
# =============================================================================

class TestCopperPrices:

    def test_all_seven_waves_present(self):
        expected = {1992, 1996, 2002, 2007, 2014, 2018, 2024}
        assert set(cfg.COPPER_PRICES.keys()) == expected

    def test_all_prices_are_positive(self):
        for year, price in cfg.COPPER_PRICES.items():
            assert price > 0, f"Price for {year} is not positive: {price}"

    def test_all_prices_are_numeric(self):
        for year, price in cfg.COPPER_PRICES.items():
            assert isinstance(price, (int, float)), f"Price for {year} not numeric"

    def test_2002_is_lowest(self):
        """2002 is the commodity trough — should be the lowest price."""
        assert cfg.COPPER_PRICES[2002] == min(cfg.COPPER_PRICES.values()), \
            "Expected 2002 to be the lowest-price year"

    def test_2007_is_boom(self):
        """2007 price should be much higher than 2002 (commodity boom)."""
        assert cfg.COPPER_PRICES[2007] > 2 * cfg.COPPER_PRICES[2002], \
            "Expected 2007 price to be >2× the 2002 price"

    def test_log_prices_are_finite(self):
        """All prices must allow a valid log (used as instrument)."""
        for year, price in cfg.COPPER_PRICES.items():
            lp = math.log(price)
            assert math.isfinite(lp), f"log(price) is not finite for {year}"

    def test_prices_plausible_range(self):
        """LME copper in USD/MT should be between $500 and $20,000."""
        for year, price in cfg.COPPER_PRICES.items():
            assert 500 <= price <= 20_000, \
                f"Price {price} for {year} outside plausible LME range"

    def test_waves_are_sorted(self):
        keys = list(cfg.COPPER_PRICES.keys())
        assert keys == sorted(keys), "COPPER_PRICES keys should be in ascending order"


# =============================================================================
# 3. PROVINCE HARMONISATION
# =============================================================================

class TestProvinceHarmonise:

    def test_exactly_nine_output_provinces(self):
        """After Muchinga merge, there must be exactly 9 harmonised provinces."""
        n_unique = len(set(cfg.PROVINCE_HARMONISE.values()))
        assert n_unique == 9, f"Expected 9 unique provinces, got {n_unique}"

    def test_muchinga_maps_to_northern(self):
        assert cfg.PROVINCE_HARMONISE["muchinga"] == "northern"

    def test_northern_maps_to_northern(self):
        assert cfg.PROVINCE_HARMONISE["northern"] == "northern"

    def test_all_three_northwestern_spellings(self):
        """All three known spellings of North-Western must resolve correctly."""
        assert cfg.PROVINCE_HARMONISE["north-western"]  == "northwestern"
        assert cfg.PROVINCE_HARMONISE["northwestern"]   == "northwestern"
        assert cfg.PROVINCE_HARMONISE["north western"]  == "northwestern"

    def test_all_output_values_are_lowercase(self):
        for val in cfg.PROVINCE_HARMONISE.values():
            assert val == val.lower(), f"Value '{val}' is not lowercase"

    def test_all_keys_are_lowercase(self):
        for key in cfg.PROVINCE_HARMONISE.keys():
            assert key == key.lower(), f"Key '{key}' is not lowercase"

    def test_copperbelt_present(self):
        assert "copperbelt" in cfg.PROVINCE_HARMONISE.values()

    def test_known_provinces_in_output(self):
        expected = {
            "central", "copperbelt", "eastern", "luapula",
            "lusaka", "northern", "northwestern", "southern", "western"
        }
        actual = set(cfg.PROVINCE_HARMONISE.values())
        assert actual == expected, f"Province mismatch:\n  Expected: {expected}\n  Got: {actual}"

    def test_no_muchinga_in_output_values(self):
        """Muchinga should not appear as an output value — it merges into northern."""
        assert "muchinga" not in cfg.PROVINCE_HARMONISE.values()


# =============================================================================
# 4. MINING PROVINCES
# =============================================================================

class TestMiningProvinces:

    def test_exactly_two_mining_provinces(self):
        assert len(cfg.MINING_PROVINCES) == 2

    def test_copperbelt_is_mining(self):
        assert "copperbelt" in cfg.MINING_PROVINCES

    def test_northwestern_is_mining(self):
        assert "northwestern" in cfg.MINING_PROVINCES

    def test_lusaka_is_not_mining(self):
        assert "lusaka" not in cfg.MINING_PROVINCES

    def test_northern_is_not_mining(self):
        assert "northern" not in cfg.MINING_PROVINCES

    def test_mining_provinces_are_subset_of_harmonised_values(self):
        harmonised_values = set(cfg.PROVINCE_HARMONISE.values())
        assert cfg.MINING_PROVINCES.issubset(harmonised_values), \
            "All mining provinces must appear in PROVINCE_HARMONISE output values"

    def test_mining_provinces_all_lowercase(self):
        for p in cfg.MINING_PROVINCES:
            assert p == p.lower()


# =============================================================================
# 5. WAVE FILES
# =============================================================================

class TestWaveFiles:

    def test_all_seven_waves_defined(self):
        expected = {1992, 1996, 2002, 2007, 2014, 2018, 2024}
        assert set(cfg.WAVE_FILES.keys()) == expected

    def test_hr_present_in_every_wave(self):
        for wave, files in cfg.WAVE_FILES.items():
            assert "HR" in files, f"HR missing from wave {wave}"

    def test_ir_present_in_every_wave(self):
        for wave, files in cfg.WAVE_FILES.items():
            assert "IR" in files, f"IR missing from wave {wave}"

    def test_kr_present_in_every_wave(self):
        for wave, files in cfg.WAVE_FILES.items():
            assert "KR" in files, f"KR missing from wave {wave}"

    def test_mr_absent_in_1992(self):
        """1992 DHS Zambia has no Men's Recode."""
        assert "MR" not in cfg.WAVE_FILES[1992]

    def test_mr_present_from_1996(self):
        for wave in [1996, 2002, 2007, 2014, 2018, 2024]:
            assert "MR" in cfg.WAVE_FILES[wave], f"MR missing from wave {wave}"

    def test_all_paths_are_strings(self):
        for wave, files in cfg.WAVE_FILES.items():
            for ftype, path in files.items():
                assert isinstance(path, str), \
                    f"Path for {ftype}/{wave} is not a string: {path!r}"

    def test_all_paths_end_in_dta(self):
        for wave, files in cfg.WAVE_FILES.items():
            for ftype, path in files.items():
                assert path.lower().endswith(".dta"), \
                    f"Path for {ftype}/{wave} does not end in .dta: {path}"

    def test_paths_contain_wave_year(self):
        """Each path should reference its own survey year."""
        for wave, files in cfg.WAVE_FILES.items():
            for ftype, path in files.items():
                assert str(wave) in path, \
                    f"Path for {ftype}/{wave} does not contain year {wave}: {path}"

    def test_path_uniqueness_within_wave(self):
        """Within a wave, all file paths must be unique."""
        for wave, files in cfg.WAVE_FILES.items():
            paths = list(files.values())
            assert len(paths) == len(set(paths)), \
                f"Duplicate paths in wave {wave}: {paths}"


# =============================================================================
# 6. DHS MISSING CODES
# =============================================================================

class TestDhsMissingCodes:

    def test_is_a_set(self):
        assert isinstance(cfg.DHS_MISSING_CODES, set)

    def test_standard_single_digit_codes(self):
        """8=DK, 9=missing must be present."""
        assert 8 in cfg.DHS_MISSING_CODES
        assert 9 in cfg.DHS_MISSING_CODES

    def test_standard_two_digit_codes(self):
        """96=other, 97=inconsistent, 98=DK, 99=missing must be present."""
        for code in [96, 97, 98, 99]:
            assert code in cfg.DHS_MISSING_CODES, f"Code {code} missing"

    def test_standard_three_digit_codes(self):
        for code in [997, 998, 999]:
            assert code in cfg.DHS_MISSING_CODES

    def test_standard_four_digit_codes(self):
        for code in [9997, 9998, 9999]:
            assert code in cfg.DHS_MISSING_CODES

    def test_zero_not_in_missing(self):
        """0 is a valid response (e.g. not vaccinated) — must not be missing."""
        assert 0 not in cfg.DHS_MISSING_CODES

    def test_one_not_in_missing(self):
        assert 1 not in cfg.DHS_MISSING_CODES

    def test_all_codes_are_positive(self):
        for code in cfg.DHS_MISSING_CODES:
            assert code > 0, f"Missing code {code} is not positive"


# =============================================================================
# 7. CLASSIFICATION CODES
# =============================================================================

class TestClassificationCodes:

    def test_improved_water_is_set(self):
        assert isinstance(cfg.IMPROVED_WATER_CODES, set)

    def test_improved_sanit_is_set(self):
        assert isinstance(cfg.IMPROVED_SANIT_CODES, set)

    def test_improved_floor_is_set(self):
        assert isinstance(cfg.IMPROVED_FLOOR_CODES, set)

    def test_improved_water_not_empty(self):
        assert len(cfg.IMPROVED_WATER_CODES) > 0

    def test_improved_water_typical_codes(self):
        """Piped water (10/11), tube well (20/21) are canonical improved sources."""
        for code in [10, 11, 20, 21]:
            assert code in cfg.IMPROVED_WATER_CODES, f"Code {code} missing from IMPROVED_WATER_CODES"

    def test_96_not_in_improved_water(self):
        """96 = other/unknown — must never be coded as improved."""
        assert 96 not in cfg.IMPROVED_WATER_CODES

    def test_99_not_in_improved_water(self):
        assert 99 not in cfg.IMPROVED_WATER_CODES

    def test_96_not_in_improved_sanit(self):
        assert 96 not in cfg.IMPROVED_SANIT_CODES

    def test_improved_sanit_typical_codes(self):
        """Flush to sewer (11) and VIP pit (21) are canonical improved sanitation."""
        for code in [11, 21]:
            assert code in cfg.IMPROVED_SANIT_CODES

    def test_improved_floor_not_empty(self):
        assert len(cfg.IMPROVED_FLOOR_CODES) > 0

    def test_no_overlap_between_missing_and_wash_codes(self):
        """DHS missing codes must not appear in WASH code sets."""
        assert cfg.DHS_MISSING_CODES.isdisjoint(cfg.IMPROVED_WATER_CODES), \
            "Overlap between DHS_MISSING_CODES and IMPROVED_WATER_CODES"
        assert cfg.DHS_MISSING_CODES.isdisjoint(cfg.IMPROVED_SANIT_CODES), \
            "Overlap between DHS_MISSING_CODES and IMPROVED_SANIT_CODES"


# =============================================================================
# 8. THRESHOLDS
# =============================================================================

class TestThresholds:

    def test_min_obs_is_positive_integer(self):
        assert isinstance(cfg.MIN_OBS, int)
        assert cfg.MIN_OBS > 0

    def test_min_clusters_is_positive_integer(self):
        assert isinstance(cfg.MIN_CLUSTERS, int)
        assert cfg.MIN_CLUSTERS > 0

    def test_min_obs_reasonable(self):
        """Should be at least 10 and no more than 10,000."""
        assert 10 <= cfg.MIN_OBS <= 10_000

    def test_min_clusters_reasonable(self):
        """With G=9 provinces, MIN_CLUSTERS must be < 9."""
        assert cfg.MIN_CLUSTERS < 9


# =============================================================================
# 9. CROSS-CONSTANT CONSISTENCY
# =============================================================================

class TestCrossConstantConsistency:

    def test_copper_waves_match_wave_files(self):
        """Every wave in COPPER_PRICES must appear in WAVE_FILES and vice versa."""
        assert set(cfg.COPPER_PRICES.keys()) == set(cfg.WAVE_FILES.keys()), \
            "COPPER_PRICES and WAVE_FILES do not cover the same waves"

    def test_mining_provinces_in_harmonise_output(self):
        harmonised_values = set(cfg.PROVINCE_HARMONISE.values())
        for p in cfg.MINING_PROVINCES:
            assert p in harmonised_values, \
                f"Mining province '{p}' not reachable via PROVINCE_HARMONISE"

    def test_bartik_instrument_is_computable(self):
        """
        Bartik = mining × ln(cu_price).
        Verify it is computable for all waves and produces valid floats.
        """
        for wave, price in cfg.COPPER_PRICES.items():
            for mining in [0, 1]:
                bartik = mining * math.log(price)
                assert math.isfinite(bartik), \
                    f"Bartik not finite for wave={wave}, mining={mining}"


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import unittest

    # Simple runner without pytest
    passed = 0
    failed = 0
    errors = []

    test_classes = [
        TestPaths, TestCopperPrices, TestProvinceHarmonise,
        TestMiningProvinces, TestWaveFiles, TestDhsMissingCodes,
        TestClassificationCodes, TestThresholds, TestCrossConstantConsistency,
    ]

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

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        print("\nFailed tests:")
        for cls_name, test_name, msg in errors:
            print(f"  {cls_name}.{test_name}: {msg}")
    sys.exit(1 if failed else 0)
