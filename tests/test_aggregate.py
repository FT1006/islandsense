"""Unit tests for M3 aggregation module.

These tests protect against subtle bugs in:
- Time → day_index mapping
- E_loss + risk score scaling
- Scenario/impact math
- Contribution sorting & CSV shapes
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock
import pandas as pd

from islandsense.aggregate import (
    compute_day_index,
    compute_daily_e_loss,
    compute_risk_score,
    get_band,
    add_risk_columns,
    compute_weekly_risk,
    apply_scenario,
    compute_impact,
    compute_sailing_contributions,
    ensure_full_grid,
)


# Test fixtures and helpers


@pytest.fixture
def mock_config():
    """Create a mock config with known parameters."""
    config = MagicMock()
    config.horizon_days = 7
    config.risk_bands = {
        "green": {"range": [0, 39]},
        "amber": {"range": [40, 69]},
        "red": {"range": [70, 100]},
    }
    config.k_hours_per_unit = 0.25
    config.units_per_trailer = 24
    config.scenarios = [
        {
            "id": "scenario_A",
            "name": "Bring forward 10%",
            "alpha": {"fresh": 0.20, "fuel": 0.15},
        }
    ]

    # Default: same scaling for both categories
    def risk_min(category):
        return 0.0

    def risk_max(category):
        return 100.0

    config.risk_expected_loss_min = risk_min
    config.risk_expected_loss_max = risk_max

    return config


# 1. test_compute_day_index_filters_and_indexes_correctly


class TestComputeDayIndex:
    def test_correct_day_index_mapping(self):
        """Day index should be (etd_date - now_date).days"""
        now = datetime(2025, 11, 15, 0, 0, 0, tzinfo=timezone.utc)

        df = pd.DataFrame(
            {
                "etd_dt": pd.to_datetime(
                    [
                        "2025-11-15T03:00:00",  # D0
                        "2025-11-16T00:00:00",  # D1
                        "2025-11-21T23:59:59",  # D6
                    ]
                ),
                "other_col": [1, 2, 3],
            }
        )

        result = compute_day_index(df, now, horizon_days=7)

        assert list(result["day_index"]) == [0, 1, 6]
        assert list(result["date"]) == ["2025-11-15", "2025-11-16", "2025-11-21"]

    def test_filters_out_past_and_beyond_horizon(self):
        """Sailings before now or at D7+ should be dropped."""
        now = datetime(2025, 11, 15, 0, 0, 0, tzinfo=timezone.utc)

        df = pd.DataFrame(
            {
                "etd_dt": pd.to_datetime(
                    [
                        "2025-11-14T23:59:59",  # D-1 (past)
                        "2025-11-15T12:00:00",  # D0 (keep)
                        "2025-11-22T00:00:00",  # D7 (beyond)
                        "2025-11-23T00:00:00",  # D8 (beyond)
                    ]
                ),
            }
        )

        result = compute_day_index(df, now, horizon_days=7)

        assert len(result) == 1
        assert result.iloc[0]["day_index"] == 0

    def test_time_of_day_invariance(self):
        """Only the date matters, not the time of now."""
        # now is mid-day
        now = datetime(2025, 11, 15, 10, 30, 0, tzinfo=timezone.utc)

        df = pd.DataFrame(
            {
                "etd_dt": pd.to_datetime(
                    [
                        "2025-11-15T23:00:00",  # Same day as now → D0
                        "2025-11-16T01:00:00",  # Next day → D1
                    ]
                ),
            }
        )

        result = compute_day_index(df, now, horizon_days=7)

        assert list(result["day_index"]) == [0, 1]


# 2. test_compute_daily_e_loss_sums_contribs_correctly


class TestComputeDailyELoss:
    def test_sums_contributions_per_day_and_category(self):
        """E_loss should be Σ contrib for each (day, category)."""
        df = pd.DataFrame(
            {
                "day_index": [0, 0, 1],
                "date": ["2025-11-15", "2025-11-15", "2025-11-16"],
                "contrib_fresh": [10.0, 5.0, 2.0],
                "contrib_fuel": [1.0, 2.0, 3.0],
            }
        )

        result = compute_daily_e_loss(df)

        # Should have 4 rows: 2 days × 2 categories
        assert len(result) == 4
        assert set(result["category"]) == {"fresh", "fuel"}

        # Day 0: fresh=15, fuel=3
        day0_fresh = result[
            (result["day_index"] == 0) & (result["category"] == "fresh")
        ].iloc[0]["E_loss"]
        day0_fuel = result[
            (result["day_index"] == 0) & (result["category"] == "fuel")
        ].iloc[0]["E_loss"]
        assert day0_fresh == 15.0
        assert day0_fuel == 3.0

        # Day 1: fresh=2, fuel=3
        day1_fresh = result[
            (result["day_index"] == 1) & (result["category"] == "fresh")
        ].iloc[0]["E_loss"]
        day1_fuel = result[
            (result["day_index"] == 1) & (result["category"] == "fuel")
        ].iloc[0]["E_loss"]
        assert day1_fresh == 2.0
        assert day1_fuel == 3.0


# 3. test_add_risk_columns_uses_per_category_scaling_and_bands


class TestRiskScalingAndBands:
    def test_compute_risk_score_scaling_and_clamping(self):
        """Risk score should scale E_loss to [0,100] and clamp."""
        # min=0, max=100
        assert compute_risk_score(0.0, 0.0, 100.0) == 0
        assert compute_risk_score(50.0, 0.0, 100.0) == 50
        assert compute_risk_score(100.0, 0.0, 100.0) == 100

        # Clamping: E_loss beyond max
        assert compute_risk_score(200.0, 0.0, 100.0) == 100

        # Clamping: E_loss below min
        assert compute_risk_score(-10.0, 0.0, 100.0) == 0

        # Edge case: max == min
        assert compute_risk_score(50.0, 100.0, 100.0) == 0

    def test_get_band_boundaries(self):
        """Band assignment at boundaries."""
        bands_config = {
            "green": {"range": [0, 39]},
            "amber": {"range": [40, 69]},
            "red": {"range": [70, 100]},
        }

        assert get_band(0, bands_config) == "green"
        assert get_band(39, bands_config) == "green"
        assert get_band(40, bands_config) == "amber"
        assert get_band(69, bands_config) == "amber"
        assert get_band(70, bands_config) == "red"
        assert get_band(100, bands_config) == "red"

    def test_add_risk_columns_per_category_scaling(self):
        """Risk score should use per-category min/max from config."""
        e_loss_df = pd.DataFrame(
            {
                "day_index": [0, 0],
                "date": ["2025-11-15", "2025-11-15"],
                "category": ["fresh", "fuel"],
                "E_loss": [10.0, 10.0],  # Same E_loss
            }
        )

        # Config with different scaling per category
        config = MagicMock()
        config.risk_bands = {
            "green": {"range": [0, 39]},
            "amber": {"range": [40, 69]},
            "red": {"range": [70, 100]},
        }

        # Fresh: 10 is near max (10) → risk 100
        # Fuel: 10 is mid-range (max=20) → risk 50
        def risk_min(cat):
            return 0.0

        def risk_max(cat):
            return 10.0 if cat == "fresh" else 20.0

        config.risk_expected_loss_min = risk_min
        config.risk_expected_loss_max = risk_max

        result = add_risk_columns(e_loss_df, config)

        fresh_risk = result[result["category"] == "fresh"].iloc[0]["risk_baseline"]
        fuel_risk = result[result["category"] == "fuel"].iloc[0]["risk_baseline"]

        assert fresh_risk == 100  # 10/10 = 1.0 → 100
        assert fuel_risk == 50  # 10/20 = 0.5 → 50
        assert fresh_risk > fuel_risk


# 4. test_compute_weekly_risk_averages_and_handles_nan


class TestComputeWeeklyRisk:
    def test_simple_average(self):
        """Weekly risk is mean of daily risk scores."""
        daily_risk_df = pd.DataFrame(
            {
                "category": ["fresh", "fresh", "fresh", "fuel", "fuel", "fuel"],
                "risk_baseline": [10, 20, 30, 40, 50, 60],
            }
        )

        result = compute_weekly_risk(daily_risk_df)

        assert result["fresh"] == 20  # mean(10,20,30)
        assert result["fuel"] == 50  # mean(40,50,60)

    def test_empty_category_returns_zero(self):
        """Empty category should return 0, not NaN error."""
        # Only fresh data, no fuel
        daily_risk_df = pd.DataFrame(
            {
                "category": ["fresh", "fresh"],
                "risk_baseline": [10, 20],
            }
        )

        result = compute_weekly_risk(daily_risk_df)

        assert result["fresh"] == 15
        assert result["fuel"] == 0  # NaN safety


# 5. test_apply_scenario_respects_alpha_per_category


class TestApplyScenario:
    def test_alpha_reduces_e_loss_per_category(self):
        """E_loss_scenario = E_loss * (1 - alpha)."""
        e_loss_df = pd.DataFrame(
            {
                "day_index": [0, 0, 1, 1],
                "date": ["2025-11-15"] * 2 + ["2025-11-16"] * 2,
                "category": ["fresh", "fuel", "fresh", "fuel"],
                "E_loss": [10.0, 20.0, 30.0, 40.0],
            }
        )

        scenario_config = {
            "id": "scenario_A",
            "alpha": {"fresh": 0.5, "fuel": 0.25},  # 50% and 25% reduction
        }

        # Config with simple scaling
        config = MagicMock()

        def risk_min(cat):
            return 0.0

        def risk_max(cat):
            return 40.0  # Simple for both

        config.risk_expected_loss_min = risk_min
        config.risk_expected_loss_max = risk_max

        scenario_df, weekly_risk = apply_scenario(e_loss_df, scenario_config, config)

        # Check E_loss_scenario
        # Fresh: 10*0.5=5, 30*0.5=15
        # Fuel: 20*0.75=15, 40*0.75=30
        fresh_scenario = scenario_df[scenario_df["category"] == "fresh"][
            "E_loss_scenario"
        ].tolist()
        fuel_scenario = scenario_df[scenario_df["category"] == "fuel"][
            "E_loss_scenario"
        ].tolist()

        assert fresh_scenario == [5.0, 15.0]
        assert fuel_scenario == [15.0, 30.0]

        # Weekly risk should be lower than baseline
        # Baseline fresh: mean(10,30)/40*100 = 50
        # Scenario fresh: mean(5,15)/40*100 = 25
        assert weekly_risk["fresh"] == 25
        # Baseline fuel: mean(20,40)/40*100 = 75
        # Scenario fuel: mean(15,30)/40*100 = 56
        assert weekly_risk["fuel"] == 56


# 6. test_compute_impact_maps_delta_to_hours_and_trailers


class TestComputeImpact:
    def test_impact_calculation_correct_sign_and_mapping(self):
        """hours_avoided and trailers_avoided computed correctly."""
        e_loss_df = pd.DataFrame(
            {
                "day_index": [0, 0],
                "category": ["fresh", "fuel"],
                "E_loss": [40.0, 60.0],
            }
        )

        scenario_df = pd.DataFrame(
            {
                "day_index": [0, 0],
                "category": ["fresh", "fuel"],
                "E_loss_scenario": [30.0, 36.0],
            }
        )

        config = MagicMock()
        config.k_hours_per_unit = 0.5
        config.units_per_trailer = 10

        result = compute_impact(e_loss_df, scenario_df, config)

        # Fresh: delta = 40 - 30 = 10, hours_avoided = 10 * 0.5 = 5.0
        assert result["fresh"]["hours_avoided"] == 5.0
        assert result["fresh"]["trailers_avoided"] == 0.0

        # Fuel: delta = 60 - 36 = 24, trailers_avoided = 24 / 10 = 2.4
        assert result["fuel"]["hours_avoided"] == 0.0
        assert result["fuel"]["trailers_avoided"] == 2.4

    def test_multiple_days_summed_correctly(self):
        """Delta E_loss should sum across all days."""
        e_loss_df = pd.DataFrame(
            {
                "day_index": [0, 0, 1, 1],
                "category": ["fresh", "fuel", "fresh", "fuel"],
                "E_loss": [10.0, 20.0, 30.0, 40.0],
            }
        )

        scenario_df = pd.DataFrame(
            {
                "day_index": [0, 0, 1, 1],
                "category": ["fresh", "fuel", "fresh", "fuel"],
                "E_loss_scenario": [8.0, 16.0, 24.0, 32.0],
            }
        )

        config = MagicMock()
        config.k_hours_per_unit = 0.25
        config.units_per_trailer = 24

        result = compute_impact(e_loss_df, scenario_df, config)

        # Fresh delta: (10-8) + (30-24) = 8
        # hours_avoided = 8 * 0.25 = 2.0
        assert result["fresh"]["hours_avoided"] == 2.0

        # Fuel delta: (20-16) + (40-32) = 12
        # trailers_avoided = 12 / 24 = 0.5
        assert result["fuel"]["trailers_avoided"] == 0.5


# 7. test_compute_sailing_contributions_shapes_and_sorts


class TestComputeSailingContributions:
    def test_all_required_columns_present(self):
        """Output should have all required columns."""
        df = pd.DataFrame(
            {
                "sailing_id": ["S1"],
                "day_index": [0],
                "date": ["2025-11-15"],
                "route": ["Portsmouth→Jersey"],
                "vessel": ["Channel Star"],
                "etd_iso": ["2025-11-15T12:00:00"],
                "p_sail": [0.5],
                "fresh_units": [10.0],
                "fuel_units": [5.0],
                "contrib_fresh": [5.0],
                "contrib_fuel": [2.5],
            }
        )

        result = compute_sailing_contributions(df)

        expected_cols = [
            "sailing_id",
            "day_index",
            "date",
            "route",
            "vessel",
            "etd_iso",
            "p_sail",
            "fresh_units",
            "fuel_units",
            "contrib_fresh",
            "contrib_fuel",
        ]
        assert list(result.columns) == expected_cols

    def test_sorted_by_total_contribution_descending(self):
        """Within each day, rows sorted by contrib_total descending."""
        df = pd.DataFrame(
            {
                "sailing_id": ["S1", "S2", "S3"],
                "day_index": [0, 0, 0],
                "date": ["2025-11-15"] * 3,
                "route": ["A", "B", "C"],
                "vessel": ["V1", "V2", "V3"],
                "etd_iso": ["2025-11-15T12:00:00"] * 3,
                "p_sail": [0.5] * 3,
                "fresh_units": [10.0, 5.0, 1.0],
                "fuel_units": [1.0, 100.0, 1.0],
                "contrib_fresh": [5.0, 2.5, 0.5],  # S1 highest fresh
                "contrib_fuel": [0.5, 50.0, 0.5],  # S2 highest fuel
            }
        )

        result = compute_sailing_contributions(df)

        # Total contrib: S1=5.5, S2=52.5, S3=1.0
        # Order should be: S2, S1, S3
        assert list(result["sailing_id"]) == ["S2", "S1", "S3"]


# 8. test_aggregate_end_to_end_smoke


class TestEnsureFullGrid:
    def test_fills_missing_days_with_zero_eloss(self):
        """Missing (day, category) pairs should be filled with E_loss=0."""
        # Only have data for day 0
        e_loss_df = pd.DataFrame(
            {
                "day_index": [0, 0],
                "date": ["2025-11-15", "2025-11-15"],
                "category": ["fresh", "fuel"],
                "E_loss": [10.0, 20.0],
            }
        )

        now_date = pd.Timestamp("2025-11-15").date()

        result = ensure_full_grid(e_loss_df, now_date, horizon_days=3)

        # Should have 3 days * 2 categories = 6 rows
        assert len(result) == 6

        # Day 1 and 2 should have E_loss = 0
        day1_fresh = result[
            (result["day_index"] == 1) & (result["category"] == "fresh")
        ].iloc[0]["E_loss"]
        day2_fuel = result[
            (result["day_index"] == 2) & (result["category"] == "fuel")
        ].iloc[0]["E_loss"]

        assert day1_fresh == 0.0
        assert day2_fuel == 0.0

        # Day 0 should retain original values
        day0_fresh = result[
            (result["day_index"] == 0) & (result["category"] == "fresh")
        ].iloc[0]["E_loss"]
        assert day0_fresh == 10.0

    def test_all_day_indices_present(self):
        """Should have all day_index from 0 to horizon_days-1."""
        e_loss_df = pd.DataFrame(
            {
                "day_index": [2],
                "date": ["2025-11-17"],
                "category": ["fresh"],
                "E_loss": [5.0],
            }
        )

        now_date = pd.Timestamp("2025-11-15").date()

        result = ensure_full_grid(e_loss_df, now_date, horizon_days=7)

        # Should have all day indices 0-6
        assert set(result["day_index"]) == {0, 1, 2, 3, 4, 5, 6}
        assert len(result) == 14  # 7 days * 2 categories
