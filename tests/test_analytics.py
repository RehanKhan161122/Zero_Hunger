"""Unit tests for deterministic Python numeric calculations in src/analytics.py."""

import pandas as pd
import pytest

from src.analytics import (
    calculate_demand_baseline,
    calculate_menu_performance,
    calculate_summary_metrics,
    calculate_waste_by_attribute,
    format_menu_summary_table,
    get_daily_waste_trend,
)


@pytest.fixture
def sample_canteen_df() -> pd.DataFrame:
    """Fixture providing a known small canteen dataset for exact numerical verification."""
    data = [
        {
            "date": pd.to_datetime("2026-01-05").date(),
            "day_of_week": "Monday",
            "menu_item": "Paneer Rice Bowl",
            "price_inr": 100,
            "meals_prepared": 100,
            "meals_sold": 80,
            "leftover_portions": 20,
            "estimated_waste_kg": 5.0,
            "waste_percentage": 20.0,
            "revenue_inr": 8000,
            "weather": "Sunny",
            "exam_period": "No",
            "college_event": "No",
        },
        {
            "date": pd.to_datetime("2026-01-05").date(),
            "day_of_week": "Monday",
            "menu_item": "Veg Thali",
            "price_inr": 80,
            "meals_prepared": 200,
            "meals_sold": 180,
            "leftover_portions": 20,
            "estimated_waste_kg": 6.0,
            "waste_percentage": 10.0,
            "revenue_inr": 14400,
            "weather": "Sunny",
            "exam_period": "No",
            "college_event": "No",
        },
        {
            "date": pd.to_datetime("2026-01-06").date(),
            "day_of_week": "Tuesday",
            "menu_item": "Paneer Rice Bowl",
            "price_inr": 100,
            "meals_prepared": 120,
            "meals_sold": 90,
            "leftover_portions": 30,
            "estimated_waste_kg": 7.5,
            "waste_percentage": 25.0,
            "revenue_inr": 9000,
            "weather": "Rainy",
            "exam_period": "Yes",
            "college_event": "No",
        },
    ]
    return pd.DataFrame(data)


def test_calculate_summary_metrics_normal(sample_canteen_df):
    metrics = calculate_summary_metrics(sample_canteen_df)

    assert metrics["total_records"] == 3
    # Total prepared: 100 + 200 + 120 = 420
    assert metrics["total_prepared"] == 420
    # Total sold: 80 + 180 + 90 = 350
    assert metrics["total_sold"] == 350
    # Total leftovers: 20 + 20 + 30 = 70
    assert metrics["total_leftovers"] == 70
    # Total waste kg: 5.0 + 6.0 + 7.5 = 18.5
    assert metrics["total_waste_kg"] == 18.5
    # Waste percentage: 70 / 420 * 100 = 16.67%
    assert metrics["waste_percentage"] == round(70 / 420 * 100, 2)
    # Revenue: 8000 + 14400 + 9000 = 31400
    assert metrics["total_revenue_inr"] == 31400.0
    # Estimated waste cost: (20*100) + (20*80) + (30*100) = 2000 + 1600 + 3000 = 6600
    assert metrics["estimated_waste_cost_inr"] == 6600.0
    assert metrics["unique_menu_items_count"] == 2
    assert metrics["date_min"] == "2026-01-05"
    assert metrics["date_max"] == "2026-01-06"


def test_calculate_summary_metrics_empty():
    empty_df = pd.DataFrame(
        columns=[
            "date",
            "day_of_week",
            "menu_item",
            "price_inr",
            "meals_prepared",
            "meals_sold",
            "leftover_portions",
            "estimated_waste_kg",
            "waste_percentage",
            "revenue_inr",
        ]
    )
    metrics = calculate_summary_metrics(empty_df)
    assert metrics["total_records"] == 0
    assert metrics["total_prepared"] == 0
    assert metrics["waste_percentage"] == 0.0
    assert metrics["total_waste_kg"] == 0.0
    assert metrics["total_revenue_inr"] == 0.0
    assert metrics["date_min"] == "N/A"


def test_calculate_menu_performance(sample_canteen_df):
    perf = calculate_menu_performance(sample_canteen_df)

    assert len(perf) == 2
    paneer_row = perf[perf["menu_item"] == "Paneer Rice Bowl"].iloc[0]
    # Paneer prepared: 100 + 120 = 220, leftovers: 20 + 30 = 50 -> 50/220 = 22.73%
    assert paneer_row["total_prepared"] == 220
    assert paneer_row["total_leftovers"] == 50
    assert paneer_row["waste_percentage"] == round(50 / 220 * 100, 2)
    assert paneer_row["total_waste_kg"] == 12.5

    thali_row = perf[perf["menu_item"] == "Veg Thali"].iloc[0]
    assert thali_row["total_prepared"] == 200
    assert thali_row["total_sold"] == 180
    assert thali_row["waste_percentage"] == 10.0


def test_calculate_demand_baseline(sample_canteen_df):
    baseline = calculate_demand_baseline(sample_canteen_df, "Paneer Rice Bowl")

    assert baseline["historical_records_count"] == 2
    assert baseline["min_meals_sold"] == 80
    assert baseline["max_meals_sold"] == 90
    assert baseline["avg_meals_sold"] == 85.0
    assert baseline["median_meals_sold"] == 85.0
    # Median 85 * 1.05 = 89.25 -> 89
    assert baseline["calculated_baseline_target"] == 89

    # Missing item
    missing = calculate_demand_baseline(sample_canteen_df, "Non-existent Dish")
    assert missing["historical_records_count"] == 0
    assert missing["calculated_baseline_target"] == 0


def test_calculate_waste_by_attribute(sample_canteen_df):
    by_weather = calculate_waste_by_attribute(sample_canteen_df, "weather")
    assert len(by_weather) == 2  # Sunny and Rainy
    sunny = by_weather[by_weather["weather"] == "Sunny"].iloc[0]
    assert sunny["total_prepared"] == 300
    assert sunny["total_leftovers"] == 40


def test_get_daily_waste_trend(sample_canteen_df):
    trend = get_daily_waste_trend(sample_canteen_df)
    assert len(trend) == 2
    day1 = trend.iloc[0]
    assert day1["total_prepared"] == 300
    assert day1["total_sold"] == 260


def test_format_menu_summary_table(sample_canteen_df):
    perf = calculate_menu_performance(sample_canteen_df)
    table_str = format_menu_summary_table(perf)
    assert "Paneer Rice Bowl" in table_str
    assert "Veg Thali" in table_str
    assert "|" in table_str
