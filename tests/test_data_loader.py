"""Unit tests for dataset loading, validation, and filtering in src/data_loader.py."""

import io
from pathlib import Path
import tempfile
import pandas as pd
import pytest

from src.data_loader import (
    filter_dataset,
    find_dataset_path,
    load_and_validate_data,
    load_uploaded_csv,
)


def test_find_dataset_path_existing():
    path = find_dataset_path()
    assert path.exists()
    assert path.name == "synthetic_canteen_data_500.csv"


def test_load_and_validate_data_actual_file():
    df = load_and_validate_data()
    assert not df.empty
    assert len(df) == 500 or len(df) == 501 or len(df) >= 400
    assert "date" in df.columns
    assert "menu_item" in df.columns
    assert "meals_prepared" in df.columns
    assert "meals_sold" in df.columns
    assert "leftover_portions" in df.columns
    assert "estimated_waste_kg" in df.columns
    assert "waste_percentage" in df.columns
    assert "revenue_inr" in df.columns


def test_load_and_validate_data_missing_column():
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write("date,menu_item,meals_prepared\n2026-01-01,Thali,100\n")
        tmp_name = f.name

    try:
        with pytest.raises(ValueError, match="missing required columns"):
            load_and_validate_data(tmp_name)
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def test_load_and_validate_data_empty_file():
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write("date,day_of_week,menu_item,price_inr,meals_prepared,meals_sold,leftover_portions,estimated_waste_kg,waste_percentage,revenue_inr,weather,exam_period,college_event\n")
        tmp_name = f.name

    try:
        with pytest.raises(ValueError, match="is empty"):
            load_and_validate_data(tmp_name)
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def test_load_uploaded_csv_valid_buffer():
    csv_text = """date,day_of_week,menu_item,price_inr,meals_prepared,meals_sold,leftover_portions,estimated_waste_kg,waste_percentage,revenue_inr,weather,exam_period,college_event
2026-02-01,Monday,Poha,40,80,70,10,2.0,12.5,2800,Sunny,No,No
"""
    buffer = io.StringIO(csv_text)
    df = load_uploaded_csv(buffer)
    assert len(df) == 1
    assert df.iloc[0]["menu_item"] == "Poha"
    assert df.iloc[0]["meals_prepared"] == 80


def test_load_uploaded_csv_aliased_columns():
    csv_text = """date,day,item,price,prepared,sold,leftover,waste_kg,waste_pct,revenue,weather,exam,event
2026-02-01,Monday,Idli Sambar,50,90,75,15,3.0,16.67,3750,Rainy,No,Yes
"""
    buffer = io.StringIO(csv_text)
    df = load_uploaded_csv(buffer)
    assert len(df) == 1
    assert df.iloc[0]["menu_item"] == "Idli Sambar"
    assert df.iloc[0]["meals_sold"] == 75
    assert df.iloc[0]["college_event"] == "Yes"


def test_load_uploaded_csv_empty():
    buffer = io.StringIO("")
    with pytest.raises(ValueError, match="Could not parse"):
        load_uploaded_csv(buffer)


def test_filter_dataset():
    df = load_and_validate_data()
    items = list(df["menu_item"].unique())
    selected_item = items[0]

    filtered = filter_dataset(df, menu_items=[selected_item])
    assert not filtered.empty
    assert (filtered["menu_item"] == selected_item).all()

    # Filter weather
    weather_val = df["weather"].iloc[0]
    filtered_w = filter_dataset(df, weather=[weather_val])
    assert (filtered_w["weather"] == weather_val).all()
