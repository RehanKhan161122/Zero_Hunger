"""Unit tests for prompt loading, schema conformity, and assembly in src/prompt_manager.py."""

import pandas as pd
import pytest

from src.prompt_manager import (
    MODULE_METADATA,
    assemble_demand_estimator_prompt,
    assemble_food_waste_prompt,
    assemble_impact_analyst_prompt,
    assemble_menu_optimizer_prompt,
    assemble_surplus_food_prompt,
    get_available_modules,
    load_template,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "date": pd.to_datetime("2026-01-05").date(),
            "day_of_week": "Monday",
            "menu_item": "Paneer Rice Bowl",
            "price_inr": 95,
            "meals_prepared": 100,
            "meals_sold": 80,
            "leftover_portions": 20,
            "estimated_waste_kg": 5.0,
            "waste_percentage": 20.0,
            "revenue_inr": 7600,
            "weather": "Sunny",
            "exam_period": "No",
            "college_event": "No",
        }
    ])


def test_modules_count_and_metadata():
    modules = get_available_modules()
    assert len(modules) == 5
    expected = [
        "food_waste_analyst",
        "demand_estimator",
        "menu_optimizer",
        "surplus_food_manager",
        "impact_analyst",
    ]
    for exp in expected:
        assert exp in modules
        assert exp in MODULE_METADATA
        assert "title" in MODULE_METADATA[exp]
        assert "description" in MODULE_METADATA[exp]


def test_load_template_all_exist():
    for mod in get_available_modules():
        content = load_template(mod)
        assert len(content) > 100
        assert "# ROLE" in content
        assert "# OBJECTIVE" in content
        assert "# OUTPUT FORMAT" in content


def test_load_template_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_template("unknown_module_xyz")


def test_assemble_food_waste_prompt(sample_df):
    prompt = assemble_food_waste_prompt(sample_df, scope_desc="Single Day Test")
    assert "Single Day Test" in prompt
    assert "Paneer Rice Bowl" in prompt
    assert "## 1. Waste Analysis & Severity" in prompt
    assert "## 2. Possible Drivers (Facts vs. Hypotheses)" in prompt
    assert "Zero Hunger" in prompt


def test_assemble_demand_estimator_prompt(sample_df):
    prompt = assemble_demand_estimator_prompt(
        sample_df,
        menu_item="Paneer Rice Bowl",
        target_day="Monday",
        weather="Rainy",
        exam_period="Yes",
    )
    assert "Paneer Rice Bowl" in prompt
    assert "## 1. Baseline Demand Evaluation" in prompt
    assert "## 3. Recommended Preparation Target" in prompt
    assert "Batching" in prompt


def test_assemble_menu_optimizer_prompt(sample_df):
    prompt = assemble_menu_optimizer_prompt(sample_df, scope_desc="Test Scope")
    assert "## 1. Menu Portfolio Diagnostics" in prompt
    assert "Paneer Rice Bowl" in prompt
    assert "## 3. Menu Scheduling & Rotation Strategy" in prompt


def test_assemble_surplus_food_prompt(sample_df):
    prompt = assemble_surplus_food_prompt(sample_df, scope_desc="Lunch Surplus")
    assert "## 1. Safety & Edibility Screening" in prompt
    assert "NEVER suggest donating or serving food that is spoiled" in prompt
    assert "Paneer Rice Bowl" in prompt


def test_assemble_impact_analyst_prompt(sample_df):
    prompt = assemble_impact_analyst_prompt(sample_df, scope_desc="Weekly Impact")
    assert "## 1. Executive Summary & Waste Footprint" in prompt
    assert "## 2. Meal Portions & Zero Hunger Relevance" in prompt
    assert "₹7600" in prompt
