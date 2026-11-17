"""Unit tests for the Sequential LLM Prompt Engineering Pipeline Engine."""

from unittest.mock import MagicMock
import pandas as pd
import pytest

from src.data_loader import load_and_validate_data
from src.ollama_client import OllamaClient
from src.pipeline_engine import (
    STAGE_METADATA,
    assemble_stage_user_prompt,
    can_run_stage,
    execute_pipeline_stage,
    generate_consolidated_report,
    initialize_pipeline_state,
    invalidate_downstream_stages,
    reset_pipeline_state,
)


@pytest.fixture
def sample_canteen_df() -> pd.DataFrame:
    """Fixture with a known sample dataframe."""
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
        },
        {
            "date": pd.to_datetime("2026-01-06").date(),
            "day_of_week": "Tuesday",
            "menu_item": "Veg Thali",
            "price_inr": 80,
            "meals_prepared": 150,
            "meals_sold": 130,
            "leftover_portions": 20,
            "estimated_waste_kg": 4.5,
            "waste_percentage": 13.33,
            "revenue_inr": 10400,
            "weather": "Rainy",
            "exam_period": "Yes",
            "college_event": "No",
        },
    ])


def test_initialize_pipeline_state():
    state = initialize_pipeline_state()
    assert len(state) == 5
    for s_id in range(1, 6):
        assert state[s_id]["status"] == "pending"
        assert state[s_id]["id"] == s_id
        assert "objective" in state[s_id]
        assert state[s_id]["raw_output"] == ""


def test_dependency_rules():
    state = initialize_pipeline_state()

    # Stage 1 has no dependencies
    can_s1, _ = can_run_stage(1, state)
    assert can_s1 is True

    # Stage 2 cannot run initially
    can_s2, reason = can_run_stage(2, state)
    assert can_s2 is False
    assert "Stage 1" in reason

    # Mark Stage 1 completed
    state[1]["status"] = "completed"
    can_s2_after, _ = can_run_stage(2, state)
    assert can_s2_after is True

    # Stage 3 requires Stage 2
    can_s3, _ = can_run_stage(3, state)
    assert can_s3 is False

    state[2]["status"] = "completed"
    can_s3_after, _ = can_run_stage(3, state)
    assert can_s3_after is True


def test_sequential_prompt_chaining(sample_canteen_df):
    state = initialize_pipeline_state()

    # Stage 1 prompt assembly
    p1, summary1, used1 = assemble_stage_user_prompt(1, sample_canteen_df, state)
    assert "Paneer Rice Bowl" in p1
    assert "Veg Thali" in p1
    assert used1 == []

    # Simulate Stage 1 output
    state[1]["status"] = "completed"
    state[1]["raw_output"] = "STAGE_1_WASTE_ANALYSIS_VERIFIED_OUTPUT"

    # Stage 2 prompt assembly
    p2, summary2, used2 = assemble_stage_user_prompt(2, sample_canteen_df, state)
    assert "STAGE_1_WASTE_ANALYSIS_VERIFIED_OUTPUT" in p2
    assert used2 == [1]

    # Simulate Stage 2 output
    state[2]["status"] = "completed"
    state[2]["raw_output"] = "STAGE_2_DEMAND_RECOMMENDATION_OUTPUT"

    # Stage 3 prompt assembly
    p3, summary3, used3 = assemble_stage_user_prompt(3, sample_canteen_df, state)
    assert "STAGE_1_WASTE_ANALYSIS_VERIFIED_OUTPUT" in p3
    assert "STAGE_2_DEMAND_RECOMMENDATION_OUTPUT" in p3
    assert used3 == [1, 2]

    # Simulate Stage 3 output
    state[3]["status"] = "completed"
    state[3]["raw_output"] = "STAGE_3_MENU_OPTIMIZER_OUTPUT"

    # Stage 4 prompt assembly
    p4, summary4, used4 = assemble_stage_user_prompt(4, sample_canteen_df, state)
    assert "STAGE_1_WASTE_ANALYSIS_VERIFIED_OUTPUT" in p4
    assert "STAGE_2_DEMAND_RECOMMENDATION_OUTPUT" in p4
    assert "STAGE_3_MENU_OPTIMIZER_OUTPUT" in p4
    assert "Food Recovery Hierarchy" in p4
    assert used4 == [1, 2, 3]

    # Simulate Stage 4 output
    state[4]["status"] = "completed"
    state[4]["raw_output"] = "STAGE_4_SURPLUS_MANAGER_OUTPUT"

    # Stage 5 prompt assembly
    p5, summary5, used5 = assemble_stage_user_prompt(5, sample_canteen_df, state)
    assert "STAGE_1_WASTE_ANALYSIS_VERIFIED_OUTPUT" in p5
    assert "STAGE_2_DEMAND_RECOMMENDATION_OUTPUT" in p5
    assert "STAGE_3_MENU_OPTIMIZER_OUTPUT" in p5
    assert "STAGE_4_SURPLUS_MANAGER_OUTPUT" in p5
    assert "SDG 2" in p5
    assert used5 == [1, 2, 3, 4]


def test_downstream_invalidation():
    state = initialize_pipeline_state()
    for s_id in range(1, 6):
        state[s_id]["status"] = "completed"
        state[s_id]["raw_output"] = f"Output for stage {s_id}"

    # Rerun Stage 2 -> Stages 3, 4, 5 must be marked stale!
    invalidate_downstream_stages(2, state)
    assert state[1]["status"] == "completed"
    assert state[2]["status"] == "completed"
    assert state[3]["status"] == "stale"
    assert state[4]["status"] == "stale"
    assert state[5]["status"] == "stale"
    assert "Stage 2" in state[3]["stale_reason"]


def test_reset_pipeline():
    state = initialize_pipeline_state()
    state[1]["status"] = "completed"
    state[1]["raw_output"] = "Some output"
    state[2]["status"] = "stale"

    reset_pipeline_state(state)
    assert state[1]["status"] == "pending"
    assert state[1]["raw_output"] == ""
    assert state[2]["status"] == "pending"


def test_execute_stage_with_mocked_ollama(sample_canteen_df):
    state = initialize_pipeline_state()

    mock_client = MagicMock(spec=OllamaClient)
    mock_client.generate.return_value = {
        "success": True,
        "response": "## 1. Waste Analysis & Severity\nWaste is moderate.",
        "duration_seconds": 1.25,
        "eval_count": 50,
        "error": None,
    }

    # Execute Stage 1
    res = execute_pipeline_stage(
        stage_id=1,
        df=sample_canteen_df,
        pipeline_state=state,
        ollama_client=mock_client,
    )
    assert res["status"] == "completed"
    assert "Waste is moderate" in res["raw_output"]
    assert res["duration_seconds"] == 1.25

    # Execute Stage 2 with mock
    mock_client.generate.return_value = {
        "success": True,
        "response": "## 1. Baseline Demand Evaluation\nDemand is steady.",
        "duration_seconds": 1.10,
        "eval_count": 45,
        "error": None,
    }
    res2 = execute_pipeline_stage(
        stage_id=2,
        df=sample_canteen_df,
        pipeline_state=state,
        ollama_client=mock_client,
    )
    assert res2["status"] == "completed"
    assert "Demand is steady" in res2["raw_output"]


def test_execute_stage_failure_preserves_earlier(sample_canteen_df):
    state = initialize_pipeline_state()
    state[1]["status"] = "completed"
    state[1]["raw_output"] = "Stage 1 Success Output"

    mock_client = MagicMock(spec=OllamaClient)
    mock_client.generate.return_value = {
        "success": False,
        "response": "",
        "duration_seconds": 0.5,
        "eval_count": 0,
        "error": "Connection to Ollama refused.",
    }

    res2 = execute_pipeline_stage(
        stage_id=2,
        df=sample_canteen_df,
        pipeline_state=state,
        ollama_client=mock_client,
    )
    assert res2["status"] == "failed"
    assert "refused" in res2["error"]
    # Verify Stage 1 output was NOT erased
    assert state[1]["status"] == "completed"
    assert state[1]["raw_output"] == "Stage 1 Success Output"


def test_generate_consolidated_report(sample_canteen_df):
    state = initialize_pipeline_state()
    for s_id in range(1, 6):
        state[s_id]["status"] = "completed"
        state[s_id]["raw_output"] = f"### Section {s_id}\nContent of stage {s_id}."
        state[s_id]["duration_seconds"] = 1.0

    report = generate_consolidated_report(sample_canteen_df, state)
    assert "# 🍲 Zero Hunger Smart Canteen" in report
    assert "Stage 1: Food Waste Analyst" in report
    assert "Stage 5: Impact Analyst" in report
    assert "Content of stage 1" in report
    assert "Content of stage 5" in report
    assert "UN Sustainable Development Goals" in report
