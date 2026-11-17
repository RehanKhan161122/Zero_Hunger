"""Prompt template loading, metadata definition, and assembly with calculated metrics."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from src.analytics import (
    calculate_demand_baseline,
    calculate_menu_performance,
    calculate_summary_metrics,
    format_menu_summary_table,
)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

MODULE_METADATA: Dict[str, Dict[str, str]] = {
    "food_waste_analyst": {
        "title": "Food Waste Analyst",
        "description": "Identify high-waste patterns, diagnose root causes, and separate empirical facts from hypotheses.",
        "icon": "🔍",
        "scope_support": "Single Record, Filtered Slice, or Full Dataset",
    },
    "demand_estimator": {
        "title": "Demand Estimator",
        "description": "Recommend target preparation quantities using transparent historical baselines and staged cooking.",
        "icon": "📈",
        "scope_support": "Item & Day Context",
    },
    "menu_optimizer": {
        "title": "Menu Optimizer",
        "description": "Recommend menu/prep adjustments balancing waste reduction, demand stability, and student variety.",
        "icon": "📋",
        "scope_support": "Menu-wide Comparative Aggregate",
    },
    "surplus_food_manager": {
        "title": "Surplus Food Manager",
        "description": "Safe, timely handling and redistribution of edible surplus with strict food safety adherence.",
        "icon": "🍱",
        "scope_support": "Service Window & Leftover Volumes",
    },
    "impact_analyst": {
        "title": "Impact Analyst",
        "description": "Summarize calculated waste/portion/revenue metrics aligned with UN SDG 2 (Zero Hunger).",
        "icon": "🌱",
        "scope_support": "Operational Period Summary",
    },
}


def get_available_modules() -> List[str]:
    """Return the list of prompt module identifiers."""
    return list(MODULE_METADATA.keys())


def load_template(module_name: str) -> str:
    """Load the raw text template from the prompts directory."""
    template_path = PROMPTS_DIR / f"{module_name}.txt"
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template file not found at: {template_path}")
    return template_path.read_text(encoding="utf-8")


def _summarize_condition(series: pd.Series) -> str:
    """Helper to summarize distribution of categorical conditions."""
    if series.empty:
        return "Not available"
    counts = series.value_counts()
    items = [f"{val}: {count}" for val, count in counts.items()]
    return ", ".join(items)


def assemble_food_waste_prompt(
    df: pd.DataFrame,
    scope_desc: str = "Current Selection",
    additional_context: str = "Standard college operating day",
) -> str:
    """Assemble the Food Waste Analyst prompt using calculated metrics."""
    template = load_template("food_waste_analyst")
    metrics = calculate_summary_metrics(df)

    if df.empty:
        return template.format(
            analysis_scope=scope_desc,
            date_range="None",
            menu_items="None",
            total_prepared=0,
            total_sold=0,
            leftover_portions=0,
            estimated_waste_kg=0.0,
            waste_percentage=0.0,
            weather_summary="No data",
            exam_period_summary="No data",
            college_event_summary="No data",
            additional_context="Empty dataset selected.",
        )

    menu_items_str = ", ".join(df["menu_item"].unique())
    date_range_str = (
        f"{metrics['date_min']} to {metrics['date_max']}"
        if metrics["date_min"] != metrics["date_max"]
        else metrics["date_min"]
    )

    weather_str = _summarize_condition(df["weather"])
    exam_str = _summarize_condition(df["exam_period"])
    event_str = _summarize_condition(df["college_event"])

    return template.format(
        analysis_scope=scope_desc,
        date_range=date_range_str,
        menu_items=menu_items_str,
        total_prepared=metrics["total_prepared"],
        total_sold=metrics["total_sold"],
        leftover_portions=metrics["total_leftovers"],
        estimated_waste_kg=metrics["total_waste_kg"],
        waste_percentage=metrics["waste_percentage"],
        weather_summary=weather_str,
        exam_period_summary=exam_str,
        college_event_summary=event_str,
        additional_context=additional_context or "None noted",
    )


def assemble_demand_estimator_prompt(
    df_full: pd.DataFrame,
    menu_item: str,
    target_day: str = "Monday",
    weather: str = "Sunny",
    exam_period: str = "No",
    college_event: str = "No",
    additional_context: str = "",
) -> str:
    """Assemble Demand Estimator prompt using historical aggregates for the item."""
    template = load_template("demand_estimator")
    baseline = calculate_demand_baseline(df_full, menu_item, day_of_week=target_day)

    return template.format(
        menu_item=menu_item,
        target_day=f"{target_day} (historical reference)",
        historical_records_count=baseline["historical_records_count"],
        avg_meals_sold=baseline["avg_meals_sold"],
        median_meals_sold=baseline["median_meals_sold"],
        min_meals_sold=baseline["min_meals_sold"],
        max_meals_sold=baseline["max_meals_sold"],
        avg_waste_percentage=baseline["avg_waste_percentage"],
        calculated_baseline_target=baseline["calculated_baseline_target"],
        weather_summary=weather,
        exam_period_summary=f"Exams in progress: {exam_period}",
        college_event_summary=f"Event scheduled: {college_event}",
        additional_context=(
            f"{baseline['baseline_strategy']} | "
            + (additional_context or "General campus routine")
        ),
    )


def assemble_menu_optimizer_prompt(
    df: pd.DataFrame,
    scope_desc: str = "Filtered Selection",
    additional_context: str = "",
) -> str:
    """Assemble Menu Optimizer prompt using aggregated menu performance."""
    template = load_template("menu_optimizer")
    metrics = calculate_summary_metrics(df)
    menu_perf = calculate_menu_performance(df)

    if menu_perf.empty:
        summary_table = "No data available."
        highest_waste = "None"
        top_revenue = "None"
    else:
        summary_table = format_menu_summary_table(menu_perf, max_rows=10)

        # Top 3 waste percentage
        top_waste_df = menu_perf.sort_values(
            by="waste_percentage", ascending=False
        ).head(3)
        highest_waste = "; ".join(
            f"{row['menu_item']} ({row['waste_percentage']}% waste, {row['total_waste_kg']} kg)"
            for _, row in top_waste_df.iterrows()
        )

        # Top 3 revenue
        top_rev_df = menu_perf.sort_values(
            by="total_revenue_inr", ascending=False
        ).head(3)
        top_revenue = "; ".join(
            f"{row['menu_item']} (₹{int(row['total_revenue_inr'])}, {int(row['total_sold'])} sold)"
            for _, row in top_rev_df.iterrows()
        )

    date_range_str = (
        f"{metrics['date_min']} to {metrics['date_max']}"
        if metrics["date_min"] != "N/A"
        else "N/A"
    )
    operating_context = (
        f"Weather: {_summarize_condition(df['weather'])} | "
        f"Exams: {_summarize_condition(df['exam_period'])} | "
        f"Events: {_summarize_condition(df['college_event'])}"
        if not df.empty
        else "N/A"
    )

    return template.format(
        analysis_scope=scope_desc,
        date_range=date_range_str,
        menu_summary_table=summary_table,
        highest_waste_items=highest_waste,
        top_revenue_items=top_revenue,
        operating_context=operating_context,
        additional_context=additional_context or "Standard kitchen capacity and storage",
    )


def assemble_surplus_food_prompt(
    df: pd.DataFrame,
    scope_desc: str = "End of Service Surplus",
    service_phase: str = "End of Lunch Service (within 1 hour)",
    logistical_context: str = "On-campus cooling and local partner shelter nearby",
    additional_context: str = "",
) -> str:
    """Assemble Surplus Food Manager prompt adhering strictly to food safety."""
    template = load_template("surplus_food_manager")
    metrics = calculate_summary_metrics(df)
    menu_perf = calculate_menu_performance(df)

    if menu_perf.empty:
        items_breakdown = "No surplus recorded."
    else:
        # Items with leftovers > 0
        surplus_df = menu_perf[menu_perf["total_leftovers"] > 0]
        if surplus_df.empty:
            items_breakdown = "Zero leftovers recorded in this selection."
        else:
            lines = []
            for _, row in surplus_df.iterrows():
                lines.append(
                    f"- {row['menu_item']}: {int(row['total_leftovers'])} portions ({row['total_waste_kg']} kg approx)"
                )
            items_breakdown = "\n".join(lines)

    weather_str = _summarize_condition(df["weather"]) if not df.empty else "Normal"
    service_window_str = (
        f"{metrics['date_min']} to {metrics['date_max']}"
        if metrics["date_min"] != "N/A"
        else "Today's service"
    )

    return template.format(
        analysis_scope=scope_desc,
        service_window=service_window_str,
        surplus_items_breakdown=items_breakdown,
        total_surplus_portions=metrics["total_leftovers"],
        total_surplus_kg=metrics["total_waste_kg"],
        weather_summary=weather_str,
        service_phase=service_phase,
        logistical_context=logistical_context,
    )


def assemble_impact_analyst_prompt(
    df: pd.DataFrame,
    scope_desc: str = "Current Scope Period",
    context_note: str = "Demonstration synthetic canteen dataset representing student dining hall operations",
) -> str:
    """Assemble Impact Analyst prompt summarizing computed metrics."""
    template = load_template("impact_analyst")
    metrics = calculate_summary_metrics(df)

    date_range_str = (
        f"{metrics['date_min']} to {metrics['date_max']}"
        if metrics["date_min"] != "N/A"
        else "N/A"
    )

    weather_str = _summarize_condition(df["weather"]) if not df.empty else "N/A"
    exam_str = _summarize_condition(df["exam_period"]) if not df.empty else "N/A"
    event_str = _summarize_condition(df["college_event"]) if not df.empty else "N/A"

    return template.format(
        analysis_scope=scope_desc,
        date_range=date_range_str,
        total_records=metrics["total_records"],
        total_prepared=metrics["total_prepared"],
        total_sold=metrics["total_sold"],
        total_leftovers=metrics["total_leftovers"],
        waste_percentage=metrics["waste_percentage"],
        estimated_waste_kg=metrics["total_waste_kg"],
        total_revenue_inr=int(metrics["total_revenue_inr"]),
        estimated_waste_cost_inr=int(metrics["estimated_waste_cost_inr"]),
        exam_period_summary=exam_str,
        college_event_summary=event_str,
        weather_summary=weather_str,
        context_note=context_note,
    )
