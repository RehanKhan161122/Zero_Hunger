"""Deterministic numeric analytics and aggregation for canteen food waste data.

All arithmetic and numerical metrics are strictly calculated here in pure Python.
The LLM is provided these calculated values and does not perform arithmetic.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def calculate_summary_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate aggregate summary metrics for a given dataframe slice.
    
    Handles empty slices and zero division safely.
    """
    if df.empty:
        return {
            "total_records": 0,
            "total_prepared": 0,
            "total_sold": 0,
            "total_leftovers": 0,
            "total_waste_kg": 0.0,
            "waste_percentage": 0.0,
            "total_revenue_inr": 0.0,
            "estimated_waste_cost_inr": 0.0,
            "avg_prepared_per_record": 0.0,
            "avg_sold_per_record": 0.0,
            "avg_waste_kg_per_record": 0.0,
            "avg_waste_percentage": 0.0,
            "unique_menu_items_count": 0,
            "date_min": "N/A",
            "date_max": "N/A",
        }

    total_records = len(df)
    total_prepared = int(df["meals_prepared"].sum())
    total_sold = int(df["meals_sold"].sum())
    total_leftovers = int(df["leftover_portions"].sum())
    total_waste_kg = round(float(df["estimated_waste_kg"].sum()), 2)
    total_revenue_inr = round(float(df["revenue_inr"].sum()), 2)

    # Waste percentage calculated from aggregate leftovers vs aggregate prepared
    waste_percentage = (
        round((total_leftovers / total_prepared) * 100.0, 2)
        if total_prepared > 0
        else 0.0
    )

    # Estimated monetary value of food waste: sum(leftovers * price)
    waste_costs = df["leftover_portions"] * df["price_inr"]
    estimated_waste_cost_inr = round(float(waste_costs.sum()), 2)

    avg_prepared = round(float(df["meals_prepared"].mean()), 2)
    avg_sold = round(float(df["meals_sold"].mean()), 2)
    avg_waste_kg = round(float(df["estimated_waste_kg"].mean()), 2)
    avg_waste_pct = round(float(df["waste_percentage"].mean()), 2)

    unique_items = int(df["menu_item"].nunique())
    date_min = str(df["date"].min())
    date_max = str(df["date"].max())

    return {
        "total_records": total_records,
        "total_prepared": total_prepared,
        "total_sold": total_sold,
        "total_leftovers": total_leftovers,
        "total_waste_kg": total_waste_kg,
        "waste_percentage": waste_percentage,
        "total_revenue_inr": total_revenue_inr,
        "estimated_waste_cost_inr": estimated_waste_cost_inr,
        "avg_prepared_per_record": avg_prepared,
        "avg_sold_per_record": avg_sold,
        "avg_waste_kg_per_record": avg_waste_kg,
        "avg_waste_percentage": avg_waste_pct,
        "unique_menu_items_count": unique_items,
        "date_min": date_min,
        "date_max": date_max,
    }


def calculate_menu_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate performance metrics per menu item.
    
    Includes sales volume, leftovers, waste kg, waste percentage, revenue, and waste cost.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "menu_item",
                "records_count",
                "total_prepared",
                "total_sold",
                "total_leftovers",
                "total_waste_kg",
                "waste_percentage",
                "total_revenue_inr",
                "estimated_waste_cost_inr",
                "avg_price_inr",
            ]
        )

    grouped = (
        df.groupby("menu_item")
        .apply(
            lambda g: pd.Series(
                {
                    "records_count": len(g),
                    "total_prepared": int(g["meals_prepared"].sum()),
                    "total_sold": int(g["meals_sold"].sum()),
                    "total_leftovers": int(g["leftover_portions"].sum()),
                    "total_waste_kg": round(float(g["estimated_waste_kg"].sum()), 2),
                    "waste_percentage": (
                        round(
                            (g["leftover_portions"].sum() / g["meals_prepared"].sum())
                            * 100.0,
                            2,
                        )
                        if g["meals_prepared"].sum() > 0
                        else 0.0
                    ),
                    "total_revenue_inr": round(float(g["revenue_inr"].sum()), 2),
                    "estimated_waste_cost_inr": round(
                        float((g["leftover_portions"] * g["price_inr"]).sum()), 2
                    ),
                    "avg_price_inr": round(float(g["price_inr"].mean()), 2),
                }
            )
        )
        .reset_index()
    )

    return grouped.sort_values(by="waste_percentage", ascending=False).reset_index(
        drop=True
    )


def calculate_demand_baseline(
    df: pd.DataFrame,
    menu_item: str,
    day_of_week: Optional[str] = None,
) -> Dict[str, Any]:
    """Calculate historical demand statistics for a menu item.
    
    Provides transparent arithmetic baseline targets (median demand) without claiming
    unverified ML statistical prediction.
    """
    subset = df[df["menu_item"] == menu_item]
    if day_of_week and not subset.empty:
        day_subset = subset[subset["day_of_week"] == day_of_week]
        if not day_subset.empty:
            subset = day_subset

    if subset.empty:
        return {
            "menu_item": menu_item,
            "day_of_week": day_of_week or "All",
            "historical_records_count": 0,
            "avg_meals_sold": 0.0,
            "median_meals_sold": 0.0,
            "min_meals_sold": 0,
            "max_meals_sold": 0,
            "std_meals_sold": 0.0,
            "avg_waste_percentage": 0.0,
            "calculated_baseline_target": 0,
            "baseline_strategy": "No historical observations available.",
        }

    records_count = len(subset)
    avg_sold = round(float(subset["meals_sold"].mean()), 1)
    median_sold = round(float(subset["meals_sold"].median()), 1)
    min_sold = int(subset["meals_sold"].min())
    max_sold = int(subset["meals_sold"].max())
    std_sold = round(float(subset["meals_sold"].std(ddof=0)), 1)
    avg_waste_pct = round(float(subset["waste_percentage"].mean()), 2)

    # Transparent baseline: Use median sales as robust central tendency,
    # plus a conservative 5% buffer rounded to nearest integer.
    calculated_baseline_target = int(round(median_sold * 1.05))

    return {
        "menu_item": menu_item,
        "day_of_week": day_of_week or "All Days",
        "historical_records_count": records_count,
        "avg_meals_sold": avg_sold,
        "median_meals_sold": median_sold,
        "min_meals_sold": min_sold,
        "max_meals_sold": max_sold,
        "std_meals_sold": std_sold,
        "avg_waste_percentage": avg_waste_pct,
        "calculated_baseline_target": calculated_baseline_target,
        "baseline_strategy": (
            f"Median historical sales ({median_sold} portions) with a conservative "
            f"5% buffer ({calculated_baseline_target} portions target)."
        ),
    }


def calculate_waste_by_attribute(df: pd.DataFrame, attribute: str) -> pd.DataFrame:
    """Group waste metrics by an operational context attribute (weather, exam_period, etc.)."""
    if df.empty or attribute not in df.columns:
        return pd.DataFrame(
            columns=[
                attribute,
                "records_count",
                "total_prepared",
                "total_sold",
                "total_leftovers",
                "total_waste_kg",
                "waste_percentage",
                "total_revenue_inr",
            ]
        )

    grouped = (
        df.groupby(attribute)
        .apply(
            lambda g: pd.Series(
                {
                    "records_count": len(g),
                    "total_prepared": int(g["meals_prepared"].sum()),
                    "total_sold": int(g["meals_sold"].sum()),
                    "total_leftovers": int(g["leftover_portions"].sum()),
                    "total_waste_kg": round(float(g["estimated_waste_kg"].sum()), 2),
                    "waste_percentage": (
                        round(
                            (g["leftover_portions"].sum() / g["meals_prepared"].sum())
                            * 100.0,
                            2,
                        )
                        if g["meals_prepared"].sum() > 0
                        else 0.0
                    ),
                    "total_revenue_inr": round(float(g["revenue_inr"].sum()), 2),
                }
            )
        )
        .reset_index()
    )

    return grouped.sort_values(by="total_waste_kg", ascending=False).reset_index(
        drop=True
    )


def get_daily_waste_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily time series aggregations."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "total_prepared",
                "total_sold",
                "total_leftovers",
                "total_waste_kg",
                "waste_percentage",
                "total_revenue_inr",
            ]
        )

    daily = (
        df.groupby("date")
        .apply(
            lambda g: pd.Series(
                {
                    "total_prepared": int(g["meals_prepared"].sum()),
                    "total_sold": int(g["meals_sold"].sum()),
                    "total_leftovers": int(g["leftover_portions"].sum()),
                    "total_waste_kg": round(float(g["estimated_waste_kg"].sum()), 2),
                    "waste_percentage": (
                        round(
                            (g["leftover_portions"].sum() / g["meals_prepared"].sum())
                            * 100.0,
                            2,
                        )
                        if g["meals_prepared"].sum() > 0
                        else 0.0
                    ),
                    "total_revenue_inr": round(float(g["revenue_inr"].sum()), 2),
                }
            )
        )
        .reset_index()
    )

    return daily.sort_values(by="date").reset_index(drop=True)


def format_menu_summary_table(df_menu: pd.DataFrame, max_rows: int = 10) -> str:
    """Format menu performance dataframe into clean text table for prompts."""
    if df_menu.empty:
        return "No menu item data available."

    top_rows = df_menu.head(max_rows)
    lines = [
        "| Menu Item | Prep | Sold | Leftovers | Waste % | Waste (kg) | Revenue (₹) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for _, row in top_rows.iterrows():
        lines.append(
            f"| {row['menu_item']} | {int(row['total_prepared'])} | {int(row['total_sold'])} | "
            f"{int(row['total_leftovers'])} | {row['waste_percentage']}% | "
            f"{row['total_waste_kg']} kg | ₹{int(row['total_revenue_inr'])} |"
        )
    return "\n".join(lines)
