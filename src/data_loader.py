"""Data loading, path resolution, schema validation, and filtering for canteen datasets."""

import io
import os
from pathlib import Path
from typing import Any, List, Optional, Union
import pandas as pd

REQUIRED_COLUMNS = [
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
    "weather",
    "exam_period",
    "college_event",
]

NUMERIC_COLUMNS = [
    "price_inr",
    "meals_prepared",
    "meals_sold",
    "leftover_portions",
    "estimated_waste_kg",
    "waste_percentage",
    "revenue_inr",
]

COLUMN_ALIASES = {
    "date": ["date", "datetime", "service_date"],
    "day_of_week": ["day_of_week", "dayofweek", "weekday", "day"],
    "menu_item": ["menu_item", "item", "dish", "food_item", "item_name"],
    "price_inr": ["price_inr", "price", "item_price", "rate_inr"],
    "meals_prepared": ["meals_prepared", "prepared", "quantity_prepared", "qty_prepared"],
    "meals_sold": ["meals_sold", "sold", "quantity_sold", "qty_sold"],
    "leftover_portions": ["leftover_portions", "leftovers", "leftover", "surplus_portions"],
    "estimated_waste_kg": ["estimated_waste_kg", "waste_kg", "waste_weight_kg"],
    "waste_percentage": ["waste_percentage", "waste_pct", "waste_%"],
    "revenue_inr": ["revenue_inr", "revenue", "sales_inr", "total_revenue"],
    "weather": ["weather", "weather_condition"],
    "exam_period": ["exam_period", "exams", "exam", "is_exam"],
    "college_event": ["college_event", "event", "events", "is_event"],
}


def find_dataset_path(custom_path: Optional[Union[str, Path]] = None) -> Path:
    """Resolve the dataset file path checking environment, defaults, and workspace."""
    candidate_paths: List[Path] = []

    if custom_path:
        candidate_paths.append(Path(custom_path))

    env_path = os.getenv("CANTEEN_DATA_PATH")
    if env_path:
        candidate_paths.append(Path(env_path))

    # Standard relative candidates
    base_dir = Path(__file__).resolve().parent.parent
    candidate_paths.extend([
        base_dir / "Data" / "synthetic_canteen_data_500.csv",
        base_dir / "data" / "synthetic_canteen_data_500.csv",
        Path("Data/synthetic_canteen_data_500.csv"),
        Path("data/synthetic_canteen_data_500.csv"),
    ])

    for path in candidate_paths:
        if path.exists() and path.is_file():
            return path.resolve()

    # Fallback search in base_dir
    found = list(base_dir.glob("**/synthetic_canteen_data*.csv"))
    if found:
        return found[0].resolve()

    searched_str = "\n".join(f" - {p}" for p in candidate_paths)
    raise FileNotFoundError(
        f"Canteen dataset not found. Checked the following locations:\n{searched_str}\n"
        f"Please verify 'data/synthetic_canteen_data_500.csv' exists or set CANTEEN_DATA_PATH."
    )


def validate_and_process_canteen_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Validate schema, apply column aliasing, cast datatypes, and format canteen data."""
    if raw_df is None or raw_df.empty:
        raise ValueError("The provided dataset is empty. Please provide a CSV with valid canteen records.")

    df = raw_df.copy()

    # Standardize column headers (lowercase, stripped)
    column_mapping = {}
    current_cols = [str(c).strip() for c in df.columns]
    df.columns = current_cols

    # Check for alias matches if required columns are not directly present
    for req_col in REQUIRED_COLUMNS:
        if req_col not in df.columns:
            # Look for alias
            for candidate in COLUMN_ALIASES.get(req_col, []):
                for col in df.columns:
                    if col.lower() == candidate.lower():
                        column_mapping[col] = req_col
                        break
                if req_col in column_mapping.values():
                    break

    if column_mapping:
        df = df.rename(columns=column_mapping)

    # Check for missing required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset is missing required columns: {', '.join(missing_cols)}.\n"
            f"Detected columns: {list(df.columns)}.\n"
            f"Expected schema: {', '.join(REQUIRED_COLUMNS)}"
        )

    # Clean string columns
    for col in ["day_of_week", "menu_item", "weather", "exam_period", "college_event"]:
        df[col] = df[col].astype(str).str.strip()

    # Parse and validate dates
    try:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    except Exception as e:
        raise ValueError(f"Error parsing 'date' column into valid dates: {e}") from e

    # Validate and cast numeric columns
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # If waste_percentage is 0 but leftovers > 0 and meals_prepared > 0, recalculate accurately
    mask = (df["waste_percentage"] == 0) & (df["meals_prepared"] > 0) & (df["leftover_portions"] > 0)
    if mask.any():
        df.loc[mask, "waste_percentage"] = (
            (df.loc[mask, "leftover_portions"] / df.loc[mask, "meals_prepared"]) * 100.0
        ).round(2)

    # If revenue_inr is 0, calculate as meals_sold * price_inr
    rev_mask = (df["revenue_inr"] == 0) & (df["meals_sold"] > 0) & (df["price_inr"] > 0)
    if rev_mask.any():
        df.loc[rev_mask, "revenue_inr"] = (df.loc[rev_mask, "meals_sold"] * df.loc[rev_mask, "price_inr"]).round(2)

    # Sort deterministically
    df = df.sort_values(by=["date", "menu_item"]).reset_index(drop=True)
    return df


def load_and_validate_data(csv_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load and validate the canteen CSV dataset from local filesystem."""
    resolved_path = find_dataset_path(csv_path)

    try:
        raw_df = pd.read_csv(resolved_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV at {resolved_path}: {e}") from e

    return validate_and_process_canteen_dataframe(raw_df)


def load_uploaded_csv(file_obj: Any) -> pd.DataFrame:
    """Load and validate an uploaded CSV file buffer (Streamlit UploadedFile or BytesIO/StringIO)."""
    if file_obj is None:
        raise ValueError("No file provided.")

    try:
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
        raw_df = pd.read_csv(file_obj)
    except Exception as e:
        raise ValueError(f"Could not parse uploaded CSV file: {str(e)}") from e

    return validate_and_process_canteen_dataframe(raw_df)


def filter_dataset(
    df: pd.DataFrame,
    start_date: Optional[Union[str, pd.Timestamp, object]] = None,
    end_date: Optional[Union[str, pd.Timestamp, object]] = None,
    menu_items: Optional[List[str]] = None,
    weather: Optional[List[str]] = None,
    exam_period: Optional[List[str]] = None,
    college_event: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Filter dataset by date range, menu items, weather, and operational flags."""
    if df.empty:
        return df.copy()

    filtered = df.copy()

    # Date filters
    if start_date is not None:
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date).date()
        filtered = filtered[filtered["date"] >= start_date]

    if end_date is not None:
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date).date()
        filtered = filtered[filtered["date"] <= end_date]

    # Categorical filters
    if menu_items:
        filtered = filtered[filtered["menu_item"].isin(menu_items)]

    if weather:
        filtered = filtered[filtered["weather"].isin(weather)]

    if exam_period:
        filtered = filtered[filtered["exam_period"].isin(exam_period)]

    if college_event:
        filtered = filtered[filtered["college_event"].isin(college_event)]

    return filtered.reset_index(drop=True)
