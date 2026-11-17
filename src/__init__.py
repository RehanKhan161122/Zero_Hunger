"""Zero Hunger Smart Canteen Food Waste Optimization package."""

from src.data_loader import (
    COLUMN_ALIASES,
    REQUIRED_COLUMNS,
    filter_dataset,
    find_dataset_path,
    load_and_validate_data,
    load_uploaded_csv,
    validate_and_process_canteen_dataframe,
)

__version__ = "1.0.0"
