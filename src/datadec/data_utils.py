"""Dataset utility functions for DataDecide.

This module contains functions for working with dataset metadata, recipe families,
and other dataset-specific operations.
"""

from typing import Dict, List, Optional

import pandas as pd

from datadec import constants as consts


def get_data_recipe_family(
    data_name: str, data_recipe_families: Optional[Dict[str, List[str]]] = None
) -> str:
    """Get the recipe family for a given data name.

    Maps individual dataset names to their broader recipe family categories
    (e.g., "Dolma1.7" -> "dolma17", "C4" -> "c4").

    Args:
        data_name: Name of the dataset to look up
        data_recipe_families: Optional custom mapping dict. If None, uses default from constants

    Returns:
        String name of the recipe family, or "unknown" if not found
    """
    if data_recipe_families is None:
        data_recipe_families = consts.DATA_RECIPE_FAMILIES

    for family, names in data_recipe_families.items():
        if data_name in names:
            return family
    return "unknown"


def load_ds_details_df(ds_details_path) -> pd.DataFrame:
    """Load and clean the dataset details CSV file.

    Reads the dataset features CSV and applies necessary data cleaning including
    column renaming and fixing inconsistent dataset names.

    Args:
        ds_details_path: Path to the dataset_features.csv file

    Returns:
        Cleaned DataFrame with dataset details and metadata
    """
    df = pd.read_csv(ds_details_path).rename(columns={"dataset": "data"})

    # Apply data name corrections to fix inconsistencies
    df["data"] = (
        df["data"]
        .str.replace("Dolma1.7 (no math code)", "Dolma1.7 (no math, code)")
        .str.replace("DCLM-Baseline (QC 7%", "DCLM-Baseline (QC 7%,")
    )

    return df
