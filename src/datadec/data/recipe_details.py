from __future__ import annotations

from importlib.resources import as_file
from importlib.resources.abc import Traversable

import pandas as pd

from datadec.data import constants as consts


def get_data_recipe_family(
    data_name: str, data_recipe_families: dict[str, list[str]] | None = None
) -> str:
    if data_recipe_families is None:
        data_recipe_families = consts.DATA_RECIPE_FAMILIES

    for family, names in data_recipe_families.items():
        if data_name in names:
            return family
    return "unknown"


def get_data_recipe_details_df(ds_details_path: Traversable) -> pd.DataFrame:
    with as_file(ds_details_path) as csv_path:
        df = pd.read_csv(csv_path).rename(columns={"dataset": "data"})

    df["data"] = (
        df["data"]
        .str.replace("Dolma1.7 (no math code)", "Dolma1.7 (no math, code)")
        .str.replace("DCLM-Baseline (QC 7%", "DCLM-Baseline (QC 7%,")
    )

    return df
