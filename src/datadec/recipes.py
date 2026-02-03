from __future__ import annotations

from pathlib import Path

import pandas as pd


def get_data_recipe_details_df(ds_details_path: Path) -> pd.DataFrame:
    df = pd.read_csv(ds_details_path).rename(columns={"dataset": "data"})

    df["data"] = (
        df["data"]
        .str.replace("Dolma1.7 (no math code)", "Dolma1.7 (no math, code)")
        .str.replace("DCLM-Baseline (QC 7%", "DCLM-Baseline (QC 7%,")
    )

    return df
