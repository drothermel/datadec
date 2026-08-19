from __future__ import annotations

from datetime import date
from importlib.resources import as_file

import pandas as pd

from datadec.config import (
    DATASET_FEATURES_CSV,
    config_file,
    load_catalog,
    load_source_manifest,
)


def test_catalog_loads_model_and_recipe_configuration() -> None:
    catalog = load_catalog()

    assert [model.name for model in catalog.models] == [
        "4M",
        "6M",
        "8M",
        "10M",
        "14M",
        "16M",
        "20M",
        "60M",
        "90M",
        "150M",
        "300M",
        "530M",
        "750M",
        "1B",
    ]
    assert catalog.data_recipe_families["c4"] == ["C4"]
    assert catalog.perplexity_name_map["eval/pile-validation/Perplexity"] == (
        "pile-valppl"
    )


def test_source_manifest_identifies_processing_inputs_and_provenance() -> None:
    manifest = load_source_manifest()

    processing = manifest.sources_for_profile("processing")
    assert [source.repo_id for source in processing] == [
        "allenai/DataDecide-ppl-results",
        "allenai/DataDecide-eval-results",
    ]
    assert [source.output for source in processing] == ["ppl_raw", "dwn_raw"]
    assert manifest.archives[0].downloaded_on == date(2025, 9, 19)


def test_dataset_features_csv_loads_from_configs() -> None:
    assert config_file("dataset_features.csv") == DATASET_FEATURES_CSV
    with as_file(DATASET_FEATURES_CSV) as csv_path:
        features = pd.read_csv(csv_path)

    assert len(features) == 25
    assert features.iloc[0]["dataset"] == "Dolma1.7"
