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

OLMES_DETAIL_RECIPES = (
    "c4",
    "dclm-baseline-25p-dolma1.7-75p",
    "dclm-baseline-50p-dolma1.7-50p",
    "dclm-baseline-75p-dolma1.7-25p",
    "dclm-baseline-top-10p",
    "dclm-baseline-top-20p",
    "dclm-baseline-top-fw-10p",
    "dclm-baseline-top-fw-3p",
    "dclm-baseline-top-fw2-7p",
    "dclm-baseline-top-fw3-7p",
    "dclm-baseline",
    "dolma1.6++",
    "dolma1.7-no-code",
    "dolma1.7-no-flan",
    "dolma1.7-no-math-no-code",
    "dolma1.7-no-reddit",
    "dolma1.7",
    "falcon-with-cc-top-10p",
    "falcon-with-cc-top-20p",
    "falcon-with-cc-top-orig-10p",
    "falcon-with-cc-top-tulu-10p",
    "falcon-with-cc",
    "falcon",
    "fineweb-edu",
    "fineweb-pro",
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


def test_source_manifest_identifies_download_inputs_and_provenance() -> None:
    manifest = load_source_manifest()

    assert manifest.ppl.model_dump() == {
        "id": "ppl",
        "provider": "datasets",
        "repo_id": "allenai/DataDecide-ppl-results",
        "revision": "c4a9fa360a0c8351e71f3ede04dd165995fab68c",
        "split": "train",
        "output": "raw/ppl.parquet",
    }
    assert manifest.olmes.model_dump() == {
        "id": "olmes",
        "provider": "datasets",
        "repo_id": "allenai/DataDecide-eval-results",
        "revision": "9919b5a0e61e57a85021263918fa82d6ceaee038",
        "split": "train",
        "output": "raw/olmes.parquet",
    }
    assert manifest.olmes_details.model_dump() == {
        "id": "olmes-details",
        "provider": "huggingface_hub",
        "repo_type": "dataset",
        "repo_id": "allenai/DataDecide-eval-instances",
        "revision": "23f3b2e186ca6c39026e3efa00e4af397680c075",
        "filename_template": "models/{recipe}.tar.gz",
        "output_root": "raw/olmes-details",
        "recipes": OLMES_DETAIL_RECIPES,
    }
    assert manifest.archives[0].downloaded_on == date(2025, 9, 19)


def test_dataset_features_csv_loads_from_configs() -> None:
    assert config_file("dataset_features.csv") == DATASET_FEATURES_CSV
    with as_file(DATASET_FEATURES_CSV) as csv_path:
        features = pd.read_csv(csv_path)

    assert len(features) == 25
    assert features.iloc[0]["dataset"] == "Dolma1.7"
