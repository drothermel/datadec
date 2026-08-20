from __future__ import annotations

from collections import Counter
from datetime import date

from datadec.config import (
    load_catalog,
    load_published_results_manifest,
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

PUBLISHED_RESULTS_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1weYlEOlHrA_fzT2OsRa40uLc4EKTGz1D"
)
SCALING_LAW_FILES = {
    "raw_data/results_ladder_5xC_seeds.csv": 1_807_261_804,
    "raw_data/results_ladder_5xC_small_seed_extras.csv": 527_061_496,
    "raw_data/results_ladder_5xC_small_seeds_extra_real.csv": 527_426_471,
}


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
    one_billion = catalog.models[-1]
    assert one_billion.nominal_parameter_count == 1_000_000_000
    assert one_billion.training_parameter_count == 1_000_000_000
    assert one_billion.exact_parameter_count == 1_176_832_000
    assert catalog.training.flops_per_token_per_parameter == 6


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


def test_published_results_inventory_pins_public_folder_contract() -> None:
    manifest = load_published_results_manifest()
    scaling_law = [file for file in manifest.files if file.category == "scaling_law"]
    published_results = [
        file for file in manifest.files if file.category == "published_results"
    ]

    assert manifest.folder_url == PUBLISHED_RESULTS_FOLDER_URL
    assert len(manifest.files) == 134
    assert len(scaling_law) == 3
    assert len(published_results) == 131
    assert sum(file.expected_size for file in manifest.files) == 11_919_102_101
    assert sum(file.expected_size for file in scaling_law) == 2_861_749_771
    assert sum(file.expected_size for file in published_results) == 9_057_352_330
    assert {file.path: file.expected_size for file in scaling_law} == (
        SCALING_LAW_FILES
    )


def test_published_results_inventory_is_disjoint_and_unique() -> None:
    manifest = load_published_results_manifest()
    ids = [file.id for file in manifest.files]
    paths = [file.path for file in manifest.files]
    category_paths = {
        category: {file.path for file in manifest.files if file.category == category}
        for category in ("scaling_law", "published_results")
    }
    destinations = [
        (
            f"raw/scaling-law/{file.path.rsplit('/', 1)[-1]}"
            if file.category == "scaling_law"
            else f"reference/published-results/{file.path}"
        )
        for file in manifest.files
    ]

    assert len(ids) == len(set(ids))
    assert len(paths) == len(set(paths))
    assert len(destinations) == len(set(destinations))
    assert category_paths["scaling_law"].isdisjoint(category_paths["published_results"])
    assert Counter(path.rsplit(".", 1)[-1] for path in paths) == {
        "csv": 43,
        "json": 11,
        "pdf": 40,
        "png": 40,
    }
    assert {
        top_level: sum(
            file.expected_size
            for file in manifest.files
            if file.path.split("/", 1)[0] == top_level
        )
        for top_level in {path.split("/", 1)[0] for path in paths}
    } == {
        "raw_data": 2_861_749_771,
        "processed_data": 4_505_490_383,
        "per_task_out": 3_892_423_987,
        "outputs2": 382_055_007,
        "cheap_decisions_stacked_rc_pred_all.csv": 277_377_057,
        "new_eval_intermediates": 5_896,
    }
