from __future__ import annotations

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from datadec.wandb_downloader import WandBDownloader
from datadec.wandb_store import WandBStore


@pytest.fixture
def wandb_store(postgresql):
    host = postgresql.info.host
    port = postgresql.info.port
    user = postgresql.info.user
    dbname = postgresql.info.dbname
    connection_string = f"postgresql+psycopg://{user}@{host}:{port}/{dbname}"
    return WandBStore(connection_string)


@pytest.fixture
def downloader(wandb_store):
    return WandBDownloader(wandb_store)


@pytest.fixture
def mock_wandb_api():
    with patch("datadec.wandb_downloader.wandb") as mock_wandb:
        mock_api = Mock()
        mock_wandb.Api.return_value = mock_api
        yield mock_api


@pytest.fixture
def sample_run():
    run = Mock()
    run.id = "test_run_123"
    run.name = "test_experiment"
    run.state = "finished"
    run.created_at = datetime.now()
    run._attrs = {"runtime": 3600}
    run.config = {"learning_rate": 0.001, "batch_size": 32}
    run.summary = Mock()
    run.summary._json_dict = {"accuracy": 0.95, "loss": 0.05}
    run.scan_history.return_value = [
        {"_step": 0, "_timestamp": datetime.now(), "loss": 1.0, "accuracy": 0.3},
        {"_step": 100, "_timestamp": datetime.now(), "loss": 0.5, "accuracy": 0.7},
        {"_step": 200, "_timestamp": datetime.now(), "loss": 0.05, "accuracy": 0.95},
    ]
    return run


class TestWandBDownloader:
    def test_initialization(self, wandb_store):
        downloader = WandBDownloader(wandb_store)
        assert downloader.store == wandb_store
        assert downloader._api is None

    def test_api_property_lazy_loading(self, downloader, mock_wandb_api):
        api = downloader.api
        assert api is not None
        api2 = downloader.api
        assert api is api2

    def test_api_authentication_error(self, downloader):
        with patch("datadec.wandb_downloader.wandb") as mock_wandb:
            import wandb as real_wandb

            mock_wandb.Api.side_effect = real_wandb.errors.UsageError(
                "api_key not configured"
            )
            mock_wandb.errors = real_wandb.errors
            with pytest.raises(RuntimeError, match="WandB API key not configured"):
                _ = downloader.api

    def test_download_run_data(self, downloader, sample_run):
        run_data = downloader.download_run_data(
            sample_run, "test_entity", "test_project"
        )
        assert run_data["run_id"] == "test_run_123"
        assert run_data["run_name"] == "test_experiment"
        assert run_data["state"] == "finished"
        assert run_data["entity"] == "test_entity"
        assert run_data["project"] == "test_project"
        assert run_data["runtime"] == 3600
        assert run_data["learning_rate"] == 0.001
        assert run_data["batch_size"] == 32
        assert run_data["accuracy"] == 0.95
        assert run_data["loss"] == 0.05

    def test_download_run_history(self, downloader, sample_run):
        history = downloader.download_run_history(sample_run)
        assert len(history) == 3
        assert history[0]["_step"] == 0
        assert history[0]["loss"] == 1.0
        assert history[1]["_step"] == 100
        assert history[2]["_step"] == 200
        assert history[2]["accuracy"] == 0.95

    def test_get_runs_to_download_force_refresh(
        self, downloader, mock_wandb_api, sample_run
    ):
        mock_wandb_api.runs.return_value = [sample_run]
        runs = downloader.get_runs_to_download(
            "test_entity", "test_project", force_refresh=True
        )
        assert len(runs) == 1
        assert runs[0] == sample_run
        mock_wandb_api.runs.assert_called_once_with("test_entity/test_project")

    def test_get_runs_to_download_incremental(
        self, downloader, wandb_store, mock_wandb_api
    ):
        existing_run_data = {
            "run_id": "existing_run_456",
            "run_name": "existing_exp",
            "state": "finished",
            "project": "test_project",
            "entity": "test_entity",
        }
        wandb_store.store_run(existing_run_data)
        unfinished_run_data = {
            "run_id": "unfinished_run_789",
            "run_name": "unfinished_exp",
            "state": "running",
            "project": "test_project",
            "entity": "test_entity",
        }
        wandb_store.store_run(unfinished_run_data)
        existing_run = Mock()
        existing_run.id = "existing_run_456"
        unfinished_run = Mock()
        unfinished_run.id = "unfinished_run_789"
        new_run = Mock()
        new_run.id = "new_run_001"
        mock_wandb_api.runs.return_value = [existing_run, unfinished_run, new_run]
        runs = downloader.get_runs_to_download(
            "test_entity", "test_project", force_refresh=False
        )
        run_ids = [r.id for r in runs]
        assert "new_run_001" in run_ids
        assert "unfinished_run_789" in run_ids
        assert "existing_run_456" not in run_ids

    def test_download_project_no_runs(self, downloader, mock_wandb_api):
        mock_wandb_api.runs.return_value = []
        stats = downloader.download_project("test_entity", "test_project")
        assert stats["total_runs"] == 0
        assert stats["new_runs"] == 0
        assert stats["updated_runs"] == 0

    def test_download_project_with_progress(
        self, downloader, wandb_store, mock_wandb_api, sample_run
    ):
        mock_wandb_api.runs.return_value = [sample_run]
        progress_calls = []

        def progress_callback(run_index, total_runs, run_name):
            progress_calls.append((run_index, total_runs, run_name))

        stats = downloader.download_project(
            "test_entity", "test_project", progress_callback=progress_callback
        )
        assert stats["total_runs"] == 1
        assert stats["new_runs"] == 1
        assert stats["updated_runs"] == 0
        assert len(progress_calls) == 1
        assert progress_calls[0] == (1, 1, "test_experiment")
        runs_df = wandb_store.get_runs(entity="test_entity", project="test_project")
        assert len(runs_df) == 1
        assert runs_df.iloc[0]["run_id"] == "test_run_123"
        history_df = wandb_store.get_history(project="test_project")
        assert len(history_df) == 3

    def test_download_project_update_existing(
        self, downloader, wandb_store, mock_wandb_api
    ):
        existing_data = {
            "run_id": "update_run_123",
            "run_name": "update_test",
            "state": "running",
            "project": "test_project",
            "entity": "test_entity",
            "accuracy": 0.8,
        }
        wandb_store.store_run(existing_data)
        updated_run = Mock()
        updated_run.id = "update_run_123"
        updated_run.name = "update_test"
        updated_run.state = "finished"
        updated_run.created_at = datetime.now()
        updated_run._attrs = {"runtime": 7200}
        updated_run.config = {"learning_rate": 0.001}
        updated_run.summary = Mock()
        updated_run.summary._json_dict = {
            "accuracy": 0.95,
            "final_loss": 0.02,
        }
        updated_run.scan_history.return_value = [
            {"_step": 300, "loss": 0.02, "accuracy": 0.95}
        ]
        mock_wandb_api.runs.return_value = [updated_run]
        stats = downloader.download_project("test_entity", "test_project")
        assert stats["total_runs"] == 1
        assert stats["new_runs"] == 0
        assert stats["updated_runs"] == 1
        runs_df = wandb_store.get_runs(entity="test_entity", project="test_project")
        assert len(runs_df) == 1
        row = runs_df.iloc[0]
        assert row["state"] == "finished"
        assert row["accuracy"] == 0.95
        assert row["final_loss"] == 0.02

    def test_get_download_summary_empty(self, downloader, wandb_store):
        summary = downloader.get_download_summary("test_entity", "test_project")
        assert summary["total_runs"] == 0
        assert summary["finished_runs"] == 0
        assert summary["running_runs"] == 0
        assert summary["failed_runs"] == 0
        assert summary["total_history_records"] == 0

    def test_get_download_summary_with_data(self, downloader, wandb_store):
        runs_data = [
            {
                "run_id": "run1",
                "run_name": "exp1",
                "state": "finished",
                "project": "test_project",
                "entity": "test_entity",
            },
            {
                "run_id": "run2",
                "run_name": "exp2",
                "state": "finished",
                "project": "test_project",
                "entity": "test_entity",
            },
            {
                "run_id": "run3",
                "run_name": "exp3",
                "state": "running",
                "project": "test_project",
                "entity": "test_entity",
            },
            {
                "run_id": "run4",
                "run_name": "exp4",
                "state": "failed",
                "project": "test_project",
                "entity": "test_entity",
            },
        ]
        for run_data in runs_data:
            wandb_store.store_run(run_data)
        wandb_store.store_history("run1", [{"_step": 0, "loss": 1.0}])
        wandb_store.store_history(
            "run2",
            [{"_step": 0, "loss": 1.0}, {"_step": 100, "loss": 0.5}],
        )
        summary = downloader.get_download_summary("test_entity", "test_project")
        assert summary["total_runs"] == 4
        assert summary["finished_runs"] == 2
        assert summary["running_runs"] == 1
        assert summary["failed_runs"] == 1
        assert summary["total_history_records"] == 3
