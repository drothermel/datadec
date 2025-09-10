from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from datadec.wandb_store import WandBStore


@pytest.fixture
def wandb_store(postgresql):
    host = postgresql.info.host
    port = postgresql.info.port
    user = postgresql.info.user
    dbname = postgresql.info.dbname

    connection_string = f"postgresql+psycopg://{user}@{host}:{port}/{dbname}"
    return WandBStore(connection_string)


class TestWandBStore:
    def test_initialization(self, wandb_store):
        assert wandb_store.engine is not None

    def test_store_run_basic(self, wandb_store):
        run_data = {
            "run_id": "test_run_123",
            "run_name": "test_experiment",
            "state": "finished",
            "project": "test_project",
            "entity": "test_entity",
            "created_at": datetime.now(),
            "runtime": 3600,
            "learning_rate": 0.001,
            "batch_size": 32,
            "accuracy": 0.95,
        }
        wandb_store.store_run(run_data)
        runs_df = wandb_store.get_runs()
        assert len(runs_df) == 1
        assert runs_df.iloc[0]["run_id"] == "test_run_123"
        assert runs_df.iloc[0]["learning_rate"] == 0.001
        assert runs_df.iloc[0]["accuracy"] == 0.95

    def test_store_run_with_complex_data(self, wandb_store):
        run_data = {
            "run_id": "complex_run_456",
            "run_name": "complex_experiment",
            "state": "running",
            "project": "test_project",
            "entity": "test_entity",
            "config": {
                "optimizer": "adam",
                "lr_schedule": {"type": "cosine", "warmup": 1000},
            },
            "summary": {
                "eval_metrics": {
                    "accuracy": 0.92,
                    "f1_score": 0.89,
                    "confusion_matrix": [[100, 5], [3, 92]],
                }
            },
        }
        wandb_store.store_run(run_data)
        runs_df = wandb_store.get_runs()
        assert len(runs_df) == 1
        assert runs_df.iloc[0]["run_id"] == "complex_run_456"
        row = runs_df.iloc[0]
        assert "config" in row
        assert "summary" in row

    def test_store_run_update(self, wandb_store):
        run_data = {
            "run_id": "update_run_789",
            "run_name": "update_test",
            "state": "running",
            "project": "test_project",
            "entity": "test_entity",
            "accuracy": 0.80,
        }
        wandb_store.store_run(run_data)
        updated_data = run_data.copy()
        updated_data.update({"state": "finished", "accuracy": 0.95, "final_loss": 0.05})
        wandb_store.store_run(updated_data)
        runs_df = wandb_store.get_runs()
        assert len(runs_df) == 1
        row = runs_df.iloc[0]
        assert row["state"] == "finished"
        assert row["accuracy"] == 0.95
        assert row["final_loss"] == 0.05

    def test_store_history(self, wandb_store):
        run_data = {
            "run_id": "history_run_001",
            "run_name": "history_test",
            "state": "finished",
            "project": "test_project",
            "entity": "test_entity",
        }
        wandb_store.store_run(run_data)
        history_data = [
            {"_step": 0, "_timestamp": datetime.now(), "loss": 1.0, "accuracy": 0.3},
            {"_step": 100, "_timestamp": datetime.now(), "loss": 0.5, "accuracy": 0.7},
            {"_step": 200, "_timestamp": datetime.now(), "loss": 0.2, "accuracy": 0.9},
        ]
        wandb_store.store_history("history_run_001", history_data)
        history_df = wandb_store.get_history()
        assert len(history_df) == 3
        assert history_df["run_id"].iloc[0] == "history_run_001"
        assert history_df["step"].tolist() == [0, 100, 200]
        assert history_df["loss"].tolist() == [1.0, 0.5, 0.2]

    def test_store_history_replace(self, wandb_store):
        run_data = {
            "run_id": "replace_run_002",
            "run_name": "replace_test",
            "state": "running",
            "project": "test_project",
            "entity": "test_entity",
        }
        wandb_store.store_run(run_data)
        history_v1 = [{"_step": 0, "loss": 1.0}, {"_step": 100, "loss": 0.8}]
        wandb_store.store_history("replace_run_002", history_v1)
        history_v2 = [
            {"_step": 0, "loss": 1.0},
            {"_step": 100, "loss": 0.7},
            {"_step": 200, "loss": 0.4},
            {"_step": 300, "loss": 0.2},
        ]
        wandb_store.store_history("replace_run_002", history_v2)
        history_df = wandb_store.get_history(run_ids=["replace_run_002"])
        assert len(history_df) == 4
        assert history_df["step"].max() == 300

    def test_get_runs_filtering(self, wandb_store):
        runs = [
            {
                "run_id": "run_1",
                "run_name": "exp_1",
                "state": "finished",
                "project": "proj_a",
                "entity": "team_1",
            },
            {
                "run_id": "run_2",
                "run_name": "exp_2",
                "state": "running",
                "project": "proj_a",
                "entity": "team_1",
            },
            {
                "run_id": "run_3",
                "run_name": "exp_3",
                "state": "finished",
                "project": "proj_b",
                "entity": "team_2",
            },
        ]
        for run in runs:
            wandb_store.store_run(run)
        all_runs = wandb_store.get_runs()
        assert len(all_runs) == 3
        proj_a_runs = wandb_store.get_runs(project="proj_a")
        assert len(proj_a_runs) == 2
        team_2_runs = wandb_store.get_runs(entity="team_2")
        assert len(team_2_runs) == 1
        finished_runs = wandb_store.get_runs(state="finished")
        assert len(finished_runs) == 2
        proj_a_finished = wandb_store.get_runs(project="proj_a", state="finished")
        assert len(proj_a_finished) == 1
        assert proj_a_finished.iloc[0]["run_id"] == "run_1"

    def test_get_existing_run_states(self, wandb_store):
        runs = [
            {
                "run_id": "state_1",
                "run_name": "exp_1",
                "state": "finished",
                "project": "test_proj",
                "entity": "test_entity",
            },
            {
                "run_id": "state_2",
                "run_name": "exp_2",
                "state": "running",
                "project": "test_proj",
                "entity": "test_entity",
            },
            {
                "run_id": "state_3",
                "run_name": "exp_3",
                "state": "failed",
                "project": "other_proj",
                "entity": "test_entity",
            },
        ]
        for run in runs:
            wandb_store.store_run(run)
        states = wandb_store.get_existing_run_states("test_entity", "test_proj")
        assert len(states) == 2
        assert states["state_1"] == "finished"
        assert states["state_2"] == "running"
        assert "state_3" not in states

    def test_export_to_parquet(self, wandb_store):
        run_data = {
            "run_id": "export_run_123",
            "run_name": "export_test",
            "state": "finished",
            "project": "export_project",
            "entity": "export_entity",
            "accuracy": 0.92,
        }
        wandb_store.store_run(run_data)
        history_data = [
            {"_step": 0, "loss": 1.0, "accuracy": 0.3},
            {"_step": 100, "loss": 0.3, "accuracy": 0.9},
        ]
        wandb_store.store_history("export_run_123", history_data)
        with tempfile.TemporaryDirectory() as temp_dir:
            wandb_store.export_to_parquet(temp_dir)

            output_path = Path(temp_dir)
            metadata_file = output_path / "runs_metadata.parquet"
            history_file = output_path / "runs_history.parquet"

            assert metadata_file.exists()
            assert history_file.exists()

            metadata_df = pd.read_parquet(metadata_file)
            assert len(metadata_df) == 1
            assert metadata_df.iloc[0]["run_id"] == "export_run_123"

            history_df = pd.read_parquet(history_file)
            assert len(history_df) == 2
            assert history_df["run_id"].iloc[0] == "export_run_123"

    def test_get_history_filtering(self, wandb_store):
        runs = [
            {
                "run_id": "hist_run_1",
                "run_name": "hist_1",
                "state": "finished",
                "project": "hist_proj",
                "entity": "hist_entity",
            },
            {
                "run_id": "hist_run_2",
                "run_name": "hist_2",
                "state": "finished",
                "project": "hist_proj",
                "entity": "hist_entity",
            },
        ]
        for run in runs:
            wandb_store.store_run(run)
        for i, run_id in enumerate(["hist_run_1", "hist_run_2"]):
            history = [
                {"_step": 0, "loss": 1.0 + i * 0.1},
                {"_step": 100, "loss": 0.5 + i * 0.1},
            ]
            wandb_store.store_history(run_id, history)
        all_history = wandb_store.get_history()
        assert len(all_history) == 4

        run_1_history = wandb_store.get_history(run_ids=["hist_run_1"])
        assert len(run_1_history) == 2
        assert all(run_1_history["run_id"] == "hist_run_1")

        proj_history = wandb_store.get_history(project="hist_proj")
        assert len(proj_history) == 4
        assert all(proj_history["project"] == "hist_proj")
