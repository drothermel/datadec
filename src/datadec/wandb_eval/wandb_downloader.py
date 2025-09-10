from __future__ import annotations

from typing import Any, Optional

import wandb

from .wandb_store import WandBStore


class WandBDownloader:
    def __init__(self, store: WandBStore):
        self.store = store
        self._api: Optional[wandb.Api] = None

    @property
    def api(self) -> wandb.Api:
        if self._api is None:
            try:
                self._api = wandb.Api()
            except wandb.errors.UsageError as e:
                if "api_key not configured" in str(e):
                    raise RuntimeError(
                        "WandB API key not configured. "
                        "Please run 'wandb login' or set WANDB_API_KEY environment variable"
                    ) from e
                raise
        return self._api

    def get_runs_to_download(
        self, entity: str, project: str, force_refresh: bool = False
    ) -> list[wandb.apis.public.Run]:
        all_runs = list(self.api.runs(f"{entity}/{project}"))
        if force_refresh:
            return all_runs
        existing_run_states = self.store.get_existing_run_states(entity, project)
        new_runs = [run for run in all_runs if run.id not in existing_run_states]
        unfinished_runs = [
            run
            for run in all_runs
            if run.id in existing_run_states
            and existing_run_states[run.id] != "finished"
        ]
        return new_runs + unfinished_runs

    def download_run_data(
        self, run: wandb.apis.public.Run, entity: str, project: str
    ) -> dict[str, Any]:
        run_data = {
            "run_id": run.id,
            "run_name": run.name,
            "state": run.state,
            "project": project,
            "entity": entity,
            "created_at": run.created_at,
            "runtime": run._attrs.get("runtime", 0),
        }
        run_data.update(run.config)
        run_data.update(run.summary._json_dict)
        return run_data

    def download_run_history(self, run: wandb.apis.public.Run) -> list[dict[str, Any]]:
        history_data = []
        for step_data in run.scan_history():
            history_data.append(step_data)
        return history_data

    def download_project(
        self,
        entity: str,
        project: str,
        force_refresh: bool = False,
        progress_callback: Optional[callable] = None,
    ) -> dict[str, int]:
        runs_to_download = self.get_runs_to_download(entity, project, force_refresh)
        stats = {"total_runs": len(runs_to_download), "new_runs": 0, "updated_runs": 0}
        if stats["total_runs"] == 0:
            return stats

        for i, run in enumerate(runs_to_download):
            if progress_callback:
                progress_callback(i + 1, len(runs_to_download), run.name)
            existing_states = self.store.get_existing_run_states(entity, project)
            if run.id not in existing_states:
                stats["new_runs"] += 1
            else:
                stats["updated_runs"] += 1
            run_data = self.download_run_data(run, entity, project)
            self.store.store_run(run_data)
            history_data = self.download_run_history(run)
            if len(history_data) > 0:
                self.store.store_history(run.id, history_data)
        return stats

    def get_download_summary(self, entity: str, project: str) -> dict[str, Any]:
        runs_df = self.store.get_runs(entity=entity, project=project)
        history_df = self.store.get_history(project=project)
        summary = {
            "total_runs": 0,
            "finished_runs": 0,
            "running_runs": 0,
            "failed_runs": 0,
            "total_history_records": 0,
        }
        if runs_df.empty:
            return summary

        state_counts = runs_df["state"].value_counts().to_dict()
        summary["total_runs"] = len(runs_df)
        summary["finished_runs"] = state_counts.get("finished", 0)
        summary["running_runs"] = state_counts.get("running", 0)
        summary["failed_runs"] = state_counts.get("failed", 0)
        summary["total_history_records"] = len(history_df)
        summary["latest_update"] = (
            runs_df["created_at"].max() if "created_at" in runs_df.columns else None,
        )
        return summary
