from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd
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
        all_runs = list(self.api.runs(f"{entity}/{project}", per_page=500))
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

        # Add proper tags from run.tags (this was missing!)
        if run.tags:
            run_data["wandb_tags"] = ",".join(run.tags)

        return run_data

    def download_run_history(self, run: wandb.apis.public.Run) -> list[dict[str, Any]]:
        history_data = []
        for step_data in run.scan_history():
            history_data.append(step_data)
        return history_data

    def download_bulk_history(
        self, entity: str, project: str, run_ids: list[str]
    ) -> pd.DataFrame:
        """
        Download ALL project histories in bulk, then filter to desired runs.
        WARNING: This downloads histories for ALL runs in the project, not just run_ids.
        Only use this for force_refresh where you're downloading most/all runs anyway.
        """
        if not run_ids:
            return pd.DataFrame()

        runs_obj = self.api.runs(f"{entity}/{project}", per_page=500)

        try:
            # This downloads ALL project histories (potentially inefficient)
            history_df = runs_obj.histories(samples=10000, format="pandas")

            if history_df is not None and not history_df.empty:
                return history_df[history_df["Run"].isin(run_ids)]
            else:
                return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

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

        existing_states = self.store.get_existing_run_states(entity, project)
        run_ids_to_download = [run.id for run in runs_to_download]

        for i, run in enumerate(runs_to_download):
            if progress_callback:
                progress_callback(i + 1, len(runs_to_download), run.name)
            if run.id not in existing_states:
                stats["new_runs"] += 1
            else:
                stats["updated_runs"] += 1
            run_data = self.download_run_data(run, entity, project)
            self.store.store_run(run_data)

        if run_ids_to_download:
            # Use bulk download only for force_refresh (downloading all runs)
            # Use individual downloads for incremental updates (much more efficient)
            if force_refresh:
                if progress_callback:
                    progress_callback(
                        len(runs_to_download),
                        len(runs_to_download),
                        f"⚠️  Starting BULK history download for all {len(run_ids_to_download)} runs. This may take 5-30 minutes with no progress updates...",
                    )

                bulk_history_df = self.download_bulk_history(
                    entity, project, run_ids_to_download
                )

                if not bulk_history_df.empty:
                    for run_id in run_ids_to_download:
                        run_history_df = bulk_history_df[
                            bulk_history_df["Run"] == run_id
                        ]
                        if not run_history_df.empty:
                            history_data = run_history_df.drop(columns=["Run"]).to_dict(
                                "records"
                            )
                            self.store.store_history(run_id, history_data)
                else:
                    if progress_callback:
                        progress_callback(
                            len(runs_to_download),
                            len(runs_to_download),
                            "Bulk history failed, falling back to individual downloads...",
                        )
                    for run in runs_to_download:
                        history_data = self.download_run_history(run)
                        if len(history_data) > 0:
                            self.store.store_history(run.id, history_data)
            else:
                # Incremental update: use individual downloads (more efficient for small numbers)
                if progress_callback:
                    progress_callback(
                        len(runs_to_download),
                        len(runs_to_download),
                        f"Starting INDIVIDUAL history downloads for {len(run_ids_to_download)} runs (incremental mode)...",
                    )

                for i, run in enumerate(runs_to_download):
                    if progress_callback:
                        progress_callback(
                            i + 1,
                            len(runs_to_download),
                            f"Downloading history for: {run.name}",
                        )
                    history_data = self.download_run_history(run)
                    if len(history_data) > 0:
                        self.store.store_history(run.id, history_data)
        return stats

    def sync_tags(
        self,
        entity: str,
        project: str,
        mode: str = "recent",
        days: int = 7,
        progress_callback: Optional[callable] = None,
    ) -> dict[str, int]:
        if progress_callback:
            progress_callback(0, 0, f"Starting tag sync - mode: {mode}")

        if mode == "all":
            all_runs = list(self.api.runs(f"{entity}/{project}", per_page=500))
            runs_to_sync = all_runs
        elif mode == "recent":
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            all_runs = list(self.api.runs(f"{entity}/{project}", per_page=500))
            runs_to_sync = []
            for run in all_runs:
                if run.created_at:
                    run_date = (
                        run.created_at.isoformat()
                        if hasattr(run.created_at, "isoformat")
                        else str(run.created_at)
                    )
                    if run_date > cutoff_date:
                        runs_to_sync.append(run)
        elif mode == "finished":
            all_runs = list(self.api.runs(f"{entity}/{project}", per_page=500))
            runs_to_sync = [run for run in all_runs if run.state == "finished"]
        else:
            raise ValueError(
                f"Invalid sync mode: {mode}. Use 'recent', 'all', or 'finished'"
            )

        if progress_callback:
            progress_callback(
                0, len(runs_to_sync), f"Found {len(runs_to_sync)} runs to sync"
            )

        stats = {"total_runs": len(runs_to_sync), "updated_runs": 0}
        if stats["total_runs"] == 0:
            return stats

        # Use bulk tag sync with progress
        updated_count = self._bulk_update_tags(
            entity, project, runs_to_sync, progress_callback
        )
        stats["updated_runs"] = updated_count

        return stats

    def _bulk_update_tags(
        self,
        entity: str,
        project: str,
        runs_list: list[wandb.apis.public.Run],
        progress_callback: Optional[callable] = None,
        batch_size: int = 50,
    ) -> int:
        updated_count = 0
        total_batches = (len(runs_list) + batch_size - 1) // batch_size

        with self.store.engine.connect() as conn:
            for i in range(0, len(runs_list), batch_size):
                batch = runs_list[i : i + batch_size]
                batch_num = (i // batch_size) + 1

                if progress_callback:
                    progress_callback(
                        batch_num,
                        total_batches,
                        f"Updating tags batch {batch_num}/{total_batches} ({len(batch)} runs)",
                    )

                # Collect updates for this batch
                updates = []
                for run in batch:
                    if run.tags:
                        tags_str = ",".join(run.tags)
                        updates.append({"tags": f'"{tags_str}"', "run_id": run.id})

                # Execute batch update
                if updates:
                    from sqlalchemy import text

                    conn.execute(
                        text(
                            "UPDATE wandb_runs SET raw_data = jsonb_set(raw_data, '{wandb_tags}', :tags) WHERE run_id = :run_id"
                        ),
                        updates,
                    )
                    conn.commit()
                    updated_count += len(updates)

        return updated_count

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
