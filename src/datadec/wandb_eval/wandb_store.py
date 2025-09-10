from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from sqlalchemy import create_engine, select, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from datadec.wandb_eval import wandb_constants as wconsts


class Base(DeclarativeBase):
    pass


class WandBRun(Base):
    __tablename__ = "wandb_runs"

    run_id: Mapped[str] = mapped_column(primary_key=True)
    run_name: Mapped[str]
    state: Mapped[str]
    project: Mapped[str]
    entity: Mapped[str]
    created_at: Mapped[Optional[datetime]]
    runtime: Mapped[Optional[int]]
    raw_data: Mapped[dict[str, Any]] = mapped_column(JSONB)


class WandBHistory(Base):
    __tablename__ = "wandb_history"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str]
    step: Mapped[Optional[int]]
    timestamp: Mapped[Optional[datetime]]
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB)


class WandBStore:
    def __init__(self, connection_string: str):
        self.engine: Engine = create_engine(connection_string)
        self.create_tables()

    def create_tables(self) -> None:
        Base.metadata.create_all(self.engine)

    def store_run(self, run_data: dict[str, Any]) -> None:
        with Session(self.engine) as session:
            run_id = run_data["run_id"]
            core_data = {
                k: run_data.get(k) for k in wconsts.CORE_RUN_FIELDS if k in run_data
            }
            raw_data = {
                k: v for k, v in run_data.items() if k not in wconsts.CORE_RUN_FIELDS
            }
            core_data["raw_data"] = raw_data
            existing_run = session.get(WandBRun, run_id)
            if existing_run:
                for key, value in core_data.items():
                    setattr(existing_run, key, value)
            else:
                new_run = WandBRun(**core_data)
                session.add(new_run)
            session.commit()

    def store_history(self, run_id: str, history_data: list[dict[str, Any]]) -> None:
        with Session(self.engine) as session:
            session.execute(
                text("DELETE FROM wandb_history WHERE run_id = :run_id"),
                {"run_id": run_id},
            )
            for step_data in history_data:
                step = step_data.get("_step")
                timestamp_raw = step_data.get("_timestamp")
                timestamp = (
                    datetime.fromtimestamp(timestamp_raw) if timestamp_raw else None
                )
                metrics = {
                    k: v
                    for k, v in step_data.items()
                    if not k.startswith("_")
                    and k not in wconsts.WANDB_REDUNDANT_HISTORY_FIELDS
                }
                history_record = WandBHistory(
                    run_id=run_id, step=step, timestamp=timestamp, metrics=metrics
                )
                session.add(history_record)
            session.commit()

    def get_runs(
        self,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        state: Optional[str] = None,
    ) -> pd.DataFrame:
        with Session(self.engine) as session:
            query = select(WandBRun)
            if project:
                query = query.where(WandBRun.project == project)
            if entity:
                query = query.where(WandBRun.entity == entity)
            if state:
                query = query.where(WandBRun.state == state)
            result = session.execute(query)
            runs = result.scalars().all()
            data = []
            for run in runs:
                row = {field: getattr(run, field) for field in wconsts.CORE_RUN_FIELDS}
                if run.raw_data:
                    row.update(run.raw_data)
                data.append(row)
            return pd.DataFrame(data)

    def get_history(
        self, run_ids: Optional[list[str]] = None, project: Optional[str] = None
    ) -> pd.DataFrame:
        with Session(self.engine) as session:
            query = select(WandBHistory, WandBRun.run_name, WandBRun.project).join(
                WandBRun, WandBHistory.run_id == WandBRun.run_id
            )
            if run_ids:
                query = query.where(WandBHistory.run_id.in_(run_ids))
            if project:
                query = query.where(WandBRun.project == project)
            result = session.execute(query)
            rows = result.all()
            data = []
            for history, run_name, project_name in rows:
                row = {
                    "run_id": history.run_id,
                    "run_name": run_name,
                    "project": project_name,
                    "step": history.step,
                    "timestamp": history.timestamp,
                }
                if history.metrics:
                    row.update(history.metrics)
                data.append(row)
            return pd.DataFrame(data)

    def get_existing_run_states(self, entity: str, project: str) -> dict[str, str]:
        with Session(self.engine) as session:
            query = select(WandBRun.run_id, WandBRun.state).where(
                WandBRun.entity == entity, WandBRun.project == project
            )
            result = session.execute(query)
            return {run_id: state for run_id, state in result.all()}

    def export_to_parquet(
        self,
        output_dir: str,
        runs_filename: str = wconsts.DEFAULT_RUNS_FILENAME,
        history_filename: str = wconsts.DEFAULT_HISTORY_FILENAME,
    ) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        runs_df = self.get_runs()
        if not runs_df.empty:
            runs_df.to_parquet(output_path / runs_filename, index=False)
        history_df = self.get_history()
        if not history_df.empty:
            history_df.to_parquet(output_path / history_filename, index=False)
