from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, PrivateAttr, computed_field

from datadec.data import model_utils
from datadec.data.ingest.enums import MMLU_SUBJECT_TASKS, Task
from datadec.data.ingest.metrics import (
    PerplexityMetrics,
    TaskEvalMetrics,
    average_task_metrics,
)

if TYPE_CHECKING:
    from datadec.data.ingest.run import TrainingRun


class EvalCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int
    perplexity: PerplexityMetrics | None = None
    task_evals: dict[Task, TaskEvalMetrics] = {}

    _run: Any = PrivateAttr(default=None)

    def _require_run(self) -> "TrainingRun":
        if self._run is None:
            raise RuntimeError(
                "EvalCheckpoint is not attached to a TrainingRun; "
                "computed fields are unavailable"
            )
        return self._run

    @computed_field
    @property
    def tokens(self) -> int:
        return self.step * self._require_run().model_details.tokens_per_step

    @computed_field
    @property
    def compute(self) -> float:
        return self.step * self._require_run().model_details.compute_per_step

    @computed_field
    @property
    def lr_at_step(self) -> float:
        details = self._require_run().model_details
        return model_utils.get_lr_at_step(
            step=self.step,
            lr_warmup_start=details.lr_warmup_start,
            lr_max=details.lr_max,
            lr_final=details.lr_final,
            warmup_steps=details.warmup_steps,
            lr_decay_steps=details.lr_decay_steps,
        )

    @computed_field
    @property
    def cumulative_lr(self) -> float:
        details = self._require_run().model_details
        return model_utils.calculate_cumulative_lr(
            step=self.step,
            lr_warmup_start=details.lr_warmup_start,
            lr_max=details.lr_max,
            lr_final=details.lr_final,
            warmup_steps=details.warmup_steps,
            lr_decay_steps=details.lr_decay_steps,
        )

    @computed_field
    @property
    def mmlu_average(self) -> TaskEvalMetrics | None:
        present = [
            self.task_evals[task]
            for task in MMLU_SUBJECT_TASKS
            if task in self.task_evals
        ]
        return average_task_metrics(present)

    @property
    def has_perplexity(self) -> bool:
        return self.perplexity is not None

    @property
    def has_task_evals(self) -> bool:
        return len(self.task_evals) > 0
