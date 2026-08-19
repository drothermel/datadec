from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from datadec.data.ingest.checkpoint import EvalCheckpoint
from datadec.data.ingest.enums import DataRecipeName, ModelSizeName, Seed
from datadec.data.ingest.registries.model_details import ModelDetails


class TrainingRun(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    params: ModelSizeName
    data: DataRecipeName
    seed: Seed
    model_details: ModelDetails
    checkpoints: list[EvalCheckpoint]

    @model_validator(mode="after")
    def _wire_back_references_and_validate_steps(self) -> Self:
        self.checkpoints.sort(key=lambda ckpt: ckpt.step)
        seen_steps: set[int] = set()
        for ckpt in self.checkpoints:
            if ckpt.step in seen_steps:
                raise ValueError(
                    f"duplicate step {ckpt.step} in TrainingRun "
                    f"({self.params}, {self.data}, {self.seed})"
                )
            if not (ckpt.has_perplexity or ckpt.has_task_evals):
                raise ValueError(
                    f"checkpoint at step {ckpt.step} has no perplexity or task evals "
                    f"in TrainingRun ({self.params}, {self.data}, {self.seed})"
                )
            seen_steps.add(ckpt.step)
            ckpt._run = self
        return self

    @property
    def steps(self) -> list[int]:
        return [ckpt.step for ckpt in self.checkpoints]

    @property
    def final_checkpoint(self) -> EvalCheckpoint:
        if len(self.checkpoints) == 0:
            raise ValueError(
                f"TrainingRun ({self.params}, {self.data}, {self.seed}) "
                f"has no checkpoints"
            )
        return self.checkpoints[-1]

    def checkpoint_at_step(self, step: int) -> EvalCheckpoint:
        for ckpt in self.checkpoints:
            if ckpt.step == step:
                return ckpt
        raise KeyError(
            f"no checkpoint at step {step} in TrainingRun "
            f"({self.params}, {self.data}, {self.seed})"
        )
