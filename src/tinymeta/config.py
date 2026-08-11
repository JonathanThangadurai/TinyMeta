"""Runtime configuration."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Thresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")
    auto_accept: float = Field(ge=0, le=1)
    human_review: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def ordered(self) -> "Thresholds":
        if self.human_review >= self.auto_accept:
            raise ValueError("human_review must be lower than auto_accept")
        return self


def load_thresholds(path: Path) -> Thresholds:
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return Thresholds.model_validate(raw["thresholds"])
