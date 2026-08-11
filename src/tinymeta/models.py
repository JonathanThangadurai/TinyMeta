"""Validated domain models shared across the profiling and tagging pipeline."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReviewStatus(StrEnum):
    AUTO_ACCEPT = "auto_accept"
    HUMAN_REVIEW = "human_review"
    UNKNOWN = "unknown"


class ColumnProfile(StrictModel):
    name: str = Field(min_length=1)
    dtype: str = Field(min_length=1)
    missing_percentage: float = Field(ge=0, le=100)
    cardinality: int = Field(ge=0)
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    standard_deviation: float | None = None
    samples: list[Any] = Field(default_factory=list)
    is_potential_timestamp: bool = False
    is_potential_geographic: bool = False
    inferred_unit: str | None = None


class DatasetProfile(StrictModel):
    filename: str = Field(min_length=1)
    file_size_bytes: int = Field(ge=0)
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    columns: list[ColumnProfile]

    @model_validator(mode="after")
    def column_count_matches(self) -> DatasetProfile:
        if self.column_count != len(self.columns):
            raise ValueError("column_count must match columns")
        return self


class MetadataTag(StrictModel):
    tag: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    method: str = Field(min_length=1)
    evidence: list[str] = Field(default_factory=list)
    taxonomy_path: list[str] = Field(min_length=1)
    status: ReviewStatus


class TaggingResult(StrictModel):
    dataset: str = Field(min_length=1)
    tags: list[MetadataTag]
    taxonomy_version: str = Field(min_length=1)
    tagger: str = Field(min_length=1)
    model_name: str | None = None
