"""Controlled taxonomy loading and validation."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class TaxonomyConcept(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    path: list[str] = Field(min_length=2)
    description: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def path_ends_in_id(self) -> "TaxonomyConcept":
        if self.path[-1] != self.id:
            raise ValueError("taxonomy path must end with concept id")
        return self


class Taxonomy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    concepts: list[TaxonomyConcept] = Field(min_length=1)

    @model_validator(mode="after")
    def ids_are_unique(self) -> "Taxonomy":
        ids = [concept.id for concept in self.concepts]
        if len(ids) != len(set(ids)):
            raise ValueError("taxonomy concept ids must be unique")
        return self

    def by_id(self) -> dict[str, TaxonomyConcept]:
        return {concept.id: concept for concept in self.concepts}


def load_taxonomy(path: Path) -> Taxonomy:
    with path.open(encoding="utf-8") as handle:
        return Taxonomy.model_validate(yaml.safe_load(handle))
