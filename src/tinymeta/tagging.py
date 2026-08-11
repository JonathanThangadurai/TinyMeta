"""Common tagger interface plus rules and embedding-similarity implementations."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from tinymeta.config import Thresholds
from tinymeta.models import DatasetProfile, MetadataTag, ReviewStatus, TaggingResult
from tinymeta.taxonomy import Taxonomy, TaxonomyConcept


class MetadataTagger(ABC):
    @abstractmethod
    def tag(self, profile: DatasetProfile, taxonomy: Taxonomy) -> TaggingResult: ...


def review_status(confidence: float, thresholds: Thresholds) -> ReviewStatus:
    if confidence >= thresholds.auto_accept:
        return ReviewStatus.AUTO_ACCEPT
    if confidence >= thresholds.human_review:
        return ReviewStatus.HUMAN_REVIEW
    return ReviewStatus.UNKNOWN


def _tokens(text: str) -> set[str]:
    normalized = re.sub(r"(?<=\d)[._](?=\d)", "", text.lower())
    return set(re.findall(r"[a-z0-9]+", normalized))


class RuleBasedTagger(MetadataTagger):
    def __init__(self, thresholds: Thresholds) -> None:
        self.thresholds = thresholds

    def tag(self, profile: DatasetProfile, taxonomy: Taxonomy) -> TaggingResult:
        sources = [("filename", profile.filename)] + [
            (f"column:{column.name}", column.name) for column in profile.columns
        ]
        tags: list[MetadataTag] = []
        for concept in taxonomy.concepts:
            needles = [concept.id, *concept.aliases]
            evidence = sorted(
                label
                for label, value in sources
                if any(_tokens(alias) <= _tokens(value) for alias in needles if _tokens(alias))
            )
            if evidence:
                confidence = 0.98 if any(item.startswith("column:") for item in evidence) else 0.9
                tags.append(
                    MetadataTag(
                        tag=concept.id,
                        confidence=confidence,
                        method="rules",
                        evidence=evidence,
                        taxonomy_path=concept.path,
                        status=review_status(confidence, self.thresholds),
                    )
                )
        tags.sort(key=lambda item: (-item.confidence, item.tag))
        return TaggingResult(
            dataset=profile.filename,
            tags=tags,
            taxonomy_version=taxonomy.version,
            tagger="rules",
        )


class Encoder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEncoder:
    """Lazy optional adapter; construction is the only point that may download a model."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("install TinyMeta with the 'embeddings' extra") from exc
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self._model.encode(list(texts), normalize_embeddings=True))


def profile_text(profile: DatasetProfile) -> str:
    columns = "; ".join(
        f"{column.name} ({column.dtype}, unit={column.inferred_unit or 'unknown'})"
        for column in profile.columns
    )
    return f"Scientific dataset file: {profile.filename}. Columns: {columns}."


def _concept_text(concept: TaxonomyConcept) -> str:
    aliases = ", ".join(concept.aliases)
    return f"Taxonomy label {concept.id}. {concept.description}. Aliases: {aliases}."


class EmbeddingTagger(MetadataTagger):
    def __init__(
        self,
        encoder: Encoder,
        thresholds: Thresholds,
        *,
        model_name: str,
        top_k: int = 5,
    ) -> None:
        self.encoder = encoder
        self.thresholds = thresholds
        self.model_name = model_name
        self.top_k = top_k

    def tag(self, profile: DatasetProfile, taxonomy: Taxonomy) -> TaggingResult:
        vectors = np.asarray(
            self.encoder.encode([profile_text(profile), *map(_concept_text, taxonomy.concepts)]),
            dtype=float,
        )
        if vectors.ndim != 2 or len(vectors) != len(taxonomy.concepts) + 1:
            raise ValueError("encoder returned an invalid shape")
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms == 0):
            raise ValueError("encoder returned a zero vector")
        normalized = vectors / norms[:, None]
        scores = normalized[1:] @ normalized[0]
        selected = sorted(
            enumerate(scores), key=lambda pair: (-pair[1], taxonomy.concepts[pair[0]].id)
        )[: self.top_k]
        tags = []
        for index, raw_score in selected:
            confidence = min(1.0, max(0.0, float(raw_score)))
            if confidence < self.thresholds.human_review:
                continue
            concept = taxonomy.concepts[index]
            tags.append(
                MetadataTag(
                    tag=concept.id,
                    confidence=round(confidence, 6),
                    method="embedding_similarity",
                    evidence=["profile_text:filename+schema+units"],
                    taxonomy_path=concept.path,
                    status=review_status(confidence, self.thresholds),
                )
            )
        return TaggingResult(
            dataset=profile.filename,
            tags=tags,
            taxonomy_version=taxonomy.version,
            tagger="embedding_similarity",
            model_name=self.model_name,
        )
