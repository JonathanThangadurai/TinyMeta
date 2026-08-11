from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from tinymeta.config import Thresholds
from tinymeta.profiling import profile_csv
from tinymeta.tagging import EmbeddingTagger, MetadataTagger, RuleBasedTagger, review_status
from tinymeta.taxonomy import Taxonomy


class FakeEncoder:
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        assert len(texts) == 3
        return np.array([[1.0, 0.0], [0.95, 0.1], [0.5, 0.5]])


def test_rule_tagger_matches_alias_and_provenance(
    csv_path: Path, taxonomy: Taxonomy, thresholds: Thresholds
) -> None:
    tagger: MetadataTagger = RuleBasedTagger(thresholds)
    result = tagger.tag(profile_csv(csv_path), taxonomy)
    assert [tag.tag for tag in result.tags] == ["nitrogen_dioxide", "temperature"]
    assert result.tags[0].evidence == ["column:NO2_ug_m3"]
    assert result.tags[0].taxonomy_path[-1] == "nitrogen_dioxide"
    assert result.tags[0].status == "auto_accept"


def test_rules_do_not_match_alias_substrings(
    tmp_path: Path, taxonomy: Taxonomy, thresholds: Thresholds
) -> None:
    path = tmp_path / "annotation.csv"
    path.write_text("annotation\ntext\n")
    assert RuleBasedTagger(thresholds).tag(profile_csv(path), taxonomy).tags == []


def test_embedding_tagger_with_fake_encoder(
    csv_path: Path, taxonomy: Taxonomy, thresholds: Thresholds
) -> None:
    result = EmbeddingTagger(FakeEncoder(), thresholds, model_name="fake", top_k=2).tag(
        profile_csv(csv_path), taxonomy
    )
    assert [tag.tag for tag in result.tags] == ["nitrogen_dioxide", "temperature"]
    assert result.tags[0].method == "embedding_similarity"
    assert result.model_name == "fake"


def test_review_boundaries(thresholds: Thresholds) -> None:
    assert review_status(0.9, thresholds) == "auto_accept"
    assert review_status(0.6, thresholds) == "human_review"
    assert review_status(0.59, thresholds) == "unknown"


def test_embedding_rejects_bad_encoder(
    csv_path: Path, taxonomy: Taxonomy, thresholds: Thresholds
) -> None:
    class BadEncoder:
        def encode(self, texts: Sequence[str]) -> np.ndarray:
            return np.array([[1.0]])

    with pytest.raises(ValueError, match="shape"):
        EmbeddingTagger(BadEncoder(), thresholds, model_name="bad").tag(
            profile_csv(csv_path), taxonomy
        )
