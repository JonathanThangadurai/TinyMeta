from pathlib import Path

import pytest

from tinymeta.config import load_thresholds
from tinymeta.profiling import profile_csv
from tinymeta.tagging import EmbeddingTagger, SentenceTransformerEncoder
from tinymeta.taxonomy import load_taxonomy


@pytest.mark.integration
def test_real_embedding_model_smoke() -> None:
    model = "sentence-transformers/all-MiniLM-L6-v2"
    tagger = EmbeddingTagger(
        SentenceTransformerEncoder(model),
        load_thresholds(Path("config/default.yaml")),
        model_name=model,
    )
    result = tagger.tag(
        profile_csv(Path("data/synthetic/amsterdam_air_quality.csv")),
        load_taxonomy(Path("taxonomy/environmental.yaml")),
    )
    assert result.tags
