from pathlib import Path

import pytest
from pydantic import ValidationError

from tinymeta.config import Thresholds, load_thresholds
from tinymeta.taxonomy import Taxonomy, load_taxonomy


def test_load_real_taxonomy() -> None:
    taxonomy = load_taxonomy(Path("taxonomy/environmental.yaml"))
    assert taxonomy.by_id()["nitrogen_dioxide"].aliases[0] == "NO2"


def test_duplicate_taxonomy_ids_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "name: x\nversion: '1'\nconcepts:\n"
        "  - {id: x, path: [root, x], description: x}\n"
        "  - {id: x, path: [root, x], description: y}\n"
    )
    with pytest.raises(ValidationError, match="unique"):
        load_taxonomy(path)


def test_path_must_end_in_id(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "name: x\nversion: '1'\nconcepts:\n  - {id: x, path: [root, y], description: x}\n"
    )
    with pytest.raises(ValidationError, match="end with concept id"):
        load_taxonomy(path)


def test_threshold_order_and_loading() -> None:
    assert load_thresholds(Path("config/default.yaml")).auto_accept == 0.9
    with pytest.raises(ValidationError, match="lower"):
        Thresholds(auto_accept=0.5, human_review=0.6)


def test_taxonomy_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Taxonomy.model_validate({"name": "x", "version": "1", "concepts": [], "surprise": 1})
