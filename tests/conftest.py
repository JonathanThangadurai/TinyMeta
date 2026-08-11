from pathlib import Path

import pytest

from tinymeta.config import Thresholds
from tinymeta.taxonomy import Taxonomy, TaxonomyConcept


@pytest.fixture
def thresholds() -> Thresholds:
    return Thresholds(auto_accept=0.9, human_review=0.6)


@pytest.fixture
def taxonomy() -> Taxonomy:
    return Taxonomy(
        name="test",
        version="1",
        concepts=[
            TaxonomyConcept(
                id="nitrogen_dioxide",
                path=["environment", "nitrogen_dioxide"],
                description="Nitrogen dioxide measurements",
                aliases=["NO2", "nitrogen dioxide"],
            ),
            TaxonomyConcept(
                id="temperature",
                path=["environment", "temperature"],
                description="Temperature measurements",
                aliases=["temp", "temperature"],
            ),
        ],
    )


@pytest.fixture
def csv_path(tmp_path: Path) -> Path:
    path = tmp_path / "station.csv"
    path.write_text("timestamp,NO2_ug_m3,temp_deg_c\n2025-01-01,10,4\n2025-01-02,,6\n")
    return path
