"""Fixtures partagées : jeux de données de test et écriture multiformat."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import sample_factory as sf  # noqa: E402


@pytest.fixture
def mixed_df() -> pd.DataFrame:
    return sf.mixed_dataset(300)


@pytest.fixture
def numeric_df() -> pd.DataFrame:
    return sf.numeric_dataset(200)


@pytest.fixture
def categorical_df() -> pd.DataFrame:
    return sf.categorical_dataset(300)


@pytest.fixture
def temporal_df() -> pd.DataFrame:
    return sf.temporal_dataset(200)


@pytest.fixture
def missing_df() -> pd.DataFrame:
    return sf.missing_dataset(200)


@pytest.fixture
def duplicates_df() -> pd.DataFrame:
    return sf.duplicates_dataset()


@pytest.fixture
def outliers_df() -> pd.DataFrame:
    return sf.outliers_dataset(200)


@pytest.fixture
def unusual_df() -> pd.DataFrame:
    return sf.unusual_names_dataset()


@pytest.fixture
def small_known() -> pd.DataFrame:
    """Jeu aux valeurs statistiques connues."""
    return pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [2.0, 4.0, 6.0, 8.0, 10.0],
        "z": [5.0, 4.0, 3.0, 2.0, 1.0],
        "g": ["a", "a", "b", "b", "b"],
    })
