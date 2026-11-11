"""Profil général du dataset : dimensions, mémoire, manquants, doublons."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from core.schema_detector import ColumnSchema, Family


@dataclass
class DatasetOverview:
    """Indicateurs généraux du jeu de données."""

    n_rows: int
    n_cols: int
    n_cells: int
    n_missing: int
    missing_ratio: float
    n_duplicates: int
    duplicate_ratio: float
    memory_mb: float
    n_numeric: int
    n_categorical: int
    n_temporal: int
    n_other: int
    n_empty_columns: int
    n_constant_columns: int
    n_complete_rows: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def count_duplicates(df: pd.DataFrame) -> int:
    """Nombre de lignes strictement identiques à une ligne précédente."""
    if df.empty:
        return 0
    try:
        return int(df.duplicated().sum())
    except TypeError:  # cellules non hachables résiduelles
        return int(df.astype(str).duplicated().sum())


def compute_overview(df: pd.DataFrame, schemas: list[ColumnSchema]) -> DatasetOverview:
    """Calcule la vue d'ensemble du dataset."""
    n_rows, n_cols = df.shape
    n_cells = n_rows * n_cols
    n_missing = int(df.isna().sum().sum())
    dup = count_duplicates(df)
    fam = [s.family for s in schemas]
    from core.schema_detector import StatType

    return DatasetOverview(
        n_rows=n_rows,
        n_cols=n_cols,
        n_cells=n_cells,
        n_missing=n_missing,
        missing_ratio=n_missing / n_cells if n_cells else 0.0,
        n_duplicates=dup,
        duplicate_ratio=dup / n_rows if n_rows else 0.0,
        memory_mb=float(df.memory_usage(deep=True).sum()) / 1024**2,
        n_numeric=fam.count(Family.NUMERIC),
        n_categorical=fam.count(Family.CATEGORICAL),
        n_temporal=fam.count(Family.TEMPORAL),
        n_other=fam.count(Family.OTHER),
        n_empty_columns=sum(1 for s in schemas if s.stat_type == StatType.EMPTY),
        n_constant_columns=sum(1 for s in schemas if s.stat_type == StatType.CONSTANT),
        n_complete_rows=int(df.notna().all(axis=1).sum()) if n_cols else 0,
    )


def missing_by_column(df: pd.DataFrame) -> pd.DataFrame:
    """Tableau des valeurs manquantes par colonne, trié par taux décroissant."""
    n = len(df)
    miss = df.isna().sum()
    out = pd.DataFrame({
        "Variable": miss.index.astype(str),
        "Valeurs manquantes": miss.values.astype(int),
        "% manquants": (100 * miss.values / n).round(2) if n else 0.0,
        "Valeurs renseignées": (n - miss.values).astype(int),
    })
    return out.sort_values("% manquants", ascending=False, kind="stable").reset_index(drop=True)


def missing_patterns(df: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Combinaisons de colonnes manquantes les plus fréquentes."""
    if df.empty:
        return pd.DataFrame(columns=["Colonnes manquantes", "Lignes", "%"])
    mask = df.isna()
    labels = mask.apply(lambda r: ", ".join(df.columns[r.values]) or "(aucune)", axis=1)
    counts = labels.value_counts().head(top)
    return pd.DataFrame({
        "Colonnes manquantes": counts.index,
        "Lignes": counts.values,
        "%": (100 * counts.values / len(df)).round(2),
    })


def dtypes_table(df: pd.DataFrame) -> pd.DataFrame:
    """Équivalent lisible de df.info()."""
    return pd.DataFrame({
        "Variable": df.columns.astype(str),
        "Type pandas": [str(t) for t in df.dtypes],
        "Non nuls": df.notna().sum().values,
        "Mémoire (Ko)": (df.memory_usage(deep=True, index=False).values / 1024).round(1),
    })
