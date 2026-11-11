"""Orchestration de l'analyse complète d'un dataset.

Point d'entrée unique utilisé par l'interface, le rapport et les tests :
détection du schéma → vue typée → profil → statistiques → corrélations →
qualité → interprétations.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from config.settings import QualityWeights
from core.data_loader import FileMeta
from core.interpretation_engine import InterpretationResult, generate_insights
from core.multivariate_engine import (
    CorrelationResult,
    correlation_matrix,
    covariance_matrix,
    recommend_method,
    strong_associations,
)
from core.profiling_engine import DatasetOverview, compute_overview, missing_by_column
from core.quality_engine import QualityReport, ValidityRule, run_quality_audit
from core.schema_detector import (
    ColumnSchema,
    apply_schema,
    categorical_columns,
    detect_schema,
    numeric_columns,
    temporal_columns,
)
from core.statistics_engine import UnivariateResults, compute_univariate


@dataclass
class AnalysisBundle:
    """Résultats complets d'une analyse (sérialisables pour les rapports)."""

    raw_df: pd.DataFrame
    typed_df: pd.DataFrame
    meta: FileMeta
    schemas: list[ColumnSchema]
    lost_conversions: dict[str, int]
    overview: DatasetOverview
    missing: pd.DataFrame
    univariate: UnivariateResults
    correlation: CorrelationResult | None
    correlation_method_reason: str
    covariance: pd.DataFrame | None
    associations: pd.DataFrame
    quality: QualityReport
    insights: InterpretationResult
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def numeric_cols(self) -> list[str]:
        return numeric_columns(self.schemas)

    @property
    def categorical_cols(self) -> list[str]:
        return categorical_columns(self.schemas)

    @property
    def temporal_cols(self) -> list[str]:
        return temporal_columns(self.schemas)


ProgressFn = Callable[[float, str], None]


def build_schema(raw_df: pd.DataFrame, overrides: dict[str, str] | None = None
                 ) -> tuple[list[ColumnSchema], pd.DataFrame, dict[str, int]]:
    """Détecte le schéma et construit la vue typée."""
    schemas = detect_schema(raw_df, overrides)
    typed, lost = apply_schema(raw_df, schemas)
    return schemas, typed, lost


def run_analysis(
    raw_df: pd.DataFrame,
    meta: FileMeta,
    overrides: dict[str, str] | None = None,
    rules: list[ValidityRule] | None = None,
    weights: QualityWeights | None = None,
    corr_method: str | None = None,
    progress: ProgressFn | None = None,
) -> AnalysisBundle:
    """Exécute toutes les analyses ; `progress(fraction, message)` est appelé à chaque étape."""
    step = progress or (lambda _f, _m: None)
    step(0.05, "Détection des types de variables…")
    schemas, typed, lost = build_schema(raw_df, overrides)
    step(0.2, "Profil général…")
    overview = compute_overview(raw_df, schemas)
    missing = missing_by_column(typed)
    step(0.35, "Statistiques descriptives…")
    uni = compute_univariate(typed, schemas)
    nums = numeric_columns(schemas)
    cats = categorical_columns(schemas)
    corr = cov = None
    reason = "Moins de deux variables quantitatives : corrélations non calculées."
    step(0.55, "Corrélations…")
    if len(nums) >= 2:
        recommended, reason = recommend_method(typed, nums)
        corr = correlation_matrix(typed, nums, corr_method or recommended)
        cov = covariance_matrix(typed, nums)
    step(0.7, "Associations…")
    assoc = strong_associations(typed, nums, cats, corr)
    step(0.8, "Audit de qualité…")
    quality = run_quality_audit(raw_df, typed, schemas, lost, rules, weights)
    step(0.92, "Interprétations…")
    insights = generate_insights(overview, uni, quality, corr)
    step(1.0, "Analyse terminée.")
    return AnalysisBundle(raw_df, typed, meta, schemas, lost, overview, missing, uni, corr, reason, cov,
                          assoc, quality, insights)
