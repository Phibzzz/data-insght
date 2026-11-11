"""Paramètres centraux de DATAINSIGHT.

Toutes les limites, seuils statistiques et pondérations du score qualité sont
regroupés ici afin d'être documentés, testables et modifiables depuis la page
« Paramètres » (les modifications de l'interface restent propres à la session).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

APP_NAME = "DATAINSIGHT"
APP_SUBTITLE = "Intelligent Dataset Analyzer"
APP_VERSION = "1.0.0"

ROOT_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT_DIR / "assets"
SAMPLE_DATA_DIR = ROOT_DIR / "sample_data"
I18N_DIR = ROOT_DIR / "i18n"

DEFAULT_LANGUAGE = "fr"


@dataclass
class LoadingSettings:
    """Limites appliquées lors de l'import des fichiers."""

    max_file_size_mb: int = 500
    # Au-delà de ce nombre de lignes, l'utilisateur est invité à échantillonner.
    large_dataset_rows: int = 1_000_000
    # Nombre de lignes chargées par défaut en mode « lecture partielle ».
    default_row_limit: int = 200_000
    # Octets lus pour détecter encodage / séparateur.
    sniff_bytes: int = 64_000
    # Protection des archives ZIP.
    zip_max_files: int = 50
    zip_max_uncompressed_mb: int = 1_000
    zip_max_ratio: int = 100
    # Nombre maximal de lignes lues depuis une table SQLite.
    sqlite_max_rows: int = 2_000_000


@dataclass
class SchemaSettings:
    """Seuils de la détection automatique des types de variables."""

    # Proportion minimale de valeurs convertibles pour accepter une conversion.
    conversion_threshold: float = 0.95
    # Nombre maximal de modalités pour considérer une variable comme catégorielle.
    max_categories: int = 30
    # Ratio d'unicité sous lequel une variable texte est catégorielle.
    category_unique_ratio: float = 0.5
    # Ratio d'unicité à partir duquel une variable est un identifiant potentiel.
    identifier_unique_ratio: float = 0.98
    # Nombre de valeurs distinctes sous lequel un entier est « discret ».
    discrete_max_unique: int = 20
    # Longueur moyenne au-delà de laquelle une chaîne est du texte libre.
    free_text_min_length: float = 40.0


@dataclass
class StatisticsSettings:
    """Seuils statistiques utilisés par les moteurs d'analyse et d'interprétation."""

    high_cv: float = 1.0
    skew_threshold: float = 1.0
    strong_correlation: float = 0.7
    moderate_correlation: float = 0.4
    significance_level: float = 0.05
    normality_max_n: int = 5000
    dominant_category_share: float = 0.8
    rare_category_share: float = 0.01
    high_missing_share: float = 0.3
    zscore_threshold: float = 3.0
    iqr_factor: float = 1.5
    outlier_share_alert: float = 0.05
    max_corr_columns: int = 60


@dataclass
class QualityWeights:
    """Pondérations du score global de qualité (somme normalisée à 1)."""

    completeness: float = 0.35
    uniqueness: float = 0.20
    validity: float = 0.20
    consistency: float = 0.15
    outliers: float = 0.10

    def normalized(self) -> dict[str, float]:
        """Retourne les pondérations normalisées pour que leur somme vaille 1."""
        weights = asdict(self)
        total = sum(weights.values()) or 1.0
        return {k: v / total for k, v in weights.items()}


@dataclass
class VisualizationSettings:
    """Paramètres d'échantillonnage des graphiques coûteux."""

    max_points_scatter: int = 20_000
    max_rows_pairplot: int = 2_000
    max_vars_pairplot: int = 6
    max_pie_categories: int = 8
    max_bar_categories: int = 30
    color_sequence: tuple[str, ...] = (
        "#4F46E5", "#0EA5E9", "#10B981", "#F59E0B", "#EF4444",
        "#8B5CF6", "#14B8A6", "#F97316", "#64748B", "#EC4899",
    )


@dataclass
class Settings:
    """Ensemble des paramètres de l'application."""

    loading: LoadingSettings = field(default_factory=LoadingSettings)
    schema: SchemaSettings = field(default_factory=SchemaSettings)
    stats: StatisticsSettings = field(default_factory=StatisticsSettings)
    quality: QualityWeights = field(default_factory=QualityWeights)
    viz: VisualizationSettings = field(default_factory=VisualizationSettings)
    random_seed: int = 42


def get_default_settings() -> Settings:
    """Construit une nouvelle instance de paramètres par défaut."""
    return Settings()


SETTINGS = get_default_settings()

SUPPORTED_EXTENSIONS: dict[str, list[str]] = {
    "Tabulaires": ["csv", "tsv", "txt"],
    "Bureautiques": ["xls", "xlsx", "xlsm", "ods"],
    "Analytiques": ["parquet", "feather", "orc"],
    "Structurés": ["json", "jsonl", "xml"],
    "Statistiques": ["sav", "dta", "sas7bdat"],
    "Bases de données": ["sqlite", "db"],
    "Archives": ["zip"],
}


def all_extensions() -> list[str]:
    """Liste à plat de toutes les extensions acceptées."""
    return [ext for exts in SUPPORTED_EXTENSIONS.values() for ext in exts]
