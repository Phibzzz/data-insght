"""Validations des entrées utilisateur et des fichiers importés."""
from __future__ import annotations

import pandas as pd

from config.settings import SETTINGS, all_extensions
from core.exceptions import EmptyDatasetError, FileValidationError
from core.file_detector import get_extension

MIN_ROWS_FOR_ANALYSIS = 1


def validate_upload(filename: str, size_bytes: int) -> None:
    """Contrôles préalables à la lecture : extension et taille."""
    ext = get_extension(filename)
    if ext and ext not in all_extensions():
        raise FileValidationError(
            f"Extension « .{ext} » non prise en charge. Formats acceptés : {', '.join(all_extensions())}."
        )
    if size_bytes <= 0:
        raise FileValidationError("Le fichier est vide.")
    limit = SETTINGS.loading.max_file_size_mb * 1024 * 1024
    if size_bytes > limit:
        raise FileValidationError(
            f"Fichier trop volumineux ({size_bytes / 1024**2:.1f} Mo > {SETTINGS.loading.max_file_size_mb} Mo)."
        )


def validate_dataframe(df: pd.DataFrame) -> list[str]:
    """Vérifie qu'un DataFrame est exploitable ; renvoie des avertissements non bloquants."""
    if df is None or df.shape[1] == 0:
        raise EmptyDatasetError("Aucune colonne exploitable.")
    if len(df) < MIN_ROWS_FOR_ANALYSIS:
        raise EmptyDatasetError("Aucune ligne de données.")
    warnings = []
    if len(df) < 10:
        warnings.append(f"Très petit échantillon ({len(df)} lignes) : les statistiques sont peu fiables.")
    if df.shape[1] == 1:
        warnings.append("Une seule variable : les analyses multivariées ne sont pas disponibles.")
    if len(df) > SETTINGS.loading.large_dataset_rows:
        warnings.append("Jeu volumineux : certains graphiques utilisent un échantillon.")
    return warnings


def validate_column_name(name: str, existing: list[str]) -> str:
    """Nom de colonne non vide et unique."""
    name = (name or "").strip()
    if not name:
        raise ValueError("Le nom de colonne ne peut pas être vide.")
    if name in existing:
        raise ValueError(f"La colonne « {name} » existe déjà.")
    return name
