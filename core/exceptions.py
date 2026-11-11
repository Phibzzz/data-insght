"""Hiérarchie d'exceptions de DATAINSIGHT.

Chaque exception porte un message destiné à l'utilisateur final (en français)
et éventuellement un détail technique pour le diagnostic.
"""
from __future__ import annotations


class DataInsightError(Exception):
    """Exception de base : message compréhensible + détail technique optionnel."""

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.user_message = message
        self.detail = detail

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.user_message


class UnsupportedFormatError(DataInsightError):
    """Le format du fichier n'est pas pris en charge."""


class FileValidationError(DataInsightError):
    """Le fichier est invalide, corrompu, trop volumineux ou suspect."""


class EmptyDatasetError(DataInsightError):
    """Le fichier ne contient aucune donnée exploitable."""


class MissingDependencyError(DataInsightError):
    """Une bibliothèque optionnelle nécessaire n'est pas installée."""


class DataLoadError(DataInsightError):
    """Erreur lors de la lecture du contenu du fichier."""


class CleaningError(DataInsightError):
    """Une opération de nettoyage ne peut pas être appliquée."""


class ReportError(DataInsightError):
    """Erreur lors de la génération d'un rapport ou d'un export."""
