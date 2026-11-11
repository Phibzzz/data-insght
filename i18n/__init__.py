"""Internationalisation minimale : dictionnaires JSON par langue et fonction t().

Le français est la langue par défaut. Une clé absente de la langue courante
retombe sur le français, puis sur la clé elle-même.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_current_language = "fr"
AVAILABLE_LANGUAGES = {"fr": "Français", "en": "English"}


@lru_cache(maxsize=8)
def _load(lang: str) -> dict[str, str]:
    path = _DIR / f"{lang}.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def set_language(lang: str) -> None:
    """Change la langue courante (si disponible)."""
    global _current_language
    if lang in AVAILABLE_LANGUAGES:
        _current_language = lang


def get_language() -> str:
    """Retourne le code de la langue courante."""
    return _current_language


def t(key: str, **kwargs: object) -> str:
    """Traduit une clé, avec formatage optionnel `{nom}`."""
    text = _load(_current_language).get(key) or _load("fr").get(key) or key
    if kwargs:
        try:
            return text.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            return text
    return text
