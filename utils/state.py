"""Gestion centralisée de l'état de session Streamlit.

* ``raw_df`` : données telles que chargées — **jamais modifiées** ;
* ``cleaning`` : :class:`CleaningSession` contenant la copie de travail et l'historique ;
* ``use_cleaned`` : analyser la copie nettoyée plutôt que l'original ;
* ``overrides`` : corrections manuelles de types (réversibles) ;
* ``data_version`` : compteur incrémenté à chaque changement de données (clé de cache).
"""
from __future__ import annotations

import copy
from dataclasses import astuple
from typing import Any

import pandas as pd
import streamlit as st

from config.settings import SETTINGS, Settings, get_default_settings
from core.cleaning_engine import CleaningSession
from core.data_loader import FileMeta
from core.pipeline import AnalysisBundle, run_analysis
from i18n import set_language

_MAX_CACHED_BUNDLES = 3


def init_state() -> None:
    """Initialise les clés de session (idempotent)."""
    defaults: dict[str, Any] = {
        "raw_df": None,
        "meta": None,
        "cleaning": None,
        "use_cleaned": False,
        "overrides": {},
        "rules": [],
        "corr_method": None,
        "data_version": 0,
        "bundles": {},
        "uploader_key": 0,
        "language": "fr",
        "settings": get_default_settings(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_session_settings() -> None:
    """Applique les paramètres de la session à l'instance de configuration.

    L'application est conçue pour un usage local mono-utilisateur : les paramètres
    sont recopiés dans l'objet global ``SETTINGS`` lu par les moteurs à chaque exécution.
    """
    session: Settings = st.session_state["settings"]
    for section in ("loading", "schema", "stats", "quality", "viz"):
        target = getattr(SETTINGS, section)
        for k, v in vars(getattr(session, section)).items():
            setattr(target, k, v)
    SETTINGS.random_seed = session.random_seed
    set_language(st.session_state.get("language", "fr"))


def has_data() -> bool:
    return st.session_state.get("raw_df") is not None


def set_dataset(df: pd.DataFrame, meta: FileMeta) -> None:
    """Enregistre un nouveau dataset et réinitialise les états dépendants."""
    st.session_state["raw_df"] = df
    st.session_state["meta"] = meta
    st.session_state["cleaning"] = CleaningSession(df)
    st.session_state["use_cleaned"] = False
    st.session_state["overrides"] = {}
    st.session_state["rules"] = []
    st.session_state["corr_method"] = None
    st.session_state["bundles"] = {}
    bump_version()


def reset_all() -> None:
    """Oublie le dataset courant et toutes les analyses."""
    for key in ("raw_df", "meta", "cleaning"):
        st.session_state[key] = None
    st.session_state["use_cleaned"] = False
    st.session_state["overrides"] = {}
    st.session_state["rules"] = []
    st.session_state["bundles"] = {}
    st.session_state["uploader_key"] += 1
    bump_version()


def bump_version() -> None:
    st.session_state["data_version"] = st.session_state.get("data_version", 0) + 1


def cleaning() -> CleaningSession:
    return st.session_state["cleaning"]


def active_df() -> pd.DataFrame:
    """Données analysées : copie nettoyée si demandée, sinon original."""
    if st.session_state.get("use_cleaned") and st.session_state.get("cleaning") is not None:
        return st.session_state["cleaning"].current
    return st.session_state["raw_df"]


def active_label() -> str:
    if st.session_state.get("use_cleaned") and cleaning() is not None and cleaning().history:
        return f"données nettoyées ({len(cleaning().history)} opération(s))"
    return "données d'origine"


def meta() -> FileMeta:
    return st.session_state["meta"]


def _bundle_key() -> tuple:
    s: Settings = st.session_state["settings"]
    return (
        st.session_state["data_version"],
        bool(st.session_state.get("use_cleaned")),
        tuple(sorted(st.session_state["overrides"].items())),
        tuple(repr(r) for r in st.session_state["rules"]),
        astuple(s.quality), astuple(s.stats), astuple(s.schema),
        st.session_state.get("corr_method"),
    )


def get_bundle(show_progress: bool = True) -> AnalysisBundle:
    """Analyse complète du dataset actif, mémorisée par session (clé : version + paramètres)."""
    key = _bundle_key()
    cache: dict = st.session_state["bundles"]
    if key in cache:
        return cache[key]
    df = active_df()
    m = copy.copy(meta())
    if show_progress:
        bar = st.progress(0.0, text="Analyse en cours…")

        def progress(frac: float, msg: str) -> None:
            bar.progress(min(frac, 1.0), text=msg)
    else:
        bar = None
        progress = None  # type: ignore[assignment]
    session: Settings = st.session_state["settings"]
    bundle = run_analysis(
        df, m, overrides=st.session_state["overrides"], rules=st.session_state["rules"],
        weights=session.quality, corr_method=st.session_state.get("corr_method"), progress=progress,
    )
    if bar is not None:
        bar.empty()
    if len(cache) >= _MAX_CACHED_BUNDLES:
        cache.pop(next(iter(cache)))
    cache[key] = bundle
    return bundle


def invalidate_analysis() -> None:
    st.session_state["bundles"] = {}


def default_settings_copy() -> Settings:
    return get_default_settings()
