"""Mises en cache Streamlit des opérations coûteuses et reproductibles.

* ``st.cache_data`` : lecture/inspection des fichiers (clé = contenu + options) ;
  les résultats sont des copies, l'original n'est donc jamais partagé entre sessions ;
* l'analyse complète est mémorisée par session dans ``utils.state.get_bundle`` (clé =
  version des données + paramètres), car elle dépend de l'état propre à l'utilisateur.
"""
from __future__ import annotations

from dataclasses import asdict

import streamlit as st

from core.data_loader import FileInspection, LoadOptions, LoadResult, inspect_file, load_dataset
from utils.sample_factory import SAMPLE_BUILDERS


@st.cache_data(show_spinner=False, max_entries=8, ttl=3600)
def cached_inspect(filename: str, content: bytes) -> FileInspection:
    """Inspection mise en cache (format, feuilles, tables, membres ZIP)."""
    return inspect_file(filename, content)


@st.cache_data(show_spinner=False, max_entries=4, ttl=3600)
def _cached_load(filename: str, content: bytes, options: dict) -> LoadResult:
    return load_dataset(filename, content, LoadOptions(**options))


def cached_load(filename: str, content: bytes, options: LoadOptions) -> LoadResult:
    """Chargement mis en cache (clé : nom, contenu et options)."""
    return _cached_load(filename, content, asdict(options))


@st.cache_data(show_spinner=False)
def cached_sample(key: str):
    """Jeu d'exemple généré en mémoire (DataFrame)."""
    return SAMPLE_BUILDERS[key][1]()
