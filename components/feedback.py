"""Gestion centralisée des erreurs et messages utilisateur dans l'interface."""
from __future__ import annotations

import traceback
from collections.abc import Iterator
from contextlib import contextmanager

import streamlit as st

from core.exceptions import DataInsightError


@contextmanager
def guarded(action: str = "L'opération") -> Iterator[None]:
    """Transforme les exceptions en messages compréhensibles (détail technique repliable)."""
    try:
        yield
    except DataInsightError as exc:
        st.error(f"**{action} a échoué.** {exc.user_message}", icon=":material/error:")
        if exc.detail:
            with st.expander("Détail technique"):
                st.code(exc.detail, language=None)
    except Exception as exc:  # noqa: BLE001 - dernier rempart de l'interface
        st.error(f"**{action} a échoué** : erreur inattendue ({type(exc).__name__}).", icon=":material/error:")
        with st.expander("Détail technique"):
            st.code("".join(traceback.format_exception(exc))[-4000:], language=None)


def require_data() -> bool:
    """Affiche une invitation à importer des données si aucun dataset n'est chargé."""
    if st.session_state.get("raw_df") is None:
        st.info("Aucun jeu de données chargé. Importez un fichier ou choisissez un jeu d'exemple.",
                icon=":material/upload_file:")
        c1, c2 = st.columns(2)
        c1.page_link("views/importation.py", label="Importer un fichier", icon=":material/upload:")
        c2.page_link("views/accueil.py", label="Jeux d'exemple", icon=":material/dataset:")
        return False
    return True


def flash(message: str, kind: str = "success") -> None:
    """Message à afficher au prochain rendu (survit à un st.rerun)."""
    st.session_state["_flash"] = (kind, message)


def show_flash() -> None:
    item = st.session_state.pop("_flash", None)
    if item:
        kind, message = item
        {"success": st.success, "info": st.info, "warning": st.warning, "error": st.error}[kind](message)
