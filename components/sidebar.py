"""Contenu de la barre latérale : état du dataset et actions globales."""
from __future__ import annotations

import streamlit as st

from config.settings import APP_NAME, APP_VERSION, ASSETS_DIR
from core.file_detector import human_size
from utils import state
from utils.helpers import fmt_int


def render_sidebar() -> None:
    logo = ASSETS_DIR / "logo.png"
    if logo.exists():
        st.logo(str(logo), size="large")
    with st.sidebar:
        if state.has_data():
            m = state.meta()
            df = state.active_df()
            with st.container(border=True):
                st.markdown(f"**:material/description: {m.filename}**")
                st.caption(f"{m.format_label} · {human_size(m.size_bytes)}")
                st.caption(f"{fmt_int(df.shape[0])} lignes × {fmt_int(df.shape[1])} colonnes")
                cl = state.cleaning()
                if cl is not None and cl.history:
                    use = st.toggle("Analyser les données nettoyées", value=bool(st.session_state["use_cleaned"]),
                                    help="Les pages d'analyse utilisent la copie de travail issue du Nettoyage.")
                    if use != st.session_state["use_cleaned"]:
                        st.session_state["use_cleaned"] = use
                        st.rerun()
                    st.caption(f"{len(cl.history)} opération(s) de nettoyage")
                if st.button("Réinitialiser l'analyse", icon=":material/restart_alt:", width="stretch"):
                    state.reset_all()
                    st.rerun()
        else:
            st.caption("Aucun jeu de données chargé.")
        st.caption(f"{APP_NAME} v{APP_VERSION} · 100 % local · aucune donnée transmise")
