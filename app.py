"""DATAINSIGHT – Intelligent Dataset Analyzer.

Point d'entrée Streamlit : configuration de la page, thème, état de session et
navigation latérale entre les 12 rubriques.

Lancement :
    streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from config.settings import APP_NAME, ASSETS_DIR
from components.sidebar import render_sidebar
from i18n import t
from utils import state

st.set_page_config(
    page_title=f"{APP_NAME} – Intelligent Dataset Analyzer",
    page_icon=":material/insights:",
    layout="wide",
    initial_sidebar_state="expanded",
)

css = ASSETS_DIR / "styles.css"
if css.exists():
    st.markdown(f"<style>{css.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

state.init_state()
state.apply_session_settings()

pages = {
    t("nav.section.start"): [
        st.Page("views/accueil.py", title=t("nav.home"), icon=":material/home:", default=True),
        st.Page("views/importation.py", title=t("nav.import"), icon=":material/upload_file:"),
        st.Page("views/vue_ensemble.py", title=t("nav.overview"), icon=":material/dashboard:"),
    ],
    t("nav.section.analysis"): [
        st.Page("views/exploration.py", title=t("nav.explore"), icon=":material/table_view:"),
        st.Page("views/profil_variables.py", title=t("nav.profile"), icon=":material/category:"),
        st.Page("views/statistiques.py", title=t("nav.stats"), icon=":material/functions:"),
        st.Page("views/analyse_multivariee.py", title=t("nav.multivariate"), icon=":material/hub:"),
        st.Page("views/qualite_donnees.py", title=t("nav.quality"), icon=":material/verified:"),
        st.Page("views/visualisations.py", title=t("nav.viz"), icon=":material/bar_chart:"),
    ],
    t("nav.section.prepare"): [
        st.Page("views/nettoyage.py", title=t("nav.cleaning"), icon=":material/cleaning_services:"),
        st.Page("views/rapports.py", title=t("nav.reports"), icon=":material/description:"),
        st.Page("views/parametres.py", title=t("nav.settings"), icon=":material/settings:"),
    ],
}

navigation = st.navigation(pages, position="sidebar", expanded=True)
render_sidebar()
navigation.run()
