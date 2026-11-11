"""Paramètres : langue, limites de chargement, seuils statistiques, pondérations du score qualité."""
from __future__ import annotations

import streamlit as st

from components.feedback import flash, show_flash
from components.kpi_cards import page_header
from config.settings import Settings, get_default_settings
from core.interpretation_engine import NullProvider
from i18n import AVAILABLE_LANGUAGES
from utils import state

page_header("Paramètres", "Réglages de la session (application locale).")
show_flash()
s: Settings = st.session_state["settings"]

with st.form("settings_form"):
    st.subheader("Interface")
    lang = st.selectbox("Langue de l'interface", list(AVAILABLE_LANGUAGES), format_func=AVAILABLE_LANGUAGES.get,
                        index=list(AVAILABLE_LANGUAGES).index(st.session_state.get("language", "fr")),
                        help="Le français est complet ; l'anglais couvre la navigation (architecture prête pour la traduction).")
    st.caption("Thème clair / sombre : menu ⋮ (en haut à droite) › Settings › Theme.")

    st.subheader("Chargement des fichiers")
    c1, c2, c3 = st.columns(3)
    max_mb = c1.number_input("Taille maximale (Mo)", 1, 5000, s.loading.max_file_size_mb,
                             help="La limite de téléversement Streamlit est fixée dans .streamlit/config.toml (maxUploadSize).")
    large = c2.number_input("Seuil « grand dataset » (lignes)", 10_000, 100_000_000, s.loading.large_dataset_rows, step=100_000)
    limit = c3.number_input("Lignes en lecture partielle", 1_000, 50_000_000, s.loading.default_row_limit, step=10_000)

    st.subheader("Détection des types")
    c1, c2, c3 = st.columns(3)
    conv = c1.slider("Part minimale de valeurs convertibles", 0.5, 1.0, s.schema.conversion_threshold, 0.01)
    max_cat = c2.number_input("Modalités max. d'une catégorielle", 2, 500, s.schema.max_categories)
    disc = c3.number_input("Valeurs distinctes max. d'une discrète", 2, 200, s.schema.discrete_max_unique)

    st.subheader("Seuils statistiques")
    c1, c2, c3, c4 = st.columns(4)
    strong = c1.slider("Corrélation forte (|r| ≥)", 0.3, 0.99, s.stats.strong_correlation, 0.05)
    alpha = c2.select_slider("Seuil de significativité α", [0.01, 0.05, 0.1], s.stats.significance_level)
    iqr_k = c3.number_input("Facteur IQR (k)", 0.5, 5.0, s.stats.iqr_factor, 0.5)
    z_thr = c4.number_input("Seuil z-score", 1.0, 6.0, s.stats.zscore_threshold, 0.5)
    c1, c2, c3, c4 = st.columns(4)
    cv = c1.number_input("CV « forte dispersion »", 0.1, 5.0, s.stats.high_cv, 0.1)
    skew = c2.number_input("|Asymétrie| marquée", 0.1, 5.0, s.stats.skew_threshold, 0.1)
    miss = c3.slider("Variable « très incomplète » (%)", 5, 95, int(100 * s.stats.high_missing_share))
    rare = c4.slider("Modalité rare (< %)", 0.1, 10.0, 100 * s.stats.rare_category_share, 0.1)

    st.subheader("Pondérations du score qualité")
    st.caption("Les pondérations sont normalisées pour que leur somme vaille 100 %.")
    c1, c2, c3, c4, c5 = st.columns(5)
    w = {
        "completeness": c1.number_input("Complétude", 0.0, 1.0, s.quality.completeness, 0.05),
        "uniqueness": c2.number_input("Unicité", 0.0, 1.0, s.quality.uniqueness, 0.05),
        "validity": c3.number_input("Validité", 0.0, 1.0, s.quality.validity, 0.05),
        "consistency": c4.number_input("Cohérence", 0.0, 1.0, s.quality.consistency, 0.05),
        "outliers": c5.number_input("Atypiques", 0.0, 1.0, s.quality.outliers, 0.05),
    }

    st.subheader("Visualisations")
    c1, c2 = st.columns(2)
    max_pts = c1.number_input("Points max. (nuages de points)", 1_000, 500_000, s.viz.max_points_scatter, step=1_000)
    max_pair = c2.number_input("Lignes max. (matrice de nuages)", 200, 20_000, s.viz.max_rows_pairplot, step=100)

    saved = st.form_submit_button("Enregistrer", type="primary", icon=":material/save:")

if saved:
    if sum(w.values()) <= 0:
        st.error("Au moins une pondération doit être strictement positive.")
    else:
        s.loading.max_file_size_mb, s.loading.large_dataset_rows, s.loading.default_row_limit = int(max_mb), int(large), int(limit)
        s.schema.conversion_threshold, s.schema.max_categories, s.schema.discrete_max_unique = float(conv), int(max_cat), int(disc)
        s.stats.strong_correlation, s.stats.significance_level = float(strong), float(alpha)
        s.stats.iqr_factor, s.stats.zscore_threshold = float(iqr_k), float(z_thr)
        s.stats.high_cv, s.stats.skew_threshold = float(cv), float(skew)
        s.stats.high_missing_share, s.stats.rare_category_share = miss / 100, rare / 100
        for k, v in w.items():
            setattr(s.quality, k, float(v))
        s.viz.max_points_scatter, s.viz.max_rows_pairplot = int(max_pts), int(max_pair)
        st.session_state["language"] = lang
        state.invalidate_analysis()
        flash("Paramètres enregistrés : les analyses seront recalculées.")
        st.rerun()

if st.button("Rétablir les valeurs par défaut", icon=":material/settings_backup_restore:"):
    st.session_state["settings"] = get_default_settings()
    st.session_state["language"] = "fr"
    state.invalidate_analysis()
    flash("Paramètres par défaut rétablis.", "info")
    st.rerun()

with st.expander("Confidentialité et enrichissement par IA"):
    st.markdown(
        f"- Fournisseur d'interprétation actif : **{NullProvider.name}**. Les commentaires sont produits par des règles "
        "locales à partir des résultats calculés.\n"
        "- L'architecture (`core/interpretation_engine.py`, protocole `LLMProvider`) permet de brancher un modèle de "
        "langage ; aucun n'est configuré et **aucune donnée n'est transmise** à un service externe.\n"
        "- Statistiques d'usage Streamlit désactivées (`gatherUsageStats = false`)."
    )
