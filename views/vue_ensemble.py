"""Vue d'ensemble : indicateurs clés, synthèse automatique, types et valeurs manquantes."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from components.charts import show_chart
from components.data_tables import show_table
from components.feedback import guarded, require_data, show_flash
from components.kpi_cards import kpi_row, page_header, score_badge
from components.report_components import show_insights
from core import visualization_engine as viz
from core.file_detector import human_size
from core.schema_detector import schema_table
from utils import state
from utils.helpers import fmt_int, fmt_num, fmt_pct

page_header("Vue d'ensemble", "Synthèse automatique du jeu de données.")
show_flash()
if not require_data():
    st.stop()

with guarded("L'analyse"):
    b = state.get_bundle()
    o, m = b.overview, b.meta
    st.caption(f"Analyse des {state.active_label()} — {m.filename} ({m.format_label}, {human_size(m.size_bytes)})")

    kpi_row([
        ("Lignes", fmt_int(o.n_rows), "lecture partielle" if m.truncated else None),
        ("Variables", fmt_int(o.n_cols), f"{o.n_numeric} quanti · {o.n_categorical} quali · {o.n_temporal} temp."),
        ("Valeurs manquantes", fmt_pct(o.missing_ratio), f"{fmt_int(o.n_missing)} cellules"),
        ("Doublons", fmt_int(o.n_duplicates), fmt_pct(o.duplicate_ratio) + " des lignes"),
        ("Score qualité", f"{fmt_num(b.quality.global_score, 1)}/100", b.quality.grade),
        ("Mémoire", f"{fmt_num(o.memory_mb)} Mo", f"{fmt_int(o.n_complete_rows)} lignes complètes"),
    ], columns=6)

    st.badge(f"Qualité {b.quality.grade.lower()}", icon=":material/verified:", color=score_badge(b.quality.global_score))
    with st.container(border=True):
        st.markdown("#### :material/summarize: Synthèse générale")
        for line in b.insights.summary:
            st.markdown(f"- {line}")

    tab_insights, tab_types, tab_missing, tab_preview = st.tabs(
        ["Interprétations automatiques", "Types de variables", "Valeurs manquantes", "Aperçu"])
    with tab_insights:
        st.caption("Commentaires calculés à partir des résultats réels. Les hypothèses sont à vérifier ; "
                   "aucun service externe n'est sollicité.")
        show_insights(b.insights)
    with tab_types:
        c1, c2 = st.columns([1, 2])
        counts = pd.Series([s.stat_type.value for s in b.schemas]).value_counts().rename_axis("Type").reset_index(name="Variables")
        fig = px.pie(counts, names="Type", values="Variables", hole=0.45,
                     color_discrete_sequence=viz.PALETTE)
        fig.update_layout(title="Répartition des types statistiques", margin={"t": 50, "l": 10, "r": 10, "b": 10})
        with c1:
            show_chart(fig, "overview_types", export=False)
        with c2:
            show_table(schema_table(b.schemas)[["Variable", "Type détecté", "Type statistique", "Valeurs uniques",
                                                "Valeurs manquantes"]], key="overview_schema")
        st.page_link("views/profil_variables.py", label="Corriger un type de variable", icon=":material/edit:")
    with tab_missing:
        if o.n_missing == 0:
            st.success("Aucune valeur manquante.", icon=":material/check_circle:")
        else:
            show_chart(viz.missing_bar(b.missing), "overview_missing")
            show_table(b.missing, key="overview_missing_tbl", export_name="valeurs_manquantes")
    with tab_preview:
        show_table(b.raw_df.head(15), key="overview_head", digits=None)
    for w in m.warnings:
        st.warning(w, icon=":material/info:")
