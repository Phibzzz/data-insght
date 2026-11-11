"""Rapports : sélection des sections, génération PDF / HTML / Excel / CSV / JSON."""
from __future__ import annotations

from datetime import datetime

import streamlit as st

from components.feedback import guarded, require_data
from components.kpi_cards import page_header
from components.report_components import section_selector, show_insights
from core.report_engine import EXPORT_FORMATS, build_report
from utils import state
from utils.exporters import safe_filename

page_header("Rapports", "Rapport d'analyse complet, sections au choix, en plusieurs formats.")
if not require_data():
    st.stop()

with guarded("La préparation du rapport"):
    b = state.get_bundle()
    st.caption(f"Le rapport porte sur les {state.active_label()}.")

    with st.container(border=True):
        title = st.text_input("Titre du rapport", f"Rapport d'analyse — {b.meta.filename}")
        st.markdown("**Sections à inclure**")
        sections = section_selector()
        c1, c2 = st.columns(2)
        formats = c1.multiselect("Formats", list(EXPORT_FORMATS), default=["PDF", "HTML"])
        charts = c2.checkbox("Inclure les graphiques", value=True,
                             help="Le HTML embarque Plotly (≈ 5 Mo, consultable hors ligne).")

    if not sections:
        st.warning("Sélectionnez au moins une section.")
    elif st.button("Générer le rapport", type="primary", icon=":material/description:", disabled=not formats):
        outputs = {}
        progress = st.progress(0.0)
        for i, fmt in enumerate(formats):
            progress.progress(i / len(formats), text=f"Génération {fmt}…")
            with guarded(f"La génération {fmt}"):
                outputs[fmt] = build_report(b, fmt, sections, title, charts)
        progress.empty()
        st.session_state["report_outputs"] = outputs
        st.session_state["report_stamp"] = datetime.now().strftime("%Y%m%d_%H%M")

    outputs = st.session_state.get("report_outputs") or {}
    if outputs:
        st.success(f"{len(outputs)} rapport(s) prêt(s) au téléchargement.", icon=":material/check_circle:")
        base = f"rapport_{safe_filename(b.meta.filename.rsplit('.', 1)[0])}_{st.session_state['report_stamp']}"
        cols = st.columns(len(outputs))
        for col, (fmt, data) in zip(cols, outputs.items()):
            ext, mime = EXPORT_FORMATS[fmt]
            col.download_button(f"{fmt} ({len(data) / 1024:,.0f} Ko)".replace(",", " "), data, f"{base}.{ext}", mime,
                                icon=":material/download:", width="stretch", key=f"dl_report_{fmt}")

    st.subheader("Aperçu des interprétations incluses")
    show_insights(b.insights, limit=15)
