"""Widgets de configuration et d'affichage liés aux rapports et interprétations."""
from __future__ import annotations

import streamlit as st

from core.interpretation_engine import Category, InterpretationResult
from core.report_engine import REPORT_SECTIONS
from utils.helpers import fmt_num

CATEGORY_STYLE = {
    Category.FINDING: (":material/fact_check:", "blue"),
    Category.INTERPRETATION: (":material/psychology:", "violet"),
    Category.HYPOTHESIS: (":material/help:", "orange"),
    Category.RECOMMENDATION: (":material/task_alt:", "green"),
}


def section_selector(key: str = "report_sections") -> list[str]:
    """Sélection des sections du rapport (toutes cochées par défaut)."""
    cols = st.columns(2)
    selected = []
    for i, (k, label) in enumerate(REPORT_SECTIONS.items()):
        if cols[i % 2].checkbox(f"{i + 1}. {label}", value=True, key=f"{key}_{k}"):
            selected.append(k)
    return selected


def show_insights(result: InterpretationResult, limit: int = 50) -> None:
    """Affiche les commentaires classés par catégorie (constats, interprétations, hypothèses, recommandations)."""
    tabs = st.tabs([f"{c.value}s ({len(result.by_category(c))})" for c in Category])
    for tab, cat in zip(tabs, Category):
        with tab:
            items = result.by_category(cat)
            if not items:
                st.caption("Aucun élément dans cette catégorie.")
            icon, color = CATEGORY_STYLE[cat]
            for item in items[:limit]:
                with st.container(border=True):
                    head = st.columns([1.6, 5])
                    head[0].badge(item.topic, icon=icon, color=color)
                    head[1].markdown(item.text)
                    if item.metric:
                        st.caption(f"Mesure source : {item.metric}"
                                   + (f" = {fmt_num(item.value, 4)}" if isinstance(item.value, float) else ""))
