"""Cartes KPI (HTML/CSS, compatibles thèmes clair et sombre)."""
from __future__ import annotations

from html import escape

import streamlit as st


def kpi_row(items: list[tuple[str, str, str | None]], columns: int | None = None) -> None:
    """Affiche une rangée de cartes KPI.

    Args:
        items: liste de (libellé, valeur, sous-texte optionnel).
    """
    cards = "".join(
        f"<div class='di-kpi'><div class='di-kpi-label'>{escape(label)}</div>"
        f"<div class='di-kpi-value'>{escape(value)}</div>"
        + (f"<div class='di-kpi-sub'>{escape(sub)}</div>" if sub else "")
        + "</div>"
        for label, value, sub in items
    )
    cols = columns or len(items)
    st.markdown(f"<div class='di-kpis' style='--di-cols:{cols}'>{cards}</div>", unsafe_allow_html=True)


def page_header(title: str, subtitle: str | None = None, icon: str | None = None) -> None:
    """En-tête de page homogène."""
    st.markdown(
        f"<div class='di-header'><h1>{escape(title)}</h1>"
        + (f"<p>{escape(subtitle)}</p>" if subtitle else "") + "</div>",
        unsafe_allow_html=True,
    )


def score_badge(score: float) -> str:
    """Couleur de badge Streamlit selon le score qualité."""
    return "green" if score >= 90 else "blue" if score >= 75 else "orange" if score >= 60 else "red"
