"""Affichage homogène des graphiques Plotly avec option d'export."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from core.visualization_engine import figure_to_html
from utils.exporters import safe_filename

PLOTLY_CONFIG = {
    "displaylogo": False,
    "toImageButtonOptions": {"format": "png", "scale": 2},
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def show_chart(fig: go.Figure, key: str, export: bool = True, height: int | None = None) -> None:
    """Affiche une figure ; export PNG via la barre d'outils et HTML interactif via un bouton."""
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, config=PLOTLY_CONFIG, key=f"chart_{key}")
    if export:
        title = (fig.layout.title.text or key) if fig.layout.title else key
        st.download_button(
            "Exporter (HTML interactif)", data=lambda: figure_to_html(fig), file_name=f"{safe_filename(title)}.html",
            mime="text/html", key=f"dl_{key}", icon=":material/download:", type="tertiary",
            help="L'export PNG est disponible via l'icône appareil photo du graphique.",
        )
