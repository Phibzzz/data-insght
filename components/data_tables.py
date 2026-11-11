"""Tableaux interactifs avec exports CSV / Excel."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.exporters import safe_filename, to_csv_bytes, to_excel_bytes
from utils.helpers import display_safe, round_numeric


def show_table(df: pd.DataFrame, key: str, export_name: str | None = None, height: int | str = "auto",
               digits: int | None = 4, hide_index: bool = True, column_config: dict | None = None,
               max_rows: int | None = None) -> None:
    """Affiche un tableau et propose son export (CSV et Excel)."""
    if df is None or df.empty:
        st.caption("Aucune donnée à afficher.")
        return
    shown = df.head(max_rows) if max_rows else df
    if digits is not None:
        shown = round_numeric(shown, digits)
    st.dataframe(display_safe(shown), height=height, hide_index=hide_index, column_config=column_config, key=f"tbl_{key}")
    if max_rows and len(df) > max_rows:
        st.caption(f"Affichage limité aux {max_rows} premières lignes sur {len(df)} (export complet ci-dessous).")
    if export_name:
        name = safe_filename(export_name)
        c1, c2, _ = st.columns([1, 1, 4])
        c1.download_button("CSV", lambda: to_csv_bytes(df), f"{name}.csv", "text/csv", key=f"csv_{key}",
                           icon=":material/download:", width="stretch")
        if len(df) <= 200_000:
            c2.download_button("Excel", lambda: to_excel_bytes({name[:31]: df}), f"{name}.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               key=f"xlsx_{key}", icon=":material/download:", width="stretch")
