"""Profil des variables : types détectés, justification et corrections manuelles réversibles."""
from __future__ import annotations

import streamlit as st

from components.data_tables import show_table
from components.feedback import flash, guarded, require_data, show_flash
from components.kpi_cards import kpi_row, page_header
from core.schema_detector import FAMILY_OF, StatType, schema_table
from utils import state
from utils.helpers import fmt_int, fmt_pct

page_header("Profil des variables", "Reconnaissance automatique des types et corrections manuelles.")
show_flash()
if not require_data():
    st.stop()

with guarded("La détection des types"):
    b = state.get_bundle()
    overrides: dict[str, str] = st.session_state["overrides"]

    st.markdown(
        "La détection combine le **type technique** (pandas), le **contenu réel** (tests de conversion), le "
        "**nombre de modalités**, la **proportion de valeurs uniques** et des indices sur le **nom** des colonnes. "
        "Les corrections sont appliquées à une vue typée : le fichier d'origine n'est jamais modifié."
    )
    table = schema_table(b.schemas)
    show_table(table, key="schema_tbl", export_name="profil_variables", column_config={
        "Confiance": st.column_config.ProgressColumn("Confiance", min_value=0.0, max_value=1.0, format="%.2f"),
        "% manquants": st.column_config.NumberColumn("% manquants", format="%.2f %%"),
    })

    if b.lost_conversions:
        st.warning("Valeurs non convertibles dans le type retenu (traitées comme manquantes dans les analyses) : "
                   + ", ".join(f"{k} ({v})" for k, v in b.lost_conversions.items()), icon=":material/warning:")

    st.subheader("Détail et correction d'une variable")
    names = [s.name for s in b.schemas]
    col = st.selectbox("Variable", names)
    sch = next(s for s in b.schemas if s.name == col)
    kpi_row([
        ("Type statistique", sch.stat_type.value, f"famille {sch.family.value.lower()}"),
        ("Type technique", sch.technical_type, sch.storage_type),
        ("Valeurs uniques", fmt_int(sch.n_unique), f"ratio {fmt_pct(sch.unique_ratio)}"),
        ("Manquants", fmt_int(sch.n_missing), fmt_pct(sch.missing_ratio)),
    ])
    st.markdown(f"**Justification :** {sch.reason}  \n**Confiance :** {sch.confidence:.2f}"
                + (f"  \n**Ordre des modalités :** {' < '.join(sch.ordinal_order)}" if sch.ordinal_order else ""))
    if sch.sample_values:
        st.caption("Exemples de valeurs : " + " · ".join(f"`{v}`" for v in sch.sample_values))

    with st.form("override_form"):
        options = [t.value for t in StatType]
        new_type = st.selectbox("Nouveau type statistique", options, index=options.index(sch.stat_type.value),
                                format_func=lambda v: f"{v} ({FAMILY_OF[StatType(v)].value})")
        c1, c2 = st.columns(2)
        apply = c1.form_submit_button("Appliquer la correction", type="primary", icon=":material/check:")
        revert = c2.form_submit_button("Revenir au type détecté", icon=":material/undo:", disabled=col not in overrides)
    if apply:
        if new_type == sch.detected_stat_type.value:
            overrides.pop(col, None)
        else:
            overrides[col] = new_type
        flash(f"Type de « {col} » : {new_type}. Les analyses ont été recalculées.")
        st.rerun()
    if revert:
        overrides.pop(col, None)
        flash(f"« {col} » est revenu au type détecté ({sch.detected_stat_type.value}).", "info")
        st.rerun()

    if overrides:
        st.subheader("Corrections actives")
        for k, v in list(overrides.items()):
            st.markdown(f"- `{k}` → **{v}**")
        if st.button("Annuler toutes les corrections", icon=":material/restart_alt:"):
            st.session_state["overrides"] = {}
            flash("Toutes les corrections ont été annulées.", "info")
            st.rerun()
