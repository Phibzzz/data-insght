"""Exploration interactive : aperçus, recherche, filtres, sélection, tri et informations."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components.data_tables import show_table
from components.feedback import guarded, require_data
from components.kpi_cards import kpi_row, page_header
from core.cleaning_engine import FILTER_OPERATORS, filter_mask
from core.profiling_engine import dtypes_table
from core.schema_detector import Family
from utils import state
from utils.helpers import fmt_int

page_header("Exploration", "Parcourez, filtrez et triez les données sans les modifier.")
if not require_data():
    st.stop()

with guarded("L'exploration"):
    b = state.get_bundle(show_progress=False)
    df = b.typed_df
    schemas = {s.name: s for s in b.schemas}
    st.caption(f"Vue typée des {state.active_label()} : les conversions de types n'affectent pas le fichier d'origine.")

    tab_view, tab_filter, tab_info = st.tabs(["Aperçus", "Rechercher, filtrer et trier", "Informations générales"])

    with tab_view:
        c1, c2 = st.columns([2, 1])
        mode = c1.segmented_control("Affichage", ["Premières lignes", "Dernières lignes", "Échantillon aléatoire"],
                                    default="Premières lignes")
        n = c2.number_input("Nombre de lignes", 1, 1000, 20)
        if mode == "Dernières lignes":
            view = df.tail(n)
        elif mode == "Échantillon aléatoire":
            seed = st.number_input("Graine aléatoire (reproductibilité)", 0, 10_000, 42)
            view = df.sample(min(n, len(df)), random_state=int(seed))
        else:
            view = df.head(n)
        show_table(view, key="explore_view", digits=None, hide_index=False)

    with tab_filter:
        cols = st.multiselect("Colonnes à afficher", list(df.columns), default=list(df.columns)[:20])
        search = st.text_input("Rechercher une valeur (dans toutes les colonnes affichées)", placeholder="ex. Paris")

        if "explore_filters" not in st.session_state:
            st.session_state["explore_filters"] = []
        with st.expander("Filtres", expanded=bool(st.session_state["explore_filters"])):
            fc1, fc2, fc3, fc4 = st.columns([2, 1.3, 2, 0.8])
            fcol = fc1.selectbox("Colonne", list(df.columns), key="f_col")
            fam = schemas[fcol].family if fcol in schemas else Family.OTHER
            f_op = fc2.selectbox("Opérateur", FILTER_OPERATORS, key="f_op")
            v1 = v2 = None
            if f_op not in ("est manquant", "n'est pas manquant"):
                if fam == Family.NUMERIC and f_op not in ("contient", "ne contient pas", "dans la liste"):
                    x = pd.to_numeric(df[fcol], errors="coerce")
                    lo, hi = (float(x.min()), float(x.max())) if x.notna().any() else (0.0, 0.0)
                    if f_op == "entre":
                        v1, v2 = fc3.slider("Intervalle", lo, hi, (lo, hi)) if lo < hi else (lo, hi)
                    else:
                        v1 = fc3.number_input("Valeur", value=lo)
                elif fam == Family.TEMPORAL and f_op not in ("contient", "ne contient pas", "dans la liste"):
                    d = pd.to_datetime(df[fcol], errors="coerce")
                    if f_op == "entre":
                        rng = fc3.date_input("Période", (d.min().date(), d.max().date())) if d.notna().any() else ()
                        if isinstance(rng, tuple) and len(rng) == 2:
                            v1, v2 = str(rng[0]), str(rng[1]) + " 23:59:59"
                    else:
                        v1 = str(fc3.date_input("Date", d.min().date() if d.notna().any() else None))
                elif fam == Family.CATEGORICAL and f_op in ("==", "!=", "dans la liste"):
                    choices = [str(v) for v in df[fcol].dropna().astype(str).unique()[:500]]
                    if f_op == "dans la liste":
                        v1 = fc3.multiselect("Valeurs", choices)
                    else:
                        v1 = fc3.selectbox("Valeur", choices)
                else:
                    v1 = fc3.text_input("Valeur", key="f_val")
            fc4.write("")
            if fc4.button("Ajouter", icon=":material/add:"):
                st.session_state["explore_filters"].append((fcol, f_op, v1, v2))
                st.rerun()
            for i, (c, o_, a, z) in enumerate(st.session_state["explore_filters"]):
                r1, r2 = st.columns([6, 1])
                r1.markdown(f"`{c}` **{o_}** {'' if a is None else a}{' et ' + str(z) if z is not None else ''}")
                if r2.button("Retirer", key=f"rm_filter_{i}", icon=":material/close:"):
                    st.session_state["explore_filters"].pop(i)
                    st.rerun()

        result = df
        for c, o_, a, z in st.session_state["explore_filters"]:
            if c in result.columns:
                result = result[filter_mask(result[c], o_, a, z)]
        shown_cols = cols or list(df.columns)
        result = result[shown_cols]
        if search:
            mask = pd.Series(False, index=result.index)
            for c in shown_cols:
                mask |= result[c].astype("string").str.contains(search, case=False, regex=False, na=False)
            result = result[mask]

        s1, s2 = st.columns([2, 1])
        sort_cols = s1.multiselect("Trier par", shown_cols)
        ascending = s2.radio("Ordre", ["Croissant", "Décroissant"], horizontal=True) == "Croissant"
        if sort_cols:
            result = result.sort_values(sort_cols, ascending=ascending, kind="stable", na_position="last")

        kpi_row([("Lignes affichées", fmt_int(len(result)), f"sur {fmt_int(len(df))}"),
                 ("Colonnes affichées", fmt_int(len(shown_cols)), f"sur {fmt_int(df.shape[1])}"),
                 ("Filtres actifs", str(len(st.session_state["explore_filters"])) + (" + recherche" if search else ""), None)])
        show_table(result, key="explore_filtered", export_name="donnees_filtrees", digits=None, hide_index=False,
                   max_rows=5000)

    with tab_info:
        kpi_row([("Dimensions", f"{fmt_int(df.shape[0])} × {fmt_int(df.shape[1])}", "lignes × colonnes"),
                 ("Cellules", fmt_int(df.size), None),
                 ("Mémoire", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} Mo", None)])
        st.markdown("**Types des variables (pandas) et valeurs non nulles**")
        info = dtypes_table(df)
        info["Type statistique"] = [schemas[c].stat_type.value if c in schemas else "" for c in info["Variable"]]
        show_table(info, key="explore_info", export_name="informations_variables")
