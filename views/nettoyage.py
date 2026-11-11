"""Nettoyage : transformations sur une copie de travail, avant/après, historique et export."""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from components.data_tables import show_table
from components.feedback import flash, guarded, require_data, show_flash
from components.kpi_cards import kpi_row, page_header
from core.cleaning_engine import FILTER_OPERATORS
from core.profiling_engine import count_duplicates
from utils import state
from utils.exporters import dataframe_to_json_bytes, safe_filename, to_csv_bytes, to_excel_bytes, to_json_bytes, to_parquet_bytes
from utils.helpers import fmt_int

page_header("Nettoyage des données", "Toutes les transformations s'appliquent à une copie de travail : l'original reste intact.")
show_flash()
if not require_data():
    st.stop()

session = state.cleaning()
work = session.current
original = session.original


def run(op_name: str, /, **params: Any) -> None:
    """Applique une opération, journalise et bascule l'analyse sur les données nettoyées."""
    with guarded("L'opération"):
        op = session.apply(op_name, **params)
        state.bump_version()
        st.session_state["use_cleaned"] = True
        flash(f"{op.description} ({op.rows_before} → {op.rows_after} lignes, {op.cols_before} → {op.cols_after} colonnes)")
        st.rerun()


cols = list(work.columns)
num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(work[c]) and not pd.api.types.is_bool_dtype(work[c])]
num_like = num_cols + [c for c in cols if c not in num_cols and pd.to_numeric(
    work[c].dropna().astype(str).str.replace(",", ".", regex=False).head(200), errors="coerce").notna().mean() > 0.9]
text_cols = [c for c in cols if pd.api.types.is_object_dtype(work[c]) or pd.api.types.is_string_dtype(work[c])]

tabs = st.tabs(["Doublons", "Valeurs manquantes", "Types", "Colonnes", "Texte", "Filtrer", "Valeurs atypiques",
                "Nouvelle variable"])

with tabs[0]:
    n_dup = count_duplicates(work)
    st.markdown(f"**{fmt_int(n_dup)}** ligne(s) dupliquée(s) dans la copie de travail.")
    subset = st.multiselect("Colonnes à comparer (vide = toutes)", cols, key="dup_subset")
    keep = st.radio("Conserver", ["first", "last", "none"], horizontal=True,
                    format_func=lambda k: {"first": "la première occurrence", "last": "la dernière occurrence",
                                           "none": "aucune (supprimer toutes les occurrences)"}[k])
    if st.button("Supprimer les doublons", type="primary", key="btn_dup"):
        run("drop_duplicates", subset=subset or None, keep=keep)

with tabs[1]:
    miss = work.isna().sum()
    miss = miss[miss > 0]
    if miss.empty:
        st.success("Aucune valeur manquante dans la copie de travail.")
    else:
        st.dataframe(pd.DataFrame({"Variable": miss.index, "Manquants": miss.values,
                                   "%": (100 * miss.values / len(work)).round(2)}), hide_index=True)
    strategy = st.selectbox("Traitement", ["drop_rows", "drop_columns", "mean", "median", "mode", "constant"],
                            format_func=lambda s: {"drop_rows": "Supprimer les lignes", "drop_columns": "Supprimer les colonnes",
                                                   "mean": "Remplacer par la moyenne", "median": "Remplacer par la médiane",
                                                   "mode": "Remplacer par le mode", "constant": "Remplacer par une valeur"}[s])
    if strategy == "drop_rows":
        target = st.multiselect("Colonnes concernées (vide = toutes)", list(miss.index) or cols)
        how = st.radio("Supprimer si", ["any", "all"], horizontal=True,
                       format_func=lambda h: "au moins une valeur manque" if h == "any" else "toutes les valeurs manquent")
        if st.button("Appliquer", type="primary", key="btn_na_rows"):
            run("drop_missing_rows", columns=target or None, how=how)
    elif strategy == "drop_columns":
        thr = st.slider("Supprimer les colonnes dont le taux de manquants dépasse (%)", 0, 100, 50)
        if st.button("Appliquer", type="primary", key="btn_na_cols"):
            run("drop_missing_columns", threshold=thr / 100)
    else:
        choices = (num_like if strategy in ("mean", "median") else cols)
        target = st.multiselect("Colonnes", [c for c in choices if c in miss.index] or choices)
        value = st.text_input("Valeur de remplacement") if strategy == "constant" else None
        if strategy in ("mean", "median"):
            st.caption("Moyenne/médiane : colonnes numériques uniquement. La moyenne est sensible aux valeurs atypiques.")
        if st.button("Appliquer", type="primary", key="btn_impute"):
            run("impute", columns=target, strategy=strategy, value=value)

with tabs[2]:
    c1, c2, c3 = st.columns(3)
    col = c1.selectbox("Colonne", cols, key="conv_col")
    target = c2.selectbox("Nouveau type", ["numeric", "integer", "datetime", "string", "category", "boolean"],
                          format_func=lambda t: {"numeric": "Numérique (réel)", "integer": "Entier", "datetime": "Date",
                                                 "string": "Texte", "category": "Catégorie", "boolean": "Booléen"}[t])
    fmt = c3.text_input("Format de date (optionnel)", placeholder="%d/%m/%Y") if target == "datetime" else None
    st.caption(f"Type actuel : `{work[col].dtype}`. Les valeurs non convertibles deviennent manquantes (nombre affiché).")
    if st.button("Convertir", type="primary", key="btn_conv"):
        run("convert_type", column=col, target=target, date_format=fmt or None)

with tabs[3]:
    st.markdown("**Renommer des colonnes**")
    editor = st.data_editor(pd.DataFrame({"Nom actuel": cols, "Nouveau nom": cols}), disabled=["Nom actuel"],
                            hide_index=True, key="rename_editor")
    if st.button("Appliquer les renommages", key="btn_rename"):
        mapping = {a: b for a, b in zip(editor["Nom actuel"], editor["Nouveau nom"]) if str(b).strip() and a != b}
        run("rename_columns", mapping=mapping)
    st.markdown("**Supprimer des colonnes**")
    to_drop = st.multiselect("Colonnes à supprimer", cols, key="drop_cols")
    if st.button("Supprimer", key="btn_drop", disabled=not to_drop):
        run("drop_columns", columns=to_drop)

with tabs[4]:
    if not text_cols:
        st.info("Aucune colonne texte.")
    else:
        target = st.multiselect("Colonnes texte", text_cols, default=text_cols[:1], key="norm_cols")
        c1, c2, c3 = st.columns(3)
        case = c1.selectbox("Casse", [None, "lower", "upper", "title", "capitalize"],
                            format_func=lambda c: {None: "Inchangée", "lower": "minuscules", "upper": "MAJUSCULES",
                                                   "title": "Première Lettre Des Mots", "capitalize": "Première lettre"}[c])
        strip = c2.checkbox("Supprimer les espaces superflus", value=True)
        accents = c3.checkbox("Supprimer les accents")
        empty = st.checkbox("Chaînes vides → valeurs manquantes", value=True)
        if st.button("Normaliser", type="primary", key="btn_norm", disabled=not target):
            run("normalize_text", columns=target, strip=strip, case=case, remove_accents=accents,
                collapse_spaces=strip, empty_as_missing=empty)
        st.markdown("**Remplacer une valeur**")
        r1, r2, r3 = st.columns(3)
        rcol = r1.selectbox("Colonne", cols, key="rep_col")
        old = r2.text_input("Valeur actuelle", key="rep_old")
        new = r3.text_input("Nouvelle valeur (vide = manquant)", key="rep_new")
        if st.button("Remplacer", key="btn_rep", disabled=not old):
            run("replace_values", column=rcol, old=old, new=new or None)

with tabs[5]:
    c1, c2, c3, c4 = st.columns([2, 1.4, 1.5, 1.5])
    fcol = c1.selectbox("Colonne", cols, key="flt_col")
    op = c2.selectbox("Condition", FILTER_OPERATORS, key="flt_op")
    v1 = c3.text_input("Valeur", key="flt_v1", disabled=op in ("est manquant", "n'est pas manquant"),
                       help="Pour « dans la liste », séparez les valeurs par ;")
    v2 = c4.text_input("Valeur max (entre)", key="flt_v2", disabled=op != "entre")
    st.caption("Les lignes qui **satisfont** la condition sont conservées.")
    if st.button("Filtrer", type="primary", key="btn_flt"):
        run("filter_rows", column=fcol, op=op, value=v1 or None, value2=v2 or None)

with tabs[6]:
    if not num_cols:
        st.info("Aucune colonne numérique (convertissez d'abord le type si nécessaire).")
    else:
        target = st.multiselect("Colonnes", num_cols, key="out_cols")
        c1, c2, c3 = st.columns(3)
        method = c1.radio("Méthode", ["iqr", "zscore"], format_func=lambda m: "IQR (Tukey)" if m == "iqr" else "Z-score")
        param = c2.number_input("k (IQR)" if method == "iqr" else "Seuil |z|", 0.5, 10.0, 1.5 if method == "iqr" else 3.0, 0.5)
        action = c3.radio("Traitement", ["cap", "nan", "remove"],
                          format_func=lambda a: {"cap": "Écrêter aux bornes (winsorisation)",
                                                 "nan": "Remplacer par manquant", "remove": "Supprimer les lignes"}[a])
        st.caption("Rappel : une valeur atypique n'est pas forcément une erreur. Préférez l'écrêtage à la suppression "
                   "si la valeur est plausible.")
        if st.button("Traiter", type="primary", key="btn_out", disabled=not target):
            run("treat_outliers", columns=target, method=method, action=action,
                factor=param if method == "iqr" else 1.5, threshold=param if method == "zscore" else 3.0)

with tabs[7]:
    name = st.text_input("Nom de la nouvelle variable", key="new_name")
    kind = st.radio("Mode de création", ["expression", "bin", "quantile_bin", "date_part"], horizontal=True,
                    format_func=lambda k: {"expression": "Formule", "bin": "Classes (bornes)",
                                           "quantile_bin": "Classes (quantiles)", "date_part": "Extraction de date"}[k])
    if kind == "expression":
        expr = st.text_input("Formule", placeholder="`prix_unitaire` * quantite")
        st.caption("Colonnes entre accents graves si leur nom contient des espaces ; opérateurs + - * / ** % // ; "
                   "fonctions log, log10, log1p, sqrt, abs, exp, round. Aucun code arbitraire n'est exécuté.")
        if st.button("Créer", type="primary", key="btn_new_expr"):
            run("create_variable", name=name, kind="expression", expression=expr)
    elif kind in ("bin", "quantile_bin"):
        src = st.selectbox("Variable source", num_like or cols, key="bin_src")
        if kind == "bin":
            edges = st.text_input("Bornes (séparées par ;) ou nombre de classes égales", "4")
            labels = st.text_input("Étiquettes (optionnel, séparées par ;)")
            if st.button("Créer", type="primary", key="btn_new_bin"):
                with guarded("La discrétisation"):
                    parts = [p.strip().replace(",", ".") for p in edges.split(";") if p.strip()]
                    bins: Any = int(parts[0]) if len(parts) == 1 else [float(p) for p in parts]
                    lab = [x.strip() for x in labels.split(";") if x.strip()] or None
                    run("create_variable", name=name, kind="bin", source=src, bins=bins, labels=lab)
        else:
            q = st.number_input("Nombre de classes", 2, 20, 4)
            if st.button("Créer", type="primary", key="btn_new_q"):
                run("create_variable", name=name, kind="quantile_bin", source=src, bins=int(q))
    else:
        src = st.selectbox("Variable de date", cols, key="date_src")
        part = st.selectbox("Élément", ["year", "month", "day", "weekday", "quarter", "week", "hour"],
                            format_func=lambda p: {"year": "Année", "month": "Mois", "day": "Jour", "weekday": "Jour de semaine (1 = lundi)",
                                                   "quarter": "Trimestre", "week": "Semaine ISO", "hour": "Heure"}[p])
        if st.button("Créer", type="primary", key="btn_new_date"):
            run("create_variable", name=name, kind="date_part", source=src, part=part)

# --- Avant / après, historique, export ---
st.divider()
st.subheader("Avant / après")
kpi_row([
    ("Lignes", f"{fmt_int(len(original))} → {fmt_int(len(work))}", f"{len(work) - len(original):+d}"),
    ("Colonnes", f"{original.shape[1]} → {work.shape[1]}", f"{work.shape[1] - original.shape[1]:+d}"),
    ("Valeurs manquantes", f"{fmt_int(original.isna().sum().sum())} → {fmt_int(work.isna().sum().sum())}", None),
    ("Doublons", f"{fmt_int(count_duplicates(original))} → {fmt_int(count_duplicates(work))}", None),
])
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Original (non modifié)**")
    show_table(original.head(10), key="before", digits=None)
with c2:
    st.markdown("**Copie de travail**")
    show_table(work.head(10), key="after", digits=None)

st.subheader("Historique des opérations")
if not session.history:
    st.caption("Aucune opération appliquée.")
else:
    show_table(session.history_table(), key="history", digits=None)
    h1, h2, h3 = st.columns(3)
    if h1.button("Annuler la dernière opération", icon=":material/undo:"):
        with guarded("L'annulation"):
            last = session.undo()
            state.bump_version()
            flash(f"Opération annulée : {last.description if last else ''}", "info")
            st.rerun()
    if h2.button("Réinitialiser la copie de travail", icon=":material/restart_alt:"):
        session.reset()
        state.bump_version()
        st.session_state["use_cleaned"] = False
        flash("Copie de travail réinitialisée à partir de l'original.", "info")
        st.rerun()
    h3.download_button("Historique (JSON)", lambda: to_json_bytes(session.history_dicts()), "historique_nettoyage.json",
                       "application/json", icon=":material/download:")

st.subheader("Exporter le dataset nettoyé")
base = safe_filename(state.meta().filename.rsplit(".", 1)[0]) + "_nettoye"
e1, e2, e3, e4 = st.columns(4)
e1.download_button("CSV", lambda: to_csv_bytes(work), f"{base}.csv", "text/csv", icon=":material/download:", width="stretch")
e2.download_button("Excel", lambda: to_excel_bytes({"Données nettoyées": work}), f"{base}.xlsx",
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", icon=":material/download:",
                   width="stretch", disabled=len(work) > 1_000_000)
e3.download_button("Parquet", lambda: to_parquet_bytes(work), f"{base}.parquet", "application/octet-stream",
                   icon=":material/download:", width="stretch")
e4.download_button("JSON", lambda: dataframe_to_json_bytes(work), f"{base}.json", "application/json",
                   icon=":material/download:", width="stretch")
st.caption("Les exports CSV et Excel neutralisent les cellules commençant par =, +, -, @ (protection contre "
           "l'injection de formules).")
