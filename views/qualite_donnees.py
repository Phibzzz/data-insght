"""Qualité des données : score transparent, audit, valeurs atypiques et règles de validité."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components.charts import show_chart
from components.data_tables import show_table
from components.feedback import flash, guarded, require_data, show_flash
from components.kpi_cards import kpi_row, page_header
from config.settings import SETTINGS
from core import visualization_engine as viz
from core.quality_engine import METHODOLOGY, SEVERITY_ORDER, SUBSCORE_LABELS, ValidityRule, iqr_outliers, zscore_outliers
from core.schema_detector import Family
from utils import state
from utils.helpers import fmt_int, fmt_num

page_header("Qualité des données", "Audit automatique, score documenté et recommandations de correction.")
show_flash()
if not require_data():
    st.stop()

with guarded("L'audit de qualité"):
    b = state.get_bundle()
    q = b.quality
    df = b.typed_df

    c1, c2 = st.columns([1, 2])
    with c1:
        show_chart(viz.quality_gauge(q.global_score), "q_gauge", export=False)
        st.caption(f"Appréciation : **{q.grade}**. Indicateur heuristique, non universel : il dépend des règles "
                   "et pondérations ci-contre.")
    with c2:
        sub = pd.DataFrame([{"Composante": SUBSCORE_LABELS[k], "Score (/100)": v, "Pondération": q.weights[k]}
                            for k, v in q.sub_scores.items()])
        sub["Contribution"] = (sub["Score (/100)"] * sub["Pondération"]).round(2)
        show_table(sub, key="q_sub", column_config={
            "Score (/100)": st.column_config.ProgressColumn("Score (/100)", min_value=0, max_value=100, format="%.1f"),
            "Pondération": st.column_config.NumberColumn("Pondération", format="%.2f"),
        })
        st.page_link("views/parametres.py", label="Modifier les pondérations", icon=":material/tune:")

    kpi_row([(k, f"{fmt_num(v, 2)} %", None) for k, v in q.indicators.items()], columns=5)

    t_issues, t_cols, t_out, t_rules, t_method = st.tabs(
        ["Problèmes détectés", "Qualité par variable", "Valeurs atypiques", "Règles de validité", "Méthodologie"])

    with t_issues:
        issues = q.issues_table()
        if issues.empty:
            st.success("Aucun problème détecté par les contrôles automatiques.", icon=":material/check_circle:")
        else:
            counts = issues["Sévérité"].value_counts()
            kpi_row([(s.capitalize(), str(int(counts.get(s, 0))), None) for s in SEVERITY_ORDER])
            sev = st.multiselect("Sévérités", list(SEVERITY_ORDER), default=list(SEVERITY_ORDER))
            show_table(issues[issues["Sévérité"].isin(sev)], key="q_issues", export_name="audit_qualite")

    with t_cols:
        show_table(q.column_scores, key="q_cols", export_name="qualite_par_variable", column_config={
            "Score (/100)": st.column_config.ProgressColumn("Score (/100)", min_value=0, max_value=100, format="%.1f")})

    with t_out:
        st.info("Une valeur atypique n'est **pas nécessairement une erreur** : elle peut refléter un phénomène réel, "
                "une sous-population ou une distribution asymétrique. Vérifiez avant toute correction.",
                icon=":material/info:")
        show_table(q.outliers_table(), key="q_out", export_name="valeurs_atypiques")
        nums = [o.column for o in q.outliers]
        if nums:
            c1, c2, c3 = st.columns(3)
            var = c1.selectbox("Variable", nums, key="out_var")
            method = c2.radio("Méthode", ["IQR", "Z-score"], horizontal=True)
            if method == "IQR":
                factor = c3.number_input("Facteur k (IQR)", 0.5, 5.0, float(SETTINGS.stats.iqr_factor), 0.5)
                mask = iqr_outliers(df[var], factor)
            else:
                thr = c3.number_input("Seuil |z|", 1.0, 6.0, float(SETTINGS.stats.zscore_threshold), 0.5)
                mask = zscore_outliers(df[var], thr)
            st.caption(f"{fmt_int(mask.sum())} valeur(s) atypique(s) selon cette méthode.")
            show_chart(viz.boxplot(df, var, points="outliers"), f"out_box_{var}")
            if mask.any():
                show_table(df[mask], key="out_rows", export_name=f"atypiques_{var}", max_rows=500, digits=None,
                           hide_index=False)

    with t_rules:
        st.caption("Définissez des règles métier : elles alimentent le taux de validité et le score global.")
        rules: list[ValidityRule] = st.session_state["rules"]
        with st.form("rule_form", clear_on_submit=True):
            col = st.selectbox("Colonne", list(df.columns))
            c1, c2 = st.columns(2)
            vmin = c1.text_input("Valeur minimale (numérique)")
            vmax = c2.text_input("Valeur maximale (numérique)")
            allowed = st.text_input("Valeurs autorisées (séparées par ;)")
            regex = st.text_input("Motif (expression régulière, ex. ^CMD-\\d{6}$)")
            not_null = st.checkbox("Valeur obligatoire (non manquante)")
            if st.form_submit_button("Ajouter la règle", icon=":material/add:"):
                try:
                    rule = ValidityRule(
                        column=col,
                        min_value=float(vmin.replace(",", ".")) if vmin.strip() else None,
                        max_value=float(vmax.replace(",", ".")) if vmax.strip() else None,
                        allowed_values=[v.strip() for v in allowed.split(";") if v.strip()] or None,
                        regex=regex.strip() or None, not_null=not_null,
                    )
                except ValueError:
                    st.error("Les bornes doivent être numériques.")
                else:
                    if any([rule.min_value is not None, rule.max_value is not None, rule.allowed_values, rule.regex,
                            rule.not_null]):
                        rules.append(rule)
                        flash(f"Règle ajoutée : {rule.describe()}")
                        st.rerun()
                    else:
                        st.warning("La règle est vide.")
        for i, rule in enumerate(rules):
            r1, r2 = st.columns([6, 1])
            n_viol = int(rule.violations(df[rule.column]).sum()) if rule.column in df.columns else 0
            r1.markdown(f"- {rule.describe()} — **{fmt_int(n_viol)}** violation(s)")
            if r2.button("Supprimer", key=f"rule_rm_{i}", icon=":material/delete:"):
                rules.pop(i)
                st.rerun()
        if not rules:
            st.caption("Aucune règle définie : le taux de validité repose uniquement sur les contrôles automatiques.")

    with t_method:
        st.markdown(METHODOLOGY)
        fam_counts = pd.Series([s.family.value for s in b.schemas]).value_counts()
        st.caption("Familles de variables auditées : " + ", ".join(f"{k} ({v})" for k, v in fam_counts.items()))
        st.caption(f"Valeurs atypiques : règle de Tukey (k = {SETTINGS.stats.iqr_factor:g}) et z-score "
                   f"(|z| > {SETTINGS.stats.zscore_threshold:g}). Modalités rares : < "
                   f"{100 * SETTINGS.stats.rare_category_share:g} %. Variables analysées : "
                   f"{sum(1 for s in b.schemas if s.family == Family.NUMERIC)} quantitatives.")
