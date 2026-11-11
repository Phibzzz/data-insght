"""Analyse multivariée : corrélations, covariance, tableaux croisés et comparaisons de groupes."""
from __future__ import annotations

import streamlit as st

from components.charts import show_chart
from components.data_tables import show_table
from components.feedback import guarded, require_data
from components.kpi_cards import kpi_row, page_header
from core import visualization_engine as viz
from core.multivariate_engine import (
    METHOD_INFO,
    contingency,
    correlation_matrix,
    cramers_v_matrix,
    group_comparison,
)
from utils import state
from utils.helpers import fmt_num

page_header("Analyse multivariée", "Relations entre variables, en ne combinant que des types compatibles.")
if not require_data():
    st.stop()

with guarded("L'analyse multivariée"):
    b = state.get_bundle()
    df = b.typed_df
    nums, cats = b.numeric_cols, [c for c in b.categorical_cols if df[c].nunique() >= 2]
    st.caption("Identifiants, constantes et textes libres sont exclus. Les valeurs manquantes sont exclues par paire : "
               "l'effectif réellement utilisé (n) est indiqué.")

    t_corr, t_cov, t_cross, t_group, t_assoc = st.tabs(
        ["Corrélations", "Covariance", "Tableaux croisés", "Quantitatif × qualitatif", "Fortes associations"])

    with t_corr:
        if len(nums) < 2:
            st.info("Au moins deux variables quantitatives sont nécessaires.")
        else:
            st.info(f"**Méthode recommandée** : {b.correlation_method_reason}", icon=":material/lightbulb:")
            methods = list(METHOD_INFO)
            current = b.correlation.method if b.correlation else "pearson"
            c1, c2 = st.columns([1, 1])
            method = c1.radio("Méthode", methods, index=methods.index(current), horizontal=True,
                              format_func=lambda m: METHOD_INFO[m]["label"],
                              help="Le choix s'applique aussi aux interprétations et aux rapports.")
            if method != current:
                st.session_state["corr_method"] = method
                st.rerun()
            selected = c2.multiselect("Variables", nums, default=nums[:15])
            info = METHOD_INFO[method]
            st.markdown(f"**{info['label']}** — {info['mesure']}  \n*Conditions :* {info['conditions']}")
            if len(selected) >= 2:
                corr = b.correlation if (b.correlation and method == b.correlation.method and
                                         selected == list(b.correlation.matrix.columns)) else \
                    correlation_matrix(df, selected, method)
                if corr.excluded:
                    st.caption("Variables exclues (moins de deux valeurs distinctes) : " + ", ".join(corr.excluded))
                show_chart(viz.correlation_heatmap(corr.matrix, f"Matrice de corrélation ({info['label']})"), "corr_heat",
                           height=max(420, 38 * len(corr.matrix) + 160))
                threshold = st.slider("Afficher les paires avec |r| ≥", 0.0, 1.0, 0.3, 0.05)
                pairs = corr.pairs(threshold)
                show_table(pairs, key="corr_pairs", export_name=f"correlations_{method}", column_config={
                    "p-value": st.column_config.NumberColumn("p-value", format="%.2e")})
                st.caption("p-value : probabilité d'observer une corrélation au moins aussi forte si la corrélation réelle "
                           "était nulle. Corrélation n'implique pas causalité.")
                if len(pairs):
                    top = pairs.iloc[0]
                    show_chart(viz.scatter(df, top["Variable 1"], top["Variable 2"]), "corr_top_scatter")

    with t_cov:
        if b.covariance is None:
            st.info("Au moins deux variables quantitatives sont nécessaires.")
        else:
            st.caption("Covariance d'échantillon (n − 1). Elle dépend des unités : pour comparer des intensités, "
                       "utilisez les corrélations.")
            show_table(b.covariance.reset_index(names="Variable"), key="cov", export_name="covariance", digits=6)

    with t_cross:
        if len(cats) < 2:
            st.info("Au moins deux variables qualitatives sont nécessaires.")
        else:
            c1, c2 = st.columns(2)
            row = c1.selectbox("Variable en ligne", cats, key="ct_row")
            col = c2.selectbox("Variable en colonne", [c for c in cats if c != row], key="ct_col")
            res = contingency(df, row, col)
            kpi_row([
                ("χ² d'indépendance", fmt_num(res.chi2), f"ddl = {res.dof}" if res.dof else None),
                ("p-value", f"{res.pvalue:.3g}" if res.pvalue is not None else "—",
                 ("dépendance significative" if res.pvalue < 0.05 else "pas de dépendance significative")
                 if res.pvalue is not None else None),
                ("V de Cramér", fmt_num(res.cramers_v, 3), "0 = indépendance · 1 = association parfaite"),
            ])
            if res.warning:
                st.warning(res.warning, icon=":material/warning:")
            view = st.segmented_control("Affichage", ["Effectifs", "% en ligne", "% en colonne", "% du total"],
                                        default="Effectifs")
            table = {"Effectifs": res.table, "% en ligne": res.row_pct, "% en colonne": res.col_pct,
                     "% du total": res.total_pct}[view or "Effectifs"]
            show_table(table.reset_index(), key="ct_table", export_name=f"contingence_{row}_{col}", digits=2)
            show_chart(viz.stacked_bar(res.table, row, col), "ct_stack")

    with t_group:
        if not nums or not cats:
            st.info("Une variable quantitative et une variable qualitative sont nécessaires.")
        else:
            c1, c2 = st.columns(2)
            num = c1.selectbox("Variable quantitative", nums, key="gc_num")
            grp = c2.selectbox("Variable de regroupement", cats, key="gc_grp")
            gc = group_comparison(df, num, grp)
            kpi_row([
                ("Groupes", str(gc.n_groups), None),
                ("η² (rapport de corrélation)", fmt_num(gc.eta_squared, 3), "part de variance expliquée par le groupe"),
                ("ANOVA (p)", f"{gc.anova_pvalue:.3g}" if gc.anova_pvalue is not None else "—", "hyp. normalité"),
                ("Kruskal-Wallis (p)", f"{gc.kruskal_pvalue:.3g}" if gc.kruskal_pvalue is not None else "—",
                 "non paramétrique"),
            ])
            show_table(gc.table, key="gc_table", export_name=f"{num}_par_{grp}")
            c1, c2 = st.columns(2)
            with c1:
                show_chart(viz.boxplot(df, num, grp), "gc_box")
            with c2:
                stat = st.radio("Statistique", ["mean", "median"], horizontal=True,
                                format_func=lambda s: {"mean": "Moyenne", "median": "Médiane"}[s])
                show_chart(viz.group_means_chart(df, num, grp, stat), "gc_bar")

    with t_assoc:
        st.caption("Associations fortes détectées automatiquement : |r| ≥ seuil (corrélations), V de Cramér ≥ 0,5 "
                   "significatif, η² ≥ 0,26.")
        show_table(b.associations, key="assoc", export_name="fortes_associations")
        if len(cats) >= 2:
            st.markdown("**Matrice des V de Cramér (variables qualitatives)**")
            vm = cramers_v_matrix(df, cats[:15])
            if len(vm) >= 2:
                show_chart(viz.generic_heatmap(vm, "Associations entre variables qualitatives (V de Cramér)",
                                               zmin=0, zmax=1, value_label="V"), "cramer")
