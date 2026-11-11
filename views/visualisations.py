"""Visualisations : suggestions automatiques et constructeur de graphiques personnalisables."""
from __future__ import annotations

import streamlit as st

from components.charts import show_chart
from components.feedback import guarded, require_data
from components.kpi_cards import page_header
from config.settings import SETTINGS
from core import visualization_engine as viz
from core.schema_detector import StatType
from utils import state

page_header("Visualisations", "Graphiques interactifs adaptés au type des variables (export PNG et HTML).")
if not require_data():
    st.stop()

with guarded("La génération des graphiques"):
    b = state.get_bundle()
    df = b.typed_df
    nums = b.numeric_cols
    cats = [c for c in b.categorical_cols if df[c].nunique() >= 1]
    dates = b.temporal_cols
    st.caption(f"Nuages de points limités à {SETTINGS.viz.max_points_scatter:,} points et matrices de nuages à "
               f"{SETTINGS.viz.max_rows_pairplot:,} lignes (échantillons aléatoires reproductibles).".replace(",", " "))

    t_auto, t_num, t_cat, t_time, t_multi = st.tabs(
        ["Suggestions automatiques", "Numériques", "Catégorielles", "Temporelles", "Multivariées"])

    with t_auto:
        suggestions = viz.suggest_charts(b.schemas)
        if not suggestions:
            st.info("Aucune suggestion : les variables détectées ne se prêtent pas à une représentation automatique.")
        cols = st.columns(2)
        for i, sug in enumerate(suggestions):
            with cols[i % 2]:
                show_chart(viz.render_suggestion(df, sug), f"auto_{i}")

    with t_num:
        if not nums:
            st.info("Aucune variable quantitative.")
        else:
            c1, c2, c3 = st.columns(3)
            var = c1.selectbox("Variable", nums, key="vn_var")
            kind = c2.selectbox("Graphique", ["Histogramme", "Courbe de densité", "Boîte à moustaches",
                                              "Violon", "Distribution cumulée (ECDF)"], key="vn_kind")
            group = c3.selectbox("Couleur / groupe (optionnel)", [None] + [c for c in cats if df[c].nunique() <= 20],
                                 key="vn_group", format_func=lambda v: "Aucun" if v is None else v)
            if kind == "Histogramme":
                p1, p2 = st.columns(2)
                bins = p1.slider("Nombre de classes", 5, 150, 30)
                log_y = p2.checkbox("Échelle logarithmique (effectifs)")
                fig = viz.histogram(df, var, bins=bins, color=group, log_y=log_y, marginal=None if group else "box")
            elif kind == "Courbe de densité":
                fig = viz.density(df, var, group)
            elif kind == "Boîte à moustaches":
                fig = viz.boxplot(df, var, group)
            elif kind == "Violon":
                fig = viz.violin(df, var, group)
            else:
                fig = viz.ecdf(df, var, group)
            show_chart(fig, "vn_chart")

    with t_cat:
        if not cats:
            st.info("Aucune variable qualitative.")
        else:
            c1, c2 = st.columns(2)
            var = c1.selectbox("Variable", cats, key="vc_var")
            kind = c2.selectbox("Graphique", ["Diagramme en barres", "Diagramme circulaire", "Diagramme de Pareto"],
                                key="vc_kind")
            n_cat = int(df[var].nunique())
            if kind == "Diagramme en barres":
                p1, p2, p3 = st.columns(3)
                top = p1.slider("Modalités affichées", 2, max(3, min(100, n_cat)), min(30, max(2, n_cat)))
                horizontal = p2.checkbox("Barres horizontales")
                percent = p3.checkbox("En pourcentage")
                sch = next(s for s in b.schemas if s.name == var)
                order = None
                if sch.stat_type == StatType.ORDINAL and var in b.univariate.categorical:
                    order = list(b.univariate.categorical[var].frequencies["Modalité"])
                fig = viz.bar_chart(df, var, top_n=top, horizontal=horizontal, percent=percent, order=order)
            elif kind == "Diagramme circulaire":
                if n_cat > SETTINGS.viz.max_pie_categories:
                    st.warning(f"{n_cat} modalités : un diagramme circulaire est peu lisible au-delà de "
                               f"{SETTINGS.viz.max_pie_categories}. Les modalités restantes sont regroupées dans « (autres) ».")
                fig = viz.pie_chart(df, var)
            else:
                fig = viz.pareto(df, var)
            show_chart(fig, "vc_chart")

    with t_time:
        if not dates:
            st.info("Aucune variable de date.")
        else:
            c1, c2 = st.columns(2)
            var = c1.selectbox("Variable de date", dates, key="vt_var")
            kind = c2.selectbox("Graphique", ["Série temporelle", "Répartition temporelle", "Heatmap calendaire"],
                                key="vt_kind")
            if kind == "Série temporelle":
                p1, p2, p3, p4 = st.columns(4)
                value = p1.selectbox("Valeur", [None] + nums, format_func=lambda v: "Nombre d'observations" if v is None else v)
                agg = p2.selectbox("Agrégation", ["sum", "mean", "median", "count"],
                                   format_func=lambda a: {"sum": "Somme", "mean": "Moyenne", "median": "Médiane",
                                                          "count": "Nombre"}[a], disabled=value is None)
                freq = p3.selectbox("Pas de temps", list(viz.FREQ_LABELS), index=2, format_func=lambda f: viz.FREQ_LABELS[f])
                rolling = p4.number_input("Moyenne mobile (périodes)", 0, 60, 0)
                fig = viz.time_series(df, var, value, agg, freq, rolling or None)
            elif kind == "Répartition temporelle":
                by = st.radio("Par", ["year", "month", "weekday", "hour"], horizontal=True,
                              format_func=lambda v: {"year": "Année", "month": "Mois", "weekday": "Jour de la semaine",
                                                     "hour": "Heure"}[v])
                fig = viz.temporal_distribution(df, var, by)
            else:
                years = sorted(df[var].dropna().dt.year.unique().tolist())
                p1, p2 = st.columns(2)
                year = p1.selectbox("Année", years, index=len(years) - 1) if years else None
                value = p2.selectbox("Valeur", [None] + nums, key="cal_val",
                                     format_func=lambda v: "Nombre d'observations" if v is None else f"Somme de {v}")
                fig = viz.calendar_heatmap(df, var, value, year)
            show_chart(fig, "vt_chart")

    with t_multi:
        kind = st.selectbox("Graphique", ["Nuage de points", "Matrice de corrélation", "Matrice de nuages (pair plot)",
                                          "Comparaison par groupe"], key="vm_kind")
        if kind == "Nuage de points":
            if len(nums) < 2:
                st.info("Au moins deux variables quantitatives sont nécessaires.")
            else:
                c1, c2, c3, c4 = st.columns(4)
                x = c1.selectbox("Axe X", nums, key="sc_x")
                y = c2.selectbox("Axe Y", [n for n in nums if n != x], key="sc_y")
                color = c3.selectbox("Couleur", [None] + [c for c in cats if df[c].nunique() <= 20], key="sc_c",
                                     format_func=lambda v: "Aucune" if v is None else v)
                trend = c4.checkbox("Droite de régression", value=True)
                show_chart(viz.scatter(df, x, y, color, trend), "vm_scatter")
        elif kind == "Matrice de corrélation":
            if b.correlation is None:
                st.info("Au moins deux variables quantitatives sont nécessaires.")
            else:
                show_chart(viz.correlation_heatmap(b.correlation.matrix,
                                                   f"Matrice de corrélation ({b.correlation.method})"), "vm_corr",
                           height=max(420, 38 * len(b.correlation.matrix) + 160))
        elif kind == "Matrice de nuages (pair plot)":
            if len(nums) < 2:
                st.info("Au moins deux variables quantitatives sont nécessaires.")
            else:
                sel = st.multiselect(f"Variables (max. {SETTINGS.viz.max_vars_pairplot})", nums,
                                     default=nums[: min(4, len(nums))], max_selections=SETTINGS.viz.max_vars_pairplot)
                color = st.selectbox("Couleur", [None] + [c for c in cats if df[c].nunique() <= 12], key="pp_c",
                                     format_func=lambda v: "Aucune" if v is None else v)
                if len(sel) >= 2:
                    show_chart(viz.pair_plot(df, sel, color), "vm_pair")
        else:
            if not nums or not cats:
                st.info("Une variable quantitative et une variable qualitative sont nécessaires.")
            else:
                c1, c2, c3 = st.columns(3)
                num = c1.selectbox("Variable quantitative", nums, key="cmp_n")
                grp = c2.selectbox("Groupe", cats, key="cmp_g")
                form = c3.selectbox("Forme", ["Boîtes", "Violons", "Moyennes (IC 95 %)", "Médianes"])
                if form == "Boîtes":
                    fig = viz.boxplot(df, num, grp)
                elif form == "Violons":
                    fig = viz.violin(df, num, grp)
                else:
                    fig = viz.group_means_chart(df, num, grp, "mean" if form.startswith("Moy") else "median")
                show_chart(fig, "vm_cmp")
