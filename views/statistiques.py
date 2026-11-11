"""Statistiques descriptives univariées : quantitatives, qualitatives et temporelles."""
from __future__ import annotations

import streamlit as st

from components.charts import show_chart
from components.data_tables import show_table
from components.feedback import guarded, require_data
from components.kpi_cards import kpi_row, page_header
from core import visualization_engine as viz
from utils import state
from utils.helpers import fmt_int, fmt_num, fmt_pct

page_header("Statistiques descriptives", "Indicateurs calculés selon le type de chaque variable.")
if not require_data():
    st.stop()

with guarded("Le calcul des statistiques"):
    b = state.get_bundle()
    uni = b.univariate
    df = b.typed_df

    with st.expander("Conventions de calcul", icon=":material/info:"):
        st.markdown(
            "- **Variance et écart-type** : estimateurs d'échantillon (dénominateur n − 1).\n"
            "- **Quartiles** : interpolation linéaire (méthode par défaut de NumPy/pandas).\n"
            "- **Coefficient de variation** : écart-type / |moyenne| (non défini si la moyenne est nulle).\n"
            "- **Asymétrie (skewness)** et **aplatissement** corrigés du biais ; l'aplatissement est un *excès* de "
            "kurtosis (0 pour une loi normale).\n"
            "- **IC 95 %** de la moyenne : loi de Student. **MAD** : écart absolu médian.\n"
            "- **Normalité** : Shapiro-Wilk (n ≤ 5 000) ou D'Agostino-Pearson ; un rejet sur un grand échantillon "
            "peut traduire un écart minime.\n"
            "- Les valeurs manquantes et infinies sont exclues ; leur nombre est indiqué."
        )

    tabs = st.tabs([f"Quantitatives ({len(uni.numeric)})", f"Qualitatives ({len(uni.categorical)})",
                    f"Temporelles ({len(uni.temporal)})"])

    with tabs[0]:
        if not uni.numeric:
            st.info("Aucune variable quantitative détectée.")
        else:
            table = uni.numeric_table()
            default = ["Variable", "Valides", "Manquants", "Moyenne", "Médiane", "Mode", "Minimum", "Maximum", "Étendue",
                       "Variance", "Écart-type", "Q1", "Q2 (médiane)", "Q3", "IQR", "Coef. de variation", "Asymétrie",
                       "Aplatissement (excès)"]
            chosen = st.multiselect("Indicateurs affichés", list(table.columns), default=default)
            show_table(table[chosen or default], key="num_stats", export_name="statistiques_quantitatives")

            st.subheader("Détail d'une variable")
            var = st.selectbox("Variable quantitative", list(uni.numeric), key="num_var")
            s = uni.numeric[var]
            kpi_row([
                ("Moyenne", fmt_num(s.mean), f"IC95 [{fmt_num(s.ci95_low)} ; {fmt_num(s.ci95_high)}]"),
                ("Médiane", fmt_num(s.median), f"Q1 {fmt_num(s.q1)} · Q3 {fmt_num(s.q3)}"),
                ("Écart-type", fmt_num(s.std), f"CV {fmt_pct(s.cv) if s.cv is not None else 'n.d.'}"),
                ("Étendue", fmt_num(s.range), f"{fmt_num(s.minimum)} → {fmt_num(s.maximum)}"),
                ("Asymétrie", fmt_num(s.skewness), f"aplatissement {fmt_num(s.kurtosis)}"),
                ("Valides", fmt_int(s.valid), f"{fmt_int(s.missing)} manquant(s)"),
            ], columns=6)
            if s.normality_pvalue is not None:
                verdict = "rejetée" if s.normality_pvalue < 0.05 else "non rejetée"
                st.caption(f"Normalité ({s.normality_test}) : p = {s.normality_pvalue:.3g} → hypothèse de normalité "
                           f"{verdict} au seuil de 5 %.")
            c1, c2 = st.columns(2)
            with c1:
                show_chart(viz.histogram(df, var, marginal=None), f"stat_hist_{var}")
            with c2:
                show_chart(viz.boxplot(df, var), f"stat_box_{var}")

    with tabs[1]:
        if not uni.categorical:
            st.info("Aucune variable qualitative détectée.")
        else:
            show_table(uni.categorical_table(), key="cat_stats", export_name="statistiques_qualitatives")
            st.subheader("Tableau de fréquences")
            var = st.selectbox("Variable qualitative", list(uni.categorical), key="cat_var")
            s = uni.categorical[var]
            kpi_row([
                ("Effectif total", fmt_int(s.count), f"{fmt_int(s.valid)} valides"),
                ("Modalités", fmt_int(s.n_categories), "ordonnées" if s.ordered else "non ordonnées"),
                ("Modalité dominante", str(s.mode), f"{fmt_int(s.mode_count)} obs. ({fmt_pct(s.mode_share)})"),
                ("Manquants", fmt_int(s.missing), None),
            ])
            if not s.ordered:
                st.caption("Fréquences cumulées non calculées : l'ordre des modalités n'est pas significatif "
                           "(corrigez le type en « ordinale » dans Profil des variables si nécessaire).")
            c1, c2 = st.columns([1, 1])
            with c1:
                show_table(s.frequencies, key=f"freq_{var}", export_name=f"frequences_{var}", digits=4)
            with c2:
                order = list(s.frequencies["Modalité"]) if s.ordered else None
                show_chart(viz.bar_chart(df, var, order=order), f"stat_bar_{var}")

    with tabs[2]:
        if not uni.temporal:
            st.info("Aucune variable de date détectée.")
        else:
            show_table(uni.temporal_table(), key="tmp_stats", export_name="statistiques_temporelles", digits=3)
            var = st.selectbox("Variable temporelle", list(uni.temporal), key="tmp_var")
            s = uni.temporal[var]
            kpi_row([
                ("Date minimale", s.minimum.strftime("%d/%m/%Y") if s.minimum is not None else "—", None),
                ("Date maximale", s.maximum.strftime("%d/%m/%Y") if s.maximum is not None else "—", None),
                ("Étendue", f"{fmt_num(s.span_days, 0)} jours", None),
                ("Dates distinctes", fmt_int(s.n_distinct), f"fréquence : {s.inferred_frequency or 'n.d.'}"),
            ])
            if s.irregularities:
                st.warning("**Irrégularités temporelles détectées :**\n" + "\n".join(f"- {i}" for i in s.irregularities),
                           icon=":material/schedule:")
            else:
                st.success("Aucune irrégularité temporelle détectée.", icon=":material/check_circle:")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Par année**")
                show_table(s.by_year, key=f"y_{var}", export_name=f"{var}_par_annee")
            with c2:
                st.markdown("**Par mois**")
                show_table(s.by_month, key=f"m_{var}", export_name=f"{var}_par_mois")
            with c3:
                st.markdown("**Par jour de la semaine**")
                show_table(s.by_weekday, key=f"w_{var}", export_name=f"{var}_par_jour")
