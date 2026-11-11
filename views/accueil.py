"""Page d'accueil : présentation, fonctionnalités, formats et jeux d'exemple."""
from __future__ import annotations

import streamlit as st

from components.feedback import guarded, show_flash
from config.settings import APP_NAME, APP_SUBTITLE, SUPPORTED_EXTENSIONS
from core.data_loader import LoadOptions
from i18n import t
from utils import state
from utils.cache import cached_load, cached_sample
from utils.sample_factory import SAMPLE_BUILDERS

show_flash()
st.markdown(
    f"""<div class="di-hero"><h1>{APP_NAME}</h1><div class="di-sub">{APP_SUBTITLE}</div>
    <p>{t("home.tagline")}</p></div>""",
    unsafe_allow_html=True,
)

c1, c2 = st.columns([1, 1])
c1.page_link("views/importation.py", label="Importer un fichier", icon=":material/upload_file:", width="stretch")
if state.has_data():
    c2.page_link("views/vue_ensemble.py", label=f"Reprendre : {state.meta().filename}", icon=":material/dashboard:",
                 width="stretch")

st.subheader("Fonctionnalités principales")
features = [
    ("Reconnaissance automatique", "Format réel, encodage, séparateur, feuilles Excel, tables SQLite, types de variables."),
    ("Statistiques descriptives", "Indicateurs adaptés à chaque type : quantitatif, qualitatif, temporel."),
    ("Analyse multivariée", "Corrélations Pearson / Spearman / Kendall, χ², V de Cramér, comparaisons de groupes."),
    ("Qualité des données", "Manquants, doublons, incohérences, valeurs atypiques et score transparent."),
    ("Visualisations", "Graphiques interactifs choisis selon le type des variables, exportables."),
    ("Interprétations", "Constats, interprétations, hypothèses et recommandations calculés sur vos données."),
    ("Nettoyage", "Transformations sur une copie de travail, historique et annulation."),
    ("Rapports", "PDF, HTML, Excel, CSV et JSON, sections au choix."),
]
cols = st.columns(4)
for i, (title, desc) in enumerate(features):
    cols[i % 4].markdown(f"<div class='di-feature'><h4>{title}</h4><p>{desc}</p></div>", unsafe_allow_html=True)
    if i == 3:
        st.write("")
        cols = st.columns(4)

st.subheader("Formats pris en charge")
fcols = st.columns(len(SUPPORTED_EXTENSIONS))
for col, (family, exts) in zip(fcols, SUPPORTED_EXTENSIONS.items()):
    col.markdown(f"**{family}**  \n" + " ".join(f"`.{e}`" for e in exts))

st.subheader("Jeux de données d'exemple")
st.caption("Générés localement et de façon reproductible : idéal pour découvrir l'application.")
scols = st.columns(4)
for i, (key, (label, _)) in enumerate(SAMPLE_BUILDERS.items()):
    if scols[i % 4].button(label, key=f"sample_{key}", icon=":material/dataset:", width="stretch"):
        with guarded("Le chargement de l'exemple"):
            df = cached_sample(key)
            content = df.to_csv(index=False).encode("utf-8")
            result = cached_load(f"{key}.csv", content, LoadOptions())
            state.set_dataset(result.df.copy(), result.meta)
            st.switch_page("views/vue_ensemble.py")

with st.expander("Confidentialité et sécurité"):
    st.markdown(
        "- Les données restent **sur votre machine** : aucune transmission vers un service externe, aucune clé API.\n"
        "- Aucun code contenu dans les fichiers n'est exécuté (macros Excel ignorées, pas de désérialisation pickle).\n"
        "- XML analysé sans DTD ni entités externes ; SQLite ouvert en lecture seule ; archives ZIP contrôlées.\n"
        "- Les exports CSV/Excel neutralisent les cellules interprétables comme des formules."
    )
