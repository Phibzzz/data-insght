"""Importation : téléversement, reconnaissance du format, options et résumé du chargement."""
from __future__ import annotations

import streamlit as st

from components.data_tables import show_table
from components.feedback import guarded, show_flash
from components.kpi_cards import kpi_row, page_header
from config.settings import SETTINGS, all_extensions
from core.data_loader import LoadOptions
from core.file_detector import SEPARATOR_LABELS, human_size
from core.profiling_engine import count_duplicates
from utils import state
from utils.cache import cached_inspect, cached_load
from utils.helpers import fmt_int
from utils.validators import validate_dataframe, validate_upload

page_header("Importation des données", "Formats tabulaires, bureautiques, analytiques, structurés, statistiques et bases SQLite.")
show_flash()

uploaded = st.file_uploader(
    "Déposez un fichier", type=all_extensions(), key=f"uploader_{st.session_state['uploader_key']}",
    help=f"Taille maximale : {SETTINGS.loading.max_file_size_mb} Mo (modifiable dans Paramètres).",
)

if uploaded is not None:
    content = uploaded.getvalue()
    inspection = None
    with guarded("La vérification du fichier"):
        validate_upload(uploaded.name, len(content))
        inspection = cached_inspect(uploaded.name, content)

    if inspection is not None:
        det = inspection.detection
        with st.container(border=True):
            st.markdown(f"#### :material/search: Fichier reconnu : **{det.label}**")
            info = [f"Taille : **{human_size(det.size_bytes)}**", f"Extension : **.{det.extension or '—'}**"]
            if det.encoding:
                info.append(f"Encodage : **{det.encoding}**")
            if det.separator:
                info.append(f"Séparateur : **{SEPARATOR_LABELS.get(det.separator, det.separator)}**")
                info.append(f"Décimale : **{det.decimal}**")
            st.markdown(" · ".join(info))
            for w in det.warnings:
                st.warning(w, icon=":material/warning:")

            options = LoadOptions()
            if inspection.sheets:
                options.sheet = st.selectbox(f"Feuille à analyser ({len(inspection.sheets)} disponible(s))",
                                             inspection.sheets)
            if inspection.tables:
                options.table = st.selectbox(f"Table à analyser ({len(inspection.tables)} disponible(s))",
                                             inspection.tables)
            if inspection.zip_members:
                options.zip_member = st.selectbox("Fichier de l'archive à analyser", inspection.zip_members)

            with st.expander("Options de lecture avancées"):
                if det.detected_format in {"csv", "tsv", "txt"}:
                    c1, c2, c3 = st.columns(3)
                    seps = [",", ";", "\t", "|"]
                    options.separator = c1.selectbox("Séparateur", seps, index=seps.index(det.separator)
                                                     if det.separator in seps else 0,
                                                     format_func=lambda s: SEPARATOR_LABELS.get(s, s))
                    encs = ["utf-8", "utf-8-sig", "cp1252", "latin-1", "utf-16"]
                    enc = det.encoding if det.encoding in encs else "utf-8"
                    options.encoding = c2.selectbox("Encodage", encs, index=encs.index(enc))
                    options.decimal = c3.selectbox("Séparateur décimal", [".", ","], index=0 if det.decimal == "." else 1)
                    header = st.checkbox("La première ligne contient les noms de colonnes", value=True)
                    options.header_row = 0 if header else None
                    try:
                        import polars  # noqa: F401

                        options.use_polars = st.checkbox("Lecture rapide avec Polars (expérimental)", value=False)
                    except ImportError:
                        st.caption("Polars non installé : lecture avec pandas (option de lecture rapide indisponible).")
                est = inspection.estimated_rows
                large = est is not None and est > SETTINGS.loading.large_dataset_rows
                partial = st.checkbox(
                    "Lecture partielle (limiter le nombre de lignes)", value=large,
                    help="Recommandé pour les très gros fichiers : l'analyse porte alors sur les premières lignes.",
                )
                if partial:
                    options.row_limit = int(st.number_input("Nombre maximal de lignes", 1_000, 50_000_000,
                                                            SETTINGS.loading.default_row_limit, step=10_000))
                if large:
                    st.info(f"Fichier volumineux (~{fmt_int(est)} lignes) : la lecture partielle est conseillée "
                            "pour préserver la mémoire.")

            if st.button("Charger et analyser", type="primary", icon=":material/play_arrow:"):
                with guarded("Le chargement"):
                    with st.spinner("Lecture du fichier…"):
                        result = cached_load(uploaded.name, content, options)
                        warnings = validate_dataframe(result.df)
                    state.set_dataset(result.df.copy(), result.meta)
                    result.meta.warnings.extend(w for w in warnings if w not in result.meta.warnings)
                    st.session_state["_just_loaded"] = True
                    st.rerun()

if state.has_data():
    df = st.session_state["raw_df"]
    m = state.meta()
    if st.session_state.pop("_just_loaded", False):
        st.success(f"« {m.filename} » chargé avec succès.", icon=":material/check_circle:")
    st.divider()
    st.subheader(f"Résumé — {m.filename}")
    n_missing = int(df.isna().sum().sum())
    kpi_row([
        ("Format détecté", m.format_label, m.sheet and f"feuille « {m.sheet} »" or m.table and f"table « {m.table} »"),
        ("Taille du fichier", human_size(m.size_bytes), m.encoding and f"encodage {m.encoding}"),
        ("Lignes", fmt_int(df.shape[0]), "lecture partielle" if m.truncated else None),
        ("Colonnes / variables", fmt_int(df.shape[1]), None),
        ("Valeurs manquantes", fmt_int(n_missing), f"{100 * n_missing / max(df.size, 1):.1f} % des cellules"),
        ("Doublons", fmt_int(count_duplicates(df)), "lignes identiques"),
    ], columns=3)
    for w in m.warnings:
        st.warning(w, icon=":material/info:")
    st.markdown("**Aperçu des premières observations**")
    show_table(df.head(20), key="import_preview", digits=None)
    c1, c2, c3 = st.columns(3)
    c1.page_link("views/vue_ensemble.py", label="Voir la vue d'ensemble", icon=":material/arrow_forward:")
    if c2.button("Charger un autre fichier", icon=":material/upload:"):
        st.session_state["uploader_key"] += 1
        st.rerun()
    if c3.button("Réinitialiser l'analyse", icon=":material/restart_alt:"):
        state.reset_all()
        st.rerun()
