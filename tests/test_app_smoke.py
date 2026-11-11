"""Smoke tests de l'interface Streamlit (exécution réelle des pages via AppTest)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

from core.cleaning_engine import CleaningSession  # noqa: E402
from core.data_loader import load_dataset  # noqa: E402
from utils.sample_factory import mixed_dataset, temporal_dataset  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")
PAGES = ["accueil", "importation", "vue_ensemble", "exploration", "profil_variables", "statistiques",
         "analyse_multivariee", "qualite_donnees", "visualisations", "nettoyage", "rapports", "parametres"]


def _app_with(df=None, name="ventes.csv") -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    if df is not None:
        res = load_dataset(name, df.to_csv(index=False).encode())
        at.session_state["raw_df"] = res.df
        at.session_state["meta"] = res.meta
        at.session_state["cleaning"] = CleaningSession(res.df)
        at.session_state["data_version"] = 1
    return at


def _goto(at: AppTest, page: str) -> AppTest:
    at.switch_page(f"views/{page}.py")
    at.run()
    return at


def _problems(at: AppTest) -> list[str]:
    return [str(e.value) for e in at.exception] + [str(e.value) for e in at.error]


def test_pages_without_data():
    at = _app_with()
    for page in PAGES:
        assert not [str(e.value) for e in _goto(at, page).exception], page


@pytest.mark.parametrize("builder", [mixed_dataset, temporal_dataset])
def test_pages_with_data(builder):
    at = _app_with(builder())
    for page in PAGES:
        assert not _problems(_goto(at, page)), page


def test_cleaning_interaction_and_history():
    df = mixed_dataset(200)
    df = pd.concat([df, df.iloc[:3]], ignore_index=True)  # 3 doublons
    at = _goto(_app_with(df), "nettoyage")
    at.button(key="btn_dup").click().run()
    assert not _problems(at)
    session = at.session_state["cleaning"]
    assert len(session.history) == 1 and len(session.current) == 200
    assert at.session_state["use_cleaned"] is True
    assert len(at.session_state["raw_df"]) == 203  # original intact


def test_type_override_interaction():
    at = _goto(_app_with(mixed_dataset(150)), "profil_variables")
    at.selectbox[0].select("quantite").run()
    # Widget de formulaire : la valeur n'est validée qu'avec la soumission (même cycle d'exécution)
    at.selectbox[1].select("Catégorielle nominale")
    [b for b in at.button if b.label == "Appliquer la correction"][0].click().run()
    assert at.session_state["overrides"].get("quantite") == "Catégorielle nominale"
    assert not _problems(at)
    [b for b in at.button if b.label == "Revenir au type détecté"][0].click().run()
    assert "quantite" not in at.session_state["overrides"]


def test_report_generation_interaction():
    at = _goto(_app_with(mixed_dataset(150)), "rapports")
    at.multiselect[0].set_value(["HTML", "JSON"]).run()
    gen = [b for b in at.button if b.label == "Générer le rapport"][0]
    gen.click().run()
    assert not _problems(at)
    outputs = at.session_state["report_outputs"]
    assert set(outputs) == {"HTML", "JSON"} and outputs["HTML"].startswith(b"<!DOCTYPE html>")
