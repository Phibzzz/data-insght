"""Tests de la détection des types de variables et des corrections manuelles."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.schema_detector import (
    Family,
    StatType,
    apply_schema,
    detect_column,
    detect_schema,
    schema_by_name,
    to_numeric_smart,
)


def st_of(name: str, values) -> StatType:
    return detect_column(name, pd.Series(values)).stat_type


def test_numeric_types():
    rng = np.random.default_rng(0)
    assert st_of("mesure", rng.normal(0, 1, 100)) == StatType.CONTINUOUS
    assert st_of("nb_enfants", rng.integers(0, 5, 100)) == StatType.DISCRETE
    assert st_of("revenu", rng.integers(1000, 90000, 200)) == StatType.CONTINUOUS


def test_text_numbers_with_french_decimals():
    sch = detect_column("prix", pd.Series(["12,5", "3,75", "1 200,00", "8"] * 10))
    assert sch.family == Family.NUMERIC and sch.conversion == "numeric"
    assert to_numeric_smart(pd.Series(["1 200,50", "3,5", "abc"])).tolist()[:2] == [1200.5, 3.5]


def test_categorical_types():
    assert st_of("ville", ["Paris", "Lyon", "Nice"] * 30) == StatType.NOMINAL
    assert st_of("sexe", ["F", "M"] * 30) == StatType.BINARY
    assert st_of("abonne", ["oui", "non", "Oui"] * 20) == StatType.BOOLEAN
    assert st_of("actif", [True, False] * 20) == StatType.BOOLEAN
    assert st_of("flag", [0, 1, 1, 0] * 10) == StatType.BOOLEAN


def test_ordinal_detection_and_order():
    sch = detect_column("satisfaction", pd.Series(["satisfait", "neutre", "très satisfait", "insatisfait"] * 10))
    assert sch.stat_type == StatType.ORDINAL
    assert sch.ordinal_order == ["insatisfait", "neutre", "satisfait", "tres satisfait"]
    assert st_of("note", [1, 2, 3, 4, 5] * 10) == StatType.ORDINAL


def test_temporal_types():
    assert st_of("date", ["2024-01-15", "2024-02-20", "2024-03-05"] * 10) == StatType.DATE
    assert st_of("d", ["15/01/2024", "20/02/2024", "31/12/2023"] * 10) == StatType.DATE
    assert st_of("ts", ["2024-01-15 10:30:00", "2024-01-16 11:00:00"] * 10) == StatType.DATETIME
    assert st_of("annee", [2019, 2020, 2021, 2022] * 10) == StatType.YEAR
    assert st_of("mois", [1, 2, 3, 12] * 10) == StatType.MONTH
    assert st_of("mois_nom", ["janvier", "février", "mars"] * 10) == StatType.MONTH
    assert st_of("dt", pd.date_range("2024-01-01", periods=30, freq="D")) == StatType.DATE


def test_other_types():
    assert st_of("id_client", [f"C{i:05d}" for i in range(100)]) == StatType.IDENTIFIER
    assert st_of("id", list(range(100))) == StatType.IDENTIFIER
    long_text = [f"Commentaire numéro {i} : le produit est arrivé en bon état et fonctionne bien." for i in range(50)]
    assert st_of("avis", long_text) == StatType.TEXT
    assert st_of("lat", np.linspace(43, 50, 40)) == StatType.LATITUDE
    assert st_of("longitude", np.linspace(-1, 7, 40)) == StatType.LONGITUDE
    assert st_of("pays", ["FR"] * 20) == StatType.CONSTANT
    assert st_of("vide", [None] * 20) == StatType.EMPTY


def test_schema_summary_counts(missing_df):
    schemas = schema_by_name(detect_schema(missing_df))
    assert schemas["revenu_annuel"].n_missing == int(missing_df["revenu_annuel"].isna().sum())
    assert schemas["categorie"].n_unique == 3
    assert schemas["colonne_vide"].stat_type == StatType.EMPTY


def test_manual_override_is_reversible(mixed_df):
    original = mixed_df.copy()
    schemas = detect_schema(mixed_df, overrides={"quantite": StatType.NOMINAL.value})
    sch = schema_by_name(schemas)["quantite"]
    assert sch.stat_type == StatType.NOMINAL and sch.overridden
    assert sch.detected_stat_type == StatType.DISCRETE
    typed, _ = apply_schema(mixed_df, schemas)
    pd.testing.assert_frame_equal(mixed_df, original)  # l'original n'est jamais modifié
    # Retour au type détecté : suppression de la surcharge
    assert schema_by_name(detect_schema(mixed_df, overrides={}))["quantite"].stat_type == StatType.DISCRETE


def test_apply_schema_conversions_and_lost_values():
    df = pd.DataFrame({"d": ["2024-01-01"] * 19 + ["pas une date"], "n": ["1,5"] * 19 + ["x"]})
    schemas = detect_schema(df)
    typed, lost = apply_schema(df, schemas)
    assert pd.api.types.is_datetime64_any_dtype(typed["d"])
    assert pd.api.types.is_float_dtype(typed["n"]) and typed["n"].iloc[0] == 1.5
    assert lost == {"d": 1, "n": 1}


def test_override_to_numeric_counts_lost_values():
    df = pd.DataFrame({"c": ["a", "1", "2", "b"]})
    schemas = detect_schema(df, overrides={"c": StatType.CONTINUOUS.value})
    typed, lost = apply_schema(df, schemas)
    assert typed["c"].tolist()[1:3] == [1.0, 2.0] and lost["c"] == 2
