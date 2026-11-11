"""Tests de l'audit de qualité, des outliers et du score global."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import QualityWeights
from core.quality_engine import ValidityRule, iqr_bounds, iqr_outliers, run_quality_audit, zscore_outliers
from core.schema_detector import apply_schema, detect_schema


def audit(df, **kw):
    schemas = detect_schema(df)
    typed, lost = apply_schema(df, schemas)
    return run_quality_audit(df, typed, schemas, lost, **kw)


def checks(report):
    return {(i.check, i.column) for i in report.issues}


def test_iqr_bounds_known_values():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
    lo, hi = iqr_bounds(s)
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    assert (lo, hi) == (pytest.approx(q1 - 1.5 * (q3 - q1)), pytest.approx(q3 + 1.5 * (q3 - q1)))
    assert iqr_outliers(s).tolist() == [False] * 9 + [True]


def test_zscore_outliers():
    s = pd.Series([10.0] * 50 + [10.5] * 50 + [100.0])
    assert zscore_outliers(s, 3).sum() == 1 and bool(zscore_outliers(s, 3).iloc[-1])
    assert zscore_outliers(pd.Series([1, 1, 1])).sum() == 0  # écart-type nul


def test_outliers_dataset(outliers_df):
    rep = audit(outliers_df)
    out = {o.column: o for o in rep.outliers}
    assert out["mesure"].n_iqr >= 5 and out["salaire"].n_iqr >= 1
    assert ("Valeurs atypiques potentielles", "mesure") in checks(rep)


def test_perfect_dataset_scores_100(numeric_df):
    clean = pd.DataFrame({"a": np.arange(100, dtype=float), "b": list("xyzw") * 25})
    rep = audit(clean)
    assert rep.sub_scores["completeness"] == 100 and rep.sub_scores["uniqueness"] == 100
    assert rep.global_score == pytest.approx(100.0)


def test_missing_duplicates_empty_constant():
    df = pd.DataFrame({
        "a": [1, 2, None, 4, 4, 6, 7, 8, 9, 10],
        "vide": [None] * 10,
        "const": ["k"] * 10,
        "b": ["x", "y", "z", "x", "x", "y", "z", "x", "y", "z"],
    })
    df.loc[4] = df.loc[3]  # doublon
    rep = audit(df)
    c = checks(rep)
    assert ("Lignes dupliquées", "(toutes)") in c
    assert ("Colonne vide", "vide") in c
    assert ("Colonne constante", "const") in c
    assert ("Valeurs manquantes", "a") in c
    assert rep.indicators["Taux de doublons (%)"] == pytest.approx(10.0)
    n_missing = int(df.isna().sum().sum())
    assert rep.sub_scores["completeness"] == pytest.approx(round(100 * (1 - n_missing / df.size), 2))


def test_inconsistent_formats_and_infinite():
    df = pd.DataFrame({
        "ville": ["Paris", "paris", " Paris", "Lyon", "LYON", "Lyon"] * 5,
        "valeur": [1.0, 2.0, np.inf, 3.0, 4.0, 5.0] * 5,
    })
    rep = audit(df)
    c = checks(rep)
    assert ("Incohérence de format", "ville") in c
    assert ("Valeurs infinies", "valeur") in c
    assert rep.sub_scores["consistency"] < 100


def test_type_issues_and_dates():
    df = pd.DataFrame({
        "montant": ["10,5", "20,0", "30,25", "12,0"] * 9 + ["10,5", "20,0", "30,25", "abc"],
        "date": ["2024-01-01", "1850-01-01", "2024-03-01", "2099-01-01"] * 10,
        "age": [25, -3, 40, 50] * 10,
    })
    rep = audit(df, now=pd.Timestamp("2025-01-01"))
    c = checks(rep)
    assert ("Valeurs incohérentes avec le type", "montant") in c
    assert ("Type de données incorrect", "montant") in c
    assert ("Problème de dates", "date") in c
    assert ("Valeurs incohérentes (hypothèse)", "age") in c
    assert rep.sub_scores["validity"] < 100


def test_rare_categories():
    df = pd.DataFrame({"cat": ["a"] * 300 + ["b"] * 200 + ["rare"]})
    assert ("Modalités rares", "cat") in checks(audit(df))


def test_validity_rules():
    df = pd.DataFrame({"age": [10, 20, 130, 40], "code": ["A1", "B2", "xx", "C3"]})
    rules = [ValidityRule("age", min_value=0, max_value=120), ValidityRule("code", regex=r"[A-Z]\d")]
    rep = audit(df, rules=rules)
    viol = [i for i in rep.issues if i.check == "Règle de validité"]
    assert {i.column: i.count for i in viol} == {"age": 1, "code": 1}
    assert rep.indicators["Taux de validité (%)"] == pytest.approx(100 * (1 - 2 / 8))


def test_weights_change_global_score(missing_df):
    base = audit(missing_df)
    only_completeness = audit(missing_df, weights=QualityWeights(1, 0, 0, 0, 0))
    assert only_completeness.global_score == pytest.approx(base.sub_scores["completeness"])
    assert sum(base.weights.values()) == pytest.approx(1.0)
