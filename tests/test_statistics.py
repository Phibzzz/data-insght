"""Tests d'exactitude des statistiques descriptives sur des valeurs connues."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from core.schema_detector import apply_schema, detect_schema
from core.statistics_engine import categorical_stats, compute_univariate, frequency_table, numeric_stats, temporal_stats


def test_numeric_known_values():
    s = numeric_stats(pd.Series([1, 2, 3, 4, 5]), "x")
    assert s.count == 5 and s.valid == 5 and s.missing == 0
    assert s.mean == 3 and s.median == 3
    assert s.minimum == 1 and s.maximum == 5 and s.range == 4
    assert s.variance == pytest.approx(2.5)
    assert s.std == pytest.approx(math.sqrt(2.5))
    assert (s.q1, s.q2, s.q3, s.iqr) == (2, 3, 4, 2)
    assert s.cv == pytest.approx(math.sqrt(2.5) / 3)
    assert s.skewness == pytest.approx(0.0)
    assert s.kurtosis == pytest.approx(-1.2)
    assert s.mode == []  # aucune valeur répétée : pas de mode
    assert s.sum == 15 and s.mad == 1


def test_numeric_mode_missing_and_infinite():
    s = numeric_stats(pd.Series([1, 2, 2, 3, np.nan, np.inf, None]), "x")
    assert s.count == 7
    assert s.missing == 2 and s.infinite == 1 and s.valid == 4  # NaN et None manquants ; inf compté à part
    assert s.mode == [2.0]
    assert s.mean == pytest.approx(2.0)


def test_numeric_matches_scipy_reference():
    rng = np.random.default_rng(1)
    x = rng.lognormal(0, 0.8, 500)
    s = numeric_stats(pd.Series(x))
    assert s.skewness == pytest.approx(sps.skew(x, bias=False))
    assert s.kurtosis == pytest.approx(sps.kurtosis(x, bias=False))
    assert s.std == pytest.approx(np.std(x, ddof=1))
    lo, hi = sps.t.interval(0.95, len(x) - 1, loc=x.mean(), scale=sps.sem(x))
    assert (s.ci95_low, s.ci95_high) == (pytest.approx(lo), pytest.approx(hi))
    assert s.normality_test == "Shapiro-Wilk" and s.normality_pvalue < 0.05


def test_numeric_edge_cases():
    empty = numeric_stats(pd.Series([np.nan, np.nan]))
    assert empty.valid == 0 and empty.mean is None
    const = numeric_stats(pd.Series([7, 7, 7, 7]))
    assert const.std == 0 and const.skewness is None and const.cv == 0
    zero_mean = numeric_stats(pd.Series([-1, 1, -1, 1]))
    assert zero_mean.cv is None  # CV non défini pour une moyenne nulle
    one = numeric_stats(pd.Series([5.0]))
    assert one.variance is None and one.mean == 5


def test_frequency_table_nominal():
    f = frequency_table(pd.Series(["a", "b", "a", "c", "a", None]), ordered=False)
    assert f["Modalité"].tolist() == ["a", "b", "c"]
    assert f["Effectif"].tolist() == [3, 1, 1]
    assert f["Fréquence"].tolist() == pytest.approx([0.6, 0.2, 0.2])
    assert f["%"].sum() == pytest.approx(100)
    assert "Effectif cumulé" not in f.columns


def test_frequency_table_ordinal_cumulative():
    s = pd.Series(pd.Categorical(["moyen", "faible", "fort", "moyen"], categories=["faible", "moyen", "fort"], ordered=True))
    f = frequency_table(s)
    assert f["Modalité"].tolist() == ["faible", "moyen", "fort"]
    assert f["Effectif cumulé"].tolist() == [1, 3, 4]
    assert f["% cumulé"].tolist() == [25.0, 75.0, 100.0]


def test_categorical_stats():
    c = categorical_stats(pd.Series(["x", "x", "x", "y", None, ""]), "v")
    assert c.count == 6 and c.valid == 4 and c.missing == 2  # chaîne vide = manquant
    assert c.n_categories == 2 and c.mode == "x" and c.mode_count == 3
    assert c.mode_share == pytest.approx(0.75)
    p = np.array([0.75, 0.25])
    assert c.entropy_normalized == pytest.approx(float(-(p * np.log(p)).sum() / np.log(2)), abs=1e-4)


def test_temporal_stats():
    dates = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-10", "2024-01-11", "2030-01-01"]))
    t = temporal_stats(dates, "d", now=pd.Timestamp("2025-01-01"))
    assert t.minimum == pd.Timestamp("2024-01-01") and t.maximum == pd.Timestamp("2030-01-01")
    assert t.n_distinct == 6 and t.median_gap_days == 1.0 and t.inferred_frequency == "journalière"
    assert t.n_gaps == 2  # 3 → 10 janvier et 11 janvier → 2030
    assert t.n_future == 1
    assert t.by_year.set_index("Année")["Effectif"].to_dict() == {"2024": 5, "2030": 1}
    assert t.by_weekday["Effectif"].sum() == 6


def test_compute_univariate_dispatch(mixed_df):
    schemas = detect_schema(mixed_df)
    typed, _ = apply_schema(mixed_df, schemas)
    uni = compute_univariate(typed, schemas)
    assert "prix_unitaire" in uni.numeric and "produit" in uni.categorical and "date_commande" in uni.temporal
    assert "id_commande" not in uni.numeric and "id_commande" not in uni.categorical
    assert uni.numeric["quantite"].mean == pytest.approx(mixed_df["quantite"].mean())
    assert uni.categorical["note_client"].ordered
