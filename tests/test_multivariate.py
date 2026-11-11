"""Tests des corrélations, tableaux croisés et comparaisons de groupes."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from core.multivariate_engine import (
    contingency,
    correlation_matrix,
    covariance_matrix,
    cramers_v_matrix,
    group_comparison,
    recommend_method,
    strong_associations,
)


def test_perfect_correlations(small_known):
    for method in ("pearson", "spearman", "kendall"):
        res = correlation_matrix(small_known, ["x", "y", "z"], method)
        assert res.matrix.loc["x", "y"] == pytest.approx(1.0)
        assert res.matrix.loc["x", "z"] == pytest.approx(-1.0)
        assert res.n_obs.loc["x", "y"] == 5


def test_correlation_matches_scipy():
    rng = np.random.default_rng(3)
    x = rng.normal(size=200)
    y = 0.5 * x + rng.normal(size=200)
    df = pd.DataFrame({"x": x, "y": y})
    assert correlation_matrix(df, ["x", "y"], "pearson").matrix.loc["x", "y"] == pytest.approx(sps.pearsonr(x, y)[0])
    assert correlation_matrix(df, ["x", "y"], "spearman").matrix.loc["x", "y"] == pytest.approx(sps.spearmanr(x, y)[0])
    assert correlation_matrix(df, ["x", "y"], "kendall").matrix.loc["x", "y"] == pytest.approx(sps.kendalltau(x, y)[0])


def test_pairwise_missing_handling():
    df = pd.DataFrame({"a": [1, 2, 3, 4, np.nan, 6], "b": [2, 4, 6, 8, 10, np.nan], "c": [1, 1, 2, 2, 3, 3]})
    res = correlation_matrix(df, ["a", "b", "c"], "pearson")
    assert res.n_obs.loc["a", "b"] == 4
    assert res.n_obs.loc["a", "c"] == 5
    assert res.matrix.loc["a", "b"] == pytest.approx(1.0)


def test_constant_columns_excluded():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "k": [5, 5, 5, 5], "b": [4, 3, 2, 1]})
    res = correlation_matrix(df, ["a", "k", "b"])
    assert res.excluded == ["k"] and "k" not in res.matrix.columns


def test_pairs_sorted_and_thresholded(small_known):
    pairs = correlation_matrix(small_known, ["x", "y", "z"]).pairs(0.5)
    assert len(pairs) == 3 and (pairs["|r|"] >= 0.5).all()
    assert set(pairs["Sens"]) == {"positive", "négative"}


def test_recommend_method():
    rng = np.random.default_rng(0)
    normal = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    assert recommend_method(normal, ["a", "b"])[0] == "pearson"
    skewed = pd.DataFrame({"a": rng.lognormal(0, 1.5, 300), "b": rng.normal(size=300)})
    assert recommend_method(skewed, ["a", "b"])[0] == "spearman"
    small = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 1, 4, 3, 5]})
    assert recommend_method(small, ["a", "b"])[0] == "kendall"


def test_covariance(small_known):
    cov = covariance_matrix(small_known, ["x", "y"])
    assert cov.loc["x", "x"] == pytest.approx(2.5)
    assert cov.loc["x", "y"] == pytest.approx(5.0)


def test_contingency_chi2_and_cramers_v():
    df = pd.DataFrame({"a": ["x"] * 50 + ["y"] * 50, "b": ["u"] * 50 + ["v"] * 50})
    res = contingency(df, "a", "b")
    assert res.table.loc["x", "u"] == 50 and res.table.loc["x", "v"] == 0
    assert res.cramers_v == pytest.approx(1.0)
    assert res.pvalue < 1e-10 and res.dof == 1
    chi2_ref = sps.chi2_contingency(res.table.to_numpy(), correction=False).statistic
    assert res.chi2 == pytest.approx(chi2_ref)
    assert res.row_pct.loc["x", "u"] == 100.0


def test_contingency_independence_and_small_counts():
    df = pd.DataFrame({"a": ["x", "x", "y", "y"] * 25, "b": ["u", "v"] * 50})
    res = contingency(df, "a", "b")
    assert res.cramers_v == pytest.approx(0.0, abs=1e-9)
    small = contingency(pd.DataFrame({"a": list("xyzxy"), "b": list("uvuvu")}), "a", "b")
    assert small.warning and "effectifs théoriques" in small.warning
    with pytest.raises(ValueError):
        contingency(df, "a", "a")


def test_group_comparison(small_known):
    gc = group_comparison(small_known, "x", "g")
    means = gc.table.set_index("g")["Moyenne"].to_dict()
    assert means == {"a": pytest.approx(1.5), "b": pytest.approx(4.0)}
    # η² = SS_inter / SS_total = (2·(1.5−3)² + 3·(4−3)²) / 10 = 7.5 / 10
    assert gc.eta_squared == pytest.approx(0.75)
    assert gc.n_groups == 2 and gc.anova_pvalue is not None


def test_cramers_matrix_and_strong_associations(mixed_df):
    vm = cramers_v_matrix(mixed_df, ["produit", "ville"])
    assert vm.shape == (2, 2) and vm.loc["produit", "produit"] == 1.0
    assoc = strong_associations(mixed_df, ["prix_unitaire", "montant_total", "quantite"], ["produit"])
    assert ((assoc["Variable 1"] == "prix_unitaire") & (assoc["Variable 2"] == "produit")).any()
