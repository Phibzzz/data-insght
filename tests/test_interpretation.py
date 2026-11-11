"""Tests du moteur d'interprétation : commentaires fondés sur les résultats réels."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from core.data_loader import FileMeta
from core.interpretation_engine import Category, InterpretationResult, NullProvider
from core.pipeline import run_analysis


def meta(name="t.csv"):
    return FileMeta(filename=name, detected_format="csv", format_label="CSV", size_bytes=100)


def test_insights_reflect_real_values():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "revenu": rng.lognormal(10, 1.2, 500),
        "x": (x := rng.normal(50, 10, 500)),
        "y": x * 2 + rng.normal(0, 1, 500),
        "cat": ["dominante"] * 450 + ["autre"] * 50,
        "trou": [np.nan] * 300 + list(rng.normal(size=200)),
    })
    b = run_analysis(df, meta())
    texts = " ".join(i.text for i in b.insights.insights)
    skew = b.univariate.numeric["revenu"].skewness
    assert "revenu" in texts and "asymétrique" in texts
    assert f"{skew:.2f}".replace(".", ",") in texts  # la valeur citée est la valeur calculée
    assert "dominante" in texts and "90,0 %" in texts
    assert "très incomplète" in texts and "60,0 %" in texts
    assert re.search(r"Corrélation (très )?forte et positive entre « x » et « y »", texts)
    cats = {i.category for i in b.insights.insights}
    assert {Category.FINDING, Category.INTERPRETATION, Category.HYPOTHESIS, Category.RECOMMENDATION} <= cats
    assert any("observations" in line for line in b.insights.summary)


def test_no_spurious_insights_on_clean_data():
    df = pd.DataFrame({"a": np.random.default_rng(1).normal(size=300), "b": ["x", "y", "z"] * 100})
    b = run_analysis(df, meta())
    assert not [i for i in b.insights.insights if i.topic in ("Dispersion", "Corrélations", "Complétude")]


def test_null_provider_is_identity():
    res = InterpretationResult(summary=["a"])
    assert NullProvider().enrich(res) is res
