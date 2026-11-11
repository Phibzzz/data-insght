"""Tests des opérations de nettoyage, de l'historique et de l'immutabilité de l'original."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.cleaning_engine import CleaningSession, apply_operation, safe_eval_expression
from core.exceptions import CleaningError


@pytest.fixture
def df():
    return pd.DataFrame({
        "a": [1.0, 2.0, np.nan, 4.0, 100.0, 1.0],
        "b": ["  Paris", "paris", None, "Lyon ", "Lyon", "  Paris"],
        "c": [10, 20, 30, 40, 50, 10],
        "d": ["2024-01-05", "2024-02-10", "2024-03-15", "x", "2024-05-20", "2024-01-05"],
    })


def op(df, op_name, /, **params):
    out, _ = apply_operation(df, op_name, **params)
    return out


def test_original_never_modified(df):
    snapshot = df.copy()
    session = CleaningSession(df)
    session.apply("impute", columns=["a"], strategy="mean")
    session.apply("drop_duplicates")
    session.apply("normalize_text", columns=["b"], case="lower")
    pd.testing.assert_frame_equal(df, snapshot)
    pd.testing.assert_frame_equal(session.original, snapshot)


def test_drop_duplicates(df):
    assert len(op(df, "drop_duplicates")) == 5
    assert len(op(df, "drop_duplicates", subset=["c"], keep="none")) == 4


def test_missing_strategies(df):
    assert op(df, "impute", columns=["a"], strategy="mean")["a"].iloc[2] == pytest.approx(np.mean([1, 2, 4, 100, 1]))
    assert op(df, "impute", columns=["a"], strategy="median")["a"].iloc[2] == 2.0
    assert op(df, "impute", columns=["a"], strategy="mode")["a"].iloc[2] == 1.0
    assert op(df, "impute", columns=["a"], strategy="constant", value="0")["a"].iloc[2] == 0.0
    assert op(df, "impute", columns=["b"], strategy="constant", value="inconnu")["b"].iloc[2] == "inconnu"
    assert len(op(df, "drop_missing_rows")) == 5
    assert "a" not in op(df, "drop_missing_columns", threshold=0.1).columns
    with pytest.raises(CleaningError):
        op(df, "impute", columns=["b"], strategy="mean")


def test_convert_type_counts_lost(df):
    out, operation = apply_operation(df, "convert_type", column="d", target="datetime")
    assert pd.api.types.is_datetime64_any_dtype(out["d"])
    assert "1 valeur(s) non convertible(s)" in operation.description
    assert op(df, "convert_type", column="c", target="string")["c"].dtype == "string"


def test_rename_and_drop(df):
    out = op(df, "rename_columns", mapping={"a": "alpha"})
    assert "alpha" in out.columns and "a" not in out.columns
    with pytest.raises(CleaningError):
        op(df, "rename_columns", mapping={"a": "b"})  # doublon de nom
    assert list(op(df, "drop_columns", columns=["a", "b"]).columns) == ["c", "d"]


def test_normalize_text(df):
    out = op(df, "normalize_text", columns=["b"], case="title", remove_accents=True)
    assert out["b"].dropna().unique().tolist() == ["Paris", "Lyon"]
    assert out["b"].isna().sum() == 1


def test_filter_rows(df):
    assert len(op(df, "filter_rows", column="c", op=">", value="20")) == 3
    assert len(op(df, "filter_rows", column="c", op="entre", value="10", value2="30")) == 4
    assert len(op(df, "filter_rows", column="b", op="contient", value="lyon")) == 2
    assert len(op(df, "filter_rows", column="a", op="est manquant")) == 1
    assert len(op(df, "filter_rows", column="b", op="dans la liste", value="Lyon;paris")) == 2


def test_outlier_treatments(df):
    capped = op(df, "treat_outliers", columns=["a"], method="iqr", action="cap")
    assert capped["a"].max() < 100
    assert len(op(df, "treat_outliers", columns=["a"], method="iqr", action="remove")) == 5
    assert op(df, "treat_outliers", columns=["a"], method="iqr", action="nan")["a"].isna().sum() == 2


def test_create_variables(df):
    out = op(df, "create_variable", name="ratio", kind="expression", expression="c / 10 + log(c)")
    assert out["ratio"].iloc[0] == pytest.approx(1 + np.log(10))
    out = op(df.rename(columns={"c": "col c"}), "create_variable", name="x2", kind="expression", expression="`col c` * 2")
    assert out["x2"].tolist() == [20, 40, 60, 80, 100, 20]
    out = op(df, "create_variable", name="classe", kind="bin", source="c", bins=[0, 25, 60], labels=["bas", "haut"])
    assert out["classe"].tolist() == ["bas", "bas", "haut", "haut", "haut", "bas"]
    out = op(df, "create_variable", name="mois", kind="date_part", source="d", part="month")
    assert out["mois"].iloc[1] == 2
    out = op(df, "create_variable", name="q", kind="quantile_bin", source="c", bins=2)
    assert out["q"].nunique() == 2


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo pwned')",
    "c.__class__",
    "open('x')",
    "[c for c in range(3)]",
    "lambda: 1",
])
def test_expression_sandbox_rejects_code(df, expr):
    with pytest.raises(CleaningError):
        safe_eval_expression(df, expr)


def test_history_undo_reset(df):
    s = CleaningSession(df)
    s.apply("drop_duplicates")
    s.apply("drop_columns", columns=["d"])
    assert len(s.history) == 2 and s.current.shape == (5, 3)
    s.undo()
    assert len(s.history) == 1 and s.current.shape == (5, 4)
    assert s.history_table()["Lignes"].iloc[0] == "6 → 5"
    s.reset()
    assert not s.history and s.current.shape == df.shape
    assert s.undo() is None
