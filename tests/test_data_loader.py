"""Tests du chargement multiformat, des options et des protections de sécurité."""
from __future__ import annotations

import io
import sqlite3
import zipfile

import pandas as pd
import pytest

from core.data_loader import (
    LoadOptions,
    inspect_file,
    load_dataset,
    sanitize_dataframe,
    supported_formats,
)
from core.exceptions import DataLoadError, EmptyDatasetError, FileValidationError
from utils import sample_factory as sf


def _bytes(writer) -> bytes:
    buf = io.BytesIO()
    writer(buf)
    return buf.getvalue()


def test_all_core_formats_registered():
    for fmt in ["csv", "tsv", "txt", "xls", "xlsx", "xlsm", "ods", "parquet", "feather", "orc", "json", "jsonl",
                "xml", "sav", "dta", "sas7bdat", "sqlite"]:
        assert fmt in supported_formats()


@pytest.mark.parametrize("name,writer", [
    ("d.csv", lambda df, b: b.write(df.to_csv(index=False).encode())),
    ("d.tsv", lambda df, b: b.write(df.to_csv(index=False, sep="\t").encode())),
    ("d.txt", lambda df, b: b.write(df.to_csv(index=False, sep="|").encode())),
    ("d.xlsx", lambda df, b: df.to_excel(b, index=False)),
    ("d.ods", lambda df, b: df.to_excel(b, index=False, engine="odf")),
    ("d.parquet", lambda df, b: df.to_parquet(b, index=False)),
    ("d.feather", lambda df, b: df.to_feather(b)),
    ("d.json", lambda df, b: b.write(df.to_json(orient="records").encode())),
    ("d.jsonl", lambda df, b: b.write(df.to_json(orient="records", lines=True).encode())),
    ("d.xml", lambda df, b: df.to_xml(b, index=False, parser="etree")),
    ("d.dta", lambda df, b: df.to_stata(b, write_index=False)),
])
def test_roundtrip_formats(name, writer, numeric_df):
    content = _bytes(lambda b: writer(numeric_df, b))
    res = load_dataset(name, content)
    assert res.df.shape == numeric_df.shape
    assert list(res.df.columns) == list(numeric_df.columns)
    assert res.df["age"].sum() == numeric_df["age"].sum()


def test_orc_or_clear_error(numeric_df):
    try:
        content = _bytes(lambda b: numeric_df.to_orc(b, index=False))
    except Exception:  # noqa: BLE001 - ORC indisponible sur la plateforme
        pytest.skip("Écriture ORC indisponible")
    assert load_dataset("d.orc", content).df.shape == numeric_df.shape


def test_sav_with_value_labels(tmp_path, categorical_df):
    pyreadstat = pytest.importorskip("pyreadstat")
    path = tmp_path / "s.sav"
    df = categorical_df.copy()
    df["code"] = (df["genre"] == "F").astype(int)
    pyreadstat.write_sav(df, str(path), variable_value_labels={"code": {0: "Homme", 1: "Femme"}})
    res = load_dataset("s.sav", path.read_bytes())
    assert res.df.shape == df.shape
    assert res.meta.value_labels["code"]["1.0"] == "Femme" or res.meta.value_labels["code"].get("1") == "Femme"


def test_french_csv_cp1252():
    df = pd.DataFrame({"prénom": ["Élodie", "Joël"], "taille": [1.65, 1.8]})
    content = df.to_csv(index=False, sep=";", decimal=",").encode("cp1252")
    res = load_dataset("fr.csv", content)
    assert list(res.df.columns) == ["prénom", "taille"]
    assert res.df["prénom"].tolist() == ["Élodie", "Joël"]
    assert res.df["taille"].tolist() == [1.65, 1.8]
    assert res.meta.separator == ";" and res.meta.decimal == ","


def test_excel_multiple_sheets(numeric_df, categorical_df):
    def write(b):
        with pd.ExcelWriter(b, engine="openpyxl") as w:
            numeric_df.to_excel(w, sheet_name="Num", index=False)
            categorical_df.to_excel(w, sheet_name="Cat", index=False)

    content = _bytes(write)
    insp = inspect_file("multi.xlsx", content)
    assert insp.sheets == ["Num", "Cat"]
    res = load_dataset("multi.xlsx", content, LoadOptions(sheet="Cat"))
    assert res.df.shape == categorical_df.shape and res.meta.sheet == "Cat"
    with pytest.raises(DataLoadError):
        load_dataset("multi.xlsx", content, LoadOptions(sheet="Inexistante"))


def test_sqlite_tables_and_injection_guard(tmp_path, numeric_df):
    db = tmp_path / "b.sqlite"
    with sqlite3.connect(db) as conn:
        numeric_df.to_sql("mesures", conn, index=False)
        numeric_df.head(5).to_sql("petite table", conn, index=False)
    conn.close()
    content = db.read_bytes()
    assert inspect_file("b.sqlite", content).tables == ["mesures", "petite table"]
    assert load_dataset("b.sqlite", content, LoadOptions(table="petite table")).df.shape == (5, numeric_df.shape[1])
    with pytest.raises(DataLoadError):
        load_dataset("b.sqlite", content, LoadOptions(table='mesures"; DROP TABLE mesures; --'))
    # La base n'a pas été modifiée (ouverture en lecture seule)
    assert load_dataset("b.sqlite", content, LoadOptions(table="mesures")).df.shape == numeric_df.shape


def test_row_limit_partial_read(numeric_df):
    content = numeric_df.to_csv(index=False).encode()
    res = load_dataset("d.csv", content, LoadOptions(row_limit=50))
    assert len(res.df) == 50 and res.meta.truncated
    content = _bytes(lambda b: numeric_df.to_parquet(b))
    res = load_dataset("d.parquet", content, LoadOptions(row_limit=30))
    assert len(res.df) == 30 and res.meta.truncated and res.meta.estimated_total_rows == len(numeric_df)


def test_nested_json_is_flattened():
    content = b'{"data": [{"id": 1, "user": {"name": "A", "age": 30}, "tags": ["x", "y"]}, ' \
              b'{"id": 2, "user": {"name": "B", "age": 40}, "tags": []}]}'
    res = load_dataset("n.json", content)
    assert {"id", "user.name", "user.age", "tags"} <= set(res.df.columns)
    assert res.df["tags"].iloc[0] == '["x", "y"]'  # listes sérialisées (cellules hachables)


def test_empty_dataset_errors():
    with pytest.raises(EmptyDatasetError):
        load_dataset("e.csv", sf.empty_dataset().to_csv(index=False).encode())
    with pytest.raises(FileValidationError):
        load_dataset("e.csv", b"")


def test_single_column_and_unusual_names(unusual_df):
    res = load_dataset("one.csv", sf.single_column_dataset().to_csv(index=False).encode())
    assert res.df.shape == (10, 1)
    res = load_dataset("u.csv", unusual_df.to_csv(index=False, sep=";").encode())
    assert res.df.shape == unusual_df.shape
    assert "Prénom de l'élève" in res.df.columns  # espaces superflus retirés
    assert "📊 score" in res.df.columns


def test_duplicate_and_empty_column_names_are_made_unique():
    df = pd.DataFrame([[1, 2, 3]], columns=["a", "a", None])
    out = sanitize_dataframe(df)
    assert len(set(out.columns)) == 3 and "colonne_3" in out.columns


def test_corrupted_file_message():
    content = b"PK\x03\x04" + b"\x00" * 200  # faux XLSX
    with pytest.raises((DataLoadError, FileValidationError)):
        load_dataset("corrompu.xlsx", content)


def test_zip_loading_and_member_choice(numeric_df, categorical_df):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("num.csv", numeric_df.to_csv(index=False))
        zf.writestr("cat.csv", categorical_df.to_csv(index=False))
    content = buf.getvalue()
    assert inspect_file("a.zip", content).zip_members == ["num.csv", "cat.csv"]
    res = load_dataset("a.zip", content, LoadOptions(zip_member="cat.csv"))
    assert res.df.shape == categorical_df.shape and res.meta.zip_member == "cat.csv"


def test_zip_path_traversal_rejected():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../../evil.csv", "a,b\n1,2\n")
    with pytest.raises(FileValidationError):
        load_dataset("evil.zip", buf.getvalue())


def test_zip_bomb_ratio_rejected():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("bomb.csv", "a\n" + "0\n" * 2_000_000)
    with pytest.raises(FileValidationError, match="compression"):
        load_dataset("bomb.zip", buf.getvalue())


def test_xml_xxe_rejected():
    xxe = b"""<?xml version="1.0"?><!DOCTYPE r [<!ENTITY x SYSTEM "file:///etc/passwd">]>
    <rows><row><a>&x;</a></row></rows>"""
    with pytest.raises(FileValidationError):
        load_dataset("x.xml", xxe)


def test_malformed_csv_lines_skipped():
    content = b"a,b\n1,2\n3,4,5,6\n7,8\n"
    res = load_dataset("m.csv", content)
    assert len(res.df) >= 2
