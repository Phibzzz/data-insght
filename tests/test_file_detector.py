"""Tests de la reconnaissance des formats, encodages et séparateurs."""
from __future__ import annotations

import io
import sqlite3
import zipfile

import pandas as pd
import pytest

from core.exceptions import FileValidationError, UnsupportedFormatError
from core.file_detector import detect_decimal, detect_encoding, detect_file, detect_separator, get_extension, human_size


def test_extension_parsing():
    assert get_extension("data.CSV") == "csv"
    assert get_extension("archive.tar.sas7bdat") == "sas7bdat"
    assert get_extension("noext") == ""


@pytest.mark.parametrize("sep", [",", ";", "\t", "|"])
def test_separator_detection(sep):
    text = sep.join(["a", "b", "c"]) + "\n" + "\n".join(sep.join(["1", "x y", "3"]) for _ in range(10))
    assert detect_separator(text) == sep


def test_separator_with_french_decimals():
    text = "taille;poids\n170,5;65,2\n180,1;80,0\n165,0;55,3\n"
    assert detect_separator(text) == ";"
    assert detect_decimal(text, ";") == ","


def test_separator_with_quoted_commas():
    text = 'nom,commentaire\nA,"bon, très bon"\nB,"moyen, sans plus"\nC,"ok"\n'
    assert detect_separator(text) == ","


def test_encoding_detection():
    assert detect_encoding("é,à\n".encode("utf-8"))[0] == "utf-8"
    assert detect_encoding("﻿a,b".encode("utf-8"))[0] == "utf-8-sig"
    enc, _ = detect_encoding("prénom;âge;numéro\nÉlodie;30;é".encode("cp1252"))
    assert enc in {"cp1252", "latin-1", "iso8859-1"}


def test_detect_csv_and_tsv(numeric_df):
    csv_bytes = numeric_df.to_csv(index=False).encode()
    det = detect_file("x.csv", csv_bytes)
    assert det.detected_format == "csv" and det.separator == ","
    det = detect_file("x.tsv", numeric_df.to_csv(index=False, sep="\t").encode())
    assert det.detected_format == "tsv" and det.separator == "\t"


def test_detect_json_and_jsonl(categorical_df):
    assert detect_file("a.json", categorical_df.to_json(orient="records").encode()).detected_format == "json"
    jl = categorical_df.to_json(orient="records", lines=True).encode()
    assert detect_file("a.jsonl", jl).detected_format == "jsonl"
    # Extension trompeuse : JSON Lines nommé .json
    assert detect_file("a.json", jl).detected_format == "jsonl"


def test_detect_binary_formats(tmp_path, numeric_df):
    buf = io.BytesIO()
    numeric_df.to_parquet(buf)
    assert detect_file("a.parquet", buf.getvalue()).detected_format == "parquet"
    buf = io.BytesIO()
    numeric_df.to_excel(buf, index=False)
    assert detect_file("a.xlsx", buf.getvalue()).detected_format == "xlsx"
    buf = io.BytesIO()
    numeric_df.to_feather(buf)
    assert detect_file("a.feather", buf.getvalue()).detected_format == "feather"
    db = tmp_path / "a.db"
    with sqlite3.connect(db) as conn:
        numeric_df.to_sql("t", conn)
    conn.close()
    assert detect_file("a.db", db.read_bytes()).detected_format == "sqlite"


def test_real_format_wins_over_extension(numeric_df):
    buf = io.BytesIO()
    numeric_df.to_parquet(buf)
    det = detect_file("trompeur.csv", buf.getvalue())
    assert det.detected_format == "parquet"
    assert det.warnings, "un avertissement doit signaler l'incohérence d'extension"


def test_zip_and_ods_detection(categorical_df):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("a.csv", "x,y\n1,2\n")
    assert detect_file("a.zip", buf.getvalue()).detected_format == "zip"
    buf = io.BytesIO()
    categorical_df.to_excel(buf, index=False, engine="odf")
    assert detect_file("a.ods", buf.getvalue()).detected_format == "ods"


def test_empty_and_unknown_binary():
    with pytest.raises(FileValidationError):
        detect_file("vide.csv", b"")
    with pytest.raises(UnsupportedFormatError):
        detect_file("bin.dat", b"\x00\x01\x02\x03" * 100)


def test_size_limit(monkeypatch):
    from config.settings import SETTINGS

    monkeypatch.setattr(SETTINGS.loading, "max_file_size_mb", 0)
    with pytest.raises(FileValidationError):
        detect_file("a.csv", b"a,b\n1,2\n")


def test_human_size():
    assert human_size(500) == "500 o"
    assert human_size(2048).startswith("2,0") or human_size(2048).startswith("2.0")


def test_xml_detection():
    xml = pd.DataFrame({"a": [1, 2]}).to_xml(parser="etree").encode()
    assert detect_file("a.xml", xml).detected_format == "xml"
