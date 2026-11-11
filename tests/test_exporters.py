"""Tests des exports sécurisés et de la génération des rapports."""
from __future__ import annotations

import io
import json
import zipfile

import pandas as pd
import pytest

from core.data_loader import load_dataset
from core.exceptions import ReportError
from core.pipeline import run_analysis
from core.report_engine import REPORT_SECTIONS, build_report
from utils.exporters import (
    neutralize_cell,
    safe_filename,
    sanitize_for_spreadsheet,
    to_csv_bytes,
    to_excel_bytes,
    to_json_bytes,
    to_parquet_bytes,
)


def test_formula_injection_neutralized():
    df = pd.DataFrame({"=cmd": ["=1+1", "+SUM(A1)", "-2", "@x", "normal", "\tTAB"], "n": [-1, 2, 3, 4, 5, 6]})
    safe = sanitize_for_spreadsheet(df)
    assert safe.columns[0] == "'=cmd"
    assert safe.iloc[:, 0].tolist() == ["'=1+1", "'+SUM(A1)", "'-2", "'@x", "normal", "'\tTAB"]
    assert safe["n"].tolist() == [-1, 2, 3, 4, 5, 6]  # nombres intacts
    text = to_csv_bytes(df).decode("utf-8-sig")
    assert "\n=1+1" not in text and "'=1+1" in text
    wb = pd.read_excel(io.BytesIO(to_excel_bytes({"x": df})))
    assert wb.iloc[0, 0] == "'=1+1"
    assert neutralize_cell(5) == 5


def test_safe_filename():
    assert safe_filename("../../etc/passwd") == "passwd"
    assert safe_filename("mon fichier (1).csv") == "mon_fichier_1_.csv"
    assert safe_filename("") == "export"


def test_json_and_parquet_exports():
    df = pd.DataFrame({"a": [1.0, float("nan")], "d": pd.to_datetime(["2024-01-01", None]), "s": ["x", None]})
    data = json.loads(to_json_bytes({"t": df, "v": float("inf")}))
    assert data["v"] is None and data["t"][1]["a"] is None
    back = pd.read_parquet(io.BytesIO(to_parquet_bytes(df)))
    assert back.shape == df.shape


@pytest.fixture(scope="module")
def bundle():
    from utils.sample_factory import mixed_dataset

    df = mixed_dataset(300)
    res = load_dataset("ventes.csv", df.to_csv(index=False).encode())
    return run_analysis(res.df, res.meta)


def test_pdf_report(bundle):
    pdf = build_report(bundle, "PDF")
    assert pdf.startswith(b"%PDF") and len(pdf) > 20_000
    fitz = pytest.importorskip("fitz")
    doc = fitz.open(stream=pdf, filetype="pdf")
    text = "".join(page.get_text() for page in doc)
    assert "Sommaire" in text and "Synthèse exécutive" in text
    for label in REPORT_SECTIONS.values():
        assert label in text
    assert doc.page_count >= 6


def test_html_report_sections_and_escaping(bundle):
    html = build_report(bundle, "HTML", sections=["presentation", "quality"], title="<script>alert(1)</script>").decode()
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;" in html
    assert REPORT_SECTIONS["quality"] in html and REPORT_SECTIONS["correlations"] not in html


def test_excel_csv_json_reports(bundle):
    xlsx = build_report(bundle, "Excel")
    sheets = pd.ExcelFile(io.BytesIO(xlsx)).sheet_names
    assert {"Synthèse", "Variables", "Stats quantitatives", "Audit qualité"} <= set(sheets)
    with zipfile.ZipFile(io.BytesIO(build_report(bundle, "CSV (ZIP)"))) as zf:
        assert any(n.endswith(".csv") for n in zf.namelist()) and len(zf.namelist()) > 5
    data = json.loads(build_report(bundle, "JSON", sections=["statistics", "quality"]))
    assert data["qualite"]["score_global"] == bundle.quality.global_score
    assert "variables" not in data
    stats = {r["Variable"]: r for r in data["statistiques_quantitatives"]}
    assert stats["quantite"]["Moyenne"] == pytest.approx(bundle.univariate.numeric["quantite"].mean)


def test_report_without_charts_and_errors(bundle):
    assert build_report(bundle, "PDF", sections=["general"], include_charts=False).startswith(b"%PDF")
    with pytest.raises(ReportError):
        build_report(bundle, "DOCX")
    with pytest.raises(ReportError):
        build_report(bundle, "PDF", sections=["inexistante"])
