"""Exports sécurisés de tableaux (CSV, Excel, JSON, Parquet).

Protection contre l'injection de formules (CSV/Formula injection) : toute
cellule texte commençant par ``= + - @`` ou une tabulation / un retour chariot
est préfixée d'une apostrophe, afin qu'un tableur ne l'interprète pas comme une
formule. Les nombres négatifs (type numérique) ne sont pas concernés.
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

_DANGEROUS_PREFIX = ("=", "+", "-", "@", "\t", "\r")
_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def neutralize_cell(value: Any) -> Any:
    """Neutralise une cellule texte susceptible d'être interprétée comme formule."""
    if isinstance(value, str) and value.startswith(_DANGEROUS_PREFIX):
        return "'" + value
    return value


def sanitize_for_spreadsheet(df: pd.DataFrame) -> pd.DataFrame:
    """Copie du DataFrame avec cellules texte et noms de colonnes neutralisés."""
    out = df.copy()
    out.columns = [neutralize_cell(str(c)) for c in out.columns]
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            out[col] = s.astype(object).map(neutralize_cell)
    if out.index.name is not None or not isinstance(out.index, pd.RangeIndex):
        out.index = out.index.map(neutralize_cell)
    return out


def safe_filename(name: str, default: str = "export") -> str:
    """Nom de fichier sans caractères dangereux (pas de chemin, pas d'espace)."""
    base = name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    base = _FILENAME_SAFE.sub("_", base).strip("._")
    return base[:80] or default


def to_csv_bytes(df: pd.DataFrame, sep: str = ",", index: bool = False, encoding: str = "utf-8-sig") -> bytes:
    """CSV neutralisé (UTF-8 avec BOM pour une ouverture correcte dans Excel)."""
    return sanitize_for_spreadsheet(df).to_csv(index=index, sep=sep).encode(encoding)


def _excel_ready(df: pd.DataFrame) -> pd.DataFrame:
    out = sanitize_for_spreadsheet(df)
    for col in out.columns:
        s = out[col]
        if isinstance(s.dtype, pd.DatetimeTZDtype):
            out[col] = s.dt.tz_localize(None)
        elif pd.api.types.is_bool_dtype(s) and s.isna().any():
            out[col] = s.astype(object)
    return out


def to_excel_bytes(sheets: dict[str, pd.DataFrame], index: bool = False) -> bytes:
    """Classeur Excel multi-feuilles (noms de feuilles tronqués à 31 caractères, uniques)."""
    buf = io.BytesIO()
    used: set[str] = set()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            sheet = re.sub(r"[\[\]:*?/\\]", "_", str(name))[:31] or "Feuille"
            base, i = sheet, 1
            while sheet in used:
                sheet = f"{base[:28]}_{i}"
                i += 1
            used.add(sheet)
            data = _excel_ready(df if df is not None and not df.empty else pd.DataFrame({"(vide)": []}))
            data.to_excel(writer, sheet_name=sheet, index=index)
            ws = writer.sheets[sheet]
            for col_cells in ws.columns:
                length = max((len(str(c.value)) for c in col_cells[:200] if c.value is not None), default=8)
                ws.column_dimensions[col_cells[0].column_letter].width = min(max(10, length + 2), 60)
            ws.freeze_panes = "A2"
    return buf.getvalue()


def json_default(obj: Any) -> Any:
    """Sérialiseur JSON pour types NumPy/pandas/dates."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) or np.isinf(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return str(obj)
    if isinstance(obj, pd.DataFrame):
        return json.loads(obj.to_json(orient="records", date_format="iso", force_ascii=False))
    if isinstance(obj, pd.Series):
        return json.loads(obj.to_json(date_format="iso", force_ascii=False))
    if isinstance(obj, set):
        return sorted(obj)
    if obj is pd.NA or obj is pd.NaT:
        return None
    if hasattr(obj, "value"):  # Enum
        return obj.value
    return str(obj)


def clean_for_json(obj: Any) -> Any:
    """Remplace récursivement NaN/inf par None pour produire un JSON valide."""
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_for_json(v) for v in obj]
    return obj


def to_json_bytes(obj: Any) -> bytes:
    """JSON UTF-8 indenté, valide (NaN → null)."""
    text = json.dumps(clean_for_json(json.loads(json.dumps(obj, default=json_default))), ensure_ascii=False, indent=2)
    return text.encode("utf-8")


def dataframe_to_json_bytes(df: pd.DataFrame) -> bytes:
    """Export JSON d'un tableau (orientation « records », dates ISO 8601)."""
    return df.to_json(orient="records", date_format="iso", force_ascii=False, indent=1).encode("utf-8")


def to_parquet_bytes(df: pd.DataFrame) -> bytes:
    """Export Parquet (types conservés)."""
    buf = io.BytesIO()
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype("string")
    out.to_parquet(buf, index=False)
    return buf.getvalue()


def zip_csvs(tables: dict[str, pd.DataFrame]) -> bytes:
    """Archive ZIP contenant un CSV neutralisé par tableau."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, df in tables.items():
            if df is None or df.empty:
                continue
            zf.writestr(f"{safe_filename(name)}.csv", to_csv_bytes(df))
    return buf.getvalue()
