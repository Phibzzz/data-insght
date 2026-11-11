"""Fonctions utilitaires de formatage (affichage à la française)."""
from __future__ import annotations

import hashlib
import math
from typing import Any

import pandas as pd

NBSP = " "


def fmt_int(n: Any) -> str:
    """Entier avec séparateur de milliers (espace fine)."""
    try:
        return f"{int(n):,}".replace(",", NBSP)
    except (TypeError, ValueError):
        return "—"


def fmt_num(x: Any, digits: int = 2) -> str:
    """Nombre décimal à la française (virgule décimale)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(v) or math.isinf(v):
        return "—"
    if abs(v) >= 1e6 or (0 < abs(v) < 10 ** -digits):
        return f"{v:.{digits}e}".replace(".", ",")
    return f"{v:,.{digits}f}".replace(",", NBSP).replace(".", ",")


def fmt_pct(x: Any, digits: int = 1, ratio: bool = True) -> str:
    """Pourcentage ; `ratio=True` si x est une proportion (0–1)."""
    try:
        v = float(x) * (100 if ratio else 1)
    except (TypeError, ValueError):
        return "—"
    return f"{v:.{digits}f} %".replace(".", ",")


def round_numeric(df: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    """Arrondit les colonnes flottantes d'un tableau pour l'affichage."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(digits)
    return out


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    """Empreinte stable d'un DataFrame (clé de cache)."""
    h = hashlib.sha256()
    h.update(str(df.shape).encode())
    h.update("|".join(map(str, df.columns)).encode())
    h.update("|".join(map(str, df.dtypes)).encode())
    try:
        h.update(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    except TypeError:
        h.update(pd.util.hash_pandas_object(df.astype(str), index=False).values.tobytes())
    return h.hexdigest()[:16]


def display_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Copie affichable par Streamlit/Arrow (types mixtes convertis en texte)."""
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if s.dtype == object:
            types = {type(v) for v in s.dropna().head(1000)}
            if len(types) > 1 or (types and not types <= {str}):
                out[col] = s.map(lambda v: None if v is None or (isinstance(v, float) and math.isnan(v)) else str(v))
    out.columns = [str(c) for c in out.columns]
    return out
