"""Chargement centralisé des jeux de données multiformats.

Architecture extensible : chaque format est servi par une fonction enregistrée
via le décorateur :func:`register_loader`. Ajouter un format consiste à écrire
une fonction ``(content, detection, options) -> (DataFrame, extras)`` puis à
déclarer son extension dans ``core/file_detector.py``.

Sécurité :
* aucun code contenu dans les fichiers n'est exécuté (pas de pickle, macros ignorées) ;
* les bases SQLite sont ouvertes en lecture seule et les noms de tables validés ;
* le XML est analysé sans DTD ni entités externes (protection XXE) ;
* les archives ZIP sont contrôlées (zip-bomb, chemins, nombre de fichiers).
"""
from __future__ import annotations

import io
import json
import os
import sqlite3
import tempfile
import zipfile
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Iterator

import pandas as pd

from config.settings import SETTINGS
from core.exceptions import (
    DataLoadError,
    EmptyDatasetError,
    FileValidationError,
    MissingDependencyError,
    UnsupportedFormatError,
)
from core.file_detector import FORMAT_LABELS, DetectionResult, detect_file


@dataclass
class LoadOptions:
    """Options de chargement choisies par l'utilisateur."""

    sheet: str | int | None = None
    table: str | None = None
    zip_member: str | None = None
    row_limit: int | None = None
    encoding: str | None = None
    separator: str | None = None
    decimal: str | None = None
    header_row: int | None = 0
    use_polars: bool = False


@dataclass
class FileMeta:
    """Métadonnées décrivant le fichier chargé."""

    filename: str
    detected_format: str
    format_label: str
    size_bytes: int
    encoding: str | None = None
    separator: str | None = None
    decimal: str = "."
    sheet: str | None = None
    sheets: list[str] = field(default_factory=list)
    table: str | None = None
    tables: list[str] = field(default_factory=list)
    zip_member: str | None = None
    truncated: bool = False
    estimated_total_rows: int | None = None
    value_labels: dict[str, dict[Any, str]] = field(default_factory=dict)
    column_labels: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Représentation sérialisable (pour les rapports)."""
        return {
            "fichier": self.filename,
            "format": self.format_label,
            "taille_octets": self.size_bytes,
            "encodage": self.encoding,
            "separateur": self.separator,
            "feuille": self.sheet,
            "table": self.table,
            "membre_zip": self.zip_member,
            "tronque": self.truncated,
            "avertissements": list(self.warnings),
        }


@dataclass
class LoadResult:
    """Résultat d'un chargement : données + métadonnées."""

    df: pd.DataFrame
    meta: FileMeta


@dataclass
class FileInspection:
    """Informations préalables au chargement (choix de feuille, table, membre ZIP)."""

    detection: DetectionResult
    sheets: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    zip_members: list[str] = field(default_factory=list)
    estimated_rows: int | None = None


LoaderFn = Callable[[bytes, DetectionResult, LoadOptions], tuple[pd.DataFrame, dict[str, Any]]]
_LOADERS: dict[str, LoaderFn] = {}


def register_loader(*formats: str) -> Callable[[LoaderFn], LoaderFn]:
    """Décorateur enregistrant une fonction de chargement pour un ou plusieurs formats."""

    def decorator(fn: LoaderFn) -> LoaderFn:
        for fmt in formats:
            _LOADERS[fmt] = fn
        return fn

    return decorator


def supported_formats() -> list[str]:
    """Formats pour lesquels un chargeur est enregistré."""
    return sorted(_LOADERS)


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

@contextmanager
def _temp_file(content: bytes, suffix: str) -> Iterator[str]:
    """Écrit le contenu dans un fichier temporaire (requis par certaines bibliothèques)."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(content)
        yield path
    finally:
        try:
            os.remove(path)
        except OSError:  # pragma: no cover - verrou Windows éventuel
            pass


def _require(module: str, fmt: str) -> Any:
    try:
        return __import__(module, fromlist=["_"])
    except ImportError as exc:
        raise MissingDependencyError(
            f"Le format {fmt} nécessite la bibliothèque « {module} » (pip install {module}).",
            detail=str(exc),
        ) from exc


def _dedupe_columns(columns: list[Any]) -> list[str]:
    """Convertit les noms de colonnes en chaînes uniques et non vides."""
    seen: dict[str, int] = {}
    result: list[str] = []
    for i, col in enumerate(columns):
        name = "" if col is None or (isinstance(col, float) and pd.isna(col)) else str(col).strip()
        if not name or name.lower().startswith("unnamed:"):
            name = f"colonne_{i + 1}"
        base = name
        if name in seen:
            seen[base] += 1
            name = f"{base}_{seen[base]}"
            while name in seen:
                seen[base] += 1
                name = f"{base}_{seen[base]}"
        seen.setdefault(name, 0)
        result.append(name)
    return result


def _is_unhashable(value: Any) -> bool:
    return isinstance(value, (list, dict, set, tuple, bytes, bytearray))


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise un DataFrame chargé : noms de colonnes, cellules non hachables."""
    df = df.copy()
    df.columns = _dedupe_columns(list(df.columns))
    for col in df.columns:
        series = df[col]
        if series.dtype == object:
            sample = series.dropna().head(500)
            if any(_is_unhashable(v) for v in sample) or series.dropna().map(_is_unhashable).any():
                df[col] = series.map(
                    lambda v: v.decode("utf-8", "replace") if isinstance(v, (bytes, bytearray))
                    else (json.dumps(v, ensure_ascii=False, default=str) if _is_unhashable(v) else v)
                )
    if isinstance(df.index, pd.MultiIndex) or not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index(drop=True)
    return df


def _count_lines(content: bytes) -> int:
    return max(content.count(b"\n") - 1, 0) + (0 if content.endswith(b"\n") else 1)


# ---------------------------------------------------------------------------
# Chargeurs par format
# ---------------------------------------------------------------------------

@register_loader("csv", "tsv", "txt")
def _load_delimited(content: bytes, det: DetectionResult, opt: LoadOptions) -> tuple[pd.DataFrame, dict]:
    encoding = opt.encoding or det.encoding or "utf-8"
    sep = opt.separator or det.separator or ","
    decimal = opt.decimal or det.decimal or "."
    extras: dict[str, Any] = {"encoding": encoding, "separator": sep, "decimal": decimal}

    if opt.use_polars:
        try:
            import polars as pl  # type: ignore

            pl_df = pl.read_csv(
                io.BytesIO(content.decode(encoding, errors="replace").encode("utf-8")),
                separator=sep, n_rows=opt.row_limit, infer_schema_length=10_000,
                has_header=opt.header_row is not None, ignore_errors=True,
            )
            df = pl_df.to_pandas()
            if decimal == ",":
                extras.setdefault("warnings", []).append(
                    "Polars ne gère pas la virgule décimale : conversion assurée par la détection des types."
                )
            return df, extras
        except ImportError:
            extras.setdefault("warnings", []).append("Polars n'est pas installé : lecture avec pandas.")

    kwargs: dict[str, Any] = dict(
        sep=sep, encoding=encoding, decimal=decimal, nrows=opt.row_limit,
        header=opt.header_row, low_memory=False, skipinitialspace=False, index_col=False,
    )
    try:
        df = pd.read_csv(io.BytesIO(content), **kwargs)
    except UnicodeDecodeError:
        kwargs["encoding"] = "latin-1"
        extras["encoding"] = "latin-1"
        extras.setdefault("warnings", []).append("Encodage incertain : lecture en latin-1.")
        df = pd.read_csv(io.BytesIO(content), **kwargs)
    except pd.errors.ParserError:
        df = pd.read_csv(io.BytesIO(content), on_bad_lines="skip", **kwargs)
        extras.setdefault("warnings", []).append(
            "Certaines lignes mal formées (nombre de champs incorrect) ont été ignorées."
        )
    if opt.row_limit is not None and len(df) >= opt.row_limit:
        extras["truncated"] = True
        extras["estimated_total_rows"] = _count_lines(content)
    return df, extras


def _excel_engine(fmt: str) -> str:
    return {"xls": "xlrd", "ods": "odf"}.get(fmt, "openpyxl")


def list_sheets(content: bytes, fmt: str) -> list[str]:
    """Liste les feuilles d'un classeur Excel/ODS."""
    engine = _excel_engine(fmt)
    _require({"xlrd": "xlrd", "odf": "odf", "openpyxl": "openpyxl"}[engine], fmt.upper())
    try:
        with pd.ExcelFile(io.BytesIO(content), engine=engine) as xls:
            return [str(s) for s in xls.sheet_names]
    except Exception as exc:
        raise DataLoadError("Impossible de lire la liste des feuilles du classeur.", detail=str(exc)) from exc


@register_loader("xls", "xlsx", "xlsm", "ods")
def _load_excel(content: bytes, det: DetectionResult, opt: LoadOptions) -> tuple[pd.DataFrame, dict]:
    sheets = list_sheets(content, det.detected_format)
    if not sheets:
        raise EmptyDatasetError("Le classeur ne contient aucune feuille.")
    sheet = opt.sheet if opt.sheet is not None else sheets[0]
    if isinstance(sheet, int):
        sheet = sheets[sheet]
    if sheet not in sheets:
        raise DataLoadError(f"La feuille « {sheet} » n'existe pas dans le classeur.")
    # openpyxl ne lit jamais les macros VBA : le contenu XLSM est traité comme des données.
    df = pd.read_excel(
        io.BytesIO(content), sheet_name=sheet, engine=_excel_engine(det.detected_format),
        nrows=opt.row_limit, header=opt.header_row,
    )
    extras: dict[str, Any] = {"sheet": sheet, "sheets": sheets}
    if det.detected_format == "xlsm":
        extras["warnings"] = ["Fichier XLSM : les macros sont ignorées et ne sont jamais exécutées."]
    if opt.row_limit is not None and len(df) >= opt.row_limit:
        extras["truncated"] = True
    return df, extras


@register_loader("parquet")
def _load_parquet(content: bytes, det: DetectionResult, opt: LoadOptions) -> tuple[pd.DataFrame, dict]:
    pq = _require("pyarrow.parquet", "Parquet")
    pf = pq.ParquetFile(io.BytesIO(content))
    total = pf.metadata.num_rows
    extras: dict[str, Any] = {"estimated_total_rows": total}
    if opt.row_limit is not None and total > opt.row_limit:
        import pyarrow as pa

        batches, read = [], 0
        for batch in pf.iter_batches(batch_size=min(65_536, opt.row_limit)):
            batches.append(batch)
            read += batch.num_rows
            if read >= opt.row_limit:
                break
        table = pa.Table.from_batches(batches).slice(0, opt.row_limit)
        extras["truncated"] = True
    else:
        table = pf.read()
    return table.to_pandas(), extras


@register_loader("feather")
def _load_feather(content: bytes, det: DetectionResult, opt: LoadOptions) -> tuple[pd.DataFrame, dict]:
    feather = _require("pyarrow.feather", "Feather")
    table = feather.read_table(io.BytesIO(content))
    extras: dict[str, Any] = {"estimated_total_rows": table.num_rows}
    if opt.row_limit is not None and table.num_rows > opt.row_limit:
        table = table.slice(0, opt.row_limit)
        extras["truncated"] = True
    return table.to_pandas(), extras


@register_loader("orc")
def _load_orc(content: bytes, det: DetectionResult, opt: LoadOptions) -> tuple[pd.DataFrame, dict]:
    try:
        from pyarrow import orc  # type: ignore
    except ImportError as exc:
        raise MissingDependencyError(
            "La lecture ORC n'est pas disponible avec votre installation de pyarrow "
            "(support absent sur certaines plateformes, notamment Windows).",
            detail=str(exc),
        ) from exc
    table = orc.ORCFile(io.BytesIO(content)).read()
    extras: dict[str, Any] = {"estimated_total_rows": table.num_rows}
    if opt.row_limit is not None and table.num_rows > opt.row_limit:
        table = table.slice(0, opt.row_limit)
        extras["truncated"] = True
    return table.to_pandas(), extras


def _json_to_frame(data: Any) -> pd.DataFrame:
    """Convertit une structure JSON arbitraire en table."""
    if isinstance(data, list):
        if not data:
            return pd.DataFrame()
        if all(isinstance(r, dict) for r in data):
            return pd.json_normalize(data, sep=".")
        return pd.DataFrame({"valeur": data})
    if isinstance(data, dict):
        # Cas fréquent : {"data": [ {...}, ... ]}
        list_keys = [k for k, v in data.items() if isinstance(v, list) and v and all(isinstance(r, dict) for r in v)]
        if len(list_keys) == 1:
            return pd.json_normalize(data[list_keys[0]], sep=".")
        # Orientation colonnes : {"col": [v1, v2]} ou {"col": {"0": v1}}
        if data and all(isinstance(v, (list, dict)) for v in data.values()):
            try:
                return pd.DataFrame(data)
            except ValueError:
                pass
        return pd.json_normalize([data], sep=".")
    return pd.DataFrame({"valeur": [data]})


@register_loader("json")
def _load_json(content: bytes, det: DetectionResult, opt: LoadOptions) -> tuple[pd.DataFrame, dict]:
    encoding = opt.encoding or det.encoding or "utf-8"
    try:
        data = json.loads(content.decode(encoding, errors="replace").lstrip("﻿"))
    except json.JSONDecodeError as exc:
        # Peut-être du JSON Lines avec une extension .json
        try:
            return _load_jsonl(content, det, opt)
        except DataLoadError:
            raise DataLoadError(
                f"JSON invalide (ligne {exc.lineno}, colonne {exc.colno}) : {exc.msg}.",
            ) from exc
    df = _json_to_frame(data)
    extras: dict[str, Any] = {"encoding": encoding}
    if opt.row_limit is not None and len(df) > opt.row_limit:
        extras["estimated_total_rows"] = len(df)
        df = df.head(opt.row_limit)
        extras["truncated"] = True
    return df, extras


@register_loader("jsonl")
def _load_jsonl(content: bytes, det: DetectionResult, opt: LoadOptions) -> tuple[pd.DataFrame, dict]:
    encoding = opt.encoding or det.encoding or "utf-8"
    records, bad = [], 0
    for line in content.decode(encoding, errors="replace").lstrip("﻿").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
        if opt.row_limit is not None and len(records) >= opt.row_limit:
            break
    if not records and bad:
        raise DataLoadError("Aucune ligne JSON valide n'a pu être lue.")
    df = _json_to_frame(records)
    extras: dict[str, Any] = {"encoding": encoding}
    if bad:
        extras["warnings"] = [f"{bad} ligne(s) JSON invalide(s) ignorée(s)."]
    if opt.row_limit is not None and len(records) >= opt.row_limit:
        extras["truncated"] = True
    return df, extras


@register_loader("xml")
def _load_xml(content: bytes, det: DetectionResult, opt: LoadOptions) -> tuple[pd.DataFrame, dict]:
    head = content[:4096].lower()
    if b"<!doctype" in head or b"<!entity" in content.lower():
        raise FileValidationError(
            "Le fichier XML contient une déclaration DTD/ENTITY : refusé par sécurité (risque XXE)."
        )
    try:
        from defusedxml import ElementTree as DET  # type: ignore

        DET.fromstring(content, forbid_dtd=True, forbid_entities=True, forbid_external=True)
    except ImportError:
        pass
    except Exception as exc:
        raise DataLoadError("Le fichier XML est mal formé ou non sécurisé.", detail=str(exc)) from exc
    try:
        df = pd.read_xml(io.BytesIO(content), parser="etree")
    except Exception as exc:
        raise DataLoadError(
            "Structure XML non tabulaire : attendu une racine contenant des éléments répétés "
            "(ex. <rows><row>…</row></rows>).",
            detail=str(exc),
        ) from exc
    extras: dict[str, Any] = {}
    if opt.row_limit is not None and len(df) > opt.row_limit:
        extras["estimated_total_rows"] = len(df)
        df = df.head(opt.row_limit)
        extras["truncated"] = True
    return df, extras


@register_loader("sav")
def _load_sav(content: bytes, det: DetectionResult, opt: LoadOptions) -> tuple[pd.DataFrame, dict]:
    pyreadstat = _require("pyreadstat", "SPSS (SAV)")
    with _temp_file(content, ".sav") as path:
        kwargs = {"row_limit": opt.row_limit} if opt.row_limit else {}
        df, meta = pyreadstat.read_sav(path, **kwargs)
    extras: dict[str, Any] = {
        "value_labels": {k: {str(a): str(b) for a, b in v.items()} for k, v in (meta.variable_value_labels or {}).items()},
        "column_labels": {k: v for k, v in zip(meta.column_names, meta.column_labels or []) if v},
        "estimated_total_rows": meta.number_rows,
    }
    if opt.row_limit and meta.number_rows and meta.number_rows > opt.row_limit:
        extras["truncated"] = True
    return df, extras


@register_loader("dta")
def _load_dta(content: bytes, det: DetectionResult, opt: LoadOptions) -> tuple[pd.DataFrame, dict]:
    extras: dict[str, Any] = {}
    try:
        with pd.io.stata.StataReader(io.BytesIO(content)) as reader:
            df = reader.read(nrows=opt.row_limit, convert_categoricals=False)
            try:
                extras["value_labels"] = {
                    k: {str(a): str(b) for a, b in v.items()} for k, v in reader.value_labels().items()
                }
                extras["column_labels"] = {k: v for k, v in reader.variable_labels().items() if v}
            except Exception:  # pragma: no cover - labels optionnels
                pass
    except Exception as exc:
        raise DataLoadError("Impossible de lire le fichier Stata.", detail=str(exc)) from exc
    if opt.row_limit is not None and len(df) >= opt.row_limit:
        extras["truncated"] = True
    return df, extras


@register_loader("sas7bdat")
def _load_sas(content: bytes, det: DetectionResult, opt: LoadOptions) -> tuple[pd.DataFrame, dict]:
    try:
        df = pd.read_sas(io.BytesIO(content), format="sas7bdat", encoding="infer")
    except Exception as exc:
        try:
            pyreadstat = _require("pyreadstat", "SAS")
            with _temp_file(content, ".sas7bdat") as path:
                df, _ = pyreadstat.read_sas7bdat(path)
        except Exception:
            raise DataLoadError("Impossible de lire le fichier SAS.", detail=str(exc)) from exc
    extras: dict[str, Any] = {}
    if opt.row_limit is not None and len(df) > opt.row_limit:
        extras["estimated_total_rows"] = len(df)
        df = df.head(opt.row_limit)
        extras["truncated"] = True
    return df, extras


def _sqlite_connect(path: str) -> sqlite3.Connection:
    uri = "file:" + path.replace("\\", "/") + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.execute("PRAGMA query_only = ON")
    return conn


def list_sqlite_tables(content: bytes) -> list[str]:
    """Liste les tables et vues d'une base SQLite (lecture seule)."""
    with _temp_file(content, ".sqlite") as path:
        conn = _sqlite_connect(path)
        try:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','view') "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise DataLoadError("Base SQLite illisible ou corrompue.", detail=str(exc)) from exc
        finally:
            conn.close()
    return [r[0] for r in rows]


@register_loader("sqlite")
def _load_sqlite(content: bytes, det: DetectionResult, opt: LoadOptions) -> tuple[pd.DataFrame, dict]:
    tables = list_sqlite_tables(content)
    if not tables:
        raise EmptyDatasetError("La base SQLite ne contient aucune table.")
    table = opt.table or tables[0]
    if table not in tables:  # validation stricte contre sqlite_master (anti-injection)
        raise DataLoadError(f"La table « {table} » n'existe pas dans la base.")
    limit = opt.row_limit or SETTINGS.loading.sqlite_max_rows
    quoted = '"' + table.replace('"', '""') + '"'
    with _temp_file(content, ".sqlite") as path:
        conn = _sqlite_connect(path)
        try:
            total = conn.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0]
            df = pd.read_sql_query(f"SELECT * FROM {quoted} LIMIT ?", conn, params=(int(limit),))
        finally:
            conn.close()
    extras: dict[str, Any] = {"table": table, "tables": tables, "estimated_total_rows": total}
    if total > limit:
        extras["truncated"] = True
    return df, extras


# ---------------------------------------------------------------------------
# Archives ZIP
# ---------------------------------------------------------------------------

def list_zip_members(content: bytes) -> list[str]:
    """Liste les fichiers exploitables d'une archive ZIP après contrôles de sécurité."""
    limits = SETTINGS.loading
    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise FileValidationError("Archive ZIP corrompue ou invalide.") from exc
    with zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        if len(infos) > limits.zip_max_files:
            raise FileValidationError(f"L'archive contient trop de fichiers (> {limits.zip_max_files}).")
        total = sum(i.file_size for i in infos)
        if total > limits.zip_max_uncompressed_mb * 1024 * 1024:
            raise FileValidationError("Le contenu décompressé de l'archive est trop volumineux.")
        members = []
        for info in infos:
            name = info.filename
            path = PurePosixPath(name.replace("\\", "/"))
            if path.is_absolute() or ".." in path.parts or ":" in name:
                raise FileValidationError(f"Chemin suspect dans l'archive : « {name} ».")
            if info.compress_size and info.file_size / max(info.compress_size, 1) > limits.zip_max_ratio:
                raise FileValidationError(
                    f"Taux de compression anormal pour « {name} » : archive potentiellement malveillante (zip bomb)."
                )
            if info.flag_bits & 0x1:
                raise FileValidationError("Les archives chiffrées ne sont pas prises en charge.")
            if path.name.startswith(".") or "__MACOSX" in path.parts:
                continue
            members.append(name)
    return members


def extract_zip_member(content: bytes, member: str) -> bytes:
    """Extrait un membre d'archive (après contrôles), sans écrire sur le disque."""
    members = list_zip_members(content)
    if member not in members:
        raise FileValidationError(f"« {member} » est absent de l'archive.")
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        info = zf.getinfo(member)
        max_bytes = SETTINGS.loading.zip_max_uncompressed_mb * 1024 * 1024
        with zf.open(info) as fh:
            data = fh.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise FileValidationError("Membre d'archive trop volumineux une fois décompressé.")
    return data


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------

def inspect_file(filename: str, content: bytes) -> FileInspection:
    """Analyse préalable : format, feuilles, tables ou membres d'archive disponibles."""
    det = detect_file(filename, content)
    insp = FileInspection(detection=det)
    fmt = det.detected_format
    if fmt in {"xls", "xlsx", "xlsm", "ods"}:
        insp.sheets = list_sheets(content, fmt)
    elif fmt == "sqlite":
        insp.tables = list_sqlite_tables(content)
    elif fmt == "zip":
        insp.zip_members = list_zip_members(content)
    elif fmt in {"csv", "tsv", "txt", "jsonl"}:
        insp.estimated_rows = _count_lines(content)
    return insp


def load_dataset(filename: str, content: bytes, options: LoadOptions | None = None) -> LoadResult:
    """Charge un fichier en DataFrame pandas après validation.

    Args:
        filename: nom d'origine du fichier.
        content: contenu binaire.
        options: options de chargement (feuille, table, limite de lignes…).

    Returns:
        LoadResult contenant le DataFrame normalisé et ses métadonnées.

    Raises:
        DataInsightError (ou sous-classe) avec un message compréhensible.
    """
    options = options or LoadOptions()
    det = detect_file(filename, content)
    zip_member = None

    if det.detected_format == "zip":
        members = list_zip_members(content)
        if not members:
            raise EmptyDatasetError("L'archive ZIP ne contient aucun fichier exploitable.")
        zip_member = options.zip_member or members[0]
        inner = extract_zip_member(content, zip_member)
        inner_name = PurePosixPath(zip_member).name
        if detect_file(inner_name, inner).detected_format == "zip":
            raise FileValidationError("Les archives ZIP imbriquées ne sont pas prises en charge.")
        result = load_dataset(inner_name, inner, options)
        result.meta.zip_member = zip_member
        result.meta.filename = f"{filename} › {zip_member}"
        result.meta.size_bytes = det.size_bytes
        return result

    loader = _LOADERS.get(det.detected_format)
    if loader is None:
        raise UnsupportedFormatError(f"Aucun chargeur disponible pour le format « {det.detected_format} ».")

    try:
        df, extras = loader(content, det, options)
    except (MissingDependencyError, FileValidationError, EmptyDatasetError, DataLoadError, UnsupportedFormatError):
        raise
    except pd.errors.EmptyDataError as exc:
        raise EmptyDatasetError("Le fichier ne contient aucune donnée.") from exc
    except MemoryError as exc:
        raise DataLoadError(
            "Mémoire insuffisante pour charger ce fichier. Activez la lecture partielle (limite de lignes).",
        ) from exc
    except Exception as exc:
        raise DataLoadError(
            f"Le fichier n'a pas pu être lu comme {FORMAT_LABELS.get(det.detected_format, det.detected_format)}. "
            "Vérifiez qu'il n'est pas corrompu ou protégé.",
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc

    if not isinstance(df, pd.DataFrame):  # pragma: no cover - défensif
        raise DataLoadError("Le chargeur n'a pas renvoyé de tableau.")
    df = sanitize_dataframe(df)
    n_before = len(df)
    df = df.dropna(axis=0, how="all").reset_index(drop=True) if len(df) else df
    if len(df) < n_before:
        extras.setdefault("warnings", []).append(
            f"{n_before - len(df)} ligne(s) entièrement vide(s) ignorée(s) à la lecture."
        )
    if df.shape[1] == 0:
        raise EmptyDatasetError("Aucune colonne n'a été trouvée dans le fichier.")
    if df.shape[0] == 0:
        raise EmptyDatasetError("Le fichier contient des en-têtes mais aucune ligne de données.")

    meta = FileMeta(
        filename=filename,
        detected_format=det.detected_format,
        format_label=det.label,
        size_bytes=det.size_bytes,
        encoding=extras.get("encoding", det.encoding),
        separator=extras.get("separator", det.separator),
        decimal=extras.get("decimal", det.decimal),
        sheet=extras.get("sheet"),
        sheets=extras.get("sheets", []),
        table=extras.get("table"),
        tables=extras.get("tables", []),
        truncated=bool(extras.get("truncated", False)),
        estimated_total_rows=extras.get("estimated_total_rows"),
        value_labels=extras.get("value_labels", {}),
        column_labels=extras.get("column_labels", {}),
        warnings=list(det.warnings) + list(extras.get("warnings", [])),
    )
    if meta.truncated:
        meta.warnings.append(
            f"Lecture partielle : {len(df):,} lignes chargées".replace(",", " ")
            + (f" sur environ {meta.estimated_total_rows:,}.".replace(",", " ") if meta.estimated_total_rows else ".")
        )
    return LoadResult(df=df, meta=meta)


def load_path(path: str | os.PathLike[str], options: LoadOptions | None = None) -> LoadResult:
    """Charge un fichier depuis le disque (jeux d'exemple, tests)."""
    with open(path, "rb") as fh:
        content = fh.read()
    return load_dataset(os.path.basename(str(path)), content, options)
