"""Reconnaissance du format réel, de l'encodage et du séparateur d'un fichier.

La détection combine l'extension déclarée et la signature binaire (« magic
bytes ») du contenu. En cas de désaccord, le format réel est privilégié et un
avertissement est émis.
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field

from config.settings import SETTINGS, all_extensions
from core.exceptions import FileValidationError, UnsupportedFormatError

# Signatures binaires connues -> famille de format.
_MAGIC_SIGNATURES: list[tuple[bytes, str]] = [
    (b"SQLite format 3\x00", "sqlite"),
    (b"PAR1", "parquet"),
    (b"ARROW1", "feather"),
    (b"ORC", "orc"),
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "ole2"),  # XLS (Office 97-2003)
    (b"PK\x03\x04", "zip"),  # XLSX / XLSM / ODS / ZIP
    (b"$FL2", "sav"),
    (b"$FL3", "sav"),
]

# Extension -> format canonique.
EXTENSION_FORMATS: dict[str, str] = {
    "csv": "csv", "tsv": "tsv", "txt": "txt",
    "xls": "xls", "xlsx": "xlsx", "xlsm": "xlsm", "ods": "ods",
    "parquet": "parquet", "pq": "parquet", "feather": "feather", "arrow": "feather", "orc": "orc",
    "json": "json", "jsonl": "jsonl", "ndjson": "jsonl", "xml": "xml",
    "sav": "sav", "zsav": "sav", "dta": "dta", "sas7bdat": "sas7bdat",
    "sqlite": "sqlite", "sqlite3": "sqlite", "db": "sqlite",
    "zip": "zip",
}

FORMAT_LABELS: dict[str, str] = {
    "csv": "CSV (valeurs séparées)", "tsv": "TSV (tabulations)", "txt": "Texte délimité",
    "xls": "Excel 97-2003 (XLS)", "xlsx": "Excel (XLSX)", "xlsm": "Excel avec macros (XLSM)",
    "ods": "OpenDocument (ODS)", "parquet": "Apache Parquet", "feather": "Feather / Arrow IPC",
    "orc": "Apache ORC", "json": "JSON", "jsonl": "JSON Lines", "xml": "XML",
    "sav": "SPSS (SAV)", "dta": "Stata (DTA)", "sas7bdat": "SAS (SAS7BDAT)",
    "sqlite": "Base SQLite", "zip": "Archive ZIP",
}

TEXT_FORMATS = {"csv", "tsv", "txt", "json", "jsonl", "xml"}

_ENCODING_FALLBACKS = ("utf-8", "cp1252", "latin-1")
_CANDIDATE_SEPARATORS = [",", ";", "\t", "|"]


@dataclass
class DetectionResult:
    """Résultat de l'identification d'un fichier."""

    filename: str
    extension: str
    declared_format: str | None
    detected_format: str
    size_bytes: int
    encoding: str | None = None
    separator: str | None = None
    decimal: str = "."
    has_bom: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return FORMAT_LABELS.get(self.detected_format, self.detected_format.upper())


def get_extension(filename: str) -> str:
    """Extension en minuscules, sans point (gère « .sas7bdat »)."""
    name = filename.lower().strip()
    if "." not in name:
        return ""
    return name.rsplit(".", 1)[-1]


def sniff_magic(head: bytes) -> str | None:
    """Identifie la famille de format via la signature binaire, sinon None."""
    for signature, fmt in _MAGIC_SIGNATURES:
        if head.startswith(signature):
            return fmt
    # SAS7BDAT : signature de 32 octets ; on vérifie un motif caractéristique.
    if len(head) >= 32 and head[:32] == bytes.fromhex(
        "000000000000000000000000c2ea8160b31411cfbd92080009c7318c181f1011"
    ):
        return "sas7bdat"
    # Stata 117+ commence par "<stata_dta>" ; les versions anciennes par un octet de version.
    if head.startswith(b"<stata_dta>"):
        return "dta"
    if len(head) > 4 and head[0] in (0x71, 0x72, 0x73) and head[1] in (0x01, 0x02) and head[2] == 0x01:
        return "dta"
    return None


def _zip_subtype(content: bytes) -> str:
    """Distingue XLSX / XLSM / ODS / ZIP générique à partir du contenu ZIP."""
    import zipfile

    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            names = set(zf.namelist())
            if "mimetype" in names:
                mime = zf.read("mimetype")[:100]
                if b"opendocument.spreadsheet" in mime:
                    return "ods"
            if "xl/workbook.xml" in names:
                return "xlsm" if "xl/vbaProject.bin" in names else "xlsx"
    except zipfile.BadZipFile:
        return "zip"
    return "zip"


def detect_encoding(sample: bytes) -> tuple[str, bool]:
    """Détecte l'encodage d'un échantillon d'octets. Retourne (encodage, BOM)."""
    if sample.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig", True
    if sample.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16", True
    try:
        sample.decode("utf-8")
        return "utf-8", False
    except UnicodeDecodeError as exc:
        # Un caractère multi-octets coupé en fin d'échantillon n'est pas une erreur.
        if exc.start >= len(sample) - 3:
            try:
                sample[: exc.start].decode("utf-8")
                return "utf-8", False
            except UnicodeDecodeError:
                pass
    try:
        from charset_normalizer import from_bytes

        best = from_bytes(sample).best()
        if best is not None and best.encoding:
            enc = best.encoding.replace("_", "-")
            # charset-normalizer renvoie parfois des encodages exotiques pour du latin-1.
            if enc.lower() in {"cp1252", "latin-1", "iso8859-1", "iso-8859-1", "cp1250", "iso8859-15"}:
                return "cp1252", False
            return enc, False
    except Exception:  # pragma: no cover - dépendance optionnelle
        pass
    for enc in _ENCODING_FALLBACKS[1:]:
        try:
            sample.decode(enc)
            return enc, False
        except UnicodeDecodeError:
            continue
    return "latin-1", False


def detect_separator(text: str) -> str:
    """Détecte le séparateur de colonnes d'un texte délimité."""
    lines = [ln for ln in text.splitlines() if ln.strip()][:50]
    if not lines:
        return ","
    sample = "\n".join(lines)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="".join(_CANDIDATE_SEPARATORS))
        sniffed = dialect.delimiter
    except csv.Error:
        sniffed = None

    # Heuristique : séparateur présent avec un nombre constant (>0) d'occurrences par ligne.
    best_sep, best_score = None, -1.0
    for sep in _CANDIDATE_SEPARATORS:
        counts = [_count_outside_quotes(ln, sep) for ln in lines]
        if not counts or max(counts) == 0:
            continue
        mode = max(set(counts), key=counts.count)
        if mode == 0:
            continue
        consistency = counts.count(mode) / len(counts)
        score = consistency * 10 + min(mode, 50) / 50
        if score > best_score:
            best_sep, best_score = sep, score
    if sniffed and best_sep and sniffed != best_sep:
        # Le Sniffer se trompe souvent sur ',' avec des décimales françaises.
        return best_sep
    return sniffed or best_sep or ","


def _count_outside_quotes(line: str, sep: str) -> int:
    count, in_quotes = 0, False
    for ch in line:
        if ch == '"':
            in_quotes = not in_quotes
        elif ch == sep and not in_quotes:
            count += 1
    return count


def detect_decimal(text: str, separator: str) -> str:
    """Détecte la virgule décimale (fréquente avec le séparateur ';')."""
    if separator == ",":
        return "."
    import re

    comma_numbers = len(re.findall(r"(?<![\d.,])-?\d+,\d+(?![\d.,])", text))
    dot_numbers = len(re.findall(r"(?<![\d.,])-?\d+\.\d+(?![\d.,])", text))
    return "," if comma_numbers > dot_numbers else "."


def _looks_like_json(text: str) -> str | None:
    stripped = text.lstrip()
    if not stripped:
        return None
    if stripped[0] in "[{":
        first_line = stripped.splitlines()[0].strip()
        # JSON Lines : chaque ligne est un objet JSON complet.
        lines = [ln for ln in stripped.splitlines()[:5] if ln.strip()]
        if len(lines) > 1 and first_line.startswith("{") and first_line.endswith("}"):
            try:
                for ln in lines:
                    json.loads(ln)
                return "jsonl"
            except json.JSONDecodeError:
                pass
        return "json"
    return None


def detect_file(filename: str, content: bytes) -> DetectionResult:
    """Identifie le format, l'encodage et le séparateur d'un fichier.

    Args:
        filename: nom d'origine du fichier (sert à lire l'extension).
        content: contenu binaire complet du fichier.

    Raises:
        FileValidationError: fichier vide ou trop volumineux.
        UnsupportedFormatError: format non reconnu.
    """
    size = len(content)
    max_bytes = SETTINGS.loading.max_file_size_mb * 1024 * 1024
    if size == 0:
        raise FileValidationError("Le fichier est vide (0 octet).")
    if size > max_bytes:
        raise FileValidationError(
            f"Le fichier dépasse la taille maximale autorisée ({SETTINGS.loading.max_file_size_mb} Mo).",
            detail="Augmentez la limite dans Paramètres ou importez un échantillon.",
        )

    ext = get_extension(filename)
    declared = EXTENSION_FORMATS.get(ext)
    head = content[:64]
    magic = sniff_magic(head)
    warnings: list[str] = []

    detected: str | None
    if magic == "zip":
        detected = _zip_subtype(content)
    elif magic == "ole2":
        detected = "xls"
    else:
        detected = magic

    encoding = separator = None
    decimal = "."
    has_bom = False
    if detected is None:
        # Fichier texte probable.
        if b"\x00" in content[:4096] and not content.startswith((b"\xff\xfe", b"\xfe\xff")):
            raise UnsupportedFormatError(
                f"Le fichier « {filename} » semble binaire mais son format n'est pas reconnu.",
            )
        sample = content[: SETTINGS.loading.sniff_bytes]
        encoding, has_bom = detect_encoding(sample)
        text = sample.decode(encoding, errors="replace")
        stripped = text.lstrip("﻿").lstrip()
        if stripped.startswith("<"):
            detected = "xml"
        else:
            detected = _looks_like_json(stripped) or (
                declared if declared in {"csv", "tsv", "txt"} else "csv"
            )
        if detected in {"csv", "tsv", "txt"}:
            separator = "\t" if detected == "tsv" else detect_separator(stripped)
            decimal = detect_decimal(stripped, separator)

    if declared is None and ext and ext not in all_extensions():
        warnings.append(f"Extension « .{ext} » inconnue : format déduit du contenu ({detected}).")
    elif declared and declared != detected:
        compatible = {
            frozenset({"csv", "txt"}), frozenset({"csv", "tsv"}), frozenset({"txt", "tsv"}),
            frozenset({"xlsx", "xlsm"}), frozenset({"json", "jsonl"}),
        }
        if frozenset({declared, detected}) not in compatible:
            warnings.append(
                f"L'extension .{ext} ne correspond pas au contenu réel : format détecté « {detected} »."
            )
    if declared in {"txt", "tsv"} and detected == "csv":
        detected = declared if declared == "txt" else detected

    if detected not in FORMAT_LABELS:
        raise UnsupportedFormatError(f"Format « {detected} » non pris en charge.")

    return DetectionResult(
        filename=filename, extension=ext, declared_format=declared, detected_format=detected,
        size_bytes=size, encoding=encoding, separator=separator, decimal=decimal,
        has_bom=has_bom, warnings=warnings,
    )


def human_size(n_bytes: int) -> str:
    """Formate une taille en octets de manière lisible."""
    size = float(n_bytes)
    for unit in ("o", "Ko", "Mo", "Go"):
        if size < 1024 or unit == "Go":
            return f"{size:,.0f} {unit}".replace(",", " ") if unit == "o" else f"{size:,.1f} {unit}".replace(",", " ")
        size /= 1024
    return f"{size:.1f} Go"  # pragma: no cover


SEPARATOR_LABELS = {",": "virgule (,)", ";": "point-virgule (;)", "\t": "tabulation", "|": "barre verticale (|)"}
