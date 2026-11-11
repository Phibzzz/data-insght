"""Détection automatique des types techniques et statistiques des variables.

La détection combine :
* le type technique pandas ;
* le contenu réel (tests de conversion numérique, date, booléen) ;
* le nombre de modalités et la proportion de valeurs uniques ;
* des indices lexicaux sur le nom de colonne (id, lat, année, mois…).

Le DataFrame d'origine n'est jamais modifié : :func:`apply_schema` produit une
*vue typée* (copie) à partir du schéma, et les corrections manuelles sont des
surcharges réversibles stockées à part.
"""
from __future__ import annotations

import re
import unicodedata
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from config.settings import SETTINGS


class StatType(str, Enum):
    """Types statistiques reconnus."""

    CONTINUOUS = "Numérique continue"
    DISCRETE = "Numérique discrète"
    BINARY = "Catégorielle binaire"
    NOMINAL = "Catégorielle nominale"
    ORDINAL = "Catégorielle ordinale"
    BOOLEAN = "Booléenne"
    DATE = "Date"
    DATETIME = "Horodatage"
    YEAR = "Année"
    MONTH = "Mois"
    TEXT = "Texte libre"
    IDENTIFIER = "Identifiant"
    LATITUDE = "Latitude"
    LONGITUDE = "Longitude"
    CONSTANT = "Constante"
    EMPTY = "Vide"


class Family(str, Enum):
    """Familles de types, utilisées pour choisir les analyses pertinentes."""

    NUMERIC = "Quantitative"
    CATEGORICAL = "Qualitative"
    TEMPORAL = "Temporelle"
    OTHER = "Autre"


FAMILY_OF: dict[StatType, Family] = {
    StatType.CONTINUOUS: Family.NUMERIC,
    StatType.DISCRETE: Family.NUMERIC,
    StatType.LATITUDE: Family.NUMERIC,
    StatType.LONGITUDE: Family.NUMERIC,
    StatType.BINARY: Family.CATEGORICAL,
    StatType.NOMINAL: Family.CATEGORICAL,
    StatType.ORDINAL: Family.CATEGORICAL,
    StatType.BOOLEAN: Family.CATEGORICAL,
    StatType.MONTH: Family.CATEGORICAL,
    StatType.DATE: Family.TEMPORAL,
    StatType.DATETIME: Family.TEMPORAL,
    StatType.YEAR: Family.TEMPORAL,
    StatType.TEXT: Family.OTHER,
    StatType.IDENTIFIER: Family.OTHER,
    StatType.CONSTANT: Family.OTHER,
    StatType.EMPTY: Family.OTHER,
}

TRUE_TOKENS = {"true", "vrai", "oui", "yes", "y", "o", "t", "x"}
FALSE_TOKENS = {"false", "faux", "non", "no", "n", "f"}

MONTH_NAMES = {
    "janvier": 1, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6, "juillet": 7,
    "aout": 8, "septembre": 9, "octobre": 10, "novembre": 11, "decembre": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7,
    "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "fev": 2, "feb": 2, "mar": 3, "avr": 4, "apr": 4, "jun": 6, "jul": 7, "juil": 7,
    "aou": 8, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

# Échelles ordinales connues (ordre croissant), comparées sans accents ni casse.
ORDINAL_SCALES: list[list[str]] = [
    ["tres faible", "faible", "moyen", "eleve", "tres eleve"],
    ["faible", "moyen", "fort"],
    ["faible", "moyenne", "forte"],
    ["bas", "moyen", "haut"],
    ["basse", "moyenne", "haute"],
    ["low", "medium", "high"],
    ["very low", "low", "medium", "high", "very high"],
    ["xs", "s", "m", "l", "xl", "xxl"],
    ["petit", "moyen", "grand"],
    ["jamais", "rarement", "parfois", "souvent", "toujours"],
    ["never", "rarely", "sometimes", "often", "always"],
    ["tres insatisfait", "insatisfait", "neutre", "satisfait", "tres satisfait"],
    ["pas du tout d'accord", "pas d'accord", "neutre", "d'accord", "tout a fait d'accord"],
    ["strongly disagree", "disagree", "neutral", "agree", "strongly agree"],
    ["mauvais", "passable", "moyen", "bon", "tres bon", "excellent"],
    ["poor", "fair", "good", "very good", "excellent"],
    ["primaire", "college", "lycee", "licence", "master", "doctorat"],
    ["bronze", "argent", "or", "platine"],
    ["bronze", "silver", "gold", "platinum"],
    ["junior", "confirme", "senior", "expert"],
    ["junior", "intermediate", "senior", "lead"],
]

_ID_HINT = re.compile(r"(^|[_\s\-.])(id|uuid|guid|code|key|cle|ref|reference|identifiant|matricule|numero|num|no)([_\s\-.]|$)|id$", re.I)
_LAT_HINT = re.compile(r"^(lat|latitude)([_\s\-.]|$)|[_\s\-.]lat(itude)?$", re.I)
_LON_HINT = re.compile(r"^(lon|lng|long|longitude)([_\s\-.]|$)|[_\s\-.](lon|lng|longitude)$", re.I)
_YEAR_HINT = re.compile(r"(annee|année|year|^an$|_an$|^an_|exercice|millesime)", re.I)
_MONTH_HINT = re.compile(r"(mois|month)", re.I)
_ORDINAL_HINT = re.compile(r"(note|rating|niveau|level|rang|rank|grade|satisfaction|echelon|priorit|stars|etoile)", re.I)
_DATE_LIKE = re.compile(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|\d{4}-\d{2}|\d{1,2}\s+[a-zéû]{3,}\.?\s+\d{2,4}|\d{8}T", re.I)

DATE_FORMATS = [
    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d", "%d.%m.%Y",
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f",
    "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M",
    "%Y-%m", "%d/%m/%y", "%Y%m%d",
]


@dataclass
class ColumnSchema:
    """Description d'une variable après détection."""

    name: str
    technical_type: str
    storage_type: str
    stat_type: StatType
    detected_stat_type: StatType
    n_rows: int
    n_missing: int
    n_unique: int
    unique_ratio: float
    confidence: float
    reason: str
    conversion: str | None = None
    date_format: str | None = None
    ordinal_order: list[str] | None = None
    overridden: bool = False
    sample_values: list[str] = field(default_factory=list)

    @property
    def family(self) -> Family:
        return FAMILY_OF[self.stat_type]

    @property
    def missing_ratio(self) -> float:
        return self.n_missing / self.n_rows if self.n_rows else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["stat_type"] = self.stat_type.value
        d["detected_stat_type"] = self.detected_stat_type.value
        d["family"] = self.family.value
        return d


# ---------------------------------------------------------------------------
# Utilitaires de normalisation et de conversion
# ---------------------------------------------------------------------------

def strip_accents(text: str) -> str:
    """Supprime les accents (é -> e)."""
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def _norm_token(value: Any) -> str:
    return strip_accents(str(value)).strip().lower()


def is_text_series(s: pd.Series) -> bool:
    """Vrai pour les colonnes de chaînes (dtype object ou str de pandas 3)."""
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


def to_numeric_smart(s: pd.Series) -> pd.Series:
    """Convertit en numérique en gérant espaces, virgule décimale, %, symboles monétaires."""
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    txt = s.astype("string").str.strip()
    txt = txt.str.replace(r"[\s  ]", "", regex=True)
    txt = txt.str.replace(r"^[€$£]|[€$£%]$", "", regex=True)
    # 1.234,56 -> 1234.56 ; 1234,56 -> 1234.56
    both = txt.str.contains(r"\d\.\d{3},\d", regex=True, na=False)
    txt = txt.where(~both, txt.str.replace(".", "", regex=False))
    txt = txt.str.replace(r"^(-?\d+),(\d+)$", r"\1.\2", regex=True)
    result = pd.to_numeric(txt, errors="coerce")
    return result.astype("float64")


def _best_date_format(values: pd.Series) -> tuple[str | None, float]:
    """Cherche le format de date qui convertit la plus grande part des valeurs."""
    best_fmt, best_ratio = None, 0.0
    n = len(values)
    if n == 0:
        return None, 0.0
    for fmt in DATE_FORMATS:
        parsed = pd.to_datetime(values, format=fmt, errors="coerce")
        ratio = parsed.notna().sum() / n
        if ratio > best_ratio:
            best_fmt, best_ratio = fmt, ratio
            if ratio == 1.0:
                break
    return best_fmt, best_ratio


def to_datetime_smart(s: pd.Series, date_format: str | None = None) -> pd.Series:
    """Convertit en dates, avec format explicite si connu, sinon ISO8601 puis 'mixed'."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    txt = s.astype("string").str.strip()
    if date_format:
        return pd.to_datetime(txt, format=date_format, errors="coerce")
    non_null = txt.dropna()
    if non_null.empty:
        return pd.to_datetime(txt, errors="coerce")
    # 1) format explicite majoritaire (évite toute ambiguïté jour/mois)
    fmt, ratio = _best_date_format(_sample(non_null, 2000))
    if fmt and ratio >= 0.5:
        return pd.to_datetime(txt, format=fmt, errors="coerce")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # 2) ISO 8601 (fuseaux, fractions de seconde…)
        try:
            out = pd.to_datetime(txt, format="ISO8601", errors="coerce")
            if out.notna().sum() >= 0.5 * len(non_null):
                return out
        except (ValueError, TypeError):
            pass
        # 3) formats mixtes ; jour en premier seulement si des dates jj/mm/aaaa sont présentes
        dayfirst = bool(non_null.str.match(r"^\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}").any())
        return pd.to_datetime(txt, format="mixed", dayfirst=dayfirst, errors="coerce")


def to_boolean_smart(s: pd.Series) -> pd.Series:
    """Convertit les jetons oui/non, vrai/faux, true/false, 1/0 en booléens nullable."""
    if pd.api.types.is_bool_dtype(s):
        return s.astype("boolean")

    def conv(v: Any) -> Any:
        if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
            return pd.NA
        tok = _norm_token(v)
        if tok in TRUE_TOKENS or tok in {"1", "1.0"}:
            return True
        if tok in FALSE_TOKENS or tok in {"0", "0.0"}:
            return False
        return pd.NA

    return s.map(conv).astype("boolean")


def _strip_tz(s: pd.Series) -> pd.Series:
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        return s.dt.tz_localize(None)
    return s


# ---------------------------------------------------------------------------
# Détection
# ---------------------------------------------------------------------------

def _storage_label(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "booléen"
    if pd.api.types.is_integer_dtype(s):
        return "entier"
    if pd.api.types.is_float_dtype(s):
        return "réel"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "date/heure"
    if isinstance(s.dtype, pd.CategoricalDtype):
        return "catégorie"
    return "texte"


def _sample(s: pd.Series, n: int = 5000) -> pd.Series:
    if len(s) <= n:
        return s
    return s.sample(n, random_state=SETTINGS.random_seed)


def _classify_numeric(name: str, values: pd.Series, n_unique: int, n_valid: int) -> tuple[StatType, float, str]:
    """Classe une série numérique non nulle."""
    cfg = SETTINGS.schema
    finite = values[np.isfinite(values.astype("float64"))]
    if finite.empty:
        return StatType.CONTINUOUS, 0.5, "Valeurs numériques non finies uniquement."
    vmin, vmax = float(finite.min()), float(finite.max())
    is_int = bool(np.all(np.mod(finite.astype("float64"), 1) == 0))
    unique_ratio = n_unique / n_valid if n_valid else 0.0

    if _LAT_HINT.search(name) and -90 <= vmin and vmax <= 90:
        return StatType.LATITUDE, 0.95, "Nom évocateur et valeurs dans [-90, 90]."
    if _LON_HINT.search(name) and -180 <= vmin and vmax <= 180:
        return StatType.LONGITUDE, 0.95, "Nom évocateur et valeurs dans [-180, 180]."
    if n_unique == 2:
        return StatType.BINARY, 0.9, f"Deux valeurs distinctes seulement ({vmin:g} / {vmax:g})."
    if is_int:
        if _YEAR_HINT.search(name) and 1800 <= vmin and vmax <= 2200:
            return StatType.YEAR, 0.95, "Nom évocateur et entiers compris entre 1800 et 2200."
        if _MONTH_HINT.search(name) and 1 <= vmin and vmax <= 12:
            return StatType.MONTH, 0.9, "Nom évocateur et entiers compris entre 1 et 12."
        if (_ID_HINT.search(name) and unique_ratio >= 0.9) or (
            unique_ratio == 1.0 and n_valid >= 20 and _is_sequence(finite)
        ):
            return StatType.IDENTIFIER, 0.85, "Entiers (quasi) uniques pour chaque ligne : identifiant probable."
        if _ORDINAL_HINT.search(name) and n_unique <= 11 and vmin >= 0 and vmax <= 10:
            return StatType.ORDINAL, 0.75, "Échelle entière courte et nom évocateur (note, niveau…)."
        if n_unique <= cfg.discrete_max_unique:
            return StatType.DISCRETE, 0.85, f"Entiers avec {n_unique} valeurs distinctes."
        return StatType.CONTINUOUS, 0.7, f"Entiers avec de nombreuses valeurs distinctes ({n_unique}) : traités comme continus."
    return StatType.CONTINUOUS, 0.9, "Nombres réels."


def _is_sequence(values: pd.Series) -> bool:
    arr = np.sort(values.to_numpy(dtype="float64"))
    if len(arr) < 2:
        return False
    diffs = np.diff(arr)
    return bool(np.all(diffs == 1))


def _ordinal_match(tokens: set[str]) -> list[str] | None:
    for scale in ORDINAL_SCALES:
        if tokens and tokens.issubset(set(scale)) and len(tokens) >= 2:
            return [v for v in scale if v in tokens]
    return None


def _classify_datetime(values: pd.Series) -> tuple[StatType, str]:
    values = _strip_tz(values)
    has_time = bool(((values.dt.hour != 0) | (values.dt.minute != 0) | (values.dt.second != 0)).any())
    if has_time:
        return StatType.DATETIME, "Valeurs date + heure."
    return StatType.DATE, "Valeurs de date (sans heure)."


def detect_column(name: str, s: pd.Series) -> ColumnSchema:
    """Détecte le type statistique d'une colonne."""
    cfg = SETTINGS.schema
    n_rows = len(s)
    non_null = s.dropna()
    if is_text_series(s):
        non_null = non_null[non_null.astype("string").str.strip() != ""]
    n_missing = n_rows - len(non_null)
    try:
        n_unique = int(non_null.nunique())
    except TypeError:
        n_unique = int(non_null.astype(str).nunique())
    n_valid = len(non_null)
    unique_ratio = n_unique / n_valid if n_valid else 0.0
    samples = [str(v)[:60] for v in non_null.drop_duplicates().head(5).tolist()]

    def make(st: StatType, conf: float, reason: str, **kw: Any) -> ColumnSchema:
        return ColumnSchema(
            name=name, technical_type=str(s.dtype), storage_type=kw.pop("storage", _storage_label(s)),
            stat_type=st, detected_stat_type=st, n_rows=n_rows, n_missing=n_missing,
            n_unique=n_unique, unique_ratio=round(unique_ratio, 4), confidence=conf, reason=reason,
            sample_values=samples, **kw,
        )

    if n_valid == 0:
        return make(StatType.EMPTY, 1.0, "Aucune valeur renseignée.")
    if n_unique == 1:
        return make(StatType.CONSTANT, 1.0, f"Une seule valeur distincte ({samples[0] if samples else ''}).")

    if pd.api.types.is_bool_dtype(s):
        return make(StatType.BOOLEAN, 1.0, "Type booléen natif.")
    if pd.api.types.is_datetime64_any_dtype(s):
        st, reason = _classify_datetime(non_null)
        return make(st, 1.0, reason)
    if pd.api.types.is_numeric_dtype(s):
        st, conf, reason = _classify_numeric(name, non_null, n_unique, n_valid)
        if st == StatType.BINARY and set(non_null.unique().tolist()) <= {0, 1}:
            st, reason = StatType.BOOLEAN, "Valeurs 0/1 uniquement : indicateur booléen."
        return make(st, conf, reason)

    # --- Colonnes texte : tests de conversion sur un échantillon ---
    sample = _sample(non_null).astype("string").str.strip()
    tokens = {_norm_token(v) for v in sample.unique()[:50]}
    threshold = cfg.conversion_threshold

    if (
        tokens
        and tokens <= (TRUE_TOKENS | FALSE_TOKENS)
        and tokens & TRUE_TOKENS
        and tokens & FALSE_TOKENS
    ):
        return make(StatType.BOOLEAN, 0.9, "Valeurs de type oui/non, vrai/faux.", conversion="boolean")

    numeric = to_numeric_smart(sample)
    num_ratio = numeric.notna().mean()
    if num_ratio >= threshold:
        full_numeric = numeric.dropna()
        st, conf, reason = _classify_numeric(name, full_numeric, n_unique, n_valid)
        if st == StatType.BINARY and set(full_numeric.unique().tolist()) <= {0.0, 1.0}:
            st, reason = StatType.BOOLEAN, "Valeurs 0/1 uniquement : indicateur booléen."
        has_comma = sample.str.contains(r"\d,\d", regex=True, na=False).any()
        return make(
            st, round(conf * num_ratio, 3),
            reason + (" Texte converti en nombre (virgule décimale détectée)." if has_comma else " Texte convertible en nombre."),
            conversion="numeric", storage="texte (numérique)",
        )

    date_like = sample.str.contains(_DATE_LIKE, na=False).mean()
    if date_like >= 0.8:
        fmt, ratio = _best_date_format(sample)
        parsed = None
        if ratio >= threshold:
            parsed = pd.to_datetime(sample, format=fmt, errors="coerce")
        else:
            alt = to_datetime_smart(sample)
            if alt.notna().mean() >= threshold:
                parsed, fmt, ratio = alt, None, float(alt.notna().mean())
        if parsed is not None:
            st, reason = _classify_datetime(parsed.dropna())
            if fmt == "%Y-%m":
                reason = "Valeurs année-mois."
            fmt_txt = f" Format : {fmt}." if fmt else " Formats multiples."
            return make(st, round(0.95 * ratio, 3), reason + fmt_txt, conversion="datetime",
                        date_format=fmt, storage="texte (date)")

    if tokens and all(t.rstrip(".") in MONTH_NAMES for t in tokens) and len(tokens) <= 24:
        return make(StatType.MONTH, 0.9, "Noms de mois.")

    ordinal = _ordinal_match({_norm_token(v) for v in non_null.unique()[:50]}) if n_unique <= 10 else None
    if ordinal:
        return make(StatType.ORDINAL, 0.8, "Modalités appartenant à une échelle ordonnée connue.", ordinal_order=ordinal)

    avg_len = float(sample.str.len().mean() or 0)
    has_spaces = float(sample.str.contains(r"\s", regex=True, na=False).mean())
    if avg_len >= cfg.free_text_min_length and unique_ratio > 0.5:
        return make(StatType.TEXT, 0.85, f"Chaînes longues (longueur moyenne {avg_len:.0f}) et variées.")
    if n_unique == 2:
        return make(StatType.BINARY, 0.9, "Deux modalités distinctes.")
    if n_valid >= 20 and unique_ratio >= cfg.identifier_unique_ratio and has_spaces < 0.2:
        return make(StatType.IDENTIFIER, 0.85, "Valeurs uniques sans espaces : identifiant probable.")
    if _ID_HINT.search(name) and unique_ratio >= 0.9:
        return make(StatType.IDENTIFIER, 0.8, "Nom évocateur et valeurs quasi uniques.")
    if n_unique <= cfg.max_categories or unique_ratio <= cfg.category_unique_ratio:
        return make(StatType.NOMINAL, 0.85, f"{n_unique} modalités distinctes.")
    if unique_ratio > 0.9 and has_spaces < 0.2:
        return make(StatType.IDENTIFIER, 0.6, "Valeurs majoritairement uniques : identifiant ou code probable.")
    return make(StatType.TEXT, 0.6, f"Nombreuses modalités ({n_unique}) : traité comme texte libre.")


def detect_schema(df: pd.DataFrame, overrides: dict[str, str] | None = None) -> list[ColumnSchema]:
    """Détecte le schéma complet d'un DataFrame, en appliquant les corrections manuelles.

    Args:
        df: données d'origine (non modifiées).
        overrides: {nom_colonne: valeur de StatType} choisies par l'utilisateur.
    """
    overrides = overrides or {}
    schemas = []
    for col in df.columns:
        sch = detect_column(str(col), df[col])
        if col in overrides:
            try:
                new_type = StatType(overrides[col])
            except ValueError:
                new_type = None
            if new_type is not None and new_type != sch.detected_stat_type:
                sch.stat_type = new_type
                sch.overridden = True
                sch.reason = f"Correction manuelle (détection initiale : {sch.detected_stat_type.value})."
                sch.confidence = 1.0
                if new_type == StatType.ORDINAL and not sch.ordinal_order:
                    sch.ordinal_order = None
        schemas.append(sch)
    return schemas


def schema_by_name(schemas: list[ColumnSchema]) -> dict[str, ColumnSchema]:
    """Index nom -> schéma."""
    return {s.name: s for s in schemas}


def columns_of(schemas: list[ColumnSchema], *families: Family) -> list[str]:
    """Noms des colonnes appartenant aux familles demandées."""
    return [s.name for s in schemas if s.family in families]


def numeric_columns(schemas: list[ColumnSchema], include_discrete: bool = True) -> list[str]:
    """Colonnes quantitatives exploitables (hors identifiants/constantes)."""
    allowed = {StatType.CONTINUOUS, StatType.LATITUDE, StatType.LONGITUDE}
    if include_discrete:
        allowed.add(StatType.DISCRETE)
    return [s.name for s in schemas if s.stat_type in allowed]


def categorical_columns(schemas: list[ColumnSchema]) -> list[str]:
    """Colonnes qualitatives exploitables."""
    return [s.name for s in schemas if s.family == Family.CATEGORICAL]


def temporal_columns(schemas: list[ColumnSchema]) -> list[str]:
    """Colonnes de dates/horodatages (hors années simples)."""
    return [s.name for s in schemas if s.stat_type in (StatType.DATE, StatType.DATETIME)]


# ---------------------------------------------------------------------------
# Vue typée
# ---------------------------------------------------------------------------

def ordered_categorical(s: pd.Series, order: list[str]) -> pd.Series:
    """Catégorielle ordonnée conservant l'orthographe d'origine des modalités.

    `order` contient les jetons normalisés (sans accents, minuscules) dans l'ordre croissant.
    """
    originals: dict[str, Any] = {}
    for v in s.dropna().unique():
        originals.setdefault(_norm_token(v), v)
    categories = [originals[tok] for tok in order if tok in originals]
    extra = [v for tok, v in originals.items() if tok not in order]
    return pd.Series(
        pd.Categorical(s, categories=categories + extra, ordered=True), index=s.index, name=s.name
    )


def convert_series(s: pd.Series, sch: ColumnSchema) -> pd.Series:
    """Convertit une série selon son type statistique (sans modifier l'original)."""
    st = sch.stat_type
    if st in (StatType.CONTINUOUS, StatType.DISCRETE, StatType.LATITUDE, StatType.LONGITUDE):
        out = to_numeric_smart(s)
        if st == StatType.DISCRETE and out.dropna().mod(1).eq(0).all():
            return out.astype("Int64") if out.isna().any() else out.astype("int64")
        return out
    if st in (StatType.DATE, StatType.DATETIME):
        return _strip_tz(to_datetime_smart(s, sch.date_format))
    if st == StatType.YEAR:
        if pd.api.types.is_datetime64_any_dtype(s):
            return s.dt.year.astype("Int64")
        return to_numeric_smart(s).round().astype("Int64")
    if st == StatType.BOOLEAN:
        return to_boolean_smart(s)
    if st == StatType.BINARY and sch.conversion == "numeric":
        return to_numeric_smart(s)
    if st == StatType.ORDINAL:
        if pd.api.types.is_numeric_dtype(s):
            return s
        if sch.ordinal_order:
            return ordered_categorical(s, sch.ordinal_order)
        return s
    if st == StatType.MONTH and sch.conversion == "numeric":
        return to_numeric_smart(s).astype("Int64")
    if st in (StatType.NOMINAL, StatType.BINARY, StatType.TEXT, StatType.IDENTIFIER, StatType.MONTH):
        if pd.api.types.is_datetime64_any_dtype(s):
            return s.astype("string")
        if is_text_series(s):
            return s.where(s.isna(), s.astype("string").str.strip()).replace("", pd.NA)
    return s


def apply_schema(df: pd.DataFrame, schemas: list[ColumnSchema]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Produit une vue typée du DataFrame et compte les valeurs perdues par conversion.

    Returns:
        (copie typée, {colonne: nb de valeurs non nulles devenues nulles})
    """
    typed = df.copy()
    lost: dict[str, int] = {}
    for sch in schemas:
        if sch.name not in typed.columns:
            continue
        original = df[sch.name]
        try:
            converted = convert_series(original, sch)
        except (ValueError, TypeError, OverflowError):
            converted = original
        before = int(original.notna().sum()) - (
            int((original.astype("string").str.strip() == "").sum()) if is_text_series(original) else 0
        )
        after = int(pd.Series(converted).notna().sum())
        if after < before:
            lost[sch.name] = before - after
        typed[sch.name] = converted
    return typed, lost


def schema_table(schemas: list[ColumnSchema]) -> pd.DataFrame:
    """Tableau de synthèse affiché à l'utilisateur."""
    return pd.DataFrame(
        [
            {
                "Variable": s.name,
                "Type détecté": s.storage_type,
                "Type statistique": s.stat_type.value,
                "Famille": s.family.value,
                "Valeurs uniques": s.n_unique,
                "Valeurs manquantes": s.n_missing,
                "% manquants": round(100 * s.missing_ratio, 2),
                "Confiance": round(s.confidence, 2),
                "Justification": s.reason,
                "Corrigé": "oui" if s.overridden else "",
            }
            for s in schemas
        ]
    )
