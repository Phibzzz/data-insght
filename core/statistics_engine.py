"""Statistiques descriptives univariées selon le type des variables.

Conventions (documentées dans l'interface et le rapport) :
* variance et écart-type **d'échantillon** (ddof = 1) ;
* quartiles par interpolation linéaire (méthode par défaut de pandas/NumPy, « type 7 ») ;
* asymétrie et aplatissement corrigés du biais (scipy, ``bias=False``) ;
  l'aplatissement est un **excès de kurtosis** (0 pour une loi normale) ;
* les valeurs manquantes et infinies sont exclues des calculs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

from config.settings import SETTINGS
from core.schema_detector import ColumnSchema, Family, StatType

WEEKDAYS_FR = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
MONTHS_FR = ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin", "Juillet", "Août",
             "Septembre", "Octobre", "Novembre", "Décembre"]


def _f(x: Any) -> float | None:
    """Convertit en float Python, None si non défini."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) or math.isinf(v) else v


# ---------------------------------------------------------------------------
# Variables quantitatives
# ---------------------------------------------------------------------------

@dataclass
class NumericStats:
    """Indicateurs d'une variable quantitative."""

    variable: str
    count: int
    valid: int
    missing: int
    infinite: int
    mean: float | None
    median: float | None
    mode: list[float]
    minimum: float | None
    maximum: float | None
    range: float | None
    variance: float | None
    std: float | None
    q1: float | None
    q2: float | None
    q3: float | None
    iqr: float | None
    cv: float | None
    skewness: float | None
    kurtosis: float | None
    sem: float | None
    ci95_low: float | None
    ci95_high: float | None
    p05: float | None
    p95: float | None
    mad: float | None
    sum: float | None
    n_zeros: int
    n_negative: int
    n_unique: int
    normality_test: str | None = None
    normality_pvalue: float | None = None

    def to_row(self) -> dict[str, Any]:
        """Ligne de tableau avec libellés français."""
        return {
            "Variable": self.variable,
            "Effectif": self.count,
            "Valides": self.valid,
            "Manquants": self.missing,
            "Moyenne": self.mean,
            "Médiane": self.median,
            "Mode": ", ".join(f"{m:g}" for m in self.mode[:3]) + ("…" if len(self.mode) > 3 else ""),
            "Minimum": self.minimum,
            "Maximum": self.maximum,
            "Étendue": self.range,
            "Variance": self.variance,
            "Écart-type": self.std,
            "Q1": self.q1,
            "Q2 (médiane)": self.q2,
            "Q3": self.q3,
            "IQR": self.iqr,
            "Coef. de variation": self.cv,
            "Asymétrie": self.skewness,
            "Aplatissement (excès)": self.kurtosis,
            "Erreur-type": self.sem,
            "IC95 bas": self.ci95_low,
            "IC95 haut": self.ci95_high,
            "P5": self.p05,
            "P95": self.p95,
            "MAD": self.mad,
            "Somme": self.sum,
            "Zéros": self.n_zeros,
            "Négatifs": self.n_negative,
            "Valeurs uniques": self.n_unique,
            "Infinis": self.infinite,
            "Test de normalité": self.normality_test,
            "p-value normalité": self.normality_pvalue,
        }


def numeric_stats(series: pd.Series, name: str | None = None) -> NumericStats:
    """Calcule les statistiques d'une série quantitative.

    Les valeurs manquantes et infinies sont exclues ; leur nombre est rapporté.
    """
    name = name or str(series.name)
    raw = pd.to_numeric(series, errors="coerce").astype("float64")
    count = int(len(raw))
    missing = int(raw.isna().sum())
    arr = raw.dropna().to_numpy()
    infinite = int(np.isinf(arr).sum())
    x = arr[np.isfinite(arr)]
    n = int(x.size)

    if n == 0:
        empty_fields = {
            f: None for f in NumericStats.__dataclass_fields__
            if f not in {"variable", "count", "valid", "missing", "infinite", "mode", "n_zeros", "n_negative", "n_unique"}
        }
        return NumericStats(variable=name, count=count, valid=0, missing=missing, infinite=infinite,
                            mode=[], n_zeros=0, n_negative=0, n_unique=0, **empty_fields)

    mean = float(np.mean(x))
    median = float(np.median(x))
    vals, counts = np.unique(x, return_counts=True)
    mode = vals[counts == counts.max()].tolist() if counts.max() > 1 else []
    q1, q2, q3 = (float(v) for v in np.quantile(x, [0.25, 0.5, 0.75]))
    p05, p95 = (float(v) for v in np.quantile(x, [0.05, 0.95]))
    var = float(np.var(x, ddof=1)) if n > 1 else None
    std = math.sqrt(var) if var is not None else None
    sem = std / math.sqrt(n) if std is not None else None
    if sem is not None and n > 1:
        t = float(sps.t.ppf(0.975, n - 1))
        ci = (mean - t * sem, mean + t * sem)
    else:
        ci = (None, None)
    cv = std / abs(mean) if std is not None and abs(mean) > 1e-12 else None
    constant = bool(np.ptp(x) == 0)
    skew = _f(sps.skew(x, bias=False)) if n > 2 and not constant else None
    kurt = _f(sps.kurtosis(x, fisher=True, bias=False)) if n > 3 and not constant else None

    test_name, pval = None, None
    if 8 <= n and not constant:
        try:
            if n <= SETTINGS.stats.normality_max_n:
                test_name, pval = "Shapiro-Wilk", _f(sps.shapiro(x).pvalue)
            else:
                test_name, pval = "D'Agostino-Pearson", _f(sps.normaltest(x).pvalue)
        except Exception:  # pragma: no cover - cas numériques extrêmes
            test_name, pval = None, None

    return NumericStats(
        variable=name, count=count, valid=n, missing=missing, infinite=infinite,
        mean=mean, median=median, mode=[float(m) for m in mode],
        minimum=float(x.min()), maximum=float(x.max()), range=float(np.ptp(x)),
        variance=var, std=std, q1=q1, q2=q2, q3=q3, iqr=q3 - q1, cv=cv,
        skewness=skew, kurtosis=kurt, sem=sem, ci95_low=ci[0], ci95_high=ci[1],
        p05=p05, p95=p95, mad=float(np.median(np.abs(x - median))), sum=float(x.sum()),
        n_zeros=int((x == 0).sum()), n_negative=int((x < 0).sum()), n_unique=int(vals.size),
        normality_test=test_name, normality_pvalue=pval,
    )


def numeric_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Tableau récapitulatif des variables quantitatives (une ligne par variable)."""
    rows = [numeric_stats(df[c], c).to_row() for c in columns if c in df.columns]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Variables qualitatives
# ---------------------------------------------------------------------------

@dataclass
class CategoricalStats:
    """Indicateurs d'une variable qualitative."""

    variable: str
    count: int
    valid: int
    missing: int
    n_categories: int
    mode: str | None
    mode_count: int
    mode_share: float
    entropy_normalized: float | None
    ordered: bool
    frequencies: pd.DataFrame = field(repr=False)

    def to_row(self) -> dict[str, Any]:
        return {
            "Variable": self.variable,
            "Effectif": self.count,
            "Valides": self.valid,
            "Manquants": self.missing,
            "Modalités": self.n_categories,
            "Modalité dominante": self.mode,
            "Effectif dominante": self.mode_count,
            "% dominante": round(100 * self.mode_share, 2),
            "Entropie normalisée": self.entropy_normalized,
            "Ordonnée": "oui" if self.ordered else "non",
        }


def _label(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "Vrai" if value else "Faux"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def frequency_table(series: pd.Series, ordered: bool | None = None, include_missing: bool = False) -> pd.DataFrame:
    """Table des fréquences absolues, relatives, pourcentages (et cumulées si ordonnée).

    Pour une variable ordonnée (catégorielle ordonnée ou numérique discrète), les
    modalités sont triées dans leur ordre naturel et les fréquences cumulées sont calculées.
    Sinon, tri par effectif décroissant, sans cumul (non pertinent).
    """
    s = series
    is_cat_ordered = isinstance(s.dtype, pd.CategoricalDtype) and s.dtype.ordered
    if ordered is None:
        ordered = bool(is_cat_ordered or (pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)))
    counts = s.value_counts(dropna=not include_missing, sort=False)
    if ordered:
        if is_cat_ordered:
            counts = counts.reindex([c for c in s.dtype.categories if c in counts.index] +
                                    [i for i in counts.index if i not in set(s.dtype.categories)])
        else:
            try:
                counts = counts.sort_index()
            except TypeError:
                counts = counts.sort_values(ascending=False)
    else:
        counts = counts.sort_values(ascending=False, kind="stable")
    counts = counts[counts > 0] if not is_cat_ordered else counts.fillna(0).astype(int)
    total = int(counts.sum())
    index_labels = ["(manquant)" if pd.isna(i) else _label(i) for i in counts.index]
    out = pd.DataFrame({
        "Modalité": index_labels,
        "Effectif": counts.values.astype(int),
        "Fréquence": (counts.values / total).round(6) if total else 0.0,
        "%": (100 * counts.values / total).round(2) if total else 0.0,
    })
    if ordered:
        out["Effectif cumulé"] = out["Effectif"].cumsum()
        out["% cumulé"] = (100 * out["Effectif cumulé"] / total).round(2) if total else 0.0
    return out


def categorical_stats(series: pd.Series, name: str | None = None, ordered: bool | None = None) -> CategoricalStats:
    """Calcule les statistiques d'une variable qualitative."""
    name = name or str(series.name)
    s = series
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        s = s.where(s.astype("string").str.strip() != "", pd.NA)
    count = int(len(s))
    valid_s = s.dropna()
    valid = int(len(valid_s))
    freq = frequency_table(s, ordered=ordered)
    if valid == 0 or freq.empty:
        return CategoricalStats(name, count, 0, count, 0, None, 0, 0.0, None, bool(ordered), freq)
    top = freq.sort_values("Effectif", ascending=False, kind="stable").iloc[0]
    p = freq["Effectif"].to_numpy() / valid
    p = p[p > 0]
    k = len(p)
    entropy = float(-(p * np.log(p)).sum() / math.log(k)) if k > 1 else 0.0
    is_ordered = "Effectif cumulé" in freq.columns
    return CategoricalStats(
        variable=name, count=count, valid=valid, missing=count - valid, n_categories=int(k),
        mode=str(top["Modalité"]), mode_count=int(top["Effectif"]), mode_share=float(top["Effectif"]) / valid,
        entropy_normalized=round(entropy, 4), ordered=is_ordered, frequencies=freq,
    )


# ---------------------------------------------------------------------------
# Variables temporelles
# ---------------------------------------------------------------------------

@dataclass
class TemporalStats:
    """Indicateurs d'une variable temporelle."""

    variable: str
    count: int
    valid: int
    missing: int
    minimum: pd.Timestamp | None
    maximum: pd.Timestamp | None
    span_days: float | None
    n_distinct: int
    inferred_frequency: str | None
    median_gap_days: float | None
    n_gaps: int
    n_duplicated_timestamps: int
    n_future: int
    n_before_1900: int
    by_year: pd.DataFrame = field(repr=False)
    by_month: pd.DataFrame = field(repr=False)
    by_weekday: pd.DataFrame = field(repr=False)
    irregularities: list[str] = field(default_factory=list)

    def to_row(self) -> dict[str, Any]:
        return {
            "Variable": self.variable,
            "Effectif": self.count,
            "Valides": self.valid,
            "Manquants": self.missing,
            "Date minimale": self.minimum,
            "Date maximale": self.maximum,
            "Étendue (jours)": self.span_days,
            "Dates distinctes": self.n_distinct,
            "Fréquence estimée": self.inferred_frequency,
            "Écart médian (jours)": self.median_gap_days,
            "Ruptures détectées": self.n_gaps,
            "Horodatages dupliqués": self.n_duplicated_timestamps,
            "Dates futures": self.n_future,
            "Dates < 1900": self.n_before_1900,
        }


_FREQ_LABELS = [
    (1 / 24 / 60, "minute"), (1 / 24, "horaire"), (1, "journalière"), (7, "hebdomadaire"),
    (30.4, "mensuelle"), (91.3, "trimestrielle"), (365.25, "annuelle"),
]


def _label_frequency(gap_days: float) -> str:
    for ref, label in _FREQ_LABELS:
        if abs(gap_days - ref) <= 0.15 * ref:
            return label
    return f"irrégulière (~{gap_days:.2f} j)"


def _distribution(values: pd.Series, labels: list[str] | None, name: str) -> pd.DataFrame:
    counts = values.value_counts().sort_index()
    total = int(counts.sum()) or 1
    idx = [labels[int(i) - (1 if labels is MONTHS_FR else 0)] if labels else str(int(i)) for i in counts.index]
    return pd.DataFrame({name: idx, "Effectif": counts.values, "%": (100 * counts.values / total).round(2)})


def temporal_stats(series: pd.Series, name: str | None = None, now: pd.Timestamp | None = None) -> TemporalStats:
    """Calcule les statistiques d'une variable de dates."""
    name = name or str(series.name)
    s = pd.to_datetime(series, errors="coerce")
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        s = s.dt.tz_localize(None)
    count = int(len(s))
    d = s.dropna()
    valid = int(len(d))
    now = now or pd.Timestamp.now()
    empty = pd.DataFrame()
    if valid == 0:
        return TemporalStats(name, count, 0, count, None, None, None, 0, None, None, 0, 0, 0, 0, empty, empty, empty)

    dmin, dmax = d.min(), d.max()
    uniq = pd.Series(np.sort(d.unique()))
    gaps = uniq.diff().dropna().dt.total_seconds() / 86400
    median_gap = float(gaps.median()) if len(gaps) else None
    freq = _label_frequency(median_gap) if median_gap else None
    n_gaps = int((gaps > 1.5 * median_gap).sum()) if median_gap else 0
    irregularities: list[str] = []
    if n_gaps:
        irregularities.append(
            f"{n_gaps} interruption(s) : écart supérieur à 1,5 × l'écart médian ({median_gap:.2f} j)."
        )
    n_dup = int(d.duplicated().sum())
    n_future = int((d > now).sum())
    n_old = int((d < pd.Timestamp("1900-01-01")).sum())
    if n_future:
        irregularities.append(f"{n_future} date(s) postérieure(s) à aujourd'hui.")
    if n_old:
        irregularities.append(f"{n_old} date(s) antérieure(s) à 1900 (valeurs par défaut ou erreurs de saisie ?).")
    if gaps.size and median_gap and gaps.std() > 0 and freq and not freq.startswith("irrég"):
        irregular_share = float((np.abs(gaps - median_gap) > 0.15 * median_gap).mean())
        if irregular_share > 0.1:
            irregularities.append(f"{100 * irregular_share:.1f} % des écarts s'éloignent de la fréquence {freq}.")

    return TemporalStats(
        variable=name, count=count, valid=valid, missing=count - valid,
        minimum=dmin, maximum=dmax, span_days=round((dmax - dmin).total_seconds() / 86400, 3),
        n_distinct=int(d.nunique()), inferred_frequency=freq,
        median_gap_days=round(median_gap, 4) if median_gap is not None else None,
        n_gaps=n_gaps, n_duplicated_timestamps=n_dup, n_future=n_future, n_before_1900=n_old,
        by_year=_distribution(d.dt.year, None, "Année"),
        by_month=_distribution(d.dt.month, MONTHS_FR, "Mois"),
        by_weekday=_distribution(d.dt.day_of_week, WEEKDAYS_FR, "Jour"),
        irregularities=irregularities,
    )


# ---------------------------------------------------------------------------
# Agrégation par type
# ---------------------------------------------------------------------------

@dataclass
class UnivariateResults:
    """Ensemble des statistiques univariées d'un dataset."""

    numeric: dict[str, NumericStats]
    categorical: dict[str, CategoricalStats]
    temporal: dict[str, TemporalStats]

    def numeric_table(self) -> pd.DataFrame:
        return pd.DataFrame([s.to_row() for s in self.numeric.values()])

    def categorical_table(self) -> pd.DataFrame:
        return pd.DataFrame([s.to_row() for s in self.categorical.values()])

    def temporal_table(self) -> pd.DataFrame:
        return pd.DataFrame([s.to_row() for s in self.temporal.values()])


def compute_univariate(typed_df: pd.DataFrame, schemas: list[ColumnSchema]) -> UnivariateResults:
    """Calcule les statistiques adaptées au type de chaque variable."""
    numeric, categorical, temporal = {}, {}, {}
    for sch in schemas:
        col = sch.name
        if col not in typed_df.columns:
            continue
        s = typed_df[col]
        if sch.family == Family.NUMERIC:
            numeric[col] = numeric_stats(s, col)
        elif sch.family == Family.CATEGORICAL:
            ordered = sch.stat_type in (StatType.ORDINAL, StatType.MONTH) or None
            categorical[col] = categorical_stats(s, col, ordered=ordered)
        elif sch.stat_type in (StatType.DATE, StatType.DATETIME):
            temporal[col] = temporal_stats(s, col)
        elif sch.stat_type == StatType.YEAR:
            numeric[col] = numeric_stats(s, col)
            categorical[col] = categorical_stats(s, col, ordered=True)
    return UnivariateResults(numeric, categorical, temporal)
