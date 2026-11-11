"""Audit automatique de la qualité des données et score global transparent.

Score global = moyenne pondérée de cinq sous-scores sur 100 :

=================  ==========================================================
Complétude         100 × (1 − cellules manquantes / cellules totales)
Unicité            100 × (1 − lignes dupliquées / lignes)
Validité           100 × (1 − cellules invalides / cellules évaluées) ; invalides =
                   valeurs non convertibles dans le type détecté, infinis, dates
                   implausibles, négatifs sur colonnes positives par nature,
                   violations des règles utilisateur
Cohérence          100 × (1 − cellules texte incohérentes / cellules texte) ;
                   incohérentes = espaces superflus, variantes de casse/accents
Valeurs atypiques  100 × (1 − valeurs hors bornes IQR / valeurs numériques)
=================  ==========================================================

Pondérations par défaut : 35 / 20 / 20 / 15 / 10 (modifiables dans Paramètres).
Ce score est un **indicateur heuristique** destiné à prioriser les actions ; il ne
constitue pas une mesure universelle de la qualité et dépend du contexte métier.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from config.settings import SETTINGS, QualityWeights
from core.profiling_engine import count_duplicates
from core.schema_detector import ColumnSchema, Family, StatType, is_text_series, strip_accents

SEVERITY_ORDER = {"critique": 0, "majeur": 1, "mineur": 2, "info": 3}
_POSITIVE_HINT = re.compile(
    r"(^|_)(age|âge|prix|price|montant|amount|quantit|qty|poids|weight|taille|height|salaire|salary|"
    r"revenu|income|nb|nombre|count|duree|durée|duration|surface|distance)", re.I,
)

METHODOLOGY = __doc__ or ""


@dataclass
class ValidityRule:
    """Règle de validité définie par l'utilisateur pour une colonne."""

    column: str
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: list[str] | None = None
    regex: str | None = None
    not_null: bool = False

    def describe(self) -> str:
        parts = []
        if self.min_value is not None:
            parts.append(f"≥ {self.min_value:g}")
        if self.max_value is not None:
            parts.append(f"≤ {self.max_value:g}")
        if self.allowed_values:
            parts.append("∈ {" + ", ".join(self.allowed_values[:6]) + ("…" if len(self.allowed_values) > 6 else "") + "}")
        if self.regex:
            parts.append(f"motif « {self.regex} »")
        if self.not_null:
            parts.append("obligatoire")
        return f"{self.column} : " + ", ".join(parts)

    def violations(self, s: pd.Series) -> pd.Series:
        """Masque des valeurs violant la règle."""
        mask = pd.Series(False, index=s.index)
        if self.not_null:
            mask |= s.isna()
        present = s.notna()
        if self.min_value is not None or self.max_value is not None:
            num = pd.to_numeric(s, errors="coerce")
            if self.min_value is not None:
                mask |= present & (num < self.min_value)
            if self.max_value is not None:
                mask |= present & (num > self.max_value)
        if self.allowed_values:
            allowed = {str(v) for v in self.allowed_values}
            mask |= present & ~s.astype(str).isin(allowed)
        if self.regex:
            try:
                mask |= present & ~s.astype(str).str.fullmatch(self.regex)
            except re.error:
                pass
        return mask.fillna(False).astype(bool)


@dataclass
class Issue:
    """Problème détecté lors de l'audit."""

    check: str
    column: str
    severity: str
    count: int
    share: float
    description: str
    recommendation: str

    def to_row(self) -> dict[str, Any]:
        return {
            "Contrôle": self.check, "Variable": self.column, "Sévérité": self.severity,
            "Nombre": self.count, "%": round(100 * self.share, 2),
            "Constat": self.description, "Recommandation": self.recommendation,
        }


@dataclass
class OutlierSummary:
    """Résultat de la détection de valeurs atypiques pour une variable."""

    column: str
    n: int
    lower_iqr: float
    upper_iqr: float
    n_iqr: int
    n_zscore: int
    n_both: int


@dataclass
class QualityReport:
    """Rapport complet d'audit qualité."""

    issues: list[Issue]
    sub_scores: dict[str, float]
    weights: dict[str, float]
    global_score: float
    indicators: dict[str, float]
    column_scores: pd.DataFrame
    outliers: list[OutlierSummary] = field(default_factory=list)
    rules: list[ValidityRule] = field(default_factory=list)

    def issues_table(self) -> pd.DataFrame:
        rows = [i.to_row() for i in sorted(self.issues, key=lambda i: (SEVERITY_ORDER[i.severity], -i.count))]
        return pd.DataFrame(rows, columns=["Contrôle", "Variable", "Sévérité", "Nombre", "%", "Constat", "Recommandation"])

    def outliers_table(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "Variable": o.column, "n": o.n, "Borne basse (IQR)": round(o.lower_iqr, 4),
            "Borne haute (IQR)": round(o.upper_iqr, 4), "Atypiques IQR": o.n_iqr,
            "% IQR": round(100 * o.n_iqr / o.n, 2) if o.n else 0.0,
            f"Atypiques |z|>{SETTINGS.stats.zscore_threshold:g}": o.n_zscore,
            "Signalées par les deux": o.n_both,
        } for o in self.outliers])

    @property
    def grade(self) -> str:
        s = self.global_score
        return "Excellente" if s >= 90 else "Bonne" if s >= 75 else "Moyenne" if s >= 60 else "Faible"


# ---------------------------------------------------------------------------
# Détection de valeurs atypiques
# ---------------------------------------------------------------------------

def iqr_bounds(s: pd.Series, factor: float | None = None) -> tuple[float, float]:
    """Bornes de Tukey : [Q1 − k·IQR ; Q3 + k·IQR]."""
    factor = SETTINGS.stats.iqr_factor if factor is None else factor
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return float("nan"), float("nan")
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    return float(q1 - factor * iqr), float(q3 + factor * iqr)


def iqr_outliers(s: pd.Series, factor: float | None = None) -> pd.Series:
    """Masque booléen des valeurs hors bornes IQR (les manquants ne sont jamais atypiques)."""
    lo, hi = iqr_bounds(s, factor)
    x = pd.to_numeric(s, errors="coerce")
    if np.isnan(lo):
        return pd.Series(False, index=s.index)
    return ((x < lo) | (x > hi)).fillna(False).astype(bool)


def zscore_outliers(s: pd.Series, threshold: float | None = None) -> pd.Series:
    """Masque booléen des valeurs dont |z| dépasse le seuil (écart-type d'échantillon)."""
    threshold = SETTINGS.stats.zscore_threshold if threshold is None else threshold
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    std = x.std(ddof=1)
    if not std or np.isnan(std):
        return pd.Series(False, index=s.index)
    z = (x - x.mean()) / std
    return (z.abs() > threshold).fillna(False).astype(bool)


# ---------------------------------------------------------------------------
# Contrôles élémentaires
# ---------------------------------------------------------------------------

def _severity_for_share(share: float, critical: float, major: float) -> str:
    if share >= critical:
        return "critique"
    if share >= major:
        return "majeur"
    return "mineur"


def _inconsistent_variants(s: pd.Series) -> tuple[int, int, list[str]]:
    """Compte les cellules avec espaces superflus et les variantes de casse/accents.

    Returns:
        (cellules avec espaces superflus, cellules variantes, exemples de groupes)
    """
    txt = s.dropna().astype(str)
    if txt.empty:
        return 0, 0, []
    n_space = int((txt != txt.str.strip()).sum() + txt.str.contains(r"\s{2,}", regex=True).sum())
    counts = txt.value_counts()
    norm = pd.Series({v: strip_accents(re.sub(r"\s+", " ", v.strip())).lower() for v in counts.index})
    groups = norm.groupby(norm.values).groups
    n_variant, examples = 0, []
    for _, members in groups.items():
        if len(members) > 1:
            member_counts = counts[list(members)].sort_values(ascending=False)
            n_variant += int(member_counts.iloc[1:].sum())
            if len(examples) < 3:
                examples.append(" / ".join(f"« {m} »" for m in member_counts.index[:3]))
    return n_space, n_variant, examples


def _issue(check: str, column: str, severity: str, count: int, total: int, description: str, reco: str) -> Issue:
    return Issue(check, column, severity, int(count), count / total if total else 0.0, description, reco)


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def run_quality_audit(
    raw_df: pd.DataFrame,
    typed_df: pd.DataFrame,
    schemas: list[ColumnSchema],
    lost_conversions: dict[str, int] | None = None,
    rules: list[ValidityRule] | None = None,
    weights: QualityWeights | None = None,
    now: pd.Timestamp | None = None,
) -> QualityReport:
    """Exécute tous les contrôles qualité.

    Args:
        raw_df: données telles que chargées (pour les contrôles de format texte).
        typed_df: vue typée issue de ``apply_schema``.
        schemas: schéma des variables.
        lost_conversions: valeurs non convertibles par colonne (sortie de ``apply_schema``).
        rules: règles de validité définies par l'utilisateur.
        weights: pondérations du score global.
    """
    cfg = SETTINGS.stats
    lost_conversions = lost_conversions or {}
    rules = rules or []
    weights_n = (weights or SETTINGS.quality).normalized()
    now = now or pd.Timestamp.now()
    n_rows, n_cols = typed_df.shape
    n_cells = n_rows * n_cols
    issues: list[Issue] = []

    col_missing: dict[str, int] = {}
    col_invalid: dict[str, int] = {}
    col_inconsistent: dict[str, int] = {}
    col_outliers: dict[str, int] = {}
    outlier_summaries: list[OutlierSummary] = []
    text_cells = numeric_cells = 0

    # Doublons
    n_dup = count_duplicates(raw_df)
    if n_dup:
        has_id = any(s.stat_type == StatType.IDENTIFIER for s in schemas)
        issues.append(_issue(
            "Lignes dupliquées", "(toutes)", _severity_for_share(n_dup / max(n_rows, 1), 0.2, 0.05), n_dup, n_rows,
            f"{n_dup} ligne(s) strictement identique(s) à une ligne précédente."
            + ("" if has_id else " Le fichier n'a pas d'identifiant : des lignes identiques peuvent être "
               "légitimes (ex. réponses identiques à une enquête)."),
            "Vérifier s'il s'agit de doublons techniques (double import) et les supprimer dans « Nettoyage ».",
        ))

    for sch in schemas:
        col = sch.name
        if col not in typed_df.columns:
            continue
        s = typed_df[col]
        raw = raw_df[col] if col in raw_df.columns else s
        miss = int(s.isna().sum())
        col_missing[col] = miss
        invalid = int(lost_conversions.get(col, 0))

        if sch.stat_type == StatType.EMPTY:
            issues.append(_issue("Colonne vide", col, "majeur", n_rows, n_rows,
                                 "La colonne ne contient aucune valeur.",
                                 "Supprimer la colonne si elle n'a pas d'utilité métier."))
            col_invalid[col] = 0
            continue
        if sch.stat_type == StatType.CONSTANT:
            issues.append(_issue("Colonne constante", col, "mineur", n_rows - miss, n_rows,
                                 f"Une seule valeur distincte ({sch.sample_values[0] if sch.sample_values else ''}).",
                                 "Colonne sans pouvoir discriminant : la supprimer pour l'analyse."))

        if miss:
            share = miss / n_rows
            issues.append(_issue(
                "Valeurs manquantes", col, _severity_for_share(share, 0.5, cfg.high_missing_share), miss, n_rows,
                f"{100 * share:.1f} % de valeurs manquantes.",
                "Supprimer la colonne (> 50 %) ou imputer (médiane pour le numérique, mode pour le "
                "catégoriel) après analyse du mécanisme de non-réponse." if share > 0.05 else
                "Faible proportion : suppression des lignes ou imputation simple envisageables.",
            ))

        if invalid:
            issues.append(_issue(
                "Valeurs incohérentes avec le type", col, "majeur" if invalid / n_rows > 0.01 else "mineur",
                invalid, n_rows,
                f"{invalid} valeur(s) non convertible(s) en {sch.stat_type.value.lower()}.",
                "Corriger les saisies erronées ou les remplacer par des valeurs manquantes.",
            ))

        if sch.conversion == "numeric" and sch.family == Family.NUMERIC:
            issues.append(_issue(
                "Type de données incorrect", col, "info", n_rows - miss, n_rows,
                "Valeurs numériques stockées sous forme de texte (virgule décimale, espaces…).",
                "Convertir la colonne en numérique (fait automatiquement dans la vue typée).",
            ))
        if sch.conversion == "datetime":
            issues.append(_issue(
                "Type de données incorrect", col, "info", n_rows - miss, n_rows,
                "Dates stockées sous forme de texte" + (f" (format {sch.date_format})." if sch.date_format else
                                                         " avec des formats multiples."),
                "Convertir en type date et harmoniser le format (ISO 8601 recommandé).",
            ))

        # Numériques : infinis, atypiques, négatifs suspects
        if sch.family == Family.NUMERIC or sch.stat_type == StatType.YEAR:
            x = pd.to_numeric(s, errors="coerce")
            numeric_cells += int(x.notna().sum())
            n_inf = int(np.isinf(x.to_numpy(dtype="float64", na_value=np.nan)).sum())
            if n_inf:
                invalid += n_inf
                issues.append(_issue("Valeurs infinies", col, "majeur", n_inf, n_rows,
                                     f"{n_inf} valeur(s) infinie(s) (±inf).",
                                     "Remplacer par des valeurs manquantes puis traiter comme telles."))
            if sch.stat_type in (StatType.CONTINUOUS, StatType.DISCRETE) and x.nunique() > 2:
                m_iqr = iqr_outliers(x)
                m_z = zscore_outliers(x)
                lo, hi = iqr_bounds(x)
                summary = OutlierSummary(col, int(x.notna().sum()), lo, hi, int(m_iqr.sum()), int(m_z.sum()),
                                         int((m_iqr & m_z).sum()))
                outlier_summaries.append(summary)
                col_outliers[col] = summary.n_iqr
                if summary.n_iqr:
                    share = summary.n_iqr / max(summary.n, 1)
                    issues.append(_issue(
                        "Valeurs atypiques potentielles", col,
                        "majeur" if share > cfg.outlier_share_alert else "mineur", summary.n_iqr, n_rows,
                        f"{summary.n_iqr} valeur(s) hors de [{lo:.4g} ; {hi:.4g}] (IQR × {cfg.iqr_factor:g}), "
                        f"dont {summary.n_both} également avec |z| > {cfg.zscore_threshold:g}.",
                        "Vérifier ces valeurs : une valeur atypique n'est pas nécessairement une erreur "
                        "(phénomène réel, sous-population). Corriger les erreurs avérées ; sinon envisager "
                        "une transformation (log) ou des méthodes robustes.",
                    ))
            if _POSITIVE_HINT.search(col):
                n_neg = int((x < 0).sum())
                if n_neg:
                    invalid += n_neg
                    issues.append(_issue(
                        "Valeurs incohérentes (hypothèse)", col, "majeur", n_neg, n_rows,
                        f"{n_neg} valeur(s) négative(s) alors que le nom suggère une grandeur positive.",
                        "Hypothèse à confirmer avec le métier : corriger ou exclure ces valeurs.",
                    ))
            if sch.stat_type == StatType.LATITUDE:
                bad = int(((x < -90) | (x > 90)).sum())
            elif sch.stat_type == StatType.LONGITUDE:
                bad = int(((x < -180) | (x > 180)).sum())
            else:
                bad = 0
            if bad:
                invalid += bad
                issues.append(_issue("Coordonnées invalides", col, "majeur", bad, n_rows,
                                     f"{bad} coordonnée(s) hors des bornes géographiques.",
                                     "Vérifier l'inversion latitude/longitude ou les erreurs de saisie."))

        # Dates
        if sch.stat_type in (StatType.DATE, StatType.DATETIME):
            d = pd.to_datetime(s, errors="coerce")
            n_future = int((d > now).sum())
            n_old = int((d < pd.Timestamp("1900-01-01")).sum())
            if n_future:
                issues.append(_issue("Problème de dates", col, "mineur", n_future, n_rows,
                                     f"{n_future} date(s) dans le futur.",
                                     "Vérifier s'il s'agit de dates prévisionnelles ou d'erreurs de saisie."))
            if n_old:
                invalid += n_old
                issues.append(_issue("Problème de dates", col, "majeur", n_old, n_rows,
                                     f"{n_old} date(s) antérieure(s) à 1900 (valeurs par défaut probables).",
                                     "Remplacer les dates sentinelles (01/01/1900…) par des valeurs manquantes."))

        # Catégorielles : modalités rares, incohérences de format
        if sch.family == Family.CATEGORICAL and sch.stat_type not in (StatType.BOOLEAN,):
            vc = s.value_counts(dropna=True)
            total = int(vc.sum())
            if total and len(vc) >= 3:
                rare = vc[vc / total < cfg.rare_category_share]
                if len(rare):
                    issues.append(_issue(
                        "Modalités rares", col, "info", int(rare.sum()), n_rows,
                        f"{len(rare)} modalité(s) représentant chacune < {100 * cfg.rare_category_share:g} % "
                        f"(ex. {', '.join(map(str, rare.index[:3]))}).",
                        "Regrouper les modalités rares dans « Autre » pour fiabiliser les analyses croisées.",
                    ))

        if is_text_series(raw) and sch.stat_type not in (StatType.TEXT,):
            n_text = int(raw.notna().sum())
            text_cells += n_text
            n_space, n_variant, examples = _inconsistent_variants(raw)
            col_inconsistent[col] = n_space + n_variant
            if n_space:
                issues.append(_issue("Incohérence de format", col, "mineur", n_space, n_rows,
                                     f"{n_space} valeur(s) avec des espaces en début/fin ou multiples.",
                                     "Appliquer « Normaliser le texte » (suppression des espaces superflus)."))
            if n_variant:
                issues.append(_issue("Incohérence de format", col, "majeur" if n_variant / n_rows > 0.05 else "mineur",
                                     n_variant, n_rows,
                                     f"Variantes d'écriture d'une même modalité : {'; '.join(examples)}.",
                                     "Harmoniser la casse et les accents (Nettoyage › Normaliser le texte)."))

        # Règles utilisateur
        for rule in rules:
            if rule.column == col:
                n_viol = int(rule.violations(s).sum())
                if n_viol:
                    invalid += n_viol
                    issues.append(_issue("Règle de validité", col, "majeur", n_viol, n_rows,
                                         f"{n_viol} valeur(s) ne respectent pas la règle {rule.describe()}.",
                                         "Corriger les valeurs ou revoir la règle métier."))
        col_invalid[col] = invalid

    # --- Indicateurs et scores ---
    n_missing = int(sum(col_missing.values()))
    n_invalid = int(sum(col_invalid.values()))
    n_inconsistent = int(sum(col_inconsistent.values()))
    n_outliers = int(sum(col_outliers.values()))
    filled = max(n_cells - n_missing, 1)
    completeness = 1 - n_missing / n_cells if n_cells else 1.0
    uniqueness = 1 - n_dup / n_rows if n_rows else 1.0
    validity = 1 - min(n_invalid / filled, 1.0)
    consistency = 1 - min(n_inconsistent / text_cells, 1.0) if text_cells else 1.0
    outlier_score = 1 - min(n_outliers / numeric_cells, 1.0) if numeric_cells else 1.0
    sub_scores = {
        "completeness": round(100 * completeness, 2),
        "uniqueness": round(100 * uniqueness, 2),
        "validity": round(100 * validity, 2),
        "consistency": round(100 * consistency, 2),
        "outliers": round(100 * outlier_score, 2),
    }
    global_score = round(sum(sub_scores[k] * weights_n[k] for k in sub_scores), 2)
    indicators = {
        "Taux de complétude (%)": round(100 * completeness, 2),
        "Taux de valeurs manquantes (%)": round(100 * (1 - completeness), 2),
        "Taux de doublons (%)": round(100 * (1 - uniqueness), 2),
        "Taux de validité (%)": round(100 * validity, 2),
        "Lignes complètes (%)": round(100 * float(typed_df.notna().all(axis=1).mean()), 2) if n_rows else 100.0,
    }

    col_rows = []
    for sch in schemas:
        col = sch.name
        n_present = n_rows - col_missing.get(col, 0)
        c_comp = 100 * n_present / n_rows if n_rows else 100.0
        c_val = 100 * (1 - min(col_invalid.get(col, 0) / max(n_present, 1), 1.0))
        c_cons = 100 * (1 - min(col_inconsistent.get(col, 0) / max(n_present, 1), 1.0))
        c_out = 100 * (1 - min(col_outliers.get(col, 0) / max(n_present, 1), 1.0))
        score = (c_comp * weights_n["completeness"] + c_val * weights_n["validity"] +
                 c_cons * weights_n["consistency"] + c_out * weights_n["outliers"]) / (
            weights_n["completeness"] + weights_n["validity"] + weights_n["consistency"] + weights_n["outliers"])
        if sch.stat_type == StatType.EMPTY:
            score = 0.0
        col_rows.append({
            "Variable": col, "Type": sch.stat_type.value, "Complétude (%)": round(c_comp, 2),
            "Validité (%)": round(c_val, 2), "Cohérence (%)": round(c_cons, 2),
            "Hors atypiques (%)": round(c_out, 2), "Score (/100)": round(score, 1),
        })
    column_scores = pd.DataFrame(col_rows).sort_values("Score (/100)", kind="stable").reset_index(drop=True) \
        if col_rows else pd.DataFrame()

    return QualityReport(issues, sub_scores, weights_n, global_score, indicators, column_scores,
                         outlier_summaries, rules)


SUBSCORE_LABELS = {
    "completeness": "Complétude",
    "uniqueness": "Unicité",
    "validity": "Validité",
    "consistency": "Cohérence",
    "outliers": "Valeurs atypiques",
}
