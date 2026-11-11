"""Génération automatique de commentaires statistiques à partir des résultats réels.

Chaque commentaire (``Insight``) est rattaché à une catégorie explicite :

* **Constat** : fait chiffré directement issu des calculs ;
* **Interprétation** : lecture statistique du constat ;
* **Hypothèse** : explication possible, à vérifier par l'utilisateur ;
* **Recommandation** : action suggérée.

Aucune valeur n'est inventée : chaque insight cite la mesure dont il découle.
Une interface :class:`LLMProvider` permet d'enrichir ultérieurement les textes
avec un modèle de langage ; le fournisseur par défaut (:class:`NullProvider`)
n'effectue aucun appel externe et ne transmet aucune donnée.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import pandas as pd

from config.settings import SETTINGS
from core.multivariate_engine import CorrelationResult, METHOD_INFO
from core.profiling_engine import DatasetOverview
from core.quality_engine import QualityReport
from core.statistics_engine import UnivariateResults


class Category(str, Enum):
    FINDING = "Constat"
    INTERPRETATION = "Interprétation"
    HYPOTHESIS = "Hypothèse"
    RECOMMENDATION = "Recommandation"


@dataclass
class Insight:
    """Commentaire statistique traçable."""

    category: Category
    topic: str
    variable: str | None
    text: str
    metric: str | None = None
    value: float | None = None
    priority: int = 2  # 1 = élevée, 3 = faible

    def to_row(self) -> dict[str, object]:
        return {
            "Catégorie": self.category.value, "Thème": self.topic, "Variable": self.variable or "",
            "Commentaire": self.text, "Mesure": self.metric or "",
            "Valeur": round(self.value, 4) if isinstance(self.value, float) else self.value,
        }


@dataclass
class InterpretationResult:
    """Ensemble des commentaires et synthèse générale."""

    insights: list[Insight] = field(default_factory=list)
    summary: list[str] = field(default_factory=list)

    def by_category(self, category: Category) -> list[Insight]:
        return sorted((i for i in self.insights if i.category == category), key=lambda i: i.priority)

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([i.to_row() for i in self.insights])


class LLMProvider(Protocol):
    """Interface d'enrichissement par un modèle de langage (optionnelle)."""

    name: str

    def enrich(self, result: InterpretationResult) -> InterpretationResult:  # pragma: no cover - protocole
        ...


class NullProvider:
    """Fournisseur par défaut : aucun appel externe, retourne les commentaires inchangés."""

    name = "Aucun (règles locales)"

    def enrich(self, result: InterpretationResult) -> InterpretationResult:
        return result


def _fmt(x: float | None, digits: int = 2) -> str:
    if x is None:
        return "n.d."
    return f"{x:,.{digits}f}".replace(",", " ").replace(".", ",")


def _int(n: int | float) -> str:
    return f"{int(n):,}".replace(",", " ")


def _pct(x: float) -> str:
    return f"{100 * x:.1f} %".replace(".", ",")


def generate_insights(
    overview: DatasetOverview,
    univariate: UnivariateResults,
    quality: QualityReport | None = None,
    correlation: CorrelationResult | None = None,
    provider: LLMProvider | None = None,
) -> InterpretationResult:
    """Produit les commentaires statistiques à partir des résultats calculés."""
    cfg = SETTINGS.stats
    res = InterpretationResult()
    add = res.insights.append

    # --- Variables quantitatives ---
    for name, st in univariate.numeric.items():
        if st.valid == 0:
            continue
        # Le CV n'a de sens que pour des grandeurs positives (échelle de rapport).
        if st.cv is not None and st.cv > cfg.high_cv and st.minimum is not None and st.minimum >= 0:
            add(Insight(Category.FINDING, "Dispersion", name,
                        f"« {name} » présente une forte dispersion : coefficient de variation de {_pct(st.cv)} "
                        f"(écart-type {_fmt(st.std)} pour une moyenne de {_fmt(st.mean)}).",
                        "Coefficient de variation", st.cv, 1))
            add(Insight(Category.INTERPRETATION, "Dispersion", name,
                        f"La moyenne de « {name} » résume mal les observations ; la médiane ({_fmt(st.median)}) "
                        "et l'écart interquartile sont plus représentatifs.", "Médiane", st.median, 2))
        if st.skewness is not None and abs(st.skewness) > cfg.skew_threshold:
            side = "à droite (queue vers les grandes valeurs)" if st.skewness > 0 else "à gauche (queue vers les petites valeurs)"
            add(Insight(Category.FINDING, "Forme de la distribution", name,
                        f"Distribution de « {name} » asymétrique {side} : skewness = {_fmt(st.skewness)}.",
                        "Asymétrie", st.skewness, 1))
            if st.skewness > 0 and st.minimum is not None and st.minimum > 0:
                add(Insight(Category.RECOMMENDATION, "Forme de la distribution", name,
                            f"Envisager une transformation logarithmique de « {name} » (valeurs strictement "
                            "positives) avant des méthodes supposant la normalité.", "Minimum", st.minimum, 3))
        if st.kurtosis is not None and st.kurtosis > 3:
            add(Insight(Category.INTERPRETATION, "Forme de la distribution", name,
                        f"« {name} » a des queues épaisses (excès d'aplatissement {_fmt(st.kurtosis)}) : "
                        "les valeurs extrêmes sont plus fréquentes que pour une loi normale.",
                        "Aplatissement", st.kurtosis, 2))
        if st.infinite:
            add(Insight(Category.FINDING, "Validité", name, f"« {name} » contient {st.infinite} valeur(s) infinie(s).",
                        "Infinis", float(st.infinite), 1))

    non_normal = [n for n, st in univariate.numeric.items()
                  if st.normality_pvalue is not None and st.normality_pvalue < cfg.significance_level and st.valid >= 30]
    if non_normal:
        add(Insight(Category.INTERPRETATION, "Normalité", None,
                    f"Les tests de normalité (Shapiro-Wilk ou D'Agostino, α = {cfg.significance_level:g}) rejettent "
                    f"l'hypothèse de normalité pour {len(non_normal)} variable(s) : {', '.join(non_normal[:8])}"
                    + ("…" if len(non_normal) > 8 else "") + ". Privilégier des méthodes non paramétriques "
                    "(médiane, Spearman, Kruskal-Wallis) pour ces variables.", "Variables non normales",
                    float(len(non_normal)), 3))

    # --- Variables qualitatives ---
    for name, st in univariate.categorical.items():
        if st.valid == 0 or st.n_categories < 2:
            continue
        if st.mode_share >= cfg.dominant_category_share:
            add(Insight(Category.FINDING, "Modalité dominante", name,
                        f"La modalité « {st.mode} » représente {_pct(st.mode_share)} des valeurs de « {name} ».",
                        "Part de la modalité dominante", st.mode_share, 1))
            add(Insight(Category.INTERPRETATION, "Modalité dominante", name,
                        f"« {name} » est peu discriminante : les comparaisons entre modalités reposeront sur "
                        "de faibles effectifs pour les modalités minoritaires.", None, None, 2))
        elif st.mode_share >= 0.5:
            add(Insight(Category.FINDING, "Modalité dominante", name,
                        f"« {st.mode} » est majoritaire dans « {name} » ({_pct(st.mode_share)}).",
                        "Part de la modalité dominante", st.mode_share, 3))
        if st.n_categories > 15 and not st.ordered:
            add(Insight(Category.RECOMMENDATION, "Cardinalité", name,
                        f"« {name} » compte {st.n_categories} modalités : regrouper les moins fréquentes "
                        "pour faciliter la lecture et les tests.", "Modalités", float(st.n_categories), 3))

    # --- Variables temporelles ---
    for name, st in univariate.temporal.items():
        if st.valid == 0:
            continue
        add(Insight(Category.FINDING, "Couverture temporelle", name,
                    f"« {name} » couvre du {st.minimum:%d/%m/%Y} au {st.maximum:%d/%m/%Y} "
                    f"({_fmt(st.span_days, 0)} jours), fréquence estimée : {st.inferred_frequency or 'n.d.'}.",
                    "Étendue (jours)", st.span_days, 2))
        for irr in st.irregularities:
            add(Insight(Category.FINDING, "Irrégularités temporelles", name, f"« {name} » : {irr}", None, None, 2))
        if st.n_gaps:
            add(Insight(Category.HYPOTHESIS, "Irrégularités temporelles", name,
                        f"Les {st.n_gaps} interruption(s) de « {name} » peuvent refléter des périodes sans "
                        "activité (week-ends, fermetures) ou des données non collectées.", None, None, 3))

    # --- Valeurs manquantes et qualité ---
    if quality is not None:
        for issue in quality.issues:
            if issue.check == "Valeurs manquantes" and issue.share >= cfg.high_missing_share:
                add(Insight(Category.FINDING, "Complétude", issue.column,
                            f"« {issue.column} » est très incomplète : {_pct(issue.share)} de valeurs manquantes.",
                            "Taux de manquants", issue.share, 1))
                add(Insight(Category.HYPOTHESIS, "Complétude", issue.column,
                            f"Les manquants de « {issue.column} » peuvent ne pas être aléatoires (question "
                            "facultative, information non applicable) : vérifier avant toute imputation.", None, None, 3))
            if issue.check == "Valeurs atypiques potentielles" and issue.share > cfg.outlier_share_alert:
                add(Insight(Category.FINDING, "Valeurs atypiques", issue.column,
                            f"« {issue.column} » : {issue.count} valeurs atypiques selon la règle IQR "
                            f"({_pct(issue.share)}).", "Part d'atypiques", issue.share, 2))
        low_cols = quality.column_scores[quality.column_scores["Score (/100)"] < 70] \
            if not quality.column_scores.empty else pd.DataFrame()
        for _, row in low_cols.iterrows():
            add(Insight(Category.FINDING, "Qualité", row["Variable"],
                        f"« {row['Variable']} » présente un risque de faible qualité (score {row['Score (/100)']}/100).",
                        "Score qualité colonne", float(row["Score (/100)"]), 1))
        if overview.n_duplicates:
            add(Insight(Category.RECOMMENDATION, "Doublons", None,
                        f"Supprimer ou justifier les {overview.n_duplicates} ligne(s) dupliquée(s) "
                        f"({_pct(overview.duplicate_ratio)} du fichier).", "Doublons", float(overview.n_duplicates), 1))
        seen = set()
        for issue in sorted(quality.issues, key=lambda i: (i.severity != "critique", i.severity != "majeur")):
            if issue.severity in ("critique", "majeur") and (issue.check, issue.column) not in seen:
                seen.add((issue.check, issue.column))
                add(Insight(Category.RECOMMENDATION, issue.check, issue.column if issue.column != "(toutes)" else None,
                            f"{issue.check} — {issue.column} : {issue.recommendation}", None, None, 2))

    # --- Corrélations ---
    if correlation is not None and not correlation.matrix.empty:
        label = METHOD_INFO[correlation.method]["label"]
        pairs = correlation.pairs(cfg.strong_correlation)
        for _, p in pairs.head(10).iterrows():
            add(Insight(Category.FINDING, "Corrélations", f"{p['Variable 1']} × {p['Variable 2']}",
                        f"Corrélation {p['Intensité']} et {p['Sens']} entre « {p['Variable 1']} » et "
                        f"« {p['Variable 2']} » ({label} r = {_fmt(p['Coefficient'], 3)}, n = {p['n']}).",
                        f"r de {label}", float(p["Coefficient"]), 1))
        if not pairs.empty:
            add(Insight(Category.INTERPRETATION, "Corrélations", None,
                        "Une corrélation mesure une association, pas une causalité : un facteur commun "
                        "ou une relation mathématique (ex. total = prix × quantité) peut l'expliquer.", None, None, 2))
            add(Insight(Category.HYPOTHESIS, "Corrélations", None,
                        "Des variables très corrélées peuvent porter une information redondante.", None, None, 3))
            very = pairs[pairs["|r|"] >= 0.9]
            if not very.empty:
                add(Insight(Category.RECOMMENDATION, "Corrélations", None,
                            f"{len(very)} paire(s) avec |r| ≥ 0,9 : en modélisation, conserver une seule "
                            "variable par paire pour limiter la multicolinéarité.", "Paires |r| ≥ 0,9", float(len(very)), 2))

    # --- Synthèse générale ---
    res.summary = _summary(overview, univariate, quality, correlation)
    return (provider or NullProvider()).enrich(res)


def _summary(
    overview: DatasetOverview, uni: UnivariateResults,
    quality: QualityReport | None, corr: CorrelationResult | None,
) -> list[str]:
    lines = [
        f"Le jeu de données contient {_int(overview.n_rows)} observations et {overview.n_cols} variables "
        f"({overview.n_numeric} quantitative(s), {overview.n_categorical} qualitative(s), "
        f"{overview.n_temporal} temporelle(s), {overview.n_other} autre(s)).",
        f"Taux global de valeurs manquantes : {_pct(overview.missing_ratio)} ; "
        f"lignes complètes : {_int(overview.n_complete_rows)} ; doublons : {_int(overview.n_duplicates)}.",
    ]
    if quality is not None:
        lines.append(
            f"Score de qualité (heuristique) : {_fmt(quality.global_score, 1)}/100 ({quality.grade.lower()}) ; "
            f"{sum(1 for i in quality.issues if i.severity in ('critique', 'majeur'))} problème(s) majeur(s) ou critique(s)."
        )
    skewed = [n for n, s in uni.numeric.items() if s.skewness is not None and abs(s.skewness) > SETTINGS.stats.skew_threshold]
    if skewed:
        lines.append(f"{len(skewed)} variable(s) quantitative(s) à distribution asymétrique : {', '.join(skewed[:5])}"
                     + ("…" if len(skewed) > 5 else "") + ".")
    if corr is not None and not corr.matrix.empty:
        pairs = corr.pairs(SETTINGS.stats.strong_correlation)
        if pairs.empty:
            lines.append("Aucune corrélation forte détectée entre variables quantitatives.")
        else:
            top = pairs.iloc[0]
            lines.append(f"{len(pairs)} corrélation(s) forte(s) ; la plus marquée : {top['Variable 1']} × "
                         f"{top['Variable 2']} (r = {_fmt(top['Coefficient'], 3)}).")
    return lines
