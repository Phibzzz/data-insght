"""Analyses multivariées : corrélations, tableaux croisés, comparaisons de groupes.

Règles de pertinence :
* corrélations (Pearson, Spearman, Kendall) et covariance : uniquement entre
  variables quantitatives (identifiants, constantes et textes exclus) ;
* qualitative × qualitative : tableau de contingence, χ² et V de Cramér ;
* quantitative × qualitative : statistiques par groupe, η², ANOVA / Kruskal-Wallis.

Les valeurs manquantes sont gérées par suppression *par paire* : chaque
coefficient est calculé sur les lignes où les deux variables sont renseignées,
et l'effectif utilisé est rapporté.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

from config.settings import SETTINGS

METHOD_INFO: dict[str, dict[str, str]] = {
    "pearson": {
        "label": "Pearson",
        "mesure": "Intensité de la relation **linéaire** entre deux variables quantitatives.",
        "conditions": "Relation linéaire, variables approximativement normales, absence de valeurs "
                      "aberrantes fortes. Sensible aux outliers.",
    },
    "spearman": {
        "label": "Spearman",
        "mesure": "Relation **monotone** (croissante ou décroissante) calculée sur les rangs.",
        "conditions": "Aucune hypothèse de normalité ; robuste aux valeurs aberrantes ; adapté aux "
                      "variables ordinales ou aux relations non linéaires monotones.",
    },
    "kendall": {
        "label": "Kendall (tau-b)",
        "mesure": "Concordance des paires d'observations (relation monotone).",
        "conditions": "Recommandé pour les petits échantillons et les nombreuses valeurs ex aequo ; "
                      "plus coûteux à calculer sur de grands volumes.",
    },
}


def correlation_strength(r: float) -> str:
    """Qualifie l'intensité d'un coefficient (seuils conventionnels de Cohen adaptés)."""
    a = abs(r)
    if a >= 0.9:
        return "très forte"
    if a >= SETTINGS.stats.strong_correlation:
        return "forte"
    if a >= SETTINGS.stats.moderate_correlation:
        return "modérée"
    if a >= 0.2:
        return "faible"
    return "négligeable"


@dataclass
class CorrelationResult:
    """Matrices de corrélation, p-values et effectifs par paire."""

    method: str
    matrix: pd.DataFrame
    pvalues: pd.DataFrame
    n_obs: pd.DataFrame
    excluded: list[str] = field(default_factory=list)

    def pairs(self, threshold: float = 0.0) -> pd.DataFrame:
        """Liste des paires distinctes triées par |r| décroissant."""
        cols = list(self.matrix.columns)
        rows = []
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                r = self.matrix.loc[a, b]
                if pd.isna(r) or abs(r) < threshold:
                    continue
                rows.append({
                    "Variable 1": a, "Variable 2": b, "Coefficient": round(float(r), 4),
                    "|r|": round(abs(float(r)), 4), "Intensité": correlation_strength(float(r)),
                    "Sens": "positive" if r > 0 else "négative",
                    "p-value": self.pvalues.loc[a, b], "n": int(self.n_obs.loc[a, b]),
                })
        out = pd.DataFrame(rows)
        return out.sort_values("|r|", ascending=False).reset_index(drop=True) if not out.empty else out


def correlation_matrix(df: pd.DataFrame, columns: list[str], method: str = "pearson") -> CorrelationResult:
    """Calcule la matrice de corrélation par paire (suppression par paire des manquants)."""
    if method not in METHOD_INFO:
        raise ValueError(f"Méthode inconnue : {method}")
    data = df[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    excluded = [c for c in columns if data[c].nunique(dropna=True) < 2]
    cols = [c for c in columns if c not in excluded][: SETTINGS.stats.max_corr_columns]
    data = data[cols]
    k = len(cols)
    mat = np.full((k, k), np.nan)
    pv = np.full((k, k), np.nan)
    nn = np.zeros((k, k), dtype=int)
    func = {"pearson": sps.pearsonr, "spearman": sps.spearmanr, "kendall": sps.kendalltau}[method]
    values = {c: data[c].to_numpy(dtype="float64") for c in cols}
    for i in range(k):
        mat[i, i], pv[i, i] = 1.0, 0.0
        nn[i, i] = int(np.isfinite(values[cols[i]]).sum())
        for j in range(i + 1, k):
            x, y = values[cols[i]], values[cols[j]]
            mask = np.isfinite(x) & np.isfinite(y)
            n = int(mask.sum())
            nn[i, j] = nn[j, i] = n
            if n < 3 or np.ptp(x[mask]) == 0 or np.ptp(y[mask]) == 0:
                continue
            res = func(x[mask], y[mask])
            r = float(res.statistic if hasattr(res, "statistic") else res[0])
            p = float(res.pvalue if hasattr(res, "pvalue") else res[1])
            mat[i, j] = mat[j, i] = r
            pv[i, j] = pv[j, i] = p
    return CorrelationResult(
        method=method,
        matrix=pd.DataFrame(mat, index=cols, columns=cols),
        pvalues=pd.DataFrame(pv, index=cols, columns=cols),
        n_obs=pd.DataFrame(nn, index=cols, columns=cols),
        excluded=excluded,
    )


def recommend_method(df: pd.DataFrame, columns: list[str]) -> tuple[str, str]:
    """Recommande une méthode de corrélation à partir des données réelles."""
    if len(columns) < 2:
        return "pearson", "Au moins deux variables quantitatives sont nécessaires."
    data = df[columns].apply(pd.to_numeric, errors="coerce")
    n = int(data.dropna(how="all").shape[0])
    non_normal, outlier_heavy, ties = 0, 0, 0
    for c in columns:
        x = data[c].dropna().to_numpy(dtype="float64")
        x = x[np.isfinite(x)]
        if x.size < 8 or np.ptp(x) == 0:
            continue
        if abs(float(sps.skew(x))) > 1:
            non_normal += 1
        q1, q3 = np.quantile(x, [0.25, 0.75])
        iqr = q3 - q1
        if iqr > 0 and ((x < q1 - 3 * iqr) | (x > q3 + 3 * iqr)).mean() > 0.01:
            outlier_heavy += 1
        if np.unique(x).size < 0.2 * x.size:
            ties += 1
    if n < 30 or ties > len(columns) / 2:
        return "kendall", (f"Petit échantillon (n = {n}) ou nombreuses valeurs ex aequo : "
                           "Kendall est le plus fiable.")
    if non_normal or outlier_heavy:
        return "spearman", (f"{non_normal} variable(s) fortement asymétrique(s) et {outlier_heavy} avec "
                            "des valeurs extrêmes : Spearman (sur les rangs) est plus robuste que Pearson.")
    return "pearson", "Distributions raisonnablement symétriques sans valeurs extrêmes marquées : Pearson convient."


def covariance_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Matrice de covariance d'échantillon (ddof = 1, suppression par paire)."""
    data = df[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return data.cov(min_periods=3)


# ---------------------------------------------------------------------------
# Qualitative × qualitative
# ---------------------------------------------------------------------------

@dataclass
class ContingencyResult:
    """Tableau croisé et test d'indépendance."""

    table: pd.DataFrame
    row_pct: pd.DataFrame
    col_pct: pd.DataFrame
    total_pct: pd.DataFrame
    chi2: float | None
    pvalue: float | None
    dof: int | None
    cramers_v: float | None
    low_expected_share: float | None
    warning: str | None


def contingency(df: pd.DataFrame, row: str, col: str, max_levels: int = 50) -> ContingencyResult:
    """Tableau de contingence, fréquences croisées, χ² et V de Cramér."""
    if row == col:
        raise ValueError("Choisissez deux variables différentes.")
    sub = df[[row, col]].dropna()
    a = sub[row].astype(str)
    b = sub[col].astype(str)
    table = pd.crosstab(a, b)
    if table.shape[0] > max_levels or table.shape[1] > max_levels:
        top_r = a.value_counts().head(max_levels).index
        top_c = b.value_counts().head(max_levels).index
        table = table.loc[table.index.isin(top_r), table.columns.isin(top_c)]
    total = table.to_numpy().sum()
    row_pct = (100 * table.div(table.sum(axis=1), axis=0)).round(2)
    col_pct = (100 * table.div(table.sum(axis=0), axis=1)).round(2)
    total_pct = (100 * table / total).round(2) if total else table * 0.0
    chi2 = p = v = low = None
    dof = None
    warning = None
    if table.shape[0] >= 2 and table.shape[1] >= 2 and total > 0:
        res = sps.chi2_contingency(table.to_numpy(), correction=False)
        chi2, p, dof = float(res.statistic), float(res.pvalue), int(res.dof)
        k = min(table.shape) - 1
        v = math.sqrt(chi2 / (total * k)) if k > 0 else None
        low = float((res.expected_freq < 5).mean())
        if low > 0.2:
            warning = (f"{100 * low:.0f} % des effectifs théoriques sont < 5 : le test du χ² est peu fiable "
                       "(regroupez des modalités ou utilisez un test exact).")
    else:
        warning = "Au moins deux modalités par variable sont nécessaires pour le test du χ²."
    return ContingencyResult(table, row_pct, col_pct, total_pct, chi2, p, dof, v, low, warning)


def cramers_v_matrix(df: pd.DataFrame, columns: list[str], max_levels: int = 50) -> pd.DataFrame:
    """Matrice des V de Cramér entre variables qualitatives."""
    cols = [c for c in columns if 2 <= df[c].nunique(dropna=True) <= max_levels]
    mat = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            try:
                v = contingency(df, a, b, max_levels).cramers_v
            except ValueError:
                v = None
            mat.loc[a, b] = mat.loc[b, a] = np.nan if v is None else v
    return mat


# ---------------------------------------------------------------------------
# Quantitative × qualitative
# ---------------------------------------------------------------------------

@dataclass
class GroupComparison:
    """Comparaison d'une variable quantitative entre groupes."""

    numeric: str
    group: str
    table: pd.DataFrame
    eta_squared: float | None
    anova_pvalue: float | None
    kruskal_pvalue: float | None
    n_groups: int


def group_comparison(df: pd.DataFrame, numeric: str, group: str, max_groups: int = 50) -> GroupComparison:
    """Moyennes, médianes et dispersion par groupe, η², ANOVA et Kruskal-Wallis."""
    sub = pd.DataFrame({
        "y": pd.to_numeric(df[numeric], errors="coerce"),
        "g": df[group].astype("string"),
    }).replace([np.inf, -np.inf], np.nan).dropna()
    top = sub["g"].value_counts().head(max_groups).index
    sub = sub[sub["g"].isin(top)]
    agg = sub.groupby("g", observed=True)["y"].agg(
        Effectif="count", Moyenne="mean", Médiane="median", **{"Écart-type": "std"}, Minimum="min", Maximum="max"
    ).sort_values("Moyenne", ascending=False)
    agg.index.name = group
    groups = [g["y"].to_numpy() for _, g in sub.groupby("g", observed=True) if len(g) >= 2]
    eta = anova_p = kw_p = None
    if len(groups) >= 2:
        grand = sub["y"].mean()
        ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        ss_total = float(((sub["y"] - grand) ** 2).sum())
        eta = float(ss_between / ss_total) if ss_total > 0 else None
        if all(np.ptp(g) > 0 for g in groups) or ss_total > 0:
            try:
                anova_p = float(sps.f_oneway(*groups).pvalue)
                kw_p = float(sps.kruskal(*groups).pvalue)
            except ValueError:
                pass
    return GroupComparison(numeric, group, agg.round(4).reset_index(), eta, anova_p, kw_p, int(agg.shape[0]))


# ---------------------------------------------------------------------------
# Synthèse des fortes associations
# ---------------------------------------------------------------------------

def strong_associations(
    df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str],
    corr: CorrelationResult | None = None, threshold: float | None = None,
) -> pd.DataFrame:
    """Recense les associations fortes (corrélations, V de Cramér, η²)."""
    threshold = threshold if threshold is not None else SETTINGS.stats.strong_correlation
    rows: list[dict[str, Any]] = []
    if corr is None and len(numeric_cols) >= 2:
        corr = correlation_matrix(df, numeric_cols, "spearman")
    if corr is not None:
        for _, r in corr.pairs(threshold).iterrows():
            rows.append({"Variable 1": r["Variable 1"], "Variable 2": r["Variable 2"],
                         "Mesure": f"Corrélation ({METHOD_INFO[corr.method]['label']})",
                         "Valeur": r["Coefficient"], "Intensité": r["Intensité"]})
    cats = [c for c in categorical_cols if 2 <= df[c].nunique(dropna=True) <= 30][:15]
    for i, a in enumerate(cats):
        for b in cats[i + 1:]:
            res = contingency(df, a, b)
            if res.cramers_v is not None and res.cramers_v >= 0.5 and (res.pvalue or 1) < SETTINGS.stats.significance_level:
                rows.append({"Variable 1": a, "Variable 2": b, "Mesure": "V de Cramér",
                             "Valeur": round(res.cramers_v, 4), "Intensité": "forte"})
    for num in numeric_cols[:20]:
        for cat in cats:
            gc = group_comparison(df, num, cat)
            if gc.eta_squared is not None and gc.eta_squared >= 0.26:
                rows.append({"Variable 1": num, "Variable 2": cat, "Mesure": "η² (rapport de corrélation)",
                             "Valeur": round(gc.eta_squared, 4), "Intensité": "forte"})
    return pd.DataFrame(rows)
