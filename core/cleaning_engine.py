"""Transformations de préparation des données, appliquées sur une copie de travail.

Principes :
* chaque opération est une fonction **pure** ``(DataFrame, **params) -> DataFrame``
  qui ne modifie jamais son entrée ;
* chaque application produit un :class:`Operation` (description, dimensions avant/après)
  stocké dans l'historique d'une :class:`CleaningSession` ;
* l'annulation rejoue l'historique depuis l'original : aucune donnée n'est
  modifiée silencieusement, l'original reste intact ;
* la création de variables par expression utilise un évaluateur restreint basé
  sur ``ast`` (opérations arithmétiques et quelques fonctions) — jamais ``eval``.
"""
from __future__ import annotations

import ast
import operator
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from core.exceptions import CleaningError
from core.quality_engine import iqr_bounds, zscore_outliers
from core.schema_detector import strip_accents, to_boolean_smart, to_datetime_smart, to_numeric_smart


@dataclass
class Operation:
    """Trace d'une transformation appliquée."""

    name: str
    params: dict[str, Any]
    description: str
    rows_before: int
    cols_before: int
    rows_after: int
    cols_after: int
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["params"] = {k: (v if isinstance(v, (str, int, float, bool, type(None), list, dict)) else str(v))
                       for k, v in self.params.items()}
        return d


OpFn = Callable[..., tuple[pd.DataFrame, str]]
OPERATIONS: dict[str, OpFn] = {}


def operation(name: str) -> Callable[[OpFn], OpFn]:
    """Décorateur d'enregistrement d'une opération de nettoyage."""

    def deco(fn: OpFn) -> OpFn:
        OPERATIONS[name] = fn
        return fn

    return deco


def _check_columns(df: pd.DataFrame, columns: list[str] | None) -> list[str]:
    columns = list(columns or [])
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise CleaningError(f"Colonne(s) introuvable(s) : {', '.join(missing)}.")
    return columns


# ---------------------------------------------------------------------------
# Opérations
# ---------------------------------------------------------------------------

@operation("drop_duplicates")
def drop_duplicates(df: pd.DataFrame, subset: list[str] | None = None, keep: str = "first") -> tuple[pd.DataFrame, str]:
    """Supprime les lignes dupliquées (sur toutes les colonnes ou un sous-ensemble)."""
    subset = _check_columns(df, subset) or None
    keep_arg: Any = False if keep == "none" else keep
    out = df.drop_duplicates(subset=subset, keep=keep_arg).reset_index(drop=True)
    scope = f"colonnes {', '.join(subset)}" if subset else "toutes les colonnes"
    return out, f"Suppression de {len(df) - len(out)} doublon(s) ({scope})."


@operation("drop_missing_rows")
def drop_missing_rows(df: pd.DataFrame, columns: list[str] | None = None, how: str = "any") -> tuple[pd.DataFrame, str]:
    """Supprime les lignes contenant des valeurs manquantes."""
    columns = _check_columns(df, columns) or None
    out = df.dropna(subset=columns, how=how).reset_index(drop=True)
    scope = f"dans {', '.join(columns)}" if columns else "dans au moins une colonne"
    return out, f"Suppression de {len(df) - len(out)} ligne(s) avec valeur(s) manquante(s) {scope}."


@operation("drop_missing_columns")
def drop_missing_columns(df: pd.DataFrame, threshold: float = 0.5) -> tuple[pd.DataFrame, str]:
    """Supprime les colonnes dont le taux de manquants dépasse le seuil (0–1)."""
    if not 0 <= threshold <= 1:
        raise CleaningError("Le seuil doit être compris entre 0 et 1.")
    share = df.isna().mean()
    to_drop = share[share > threshold].index.tolist()
    return df.drop(columns=to_drop), (
        f"Suppression de {len(to_drop)} colonne(s) avec plus de {100 * threshold:.0f} % de manquants"
        + (f" : {', '.join(map(str, to_drop))}." if to_drop else ".")
    )


@operation("drop_columns")
def drop_columns(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, str]:
    """Supprime des colonnes."""
    columns = _check_columns(df, columns)
    if not columns:
        raise CleaningError("Sélectionnez au moins une colonne à supprimer.")
    return df.drop(columns=columns), f"Suppression des colonnes : {', '.join(columns)}."


@operation("impute")
def impute(df: pd.DataFrame, columns: list[str], strategy: str, value: Any = None) -> tuple[pd.DataFrame, str]:
    """Remplace les valeurs manquantes (moyenne, médiane, mode ou valeur personnalisée)."""
    columns = _check_columns(df, columns)
    if not columns:
        raise CleaningError("Sélectionnez au moins une colonne.")
    out = df.copy()
    details = []
    for col in columns:
        s = out[col]
        n_missing = int(s.isna().sum())
        if n_missing == 0:
            continue
        if strategy in ("mean", "median"):
            num = pd.to_numeric(s, errors="coerce")
            if num.notna().sum() == 0:
                raise CleaningError(f"« {col} » n'est pas numérique : moyenne/médiane impossibles.")
            fill = float(num.mean() if strategy == "mean" else num.median())
            if pd.api.types.is_integer_dtype(s) and float(fill).is_integer():
                fill = int(fill)
            out[col] = num.fillna(fill) if not pd.api.types.is_numeric_dtype(s) else s.fillna(fill)
        elif strategy == "mode":
            modes = s.mode(dropna=True)
            if modes.empty:
                raise CleaningError(f"« {col} » ne contient aucune valeur : mode indéfini.")
            fill = modes.iloc[0]
            out[col] = s.fillna(fill)
        elif strategy == "constant":
            if value is None or value == "":
                raise CleaningError("Indiquez une valeur de remplacement.")
            fill = value
            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                try:
                    fill = float(str(value).replace(",", "."))
                except ValueError as exc:
                    raise CleaningError(f"« {value} » n'est pas une valeur numérique valide pour « {col} ».") from exc
            elif pd.api.types.is_datetime64_any_dtype(s):
                fill = pd.to_datetime(value, errors="coerce")
                if pd.isna(fill):
                    raise CleaningError(f"« {value} » n'est pas une date valide.")
            out[col] = s.fillna(fill)
        else:
            raise CleaningError(f"Stratégie d'imputation inconnue : {strategy}.")
        details.append(f"{col} ({n_missing} → {fill!s:.30})")
    labels = {"mean": "moyenne", "median": "médiane", "mode": "mode", "constant": "valeur personnalisée"}
    return out, f"Imputation par {labels.get(strategy, strategy)} : " + (", ".join(details) or "aucune valeur manquante.")


@operation("convert_type")
def convert_type(df: pd.DataFrame, column: str, target: str, date_format: str | None = None) -> tuple[pd.DataFrame, str]:
    """Convertit le type d'une colonne. Les valeurs non convertibles deviennent manquantes (comptées)."""
    _check_columns(df, [column])
    out = df.copy()
    s = out[column]
    before = int(s.notna().sum())
    if target == "numeric":
        new = to_numeric_smart(s)
    elif target == "integer":
        new = to_numeric_smart(s).round().astype("Int64")
    elif target == "datetime":
        new = to_datetime_smart(s, date_format or None)
    elif target == "string":
        new = s.astype("string")
    elif target == "category":
        new = s.astype("category")
    elif target == "boolean":
        new = to_boolean_smart(s)
    else:
        raise CleaningError(f"Type cible inconnu : {target}.")
    lost = before - int(new.notna().sum())
    out[column] = new
    labels = {"numeric": "numérique", "integer": "entier", "datetime": "date", "string": "texte",
              "category": "catégorie", "boolean": "booléen"}
    msg = f"Conversion de « {column} » en {labels[target]}."
    if lost > 0:
        msg += f" {lost} valeur(s) non convertible(s) remplacée(s) par des manquants."
    return out, msg


@operation("rename_columns")
def rename_columns(df: pd.DataFrame, mapping: dict[str, str]) -> tuple[pd.DataFrame, str]:
    """Renomme des colonnes (les nouveaux noms doivent être uniques et non vides)."""
    mapping = {k: str(v).strip() for k, v in mapping.items() if str(v).strip() and k != str(v).strip()}
    _check_columns(df, list(mapping))
    new_cols = [mapping.get(c, c) for c in df.columns]
    if len(set(new_cols)) != len(new_cols):
        raise CleaningError("Le renommage créerait des noms de colonnes en double.")
    if not mapping:
        raise CleaningError("Aucun renommage à appliquer.")
    return df.rename(columns=mapping), "Renommage : " + ", ".join(f"{a} → {b}" for a, b in mapping.items()) + "."


@operation("normalize_text")
def normalize_text(
    df: pd.DataFrame, columns: list[str], strip: bool = True, case: str | None = None,
    remove_accents: bool = False, collapse_spaces: bool = True, empty_as_missing: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Normalise des colonnes texte : espaces, casse, accents, chaînes vides."""
    columns = _check_columns(df, columns)
    out = df.copy()
    changed = 0
    for col in columns:
        s = out[col]
        mask = s.notna()
        txt = s[mask].astype(str)
        new = txt
        if collapse_spaces:
            new = new.str.replace(r"\s+", " ", regex=True)
        if strip:
            new = new.str.strip()
        if case == "lower":
            new = new.str.lower()
        elif case == "upper":
            new = new.str.upper()
        elif case == "title":
            new = new.str.title()
        elif case == "capitalize":
            new = new.str.capitalize()
        if remove_accents:
            new = new.map(strip_accents)
        changed += int((new != txt).sum())
        result = s.astype(object).copy()
        result[mask] = new.values
        if empty_as_missing:
            result = result.where(~result.isin([""]), None)
        out[col] = result
    return out, f"Normalisation du texte de {', '.join(columns)} : {changed} valeur(s) modifiée(s)."


@operation("replace_values")
def replace_values(df: pd.DataFrame, column: str, old: Any, new: Any) -> tuple[pd.DataFrame, str]:
    """Remplace une valeur précise par une autre dans une colonne (`None` = manquant)."""
    _check_columns(df, [column])
    out = df.copy()
    s = out[column]
    mask = s.astype(str) == str(old)
    n = int(mask.sum())
    out[column] = s.astype(object).where(~mask, None if new in (None, "") else new)
    return out, f"Remplacement de « {old} » par « {new if new not in (None, '') else 'manquant'} » dans {column} ({n} valeur(s))."


FILTER_OPERATORS = ["==", "!=", ">", ">=", "<", "<=", "entre", "contient", "ne contient pas",
                    "dans la liste", "est manquant", "n'est pas manquant"]


def filter_mask(s: pd.Series, op: str, value: Any = None, value2: Any = None) -> pd.Series:
    """Masque booléen des lignes à conserver pour un filtre (colonne, opérateur, valeur)."""
    def num(v: Any) -> Any:
        if pd.api.types.is_datetime64_any_dtype(s):
            return pd.to_datetime(v)
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            return float(str(v).replace(",", "."))
        return v

    try:
        if op == "est manquant":
            m = s.isna()
        elif op == "n'est pas manquant":
            m = s.notna()
        elif op in ("contient", "ne contient pas"):
            contains = s.astype("string").str.contains(str(value), case=False, regex=False, na=False)
            m = contains if op == "contient" else ~contains
        elif op == "dans la liste":
            values = [v.strip() for v in str(value).split(";") if v.strip()] if not isinstance(value, list) else value
            m = s.astype(str).isin([str(v) for v in values])
        elif op == "entre":
            m = (s >= num(value)) & (s <= num(value2))
        else:
            ops = {"==": operator.eq, "!=": operator.ne, ">": operator.gt, ">=": operator.ge,
                   "<": operator.lt, "<=": operator.le}
            if op not in ops:
                raise CleaningError(f"Opérateur inconnu : {op}.")
            target = num(value)
            m = ops[op](s.astype(str), str(target)) if s.dtype == object and op in ("==", "!=") else ops[op](s, target)
    except (ValueError, TypeError) as exc:
        raise CleaningError(f"Filtre impossible : valeur « {value} » incompatible avec la colonne.", detail=str(exc)) from exc
    return pd.Series(m, index=s.index).fillna(False).astype(bool)


@operation("filter_rows")
def filter_rows(df: pd.DataFrame, column: str, op: str, value: Any = None, value2: Any = None) -> tuple[pd.DataFrame, str]:
    """Conserve les lignes satisfaisant une condition."""
    _check_columns(df, [column])
    mask = filter_mask(df[column], op, value, value2)
    out = df[mask].reset_index(drop=True)
    cond = f"{column} {op}" + (f" {value}" if value not in (None, "") else "") + (f" et {value2}" if op == "entre" else "")
    return out, f"Filtre « {cond} » : {len(out)} ligne(s) conservée(s), {len(df) - len(out)} retirée(s)."


@operation("treat_outliers")
def treat_outliers(
    df: pd.DataFrame, columns: list[str], method: str = "iqr", action: str = "cap",
    factor: float = 1.5, threshold: float = 3.0,
) -> tuple[pd.DataFrame, str]:
    """Traite les valeurs atypiques : suppression des lignes, écrêtage (winsorisation) ou mise à manquant."""
    columns = _check_columns(df, columns)
    out = df.copy()
    remove_mask = pd.Series(False, index=df.index)
    details = []
    for col in columns:
        x = pd.to_numeric(out[col], errors="coerce")
        if x.notna().sum() == 0:
            raise CleaningError(f"« {col} » n'est pas numérique.")
        if method == "iqr":
            lo, hi = iqr_bounds(x, factor)
        elif method == "zscore":
            mean, std = x.mean(), x.std(ddof=1)
            lo, hi = mean - threshold * std, mean + threshold * std
        else:
            raise CleaningError(f"Méthode inconnue : {method}.")
        mask = ((x < lo) | (x > hi)) if method == "iqr" else zscore_outliers(x, threshold)
        mask = mask.fillna(False)
        n = int(mask.sum())
        details.append(f"{col} : {n}")
        if action == "remove":
            remove_mask |= mask
        elif action == "cap":
            out[col] = x.clip(lower=lo, upper=hi)
        elif action == "nan":
            out[col] = x.mask(mask)
        else:
            raise CleaningError(f"Action inconnue : {action}.")
    if action == "remove":
        out = out[~remove_mask].reset_index(drop=True)
    labels = {"remove": "suppression des lignes", "cap": "écrêtage aux bornes", "nan": "remplacement par manquant"}
    mlabel = f"IQR × {factor:g}" if method == "iqr" else f"|z| > {threshold:g}"
    return out, f"Valeurs atypiques ({mlabel}, {labels[action]}) — " + ", ".join(details) + "."


# ---------------------------------------------------------------------------
# Création de variables
# ---------------------------------------------------------------------------

_BIN_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
            ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS: dict[str, Callable[..., Any]] = {
    "log": np.log, "log10": np.log10, "log1p": np.log1p, "sqrt": np.sqrt, "abs": np.abs, "exp": np.exp,
    "round": lambda x, d=0: np.round(x, int(d)),
}
_COL_REF = re.compile(r"`([^`]+)`")


def safe_eval_expression(df: pd.DataFrame, expression: str) -> pd.Series:
    """Évalue une expression arithmétique sur les colonnes, de façon sûre.

    Syntaxe : colonnes entre accents graves (`prix unitaire`) ou noms simples,
    opérateurs + - * / ** % //, fonctions log, log10, log1p, sqrt, abs, exp, round.
    Exemple : ``\\`prix_unitaire\\` * quantite`` ou ``log(revenu)``.
    """
    if len(expression) > 500:
        raise CleaningError("Expression trop longue.")
    names: dict[str, str] = {}

    def repl(m: re.Match[str]) -> str:
        col = m.group(1)
        if col not in df.columns:
            raise CleaningError(f"Colonne inconnue dans l'expression : « {col} ».")
        key = f"__c{len(names)}"
        names[key] = col
        return key

    prepared = _COL_REF.sub(repl, expression)
    try:
        tree = ast.parse(prepared, mode="eval")
    except SyntaxError as exc:
        raise CleaningError(f"Expression invalide : {exc.msg}.") from exc

    def ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.Name):
            col = names.get(node.id, node.id)
            if col not in df.columns:
                raise CleaningError(f"Colonne inconnue dans l'expression : « {node.id} ».")
            values = pd.to_numeric(df[col], errors="coerce")
            return values.astype("float64")
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise CleaningError(f"Élément non autorisé dans l'expression : {type(node).__name__}.")

    with np.errstate(all="ignore"):
        result = ev(tree)
    if not isinstance(result, pd.Series):
        result = pd.Series(result, index=df.index, dtype="float64")
    return result.replace([np.inf, -np.inf], np.nan)


@operation("create_variable")
def create_variable(
    df: pd.DataFrame, name: str, kind: str, expression: str | None = None, source: str | None = None,
    bins: int | list[float] | None = None, labels: list[str] | None = None, part: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Crée une nouvelle variable : expression, discrétisation (classes) ou extraction de date."""
    name = (name or "").strip()
    if not name:
        raise CleaningError("Indiquez le nom de la nouvelle variable.")
    if name in df.columns:
        raise CleaningError(f"La colonne « {name} » existe déjà.")
    out = df.copy()
    if kind == "expression":
        if not expression:
            raise CleaningError("Indiquez une expression.")
        out[name] = safe_eval_expression(df, expression)
        desc = f"Création de « {name} » = {expression}."
    elif kind == "bin":
        _check_columns(df, [source])
        x = pd.to_numeric(df[source], errors="coerce")
        if isinstance(bins, int) and bins < 2:
            raise CleaningError("Au moins 2 classes sont nécessaires.")
        try:
            out[name] = pd.cut(x, bins=bins or 4, labels=labels, include_lowest=True).astype("string")
        except ValueError as exc:
            raise CleaningError("Discrétisation impossible (bornes ou étiquettes invalides).", detail=str(exc)) from exc
        desc = f"Création de « {name} » : discrétisation de « {source} » en classes."
    elif kind == "quantile_bin":
        _check_columns(df, [source])
        x = pd.to_numeric(df[source], errors="coerce")
        q = int(bins) if isinstance(bins, int) else 4
        out[name] = pd.qcut(x, q=q, duplicates="drop").astype("string")
        desc = f"Création de « {name} » : {q} classes d'effectifs égaux (quantiles) de « {source} »."
    elif kind == "date_part":
        _check_columns(df, [source])
        d = to_datetime_smart(df[source])
        parts = {"year": d.dt.year, "month": d.dt.month, "day": d.dt.day, "weekday": d.dt.day_of_week + 1,
                 "quarter": d.dt.quarter, "week": d.dt.isocalendar().week.astype("Int64") if d.notna().any() else d.dt.month,
                 "hour": d.dt.hour}
        if part not in parts:
            raise CleaningError("Partie de date inconnue.")
        out[name] = parts[part].astype("Int64")
        desc = f"Création de « {name} » : extraction ({part}) de « {source} »."
    else:
        raise CleaningError(f"Type de création inconnu : {kind}.")
    return out, desc


# ---------------------------------------------------------------------------
# Session de nettoyage (historique, annulation)
# ---------------------------------------------------------------------------

def apply_operation(df: pd.DataFrame, op_name: str, /, **params: Any) -> tuple[pd.DataFrame, Operation]:
    """Applique une opération enregistrée et retourne (nouveau DataFrame, trace)."""
    fn = OPERATIONS.get(op_name)
    if fn is None:
        raise CleaningError(f"Opération inconnue : {op_name}.")
    new_df, description = fn(df, **params)
    op = Operation(op_name, params, description, len(df), df.shape[1], len(new_df), new_df.shape[1])
    return new_df, op


class CleaningSession:
    """Copie de travail + historique des opérations, avec annulation et réinitialisation."""

    def __init__(self, original: pd.DataFrame) -> None:
        self._original = original
        self.current = original.copy()
        self.history: list[Operation] = []

    @property
    def original(self) -> pd.DataFrame:
        return self._original

    def apply(self, op_name: str, /, **params: Any) -> Operation:
        new_df, op = apply_operation(self.current, op_name, **params)
        self.current = new_df
        self.history.append(op)
        return op

    def undo(self) -> Operation | None:
        """Annule la dernière opération en rejouant l'historique depuis l'original."""
        if not self.history:
            return None
        last = self.history.pop()
        df = self._original.copy()
        for op in self.history:
            df, _ = apply_operation(df, op.name, **op.params)
        self.current = df
        return last

    def reset(self) -> None:
        self.current = self._original.copy()
        self.history.clear()

    def history_table(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "#": i + 1, "Horodatage": op.timestamp, "Opération": op.description,
            "Lignes": f"{op.rows_before} → {op.rows_after}", "Colonnes": f"{op.cols_before} → {op.cols_after}",
        } for i, op in enumerate(self.history)])

    def history_dicts(self) -> list[dict[str, Any]]:
        return [op.to_dict() for op in self.history]
