"""Fabriques de graphiques adaptés au type des variables.

* Plotly (interactif) pour l'interface et le rapport HTML ;
* matplotlib (statique, sans dépendance à un navigateur) pour le rapport PDF.

Les graphiques coûteux (nuages de points, pair plots) sont calculés sur un
échantillon aléatoire reproductible dont la taille est indiquée dans le titre.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats as sps

from config.settings import SETTINGS
from core.schema_detector import ColumnSchema, Family, StatType

PALETTE = list(SETTINGS.viz.color_sequence)
WEEKDAYS = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
MONTHS = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]


def sample_df(df: pd.DataFrame, max_rows: int) -> tuple[pd.DataFrame, bool]:
    """Échantillon aléatoire reproductible si le DataFrame dépasse `max_rows`."""
    if len(df) <= max_rows:
        return df, False
    return df.sample(max_rows, random_state=SETTINGS.random_seed), True


def _layout(fig: go.Figure, title: str, x: str | None = None, y: str | None = None, legend: bool = True) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0.0, "xanchor": "left", "font": {"size": 16}},
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
        colorway=PALETTE,
        showlegend=legend,
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.3},
        hoverlabel={"namelength": -1},
    )
    # Titres sur les axes principaux uniquement (pas sur les graphiques marginaux)
    if x is not None:
        fig.update_layout(xaxis_title_text=x)
    if y is not None:
        fig.update_layout(yaxis_title_text=y)
    return fig


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _cat(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.map({True: "Vrai", False: "Faux"})
    return s.astype("string")


# ---------------------------------------------------------------------------
# Variables numériques
# ---------------------------------------------------------------------------

def histogram(df: pd.DataFrame, col: str, bins: int = 30, color: str | None = None,
              log_y: bool = False, marginal: str | None = "box") -> go.Figure:
    data = pd.DataFrame({col: _num(df, col)})
    if color:
        data[color] = _cat(df[color])
    fig = px.histogram(data.dropna(subset=[col]), x=col, color=color, nbins=bins, log_y=log_y,
                       marginal=marginal, opacity=0.85, barmode="overlay" if color else "relative")
    return _layout(fig, f"Distribution de {col}", col, "Effectif", legend=bool(color))


def density(df: pd.DataFrame, col: str, group: str | None = None, points: int = 200) -> go.Figure:
    fig = go.Figure()
    groups = [(col, _num(df, col))] if not group else [
        (str(g), _num(sub, col)) for g, sub in df.groupby(_cat(df[group]), observed=True)
    ][:10]
    for label, x in groups:
        x = x.dropna().to_numpy()
        if x.size < 3 or np.ptp(x) == 0:
            continue
        kde = sps.gaussian_kde(x)
        grid = np.linspace(x.min(), x.max(), points)
        fig.add_trace(go.Scatter(x=grid, y=kde(grid), mode="lines", name=label, fill="tozeroy",
                                 hovertemplate=f"{col}=%{{x:.4g}}<br>densité=%{{y:.4g}}<extra>{label}</extra>"))
    return _layout(fig, f"Courbe de densité de {col}" + (f" par {group}" if group else ""), col, "Densité",
                   legend=bool(group))


def boxplot(df: pd.DataFrame, col: str, group: str | None = None, points: str | bool = "outliers") -> go.Figure:
    data = pd.DataFrame({col: _num(df, col)})
    if group:
        data[group] = _cat(df[group])
    fig = px.box(data.dropna(subset=[col]), y=col, x=group, color=group, points=points)
    return _layout(fig, f"Boîte à moustaches de {col}" + (f" par {group}" if group else ""), group, col, legend=False)


def violin(df: pd.DataFrame, col: str, group: str | None = None) -> go.Figure:
    data = pd.DataFrame({col: _num(df, col)})
    if group:
        data[group] = _cat(df[group])
    fig = px.violin(data.dropna(subset=[col]), y=col, x=group, color=group, box=True, points=False)
    return _layout(fig, f"Diagramme en violon de {col}" + (f" par {group}" if group else ""), group, col, legend=False)


def ecdf(df: pd.DataFrame, col: str, group: str | None = None) -> go.Figure:
    data = pd.DataFrame({col: _num(df, col)})
    if group:
        data[group] = _cat(df[group])
    fig = px.ecdf(data.dropna(subset=[col]), x=col, color=group)
    return _layout(fig, f"Distribution cumulée (ECDF) de {col}", col, "Proportion cumulée", legend=bool(group))


def scatter(df: pd.DataFrame, x: str, y: str, color: str | None = None, trendline: bool = True,
            max_points: int | None = None) -> go.Figure:
    max_points = max_points or SETTINGS.viz.max_points_scatter
    data = pd.DataFrame({x: _num(df, x), y: _num(df, y)})
    if color:
        data[color] = _cat(df[color]) if color not in (x, y) else data[color]
    data = data.dropna(subset=[x, y])
    data, sampled = sample_df(data, max_points)
    fig = px.scatter(data, x=x, y=y, color=color, opacity=0.7)
    title = f"{y} en fonction de {x}"
    if trendline and len(data) >= 3 and data[x].nunique() > 1:
        slope, intercept, r, p, _ = sps.linregress(data[x], data[y])
        grid = np.linspace(data[x].min(), data[x].max(), 50)
        fig.add_trace(go.Scatter(x=grid, y=intercept + slope * grid, mode="lines", name=f"Droite MCO (r = {r:.3f})",
                                 line={"color": "#EF4444", "width": 2, "dash": "dash"}))
        title += f" — r de Pearson = {r:.3f}"
    if sampled:
        title += f" (échantillon de {len(data):,} points)".replace(",", " ")
    return _layout(fig, title, x, y)


# ---------------------------------------------------------------------------
# Variables catégorielles
# ---------------------------------------------------------------------------

def _counts(df: pd.DataFrame, col: str, top_n: int) -> pd.DataFrame:
    s = _cat(df[col])
    vc = s.value_counts(dropna=True)
    if len(vc) > top_n:
        other = int(vc.iloc[top_n:].sum())
        vc = pd.concat([vc.iloc[:top_n], pd.Series({"(autres)": other})])
    out = vc.rename_axis(col).reset_index(name="Effectif")
    out["%"] = (100 * out["Effectif"] / out["Effectif"].sum()).round(2)
    return out


def bar_chart(df: pd.DataFrame, col: str, top_n: int | None = None, horizontal: bool = False,
              percent: bool = False, order: list[str] | None = None) -> go.Figure:
    top_n = top_n or SETTINGS.viz.max_bar_categories
    data = _counts(df, col, top_n)
    if order:
        rank = {v: i for i, v in enumerate(map(str, order))}
        data = data.sort_values(col, key=lambda s: s.map(lambda v: rank.get(str(v), len(rank))))
    value = "%" if percent else "Effectif"
    if horizontal:
        fig = px.bar(data, y=col, x=value, orientation="h", text=value, hover_data=["Effectif", "%"])
        fig.update_yaxes(categoryorder="array", categoryarray=list(data[col])[::-1])
    else:
        fig = px.bar(data, x=col, y=value, text=value, hover_data=["Effectif", "%"])
    fig.update_traces(marker_color=PALETTE[0], texttemplate="%{text:.3s}" if not percent else "%{text:.1f} %")
    return _layout(fig, f"Répartition de {col}", value if horizontal else col, col if horizontal else value, legend=False)


def pie_chart(df: pd.DataFrame, col: str, max_categories: int | None = None) -> go.Figure:
    max_categories = max_categories or SETTINGS.viz.max_pie_categories
    data = _counts(df, col, max_categories)
    fig = px.pie(data, names=col, values="Effectif", hole=0.4)
    fig.update_traces(textinfo="percent+label")
    return _layout(fig, f"Part de chaque modalité de {col}")


def pareto(df: pd.DataFrame, col: str, top_n: int = 30) -> go.Figure:
    data = _cat(df[col]).value_counts().head(top_n).rename_axis(col).reset_index(name="Effectif")
    total = int(_cat(df[col]).notna().sum()) or 1
    data["% cumulé"] = 100 * data["Effectif"].cumsum() / total
    fig = go.Figure()
    fig.add_bar(x=data[col], y=data["Effectif"], name="Effectif", marker_color=PALETTE[0])
    fig.add_scatter(x=data[col], y=data["% cumulé"], name="% cumulé", yaxis="y2", mode="lines+markers",
                    line={"color": PALETTE[4]})
    fig.add_hline(y=80, line_dash="dot", line_color="#94A3B8", yref="y2")
    fig.update_layout(yaxis2={"overlaying": "y", "side": "right", "range": [0, 105], "title": "% cumulé"})
    return _layout(fig, f"Diagramme de Pareto de {col}", col, "Effectif")


# ---------------------------------------------------------------------------
# Variables temporelles
# ---------------------------------------------------------------------------

FREQ_LABELS = {"D": "jour", "W": "semaine", "MS": "mois", "QS": "trimestre", "YS": "année"}


def time_series(df: pd.DataFrame, date_col: str, value_col: str | None = None, agg: str = "count",
                freq: str = "MS", rolling: int | None = None) -> go.Figure:
    d = pd.to_datetime(df[date_col], errors="coerce")
    if value_col:
        data = pd.DataFrame({"date": d, "v": _num(df, value_col)}).dropna(subset=["date"])
        series = data.set_index("date")["v"].resample(freq).agg(agg)
        agg_label = {"sum": "Somme", "mean": "Moyenne", "median": "Médiane", "count": "Nombre"}[agg]
        ylabel = f"{agg_label} de {value_col}"
    else:
        series = pd.Series(1, index=d.dropna()).resample(freq).sum()
        ylabel = "Nombre d'observations"
    out = series.rename("valeur").reset_index()
    fig = px.line(out, x=out.columns[0], y="valeur", markers=len(out) < 60)
    if rolling and rolling > 1 and len(out) > rolling:
        fig.add_scatter(x=out[out.columns[0]], y=out["valeur"].rolling(rolling, center=True).mean(),
                        name=f"Moyenne mobile ({rolling})", line={"dash": "dash", "color": PALETTE[4]})
    fig.update_traces(selector={"type": "scatter", "mode": "lines+markers"}, name=ylabel)
    return _layout(fig, f"Évolution par {FREQ_LABELS.get(freq, freq)} — {ylabel}", date_col, ylabel,
                   legend=bool(rolling))


def temporal_distribution(df: pd.DataFrame, date_col: str, by: str = "month") -> go.Figure:
    d = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if by == "weekday":
        counts = d.dt.day_of_week.value_counts().reindex(range(7), fill_value=0)
        labels, xl = WEEKDAYS, "Jour de la semaine"
    elif by == "month":
        counts = d.dt.month.value_counts().reindex(range(1, 13), fill_value=0)
        labels, xl = MONTHS, "Mois"
    elif by == "hour":
        counts = d.dt.hour.value_counts().reindex(range(24), fill_value=0)
        labels, xl = [f"{h}h" for h in range(24)], "Heure"
    else:
        counts = d.dt.year.value_counts().sort_index()
        labels, xl = [str(y) for y in counts.index], "Année"
    fig = px.bar(x=labels, y=counts.values, labels={"x": xl, "y": "Effectif"})
    fig.update_traces(marker_color=PALETTE[1])
    return _layout(fig, f"Répartition de {date_col} par {xl.lower()}", xl, "Effectif", legend=False)


def calendar_heatmap(df: pd.DataFrame, date_col: str, value_col: str | None = None, year: int | None = None) -> go.Figure:
    d = pd.to_datetime(df[date_col], errors="coerce")
    data = pd.DataFrame({"date": d.dt.normalize(), "v": _num(df, value_col) if value_col else 1}).dropna(subset=["date"])
    if data.empty:
        return _layout(go.Figure(), "Aucune date exploitable")
    year = year or int(data["date"].dt.year.mode().iloc[0])
    data = data[data["date"].dt.year == year]
    daily = data.groupby("date")["v"].sum()
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    daily = daily.reindex(days)
    week = days.isocalendar().week.to_numpy().astype(int)
    week = np.where((days.month == 1) & (week > 50), 0, week)
    week = np.where((days.month == 12) & (week == 1), 53, week)
    grid = pd.DataFrame({"week": week, "dow": days.day_of_week, "v": daily.values, "date": days.strftime("%d/%m/%Y")})
    z = grid.pivot(index="dow", columns="week", values="v")
    txt = grid.pivot(index="dow", columns="week", values="date")
    fig = go.Figure(go.Heatmap(z=z.values, x=z.columns, y=WEEKDAYS, text=txt.values, colorscale="Blues",
                               hovertemplate="%{text}<br>valeur=%{z}<extra></extra>", xgap=2, ygap=2))
    fig.update_yaxes(autorange="reversed")
    label = f"somme de {value_col}" if value_col else "nombre d'observations"
    return _layout(fig, f"Calendrier {year} — {label}", "Semaine", "Jour", legend=False)


# ---------------------------------------------------------------------------
# Multivarié
# ---------------------------------------------------------------------------

def correlation_heatmap(matrix: pd.DataFrame, title: str = "Matrice de corrélation", annotate: bool = True) -> go.Figure:
    z = matrix.to_numpy(dtype=float)
    fig = go.Figure(go.Heatmap(
        z=z, x=list(matrix.columns), y=list(matrix.index), zmin=-1, zmax=1, colorscale="RdBu", reversescale=False,
        text=np.vectorize(lambda v: "" if np.isnan(v) else f"{v + 0.0:.2f}".replace("-0.00", "0.00"))(z)
        if annotate and len(matrix) <= 20 else None,
        texttemplate="%{text}" if annotate and len(matrix) <= 20 else None,
        hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>", colorbar={"title": "r"},
    ))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, title, legend=False)


def generic_heatmap(matrix: pd.DataFrame, title: str, colorscale: str = "Blues", zmin: float | None = None,
                    zmax: float | None = None, value_label: str = "valeur") -> go.Figure:
    fig = go.Figure(go.Heatmap(z=matrix.to_numpy(dtype=float), x=[str(c) for c in matrix.columns],
                               y=[str(i) for i in matrix.index], colorscale=colorscale, zmin=zmin, zmax=zmax,
                               texttemplate="%{z:.3g}" if matrix.size <= 400 else None,
                               hovertemplate=f"%{{y}} × %{{x}}<br>{value_label} = %{{z:.4g}}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, title, legend=False)


def pair_plot(df: pd.DataFrame, columns: list[str], color: str | None = None) -> go.Figure:
    cols = columns[: SETTINGS.viz.max_vars_pairplot]
    data = df[cols].apply(pd.to_numeric, errors="coerce")
    if color:
        data[color] = _cat(df[color])
    data, sampled = sample_df(data.dropna(subset=cols), SETTINGS.viz.max_rows_pairplot)
    fig = px.scatter_matrix(data, dimensions=cols, color=color, opacity=0.6)
    fig.update_traces(diagonal_visible=False, marker={"size": 4})
    title = "Matrice de nuages de points" + (f" (échantillon de {len(data)} lignes)" if sampled else "")
    fig = _layout(fig, title)
    fig.update_layout(height=max(500, 160 * len(cols)))
    return fig


def group_means_chart(df: pd.DataFrame, numeric: str, group: str, stat: str = "mean", top_n: int = 30) -> go.Figure:
    data = pd.DataFrame({"y": _num(df, numeric), "g": _cat(df[group])}).dropna()
    agg = data.groupby("g", observed=True)["y"].agg([stat, "count", "std"]).sort_values(stat, ascending=False).head(top_n)
    err = (agg["std"] / np.sqrt(agg["count"])) * 1.96 if stat == "mean" else None
    fig = go.Figure(go.Bar(x=agg.index.astype(str), y=agg[stat], marker_color=PALETTE[0],
                           error_y={"type": "data", "array": err, "visible": err is not None} if err is not None else None,
                           customdata=agg["count"], hovertemplate="%{x}<br>%{y:.4g}<br>n = %{customdata}<extra></extra>"))
    lbl = {"mean": "Moyenne", "median": "Médiane", "sum": "Somme"}[stat]
    return _layout(fig, f"{lbl} de {numeric} par {group}" + (" (IC 95 %)" if stat == "mean" else ""), group,
                   f"{lbl} de {numeric}", legend=False)


def stacked_bar(table: pd.DataFrame, row: str, col: str, percent: bool = True) -> go.Figure:
    data = (table.div(table.sum(axis=1), axis=0) * 100) if percent else table
    long = data.reset_index().melt(id_vars=data.index.name or "index", var_name=col, value_name="valeur")
    x = data.index.name or "index"
    fig = px.bar(long, x=x, y="valeur", color=col, barmode="stack")
    return _layout(fig, f"{col} selon {row}" + (" (% en ligne)" if percent else ""), row, "%" if percent else "Effectif")


def missing_bar(missing_table: pd.DataFrame) -> go.Figure:
    data = missing_table[missing_table["Valeurs manquantes"] > 0]
    fig = px.bar(data, x="Variable", y="% manquants", text="% manquants", hover_data=["Valeurs manquantes"])
    fig.update_traces(marker_color=PALETTE[4], texttemplate="%{text:.1f} %")
    fig.update_yaxes(range=[0, 105])
    return _layout(fig, "Taux de valeurs manquantes par variable", "Variable", "% manquants", legend=False)


def missing_matrix(df: pd.DataFrame, max_rows: int = 500) -> go.Figure:
    data, sampled = sample_df(df, max_rows)
    data = data.sort_index()
    z = data.isna().astype(int).to_numpy()
    fig = go.Figure(go.Heatmap(z=z, x=[str(c) for c in data.columns], colorscale=[[0, "#E2E8F0"], [1, "#EF4444"]],
                               showscale=False, hovertemplate="ligne %{y}<br>%{x}<br>manquant = %{z}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title_text="Ligne" + (" (échantillon)" if sampled else ""))
    return _layout(fig, "Carte des valeurs manquantes (rouge = manquant)", legend=False)


def quality_gauge(score: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number={"suffix": " /100"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": PALETTE[0]},
               "steps": [{"range": [0, 60], "color": "#FECACA"}, {"range": [60, 75], "color": "#FDE68A"},
                         {"range": [75, 90], "color": "#BBF7D0"}, {"range": [90, 100], "color": "#86EFAC"}]},
    ))
    fig.update_layout(height=260, margin={"l": 30, "r": 40, "t": 40, "b": 10}, title="Score global de qualité")
    return fig


# ---------------------------------------------------------------------------
# Suggestions automatiques
# ---------------------------------------------------------------------------

@dataclass
class ChartSuggestion:
    title: str
    kind: str
    params: dict[str, Any]


def suggest_charts(schemas: list[ColumnSchema], max_charts: int = 12) -> list[ChartSuggestion]:
    """Propose des graphiques pertinents selon le type des variables."""
    out: list[ChartSuggestion] = []
    nums = [s for s in schemas if s.stat_type in (StatType.CONTINUOUS, StatType.DISCRETE)]
    continuous = [s for s in nums if s.stat_type == StatType.CONTINUOUS] or nums
    cats = [s for s in schemas if s.family == Family.CATEGORICAL and 2 <= s.n_unique <= 30]
    dates = [s for s in schemas if s.stat_type in (StatType.DATE, StatType.DATETIME)]
    for s in nums[:4]:
        out.append(ChartSuggestion(f"Histogramme — {s.name}", "histogram", {"col": s.name}))
    for s in cats[:3]:
        kind = "pie" if s.n_unique <= 5 else "bar"
        out.append(ChartSuggestion(f"{'Camembert' if kind == 'pie' else 'Barres'} — {s.name}", kind, {"col": s.name}))
    for d in dates[:2]:
        out.append(ChartSuggestion(f"Série temporelle — {d.name}", "time_series", {"date_col": d.name}))
    if len(continuous) >= 2:
        out.append(ChartSuggestion(f"Nuage — {continuous[1].name} vs {continuous[0].name}", "scatter",
                                   {"x": continuous[0].name, "y": continuous[1].name}))
    if continuous and cats:
        out.append(ChartSuggestion(f"Boîtes — {continuous[0].name} par {cats[0].name}", "box",
                                   {"col": continuous[0].name, "group": cats[0].name}))
    return out[:max_charts]


def render_suggestion(df: pd.DataFrame, sug: ChartSuggestion) -> go.Figure:
    """Construit la figure correspondant à une suggestion."""
    p = sug.params
    if sug.kind == "histogram":
        return histogram(df, p["col"])
    if sug.kind == "pie":
        return pie_chart(df, p["col"])
    if sug.kind == "bar":
        return bar_chart(df, p["col"])
    if sug.kind == "time_series":
        return time_series(df, p["date_col"])
    if sug.kind == "scatter":
        return scatter(df, p["x"], p["y"])
    if sug.kind == "box":
        return boxplot(df, p["col"], p.get("group"))
    raise ValueError(sug.kind)


def figure_to_html(fig: go.Figure, full: bool = True) -> str:
    """HTML autonome d'une figure (Plotly embarqué, utilisable hors ligne si full=True)."""
    return fig.to_html(include_plotlyjs=True if full else "cdn", full_html=full)


# ---------------------------------------------------------------------------
# Graphiques statiques (matplotlib) pour le PDF
# ---------------------------------------------------------------------------

def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 10, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _png(fig) -> bytes:
    plt = _mpl()
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()


def static_histogram(df: pd.DataFrame, col: str, bins: int = 30) -> bytes:
    plt = _mpl()
    x = _num(df, col).dropna()
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.hist(x, bins=bins, color=PALETTE[0], alpha=0.85, edgecolor="white")
    ax.axvline(x.mean(), color=PALETTE[4], linestyle="--", linewidth=1, label=f"moyenne {x.mean():.4g}")
    ax.axvline(x.median(), color=PALETTE[2], linestyle=":", linewidth=1.2, label=f"médiane {x.median():.4g}")
    ax.set_title(f"Distribution de {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Effectif")
    ax.legend(frameon=False)
    return _png(fig)


def static_bar(df: pd.DataFrame, col: str, top_n: int = 15) -> bytes:
    plt = _mpl()
    data = _counts(df, col, top_n)
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.barh(data[col].astype(str)[::-1], data["Effectif"][::-1], color=PALETTE[1])
    ax.set_title(f"Répartition de {col}")
    ax.set_xlabel("Effectif")
    return _png(fig)


def static_correlation(matrix: pd.DataFrame, title: str = "Matrice de corrélation") -> bytes:
    plt = _mpl()
    k = len(matrix)
    size = min(6.5, 1.6 + 0.45 * k)
    fig, ax = plt.subplots(figsize=(size + 0.8, size))
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap="RdBu", vmin=-1, vmax=1)
    ax.set_xticks(range(k), [str(c)[:18] for c in matrix.columns], rotation=45, ha="right")
    ax.set_yticks(range(k), [str(c)[:18] for c in matrix.index])
    if k <= 12:
        for i in range(k):
            for j in range(k):
                v = matrix.iat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)
    return _png(fig)


def static_missing(missing_table: pd.DataFrame) -> bytes:
    plt = _mpl()
    data = missing_table[missing_table["Valeurs manquantes"] > 0].head(25)
    fig, ax = plt.subplots(figsize=(5.2, max(1.8, 0.25 * len(data) + 0.8)))
    ax.barh(data["Variable"].astype(str)[::-1], data["% manquants"][::-1], color=PALETTE[4])
    ax.set_xlim(0, 100)
    ax.set_xlabel("% de valeurs manquantes")
    ax.set_title("Valeurs manquantes par variable")
    return _png(fig)


def static_time_series(df: pd.DataFrame, date_col: str) -> bytes:
    plt = _mpl()
    d = pd.to_datetime(df[date_col], errors="coerce").dropna()
    span = (d.max() - d.min()).days if len(d) else 0
    freq = "D" if span <= 120 else "W" if span <= 900 else "MS"
    series = pd.Series(1, index=d).resample(freq).sum()
    fig, ax = plt.subplots(figsize=(5.2, 2.4))
    ax.plot(series.index, series.values, color=PALETTE[0], linewidth=1.2)
    ax.set_title(f"Nombre d'observations par {FREQ_LABELS[freq]} — {date_col}")
    ax.set_ylabel("Observations")
    fig.autofmt_xdate()
    return _png(fig)


def static_boxplots(df: pd.DataFrame, columns: list[str]) -> bytes:
    plt = _mpl()
    cols = columns[:8]
    fig, axes = plt.subplots(1, len(cols), figsize=(min(7, 1.3 * len(cols) + 1), 2.6), squeeze=False)
    for ax, col in zip(axes[0], cols):
        ax.boxplot(_num(df, col).dropna(), widths=0.6, patch_artist=True,
                   boxprops={"facecolor": PALETTE[1], "alpha": 0.6}, flierprops={"markersize": 2})
        ax.set_title(str(col)[:16], fontsize=7)
        ax.set_xticks([])
    fig.suptitle("Boîtes à moustaches (détection graphique des valeurs atypiques)", fontsize=9)
    return _png(fig)
