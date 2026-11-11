"""Génération des rapports d'analyse : PDF, HTML, Excel, CSV (ZIP) et JSON.

Les sections sont sélectionnables (clés de :data:`REPORT_SECTIONS`). Le contenu
provient exclusivement d'un :class:`~core.pipeline.AnalysisBundle` : aucun
résultat n'est inventé. Le HTML est autonome (Plotly embarqué, consultable hors
ligne) et tout texte issu des données est échappé (Jinja2 autoescape / XML).
"""
from __future__ import annotations

import io
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import pandas as pd

from config.settings import APP_NAME, APP_SUBTITLE, APP_VERSION
from core import visualization_engine as viz
from core.exceptions import ReportError
from core.file_detector import SEPARATOR_LABELS, human_size
from core.interpretation_engine import Category
from core.multivariate_engine import METHOD_INFO
from core.pipeline import AnalysisBundle
from core.quality_engine import SUBSCORE_LABELS
from core.schema_detector import schema_table
from utils.exporters import to_excel_bytes, to_json_bytes, zip_csvs
from utils.helpers import fmt_int, fmt_num, fmt_pct

REPORT_SECTIONS: dict[str, str] = {
    "presentation": "Présentation du dataset",
    "general": "Caractéristiques générales",
    "variables": "Description des variables",
    "statistics": "Statistiques descriptives",
    "missing": "Analyse des valeurs manquantes",
    "quality": "Audit de qualité",
    "correlations": "Analyse des corrélations",
    "visualizations": "Visualisations principales",
    "interpretations": "Interprétations statistiques",
    "recommendations": "Recommandations",
}


def _sections(selected: list[str] | None) -> list[str]:
    if not selected:
        return list(REPORT_SECTIONS)
    unknown = [s for s in selected if s not in REPORT_SECTIONS]
    if unknown:
        raise ReportError(f"Section(s) inconnue(s) : {', '.join(unknown)}.")
    return [s for s in REPORT_SECTIONS if s in selected]


# ---------------------------------------------------------------------------
# Contenu commun
# ---------------------------------------------------------------------------

def executive_summary(b: AnalysisBundle) -> list[str]:
    """Synthèse exécutive (phrases chiffrées issues des résultats)."""
    lines = list(b.insights.summary)
    top_issues = [i for i in b.quality.issues if i.severity in ("critique", "majeur")][:3]
    if top_issues:
        lines.append("Points d'attention prioritaires : " + " ; ".join(
            f"{i.check.lower()} ({i.column})" for i in top_issues) + ".")
    return lines


def general_table(b: AnalysisBundle) -> pd.DataFrame:
    o, m = b.overview, b.meta
    rows = [
        ("Fichier", m.filename), ("Format détecté", m.format_label), ("Taille du fichier", human_size(m.size_bytes)),
        ("Encodage", m.encoding or "—"), ("Séparateur", SEPARATOR_LABELS.get(m.separator or "", m.separator or "—")),
        ("Feuille / table", m.sheet or m.table or "—"),
        ("Lignes", fmt_int(o.n_rows)), ("Colonnes", fmt_int(o.n_cols)),
        ("Cellules", fmt_int(o.n_cells)), ("Valeurs manquantes", f"{fmt_int(o.n_missing)} ({fmt_pct(o.missing_ratio)})"),
        ("Lignes dupliquées", f"{fmt_int(o.n_duplicates)} ({fmt_pct(o.duplicate_ratio)})"),
        ("Lignes complètes", fmt_int(o.n_complete_rows)), ("Mémoire", f"{fmt_num(o.memory_mb)} Mo"),
        ("Variables quantitatives", str(o.n_numeric)), ("Variables qualitatives", str(o.n_categorical)),
        ("Variables temporelles", str(o.n_temporal)), ("Autres variables", str(o.n_other)),
        ("Lecture partielle", "oui" if m.truncated else "non"),
    ]
    return pd.DataFrame(rows, columns=["Caractéristique", "Valeur"])


def recommendations(b: AnalysisBundle) -> list[str]:
    recos = [i.text for i in b.insights.by_category(Category.RECOMMENDATION)]
    if not recos:
        recos = ["Aucune action corrective prioritaire n'a été identifiée par les règles automatiques."]
    return recos


def _bundle_tables(b: AnalysisBundle, sections: list[str]) -> dict[str, pd.DataFrame]:
    """Tableaux de résultats par section (utilisés par Excel/CSV)."""
    t: dict[str, pd.DataFrame] = {}
    if "presentation" in sections or "general" in sections:
        t["Caractéristiques"] = general_table(b)
    if "variables" in sections:
        t["Variables"] = schema_table(b.schemas)
    if "statistics" in sections:
        t["Stats quantitatives"] = b.univariate.numeric_table()
        t["Stats qualitatives"] = b.univariate.categorical_table()
        t["Stats temporelles"] = b.univariate.temporal_table()
        for name, st in list(b.univariate.categorical.items())[:20]:
            t[f"Fréq {name}"] = st.frequencies
    if "missing" in sections:
        t["Valeurs manquantes"] = b.missing
    if "quality" in sections:
        t["Audit qualité"] = b.quality.issues_table()
        t["Qualité par colonne"] = b.quality.column_scores
        t["Valeurs atypiques"] = b.quality.outliers_table()
        t["Score qualité"] = pd.DataFrame([
            {"Composante": SUBSCORE_LABELS[k], "Score (/100)": v, "Pondération": round(b.quality.weights[k], 3)}
            for k, v in b.quality.sub_scores.items()
        ] + [{"Composante": "Score global", "Score (/100)": b.quality.global_score, "Pondération": 1.0}])
    if "correlations" in sections and b.correlation is not None:
        t[f"Corrélation {b.correlation.method}"] = b.correlation.matrix.round(4).reset_index(names="Variable")
        t["Paires corrélées"] = b.correlation.pairs()
        if b.covariance is not None:
            t["Covariance"] = b.covariance.round(6).reset_index(names="Variable")
        if not b.associations.empty:
            t["Fortes associations"] = b.associations
    if "interpretations" in sections or "recommendations" in sections:
        t["Interprétations"] = b.insights.table()
    return t


def report_dict(b: AnalysisBundle, sections: list[str] | None = None) -> dict[str, Any]:
    """Représentation structurée complète du rapport (base des exports JSON)."""
    sections = _sections(sections)
    d: dict[str, Any] = {
        "application": f"{APP_NAME} – {APP_SUBTITLE} v{APP_VERSION}",
        "date_generation": datetime.now().isoformat(timespec="seconds"),
        "sections": sections,
        "synthese_executive": executive_summary(b),
    }
    if "presentation" in sections or "general" in sections:
        d["fichier"] = b.meta.to_dict()
        d["caracteristiques"] = b.overview.to_dict()
    if "variables" in sections:
        d["variables"] = [s.to_dict() for s in b.schemas]
    if "statistics" in sections:
        d["statistiques_quantitatives"] = b.univariate.numeric_table()
        d["statistiques_qualitatives"] = {
            k: {"resume": v.to_row(), "frequences": v.frequencies} for k, v in b.univariate.categorical.items()
        }
        d["statistiques_temporelles"] = {
            k: {"resume": v.to_row(), "irregularites": v.irregularities, "par_annee": v.by_year,
                "par_mois": v.by_month, "par_jour": v.by_weekday} for k, v in b.univariate.temporal.items()
        }
    if "missing" in sections:
        d["valeurs_manquantes"] = b.missing
    if "quality" in sections:
        d["qualite"] = {
            "score_global": b.quality.global_score, "appreciation": b.quality.grade,
            "sous_scores": b.quality.sub_scores, "ponderations": b.quality.weights,
            "indicateurs": b.quality.indicators, "problemes": b.quality.issues_table(),
            "scores_colonnes": b.quality.column_scores, "valeurs_atypiques": b.quality.outliers_table(),
            "avertissement": "Score heuristique : ni universel, ni substitut à une validation métier.",
        }
    if "correlations" in sections and b.correlation is not None:
        d["correlations"] = {
            "methode": b.correlation.method, "justification": b.correlation_method_reason,
            "matrice": b.correlation.matrix.round(6).reset_index(names="variable"),
            "paires": b.correlation.pairs(), "associations_fortes": b.associations,
        }
    if "interpretations" in sections:
        d["interpretations"] = {c.value: [i.text for i in b.insights.by_category(c)] for c in Category}
    if "recommendations" in sections:
        d["recommandations"] = recommendations(b)
    return d


# ---------------------------------------------------------------------------
# Exports tabulaires
# ---------------------------------------------------------------------------

def build_report_json(b: AnalysisBundle, sections: list[str] | None = None) -> bytes:
    return to_json_bytes(report_dict(b, sections))


def build_report_excel(b: AnalysisBundle, sections: list[str] | None = None) -> bytes:
    sections = _sections(sections)
    sheets = {"Synthèse": pd.DataFrame({"Synthèse exécutive": executive_summary(b)})}
    sheets.update(_bundle_tables(b, sections))
    if "recommendations" in sections:
        sheets["Recommandations"] = pd.DataFrame({"Recommandation": recommendations(b)})
    return to_excel_bytes(sheets)


def build_report_csv_zip(b: AnalysisBundle, sections: list[str] | None = None) -> bytes:
    sections = _sections(sections)
    tables = {"00_synthese": pd.DataFrame({"Synthèse exécutive": executive_summary(b)})}
    for i, (name, df) in enumerate(_bundle_tables(b, sections).items(), start=1):
        tables[f"{i:02d}_{name}"] = df
    return zip_csvs(tables)


# ---------------------------------------------------------------------------
# Graphiques principaux
# ---------------------------------------------------------------------------

def _main_charts(b: AnalysisBundle, max_numeric: int = 4, max_cat: int = 3) -> list[tuple[str, Callable[[], Any], Callable[[], bytes]]]:
    """(titre, figure Plotly, PNG matplotlib) des graphiques principaux."""
    df = b.typed_df
    charts: list[tuple[str, Callable[[], Any], Callable[[], bytes]]] = []
    nums = [c for c in b.numeric_cols if df[c].notna().sum() > 1][:max_numeric]
    cats = [c for c in b.categorical_cols if 2 <= df[c].nunique() <= 50][:max_cat]
    for c in nums:
        charts.append((f"Distribution de {c}", lambda c=c: viz.histogram(df, c), lambda c=c: viz.static_histogram(df, c)))
    for c in cats:
        charts.append((f"Répartition de {c}", lambda c=c: viz.bar_chart(df, c), lambda c=c: viz.static_bar(df, c)))
    for c in b.temporal_cols[:1]:
        charts.append((f"Évolution temporelle de {c}", lambda c=c: viz.time_series(df, c),
                       lambda c=c: viz.static_time_series(df, c)))
    if nums:
        charts.append(("Boîtes à moustaches", lambda: viz.boxplot(df, nums[0]), lambda: viz.static_boxplots(df, nums)))
    return charts


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }}</title>
<style>
:root{--p:#4F46E5;--bg:#F8FAFC;--card:#fff;--txt:#0F172A;--mut:#64748B;--bd:#E2E8F0}
@media (prefers-color-scheme: dark){:root{--bg:#0B1120;--card:#111827;--txt:#E5E7EB;--mut:#94A3B8;--bd:#1F2937}}
*{box-sizing:border-box}body{margin:0;font-family:Inter,Segoe UI,Roboto,Arial,sans-serif;background:var(--bg);color:var(--txt);line-height:1.5}
header{background:linear-gradient(135deg,#4F46E5,#0EA5E9);color:#fff;padding:48px 24px}
header h1{margin:0;font-size:32px}header p{margin:6px 0 0;opacity:.9}
main{max-width:1100px;margin:0 auto;padding:24px 16px}
section{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:20px 24px;margin:18px 0}
h2{color:var(--p);margin-top:0;border-bottom:2px solid var(--bd);padding-bottom:6px}
h3{margin-bottom:6px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.kpi{border:1px solid var(--bd);border-radius:10px;padding:12px}.kpi b{display:block;font-size:22px}.kpi span{color:var(--mut);font-size:13px}
.tbl{overflow-x:auto;margin:8px 0}table{border-collapse:collapse;width:100%;font-size:13px}
th{background:var(--p);color:#fff;text-align:left;padding:6px 8px;position:sticky;top:0}
td{border-bottom:1px solid var(--bd);padding:5px 8px;vertical-align:top}
tr:nth-child(even) td{background:rgba(100,116,139,.06)}
.tag{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;font-weight:600}
.cat-Constat{background:#DBEAFE;color:#1E3A8A}.cat-Interprétation{background:#EDE9FE;color:#4C1D95}
.cat-Hypothèse{background:#FEF3C7;color:#78350F}.cat-Recommandation{background:#DCFCE7;color:#14532D}
.note{color:var(--mut);font-size:13px}nav ol{columns:2}nav a{color:var(--p);text-decoration:none}
.chart{margin:12px 0}
</style></head><body>
<header><h1>{{ title }}</h1><p>{{ app }} — rapport généré le {{ date }}</p><p>{{ filename }}</p></header>
<main>
<section><h2>Synthèse exécutive</h2><ul>{% for l in summary %}<li>{{ l }}</li>{% endfor %}</ul></section>
<nav><section><h2>Sommaire</h2><ol>{% for key, label in toc %}<li><a href="#{{ key }}">{{ label }}</a></li>{% endfor %}</ol></section></nav>
{% for s in blocks %}<section id="{{ s.key }}"><h2>{{ loop.index }}. {{ s.label }}</h2>{{ s.html | safe }}</section>{% endfor %}
<p class="note">Rapport produit localement par {{ app }}. Aucune donnée n'a été transmise à un service externe.</p>
</main></body></html>"""


def _html_table(df: pd.DataFrame, max_rows: int = 200) -> str:
    if df is None or df.empty:
        return "<p class='note'>Aucune donnée.</p>"
    out = df.head(max_rows).copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(4)
    html = out.to_html(index=False, escape=True, na_rep="—", border=0)
    more = f"<p class='note'>{len(df) - max_rows} ligne(s) supplémentaire(s) non affichée(s).</p>" if len(df) > max_rows else ""
    return f"<div class='tbl'>{html}</div>{more}"


def _p(text: str) -> str:
    return f"<p>{escape(text)}</p>"


def build_report_html(b: AnalysisBundle, sections: list[str] | None = None, title: str | None = None,
                      include_charts: bool = True) -> str:
    """Rapport HTML autonome."""
    from jinja2 import Environment

    sections = _sections(sections)
    title = title or f"Rapport d'analyse — {b.meta.filename}"
    blocks = []
    plotly_included = False

    def fig_html(fig: Any) -> str:
        nonlocal plotly_included
        html = fig.to_html(full_html=False, include_plotlyjs=not plotly_included, config={"displaylogo": False})
        plotly_included = True
        return f"<div class='chart'>{html}</div>"

    for key in sections:
        h = ""
        if key == "presentation":
            o = b.overview
            h += ("<div class='kpis'>" + "".join(
                f"<div class='kpi'><b>{escape(v)}</b><span>{escape(k)}</span></div>" for k, v in [
                    ("Lignes", fmt_int(o.n_rows)), ("Colonnes", fmt_int(o.n_cols)),
                    ("Valeurs manquantes", fmt_pct(o.missing_ratio)), ("Doublons", fmt_int(o.n_duplicates)),
                    ("Score qualité", f"{fmt_num(b.quality.global_score, 1)}/100"),
                ]) + "</div>")
            h += _p(f"Fichier « {b.meta.filename} » ({b.meta.format_label}, {human_size(b.meta.size_bytes)}).")
            for w in b.meta.warnings:
                h += f"<p class='note'>⚠ {escape(w)}</p>"
            h += "<h3>Aperçu des premières lignes</h3>" + _html_table(b.raw_df.head(10))
        elif key == "general":
            h += _html_table(general_table(b))
        elif key == "variables":
            h += _html_table(schema_table(b.schemas))
        elif key == "statistics":
            h += _p("Conventions : variance et écart-type d'échantillon (n − 1), quartiles par interpolation "
                    "linéaire, asymétrie et excès d'aplatissement corrigés du biais.")
            h += "<h3>Variables quantitatives</h3>" + _html_table(b.univariate.numeric_table())
            h += "<h3>Variables qualitatives</h3>" + _html_table(b.univariate.categorical_table())
            for name, st in list(b.univariate.categorical.items())[:10]:
                h += f"<h3>Fréquences — {escape(name)}</h3>" + _html_table(st.frequencies, 30)
            if b.univariate.temporal:
                h += "<h3>Variables temporelles</h3>" + _html_table(b.univariate.temporal_table())
        elif key == "missing":
            h += _html_table(b.missing)
            if include_charts and b.overview.n_missing:
                h += fig_html(viz.missing_bar(b.missing))
        elif key == "quality":
            h += _p(f"Score global : {fmt_num(b.quality.global_score, 1)}/100 ({b.quality.grade}). Ce score est un "
                    "indicateur heuristique (pondérations documentées ci-dessous), pas une mesure universelle.")
            h += _html_table(pd.DataFrame([{"Composante": SUBSCORE_LABELS[k], "Score (/100)": v,
                                            "Pondération (%)": round(100 * b.quality.weights[k], 1)}
                                           for k, v in b.quality.sub_scores.items()]))
            h += "<h3>Problèmes détectés</h3>" + _html_table(b.quality.issues_table())
            h += "<h3>Valeurs atypiques</h3>" + _p("Une valeur atypique n'est pas nécessairement une erreur.")
            h += _html_table(b.quality.outliers_table())
            h += "<h3>Score par variable</h3>" + _html_table(b.quality.column_scores)
        elif key == "correlations":
            if b.correlation is None:
                h += _p(b.correlation_method_reason)
            else:
                info = METHOD_INFO[b.correlation.method]
                h += _p(f"Méthode : {info['label']}. {b.correlation_method_reason}")
                h += _p(f"Conditions d'utilisation : {info['conditions']}")
                if include_charts:
                    h += fig_html(viz.correlation_heatmap(b.correlation.matrix, f"Matrice de corrélation ({info['label']})"))
                h += "<h3>Paires de variables</h3>" + _html_table(b.correlation.pairs(), 50)
                if not b.associations.empty:
                    h += "<h3>Fortes associations</h3>" + _html_table(b.associations)
        elif key == "visualizations":
            if include_charts:
                for _title, fig_fn, _ in _main_charts(b):
                    try:
                        h += fig_html(fig_fn())
                    except Exception as exc:  # noqa: BLE001 - un graphique défaillant ne bloque pas le rapport
                        h += f"<p class='note'>Graphique « {escape(_title)} » indisponible : {escape(str(exc))}</p>"
            else:
                h += _p("Graphiques désactivés pour ce rapport.")
        elif key == "interpretations":
            for cat in (Category.FINDING, Category.INTERPRETATION, Category.HYPOTHESIS):
                items = b.insights.by_category(cat)
                if items:
                    h += f"<h3><span class='tag cat-{cat.value}'>{cat.value}s</span></h3><ul>" + "".join(
                        f"<li>{escape(i.text)}</li>" for i in items) + "</ul>"
            if not b.insights.insights:
                h += _p("Aucun commentaire automatique.")
        elif key == "recommendations":
            h += "<ul>" + "".join(f"<li>{escape(r)}</li>" for r in recommendations(b)) + "</ul>"
        blocks.append({"key": key, "label": REPORT_SECTIONS[key], "html": h})

    env = Environment(autoescape=True)
    tpl = env.from_string(_HTML_TEMPLATE)
    return tpl.render(
        title=title, app=f"{APP_NAME} – {APP_SUBTITLE}", date=datetime.now().strftime("%d/%m/%Y à %H:%M"),
        filename=b.meta.filename, summary=executive_summary(b), toc=[(k, REPORT_SECTIONS[k]) for k in sections],
        blocks=blocks,
    )


# ---------------------------------------------------------------------------
# PDF (reportlab)
# ---------------------------------------------------------------------------

def _register_fonts() -> tuple[str, str]:
    """Police Unicode (DejaVu, fournie avec matplotlib) ; repli sur Helvetica."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    try:
        import matplotlib

        base = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        if "DejaVu" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont("DejaVu", str(base / "DejaVuSans.ttf")))
            pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(base / "DejaVuSans-Bold.ttf")))
        return "DejaVu", "DejaVu-Bold"
    except Exception:  # pragma: no cover - matplotlib absent
        return "Helvetica", "Helvetica-Bold"


def _pdf_text(value: Any, max_len: int = 60) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e15:
            text = fmt_int(value)
        elif abs(value) >= 1e6 or abs(value) < 1e-4:
            text = f"{value:.4g}".replace(".", ",")
        else:
            text = f"{value:,.4f}".rstrip("0").rstrip(".").replace(",", " ").replace(".", ",")
    elif isinstance(value, pd.Timestamp):
        text = value.strftime("%d/%m/%Y %H:%M") if (value.hour or value.minute) else value.strftime("%d/%m/%Y")
    else:
        text = str(value)
    text = text if len(text) <= max_len else text[: max_len - 1] + "…"
    return escape(text)


def build_report_pdf(b: AnalysisBundle, sections: list[str] | None = None, title: str | None = None,
                     include_charts: bool = True) -> bytes:
    """Rapport PDF : couverture, sommaire, synthèse exécutive, sections, tableaux et graphiques."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        BaseDocTemplate, Frame, Image, KeepTogether, NextPageTemplate, PageBreak, PageTemplate,
        Paragraph, Spacer, Table, TableStyle,
    )
    from reportlab.platypus.tableofcontents import TableOfContents

    sections = _sections(sections)
    font, bold = _register_fonts()
    title = title or "Rapport d'analyse de données"
    primary = colors.HexColor("#4F46E5")
    muted = colors.HexColor("#64748B")
    gen_date = datetime.now().strftime("%d/%m/%Y à %H:%M")

    ss = getSampleStyleSheet()
    st = {
        "body": ParagraphStyle("body", parent=ss["BodyText"], fontName=font, fontSize=9, leading=12.5),
        "small": ParagraphStyle("small", parent=ss["BodyText"], fontName=font, fontSize=7.5, leading=9.5, textColor=muted),
        "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontName=bold, fontSize=16, textColor=primary, spaceBefore=6, spaceAfter=8),
        "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName=bold, fontSize=11.5, spaceBefore=8, spaceAfter=4,
                             keepWithNext=1),
        "h1toc": ParagraphStyle("h1toc", parent=ss["Heading1"], fontName=bold, fontSize=16, textColor=primary,
                                spaceAfter=8),
        "small_c": ParagraphStyle("small_c", parent=ss["BodyText"], fontName=font, fontSize=7.5, leading=9.5,
                                  textColor=muted, alignment=TA_CENTER),
        "cell": ParagraphStyle("cell", fontName=font, fontSize=6.8, leading=8.2),
        "cellh": ParagraphStyle("cellh", fontName=bold, fontSize=6.8, leading=8.2, textColor=colors.white),
        "cover_t": ParagraphStyle("ct", fontName=bold, fontSize=30, leading=36, textColor=primary, alignment=TA_CENTER),
        "cover_s": ParagraphStyle("cs", fontName=font, fontSize=13, leading=18, textColor=muted, alignment=TA_CENTER),
        "bullet": ParagraphStyle("bullet", parent=ss["BodyText"], fontName=font, fontSize=9, leading=12.5,
                                 leftIndent=12, bulletIndent=2),
        "toc1": ParagraphStyle("toc1", fontName=font, fontSize=11, leading=18, leftIndent=10),
    }

    class Doc(BaseDocTemplate):
        def afterFlowable(self, flowable: Any) -> None:  # noqa: N802 - API reportlab
            if isinstance(flowable, Paragraph) and flowable.style.name == "h1":
                key = getattr(flowable, "_bookmark", None)
                text = flowable.getPlainText()
                self.notify("TOCEntry", (0, text, self.page, key))
                if key:
                    self.canv.bookmarkPage(key)
                    self.canv.addOutlineEntry(text, key, level=0)

    buf = io.BytesIO()
    doc = Doc(buf, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.8 * cm,
              bottomMargin=1.8 * cm, title=title, author=APP_NAME, subject=b.meta.filename)
    width = A4[0] - doc.leftMargin - doc.rightMargin
    frame = Frame(doc.leftMargin, doc.bottomMargin, width, A4[1] - doc.topMargin - doc.bottomMargin, id="f")

    def on_cover(canv: Any, _doc: Any) -> None:
        canv.saveState()
        canv.setFillColor(primary)
        canv.rect(0, A4[1] - 7 * cm, A4[0], 7 * cm, stroke=0, fill=1)
        canv.setFillColor(colors.HexColor("#0EA5E9"))
        canv.rect(0, A4[1] - 7.3 * cm, A4[0], 0.3 * cm, stroke=0, fill=1)
        canv.setFillColor(colors.white)
        canv.setFont(bold, 26)
        canv.drawString(2 * cm, A4[1] - 3.6 * cm, APP_NAME)
        canv.setFont(font, 13)
        canv.drawString(2 * cm, A4[1] - 4.5 * cm, APP_SUBTITLE)
        canv.restoreState()

    def on_page(canv: Any, _doc: Any) -> None:
        canv.saveState()
        canv.setFont(font, 7.5)
        canv.setFillColor(muted)
        canv.drawString(doc.leftMargin, 1.0 * cm, f"{APP_NAME} — {b.meta.filename[:70]}")
        canv.drawRightString(A4[0] - doc.rightMargin, 1.0 * cm, f"Page {canv.getPageNumber()} — {gen_date}")
        canv.setStrokeColor(colors.HexColor("#E2E8F0"))
        canv.line(doc.leftMargin, 1.35 * cm, A4[0] - doc.rightMargin, 1.35 * cm)
        canv.restoreState()

    doc.addPageTemplates([PageTemplate("cover", [frame], onPage=on_cover), PageTemplate("normal", [frame], onPage=on_page)])

    def para(text: str, style: str = "body") -> Paragraph:
        return Paragraph(escape(text), st[style])

    def table(df: pd.DataFrame, max_rows: int = 40, col_widths: list[float] | None = None, max_len: int = 60) -> list[Any]:
        if df is None or df.empty:
            return [para("Aucune donnée.", "small")]
        data = df.head(max_rows)
        header = [Paragraph(escape(str(c)), st["cellh"]) for c in data.columns]
        rows = [[Paragraph(_pdf_text(v, max_len), st["cell"]) for v in r] for r in data.itertuples(index=False)]
        t = Table([header] + rows, repeatRows=1, colWidths=col_widths or [width / len(data.columns)] * len(data.columns))
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), primary),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")]),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        out: list[Any] = [t]
        if len(df) > max_rows:
            out.append(para(f"… {len(df) - max_rows} ligne(s) supplémentaire(s) disponibles dans l'export Excel.", "small"))
        out.append(Spacer(1, 6))
        return out

    def wide_table(df: pd.DataFrame, id_col: str, chunk: int = 5) -> list[Any]:
        """Tableau transposé (indicateurs en lignes, variables en colonnes) découpé par blocs."""
        if df is None or df.empty:
            return [para("Aucune donnée.", "small")]
        tdf = df.set_index(id_col).T.reset_index().rename(columns={"index": "Indicateur"})
        cols = list(tdf.columns[1:])
        out: list[Any] = []
        for i in range(0, len(cols), chunk):
            part = tdf[["Indicateur"] + cols[i:i + chunk]]
            first = width * 0.26
            widths = [first] + [(width - first) / (len(part.columns) - 1)] * (len(part.columns) - 1)
            out += table(part, max_rows=60, col_widths=widths, max_len=40)
        return out

    def image(png: bytes, max_w: float | None = None) -> Image:
        from reportlab.lib.utils import ImageReader

        reader = ImageReader(io.BytesIO(png))
        iw, ih = reader.getSize()
        w = min(max_w or width, width)
        h = w * ih / iw
        if h > 11 * cm:
            h = 11 * cm
            w = h * iw / ih
        return Image(io.BytesIO(png), width=w, height=h)

    def h1(text: str, key: str) -> Paragraph:
        p = Paragraph(escape(text), st["h1"])
        p._bookmark = key  # type: ignore[attr-defined]
        return p

    story: list[Any] = [Spacer(1, 6.5 * cm), Paragraph(escape(title), st["cover_t"]), Spacer(1, 0.6 * cm),
                        Paragraph(escape(b.meta.filename), st["cover_s"]), Spacer(1, 0.3 * cm),
                        Paragraph(f"{escape(b.meta.format_label)} — {fmt_int(b.overview.n_rows)} lignes × "
                                  f"{fmt_int(b.overview.n_cols)} colonnes", st["cover_s"]),
                        Spacer(1, 2.5 * cm)]
    kpi = Table([[Paragraph(f"<b>{v}</b>", st["cover_s"]) for v in [
        f"{fmt_num(b.quality.global_score, 1)}/100", fmt_pct(b.overview.missing_ratio), fmt_int(b.overview.n_duplicates)]],
        [Paragraph(escape(v), st["small_c"]) for v in ["Score qualité (heuristique)", "Valeurs manquantes", "Doublons"]]],
        colWidths=[width / 3] * 3)
    kpi.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"),
                             ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                             ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC"))]))
    story += [kpi, Spacer(1, 3 * cm), Paragraph(f"Rapport généré le {gen_date}", st["cover_s"]),
              Paragraph(f"{APP_NAME} v{APP_VERSION} — analyse 100 % locale", st["cover_s"]),
              NextPageTemplate("normal"), PageBreak()]

    toc = TableOfContents()
    toc.levelStyles = [st["toc1"]]
    story += [Paragraph("Sommaire", st["h1toc"]), toc, PageBreak()]

    story.append(h1("Synthèse exécutive", "synthese"))
    for line in executive_summary(b):
        story.append(Paragraph(escape(line), st["bullet"], bulletText="•"))
    story.append(Spacer(1, 8))

    n = 0
    for key in sections:
        n += 1
        story.append(h1(f"{n}. {REPORT_SECTIONS[key]}", key))
        if key == "presentation":
            story.append(para(f"Ce rapport présente l'analyse automatique du fichier « {b.meta.filename} » "
                              f"({b.meta.format_label}, {human_size(b.meta.size_bytes)}) : structure, statistiques "
                              "descriptives, qualité des données, relations entre variables et recommandations."))
            for w in b.meta.warnings:
                story.append(para(f"⚠ {w}", "small"))
            story.append(Paragraph("Aperçu des premières observations", st["h2"]))
            prev = b.raw_df.iloc[:8, :8]
            story += table(prev, max_len=25)
            if b.raw_df.shape[1] > 8:
                story.append(para(f"({b.raw_df.shape[1] - 8} colonne(s) supplémentaire(s) non affichée(s).)", "small"))
        elif key == "general":
            story += table(general_table(b), col_widths=[width * 0.4, width * 0.6])
        elif key == "variables":
            vt = schema_table(b.schemas)[["Variable", "Type détecté", "Type statistique", "Valeurs uniques",
                                          "Valeurs manquantes", "Justification"]]
            story += table(vt, max_rows=80, col_widths=[width * w for w in (0.18, 0.12, 0.17, 0.09, 0.1, 0.34)])
        elif key == "statistics":
            story.append(para("Conventions : variance et écart-type d'échantillon (n − 1) ; quartiles par interpolation "
                              "linéaire ; asymétrie et excès d'aplatissement corrigés du biais ; valeurs manquantes "
                              "et infinies exclues.", "small"))
            nt = b.univariate.numeric_table()
            if not nt.empty:
                story.append(Paragraph("Variables quantitatives", st["h2"]))
                keep = ["Variable", "Valides", "Manquants", "Moyenne", "Médiane", "Écart-type", "Minimum", "Q1", "Q3",
                        "Maximum", "IQR", "Coef. de variation", "Asymétrie", "Aplatissement (excès)"]
                story += wide_table(nt[[c for c in keep if c in nt.columns]], "Variable")
            ct = b.univariate.categorical_table()
            if not ct.empty:
                story.append(Paragraph("Variables qualitatives", st["h2"]))
                story += table(ct.drop(columns=["Effectif"], errors="ignore"), max_rows=40)
                for name, cs in list(b.univariate.categorical.items())[:6]:
                    story.append(KeepTogether([Paragraph(f"Fréquences — {escape(name)}", st["h2"])]
                                              + table(cs.frequencies, max_rows=15)))
            tt = b.univariate.temporal_table()
            if not tt.empty:
                story.append(Paragraph("Variables temporelles", st["h2"]))
                story += wide_table(tt, "Variable", chunk=3)
                for name, ts in b.univariate.temporal.items():
                    for irr in ts.irregularities:
                        story.append(para(f"{name} : {irr}", "small"))
        elif key == "missing":
            mt = b.missing[b.missing["Valeurs manquantes"] > 0]
            if mt.empty:
                story.append(para("Aucune valeur manquante détectée."))
            else:
                story.append(para(f"{fmt_int(b.overview.n_missing)} valeurs manquantes au total "
                                  f"({fmt_pct(b.overview.missing_ratio)} des cellules), réparties sur {len(mt)} variable(s)."))
                story += table(mt, max_rows=40)
                if include_charts:
                    story.append(image(viz.static_missing(b.missing), width * 0.85))
        elif key == "quality":
            q = b.quality
            story.append(para(f"Score global : {fmt_num(q.global_score, 1)}/100 — appréciation : {q.grade.lower()}."))
            story.append(para("Ce score est un indicateur heuristique destiné à prioriser les actions : moyenne pondérée "
                              "de sous-scores (règles ci-dessous). Il ne constitue pas une mesure universelle de la "
                              "qualité et dépend du contexte métier.", "small"))
            story += table(pd.DataFrame([{"Composante": SUBSCORE_LABELS[k], "Score (/100)": v,
                                          "Pondération (%)": round(100 * q.weights[k], 1)} for k, v in q.sub_scores.items()]),
                           col_widths=[width * 0.5, width * 0.25, width * 0.25])
            story.append(Paragraph("Problèmes détectés", st["h2"]))
            it = q.issues_table()
            story += table(it[["Contrôle", "Variable", "Sévérité", "Nombre", "Constat"]] if not it.empty else it, max_rows=50,
                           col_widths=[width * w for w in (0.18, 0.16, 0.09, 0.08, 0.49)], max_len=200)
            ot = q.outliers_table()
            if not ot.empty:
                story.append(Paragraph("Valeurs atypiques (IQR et z-score)", st["h2"]))
                story.append(para("Les valeurs atypiques ne sont pas nécessairement des erreurs : elles peuvent refléter "
                                  "un phénomène réel ou une sous-population.", "small"))
                story += table(ot, max_rows=30)
                if include_charts:
                    story.append(image(viz.static_boxplots(b.typed_df, list(ot["Variable"])), width * 0.95))
        elif key == "correlations":
            if b.correlation is None:
                story.append(para(b.correlation_method_reason))
            else:
                info = METHOD_INFO[b.correlation.method]
                story.append(para(f"Méthode retenue : {info['label']}. {b.correlation_method_reason}"))
                story.append(para(f"Conditions d'utilisation : {info['conditions']}", "small"))
                story.append(para("Les valeurs manquantes sont exclues paire par paire ; l'effectif utilisé (n) est "
                                  "indiqué pour chaque paire. Corrélation n'implique pas causalité.", "small"))
                if include_charts and len(b.correlation.matrix) >= 2:
                    story.append(image(viz.static_correlation(b.correlation.matrix, f"Corrélations ({info['label']})"),
                                       width * 0.8))
                pairs = b.correlation.pairs()
                if not pairs.empty:
                    story.append(Paragraph("Principales paires", st["h2"]))
                    story += table(pairs.head(15), max_rows=15)
                if not b.associations.empty:
                    story.append(Paragraph("Fortes associations (toutes mesures)", st["h2"]))
                    story += table(b.associations, max_rows=20)
        elif key == "visualizations":
            if include_charts:
                for ctitle, _, png_fn in _main_charts(b):
                    try:
                        story.append(KeepTogether([Paragraph(escape(ctitle), st["h2"]), image(png_fn(), width * 0.85)]))
                    except Exception as exc:  # noqa: BLE001
                        story.append(para(f"Graphique « {ctitle} » indisponible : {exc}", "small"))
            else:
                story.append(para("Graphiques désactivés pour ce rapport."))
        elif key == "interpretations":
            story.append(para("Les commentaires sont générés par des règles appliquées aux résultats calculés. Ils "
                              "distinguent les constats (faits chiffrés), les interprétations, les hypothèses (à "
                              "vérifier) et les recommandations.", "small"))
            for cat in (Category.FINDING, Category.INTERPRETATION, Category.HYPOTHESIS):
                items = b.insights.by_category(cat)
                if items:
                    story.append(Paragraph(f"{cat.value}s", st["h2"]))
                    for i in items[:30]:
                        story.append(Paragraph(escape(i.text), st["bullet"], bulletText="•"))
        elif key == "recommendations":
            for r in recommendations(b):
                story.append(Paragraph(escape(r), st["bullet"], bulletText="•"))
        story.append(Spacer(1, 6))

    try:
        doc.multiBuild(story)
    except Exception as exc:
        raise ReportError("La génération du PDF a échoué.", detail=f"{type(exc).__name__}: {exc}") from exc
    return buf.getvalue()


EXPORT_FORMATS = {
    "PDF": ("pdf", "application/pdf"),
    "HTML": ("html", "text/html"),
    "Excel": ("xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    "CSV (ZIP)": ("zip", "application/zip"),
    "JSON": ("json", "application/json"),
}


def build_report(b: AnalysisBundle, fmt: str, sections: list[str] | None = None, title: str | None = None,
                 include_charts: bool = True) -> bytes:
    """Point d'entrée unique de génération d'un rapport au format demandé."""
    if fmt == "PDF":
        return build_report_pdf(b, sections, title, include_charts)
    if fmt == "HTML":
        return build_report_html(b, sections, title, include_charts).encode("utf-8")
    if fmt == "Excel":
        return build_report_excel(b, sections)
    if fmt == "CSV (ZIP)":
        return build_report_csv_zip(b, sections)
    if fmt == "JSON":
        return build_report_json(b, sections)
    raise ReportError(f"Format de rapport inconnu : {fmt}.")

