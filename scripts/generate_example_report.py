"""Génère des exemples de rapports (PDF, HTML, Excel, JSON) dans docs/ à partir du jeu « ventes »."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.data_loader import load_path  # noqa: E402
from core.pipeline import run_analysis  # noqa: E402
from core.report_engine import EXPORT_FORMATS, build_report  # noqa: E402


def main() -> None:
    src = ROOT / "sample_data" / "ventes_mixte.csv"
    if not src.exists():
        raise SystemExit("Lancez d'abord : python scripts/generate_samples.py")
    res = load_path(src)
    bundle = run_analysis(res.df, res.meta)
    out = ROOT / "docs"
    out.mkdir(exist_ok=True)
    for fmt in ("PDF", "HTML", "Excel", "JSON"):
        ext = EXPORT_FORMATS[fmt][0]
        path = out / f"exemple_rapport.{ext}"
        path.write_bytes(build_report(bundle, fmt, title="Exemple — analyse du jeu « ventes »"))
        print(f"  ✓ {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
