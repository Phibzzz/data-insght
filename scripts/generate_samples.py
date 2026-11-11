"""Génère les jeux de données d'exemple dans sample_data/ (plusieurs formats).

Usage :
    python scripts/generate_samples.py            # jeux standards
    python scripts/generate_samples.py --large    # + grand jeu (500 000 lignes, Parquet)
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from utils.sample_factory import (  # noqa: E402
    categorical_dataset,
    duplicates_dataset,
    large_dataset,
    missing_dataset,
    mixed_dataset,
    numeric_dataset,
    outliers_dataset,
    single_column_dataset,
    temporal_dataset,
    unusual_names_dataset,
)

OUT = ROOT / "sample_data"


def _write(name: str, fn) -> None:
    path = OUT / name
    try:
        fn(path)
        print(f"  ✓ {name}")
    except Exception as exc:  # noqa: BLE001 - un format optionnel peut manquer
        print(f"  ✗ {name} ignoré ({type(exc).__name__}: {exc})")


def main(large: bool = False) -> None:
    OUT.mkdir(exist_ok=True)
    mixed = mixed_dataset()
    numeric = numeric_dataset()
    cat = categorical_dataset()
    temporal = temporal_dataset()
    print(f"Génération dans {OUT}")

    # Formats tabulaires
    _write("ventes_mixte.csv", lambda p: mixed.to_csv(p, index=False))
    _write("mesures_numeriques_fr.csv",
           lambda p: numeric.to_csv(p, index=False, sep=";", decimal=",", encoding="cp1252"))
    _write("enquete_satisfaction.tsv", lambda p: cat.to_csv(p, index=False, sep="\t"))
    _write("trafic_journalier.txt", lambda p: temporal.to_csv(p, index=False, sep="|"))
    _write("donnees_manquantes.csv", lambda p: missing_dataset().to_csv(p, index=False))
    _write("valeurs_aberrantes.csv", lambda p: outliers_dataset().to_csv(p, index=False))
    _write("doublons.csv", lambda p: duplicates_dataset().to_csv(p, index=False))
    _write("une_colonne.csv", lambda p: single_column_dataset().to_csv(p, index=False))
    _write("noms_inhabituels.csv", lambda p: unusual_names_dataset().to_csv(p, index=False, sep=";"))

    # Bureautiques
    def excel(p: Path) -> None:
        with pd.ExcelWriter(p, engine="openpyxl") as w:
            mixed.to_excel(w, sheet_name="Ventes", index=False)
            numeric.to_excel(w, sheet_name="Mesures", index=False)
            cat.to_excel(w, sheet_name="Enquête", index=False)

    _write("classeur_multi_feuilles.xlsx", excel)
    _write("enquete_satisfaction.ods", lambda p: cat.to_excel(p, index=False, engine="odf"))

    # Analytiques
    _write("ventes_mixte.parquet", lambda p: mixed.to_parquet(p, index=False))
    _write("trafic_journalier.feather", lambda p: temporal.to_feather(p))
    _write("mesures_numeriques.orc", lambda p: numeric.to_orc(p, index=False))

    # Structurés
    _write("ventes_mixte.json", lambda p: mixed.head(300).to_json(p, orient="records", force_ascii=False, indent=1))
    _write("enquete_satisfaction.jsonl", lambda p: cat.to_json(p, orient="records", lines=True, force_ascii=False))
    _write("mesures_numeriques.xml", lambda p: numeric.head(200).to_xml(p, index=False, root_name="mesures", row_name="mesure", parser="etree"))

    # Statistiques
    _write("mesures_numeriques.dta", lambda p: numeric.to_stata(p, write_index=False))

    def sav(p: Path) -> None:
        import pyreadstat

        df = cat.copy()
        df["satisfaction_code"] = df["satisfaction"].map(
            {"très insatisfait": 1, "insatisfait": 2, "neutre": 3, "satisfait": 4, "très satisfait": 5})
        pyreadstat.write_sav(df, str(p), variable_value_labels={
            "satisfaction_code": {1: "très insatisfait", 2: "insatisfait", 3: "neutre", 4: "satisfait", 5: "très satisfait"}})

    _write("enquete_satisfaction.sav", sav)

    # Base de données
    def sqlite(p: Path) -> None:
        if p.exists():
            p.unlink()
        with sqlite3.connect(p) as conn:
            mixed.to_sql("ventes", conn, index=False)
            numeric.to_sql("mesures", conn, index=False)
        conn.close()

    _write("base_exemple.sqlite", sqlite)

    # Archive
    def zipped(p: Path) -> None:
        with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("ventes_mixte.csv", mixed.to_csv(index=False))
            zf.writestr("mesures.csv", numeric.to_csv(index=False))

    _write("archive_exemple.zip", zipped)

    if large:
        _write("grand_dataset.parquet", lambda p: large_dataset().to_parquet(p, index=False))
    print("Terminé.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--large", action="store_true", help="Génère aussi un grand jeu de 500 000 lignes")
    main(parser.parse_args().large)
