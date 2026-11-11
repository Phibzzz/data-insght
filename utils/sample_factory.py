"""Fabrique de jeux de données d'exemple et de test (reproductibles, graine fixe).

Ces jeux couvrent les cas demandés : numérique, catégoriel, mixte, temporel,
valeurs manquantes, doublons, valeurs aberrantes, vide, une seule colonne,
noms de colonnes inhabituels et grande taille.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42


def numeric_dataset(n: int = 300) -> pd.DataFrame:
    """Mesures physiologiques purement numériques."""
    rng = np.random.default_rng(SEED)
    taille = rng.normal(170, 9, n).round(1)
    poids = (0.9 * (taille - 100) + rng.normal(0, 7, n)).round(1)
    return pd.DataFrame({
        "taille_cm": taille,
        "poids_kg": poids,
        "age": rng.integers(18, 80, n),
        "frequence_cardiaque": rng.normal(72, 10, n).round(0),
        "revenu_annuel": rng.lognormal(10.3, 0.5, n).round(2),
        "nb_enfants": rng.poisson(1.3, n),
    })


def categorical_dataset(n: int = 400) -> pd.DataFrame:
    """Enquête de satisfaction catégorielle."""
    rng = np.random.default_rng(SEED)
    return pd.DataFrame({
        "region": rng.choice(["Île-de-France", "Occitanie", "Bretagne", "Grand Est", "Corse"], n, p=[0.35, 0.25, 0.2, 0.18, 0.02]),
        "genre": rng.choice(["F", "M"], n),
        "abonne": rng.choice(["oui", "non"], n, p=[0.3, 0.7]),
        "satisfaction": rng.choice(
            ["très insatisfait", "insatisfait", "neutre", "satisfait", "très satisfait"], n,
            p=[0.05, 0.15, 0.3, 0.35, 0.15],
        ),
        "canal": rng.choice(["web", "magasin", "téléphone"], n, p=[0.6, 0.3, 0.1]),
        "niveau": rng.choice(["faible", "moyen", "fort"], n),
    })


def mixed_dataset(n: int = 1000) -> pd.DataFrame:
    """Ventes : identifiants, dates, catégories, montants, booléens, coordonnées, texte."""
    rng = np.random.default_rng(SEED)
    dates = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 730, n), unit="D")
    produit = rng.choice(["Ordinateur", "Écran", "Clavier", "Souris", "Casque"], n, p=[0.15, 0.2, 0.25, 0.3, 0.1])
    prix_base = {"Ordinateur": 900, "Écran": 220, "Clavier": 45, "Souris": 25, "Casque": 80}
    quantite = rng.integers(1, 6, n)
    prix = np.array([prix_base[p] for p in produit]) * rng.normal(1, 0.08, n)
    ville = rng.choice(["Paris", "Lyon", "Marseille", "Toulouse", "Lille"], n)
    coords = {"Paris": (48.8566, 2.3522), "Lyon": (45.764, 4.8357), "Marseille": (43.2965, 5.3698),
              "Toulouse": (43.6047, 1.4442), "Lille": (50.6292, 3.0573)}
    commentaires = [
        "Livraison rapide et produit conforme à la description, je recommande vivement.",
        "Emballage abîmé mais le produit fonctionne correctement après plusieurs jours d'utilisation.",
        "Service client réactif, remboursement effectué sans difficulté particulière.",
        "Qualité décevante pour le prix, je ne commanderai probablement plus ce modèle.",
        None,
    ]
    df = pd.DataFrame({
        "id_commande": [f"CMD-{100000 + i}" for i in range(n)],
        "date_commande": dates.strftime("%Y-%m-%d"),
        "produit": produit,
        "ville": ville,
        "latitude": [coords[v][0] + rng.normal(0, 0.02) for v in ville],
        "longitude": [coords[v][1] + rng.normal(0, 0.02) for v in ville],
        "quantite": quantite,
        "prix_unitaire": prix.round(2),
        "montant_total": (prix * quantite).round(2),
        "remise_appliquee": rng.choice([True, False], n, p=[0.25, 0.75]),
        "note_client": rng.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.1, 0.2, 0.35, 0.3]),
        "commentaire": [commentaires[i % len(commentaires)] if rng.random() < 0.6 else None for i in range(n)],
    })
    return df


def temporal_dataset(days: int = 730) -> pd.DataFrame:
    """Série journalière avec tendance, saisonnalité et quelques jours manquants."""
    rng = np.random.default_rng(SEED)
    dates = pd.date_range("2022-01-01", periods=days, freq="D")
    t = np.arange(days)
    visites = 1000 + 0.8 * t + 150 * np.sin(2 * np.pi * t / 365) + 80 * (dates.day_of_week >= 5) + rng.normal(0, 40, days)
    df = pd.DataFrame({
        "date": dates,
        "visites": visites.round(0).astype(int),
        "temperature": (12 + 9 * np.sin(2 * np.pi * (t - 100) / 365) + rng.normal(0, 2.5, days)).round(1),
        "horodatage_maj": dates + pd.to_timedelta(rng.integers(0, 86_400, days), unit="s"),
    })
    drop = rng.choice(days, 15, replace=False)  # trous dans la série
    return df.drop(index=drop).reset_index(drop=True)


def missing_dataset(n: int = 200) -> pd.DataFrame:
    """Jeu avec valeurs manquantes à différents taux."""
    rng = np.random.default_rng(SEED)
    df = numeric_dataset(n).copy()
    df["categorie"] = rng.choice(["A", "B", "C"], n)
    for col, rate in [("taille_cm", 0.05), ("poids_kg", 0.2), ("revenu_annuel", 0.6), ("categorie", 0.1)]:
        mask = rng.random(n) < rate
        df[col] = df[col].astype(object).where(~mask, None) if col == "categorie" else df[col].mask(mask)
    df["colonne_vide"] = np.nan
    return df


def duplicates_dataset() -> pd.DataFrame:
    """Jeu contenant des lignes dupliquées."""
    base = pd.DataFrame({
        "client": ["Alice", "Bob", "Chloé", "David", "Emma", "Bob", "Alice", "Farid"],
        "ville": ["Paris", "Lyon", "Nantes", "Lyon", "Paris", "Lyon", "Paris", "Lille"],
        "achats": [3, 5, 2, 7, 1, 5, 3, 4],
    })
    return base


def outliers_dataset(n: int = 200) -> pd.DataFrame:
    """Jeu contenant des valeurs aberrantes connues."""
    rng = np.random.default_rng(SEED)
    values = rng.normal(50, 5, n)
    values[:5] = [150, 160, -40, 170, 200]
    salaires = rng.normal(2500, 300, n).round(0)
    salaires[10] = 25_000
    return pd.DataFrame({"mesure": values.round(2), "salaire": salaires, "age": rng.integers(20, 60, n)})


def empty_dataset() -> pd.DataFrame:
    """Jeu vide (en-têtes sans lignes)."""
    return pd.DataFrame({"a": pd.Series(dtype=float), "b": pd.Series(dtype=str)})


def single_column_dataset() -> pd.DataFrame:
    """Jeu à une seule colonne."""
    return pd.DataFrame({"valeur": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]})


def unusual_names_dataset() -> pd.DataFrame:
    """Noms de colonnes inhabituels : espaces, accents, symboles, nombres, emoji."""
    return pd.DataFrame({
        "  Prénom de l'élève ": ["Léa", "Noé", "Zoë", "Ana"],
        "Note (sur 20) %": ["12,5", "15", "8,75", "19"],
        "123": [1, 2, 3, 4],
        "colonne-avec/slash": ["a", "b", "a", "b"],
        "=SUM(A1)": ["x", "y", "z", "w"],
        "📊 score": [0.1, 0.2, 0.3, 0.4],
    })


def large_dataset(n: int = 500_000) -> pd.DataFrame:
    """Grand jeu de données synthétique (non versionné)."""
    rng = np.random.default_rng(SEED)
    return pd.DataFrame({
        "id": np.arange(n),
        "groupe": rng.choice(list("ABCDEFGH"), n),
        "x": rng.normal(0, 1, n),
        "y": rng.normal(10, 3, n),
        "compteur": rng.poisson(4, n),
        "date": pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 1500, n), unit="D"),
    })


SAMPLE_BUILDERS = {
    "ventes_mixte": ("Ventes (dataset mixte)", mixed_dataset),
    "mesures_numeriques": ("Mesures (dataset numérique)", numeric_dataset),
    "enquete_satisfaction": ("Enquête (dataset catégoriel)", categorical_dataset),
    "trafic_journalier": ("Trafic web (dataset temporel)", temporal_dataset),
    "donnees_manquantes": ("Valeurs manquantes", missing_dataset),
    "valeurs_aberrantes": ("Valeurs aberrantes", outliers_dataset),
    "doublons": ("Doublons", duplicates_dataset),
}
