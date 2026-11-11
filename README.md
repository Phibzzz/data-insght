# DATAINSIGHT – Intelligent Dataset Analyzer

Application Streamlit qui importe un jeu de données, en reconnaît automatiquement la structure,
évalue sa qualité, calcule des statistiques descriptives et multivariées, génère des graphiques
interactifs, rédige des interprétations chiffrées et produit des rapports PDF / HTML / Excel / CSV / JSON.

**Tout est local** : aucune clé API, aucune donnée transmise à un service externe.

---

## Installation

Prérequis : Python 3.10 ou plus récent (testé avec Python 3.13).

```bash
python -m venv .venv
# Windows : .venv\Scripts\activate    —    macOS / Linux : source .venv/bin/activate
pip install -r requirements.txt
```

## Lancement

```bash
streamlit run app.py
```

L'application s'ouvre sur <http://localhost:8501>. Pour la découvrir sans fichier, utilisez les
**jeux d'exemple** de la page d'accueil.

Scripts utiles :

| Commande | Rôle |
|---|---|
| `python scripts/generate_samples.py` | (Re)génère les fichiers d'exemple de `sample_data/` dans tous les formats |
| `python scripts/generate_samples.py --large` | Ajoute un jeu de 500 000 lignes (Parquet, non versionné) |
| `python scripts/generate_example_report.py` | Régénère les rapports d'exemple de `docs/` |
| `python scripts/make_logo.py` | Régénère `assets/logo.png` |
| `pytest` | Lance la suite de tests (116 tests) |

---

## Fonctionnalités

| Rubrique | Contenu |
|---|---|
| Accueil | Présentation, fonctionnalités, formats, jeux d'exemple |
| Importation | Détection du format réel, de l'encodage, du séparateur et de la décimale ; choix de la feuille / table / fichier d'archive ; lecture partielle des gros fichiers ; résumé (format, taille, lignes, colonnes, manquants, doublons, aperçu) |
| Vue d'ensemble | Indicateurs clés, synthèse générale, interprétations automatiques, types, manquants |
| Exploration | Premières / dernières lignes, échantillon, recherche, filtres, sélection et tri de colonnes, export du résultat filtré |
| Profil des variables | Type technique + type statistique, justification, confiance ; corrections manuelles réversibles |
| Statistiques descriptives | Quantitatives (25+ indicateurs), qualitatives (fréquences, cumul si ordonnée), temporelles (étendue, répartitions, irrégularités) |
| Analyse multivariée | Pearson / Spearman / Kendall avec méthode recommandée, covariance, contingence + χ² + V de Cramér, comparaisons de groupes (η², ANOVA, Kruskal-Wallis), fortes associations |
| Qualité des données | Score global transparent, 15 contrôles, outliers IQR / z-score, règles de validité métier |
| Visualisations | Suggestions automatiques + constructeur (histogramme, densité, boîte, violon, ECDF, barres, camembert, Pareto, séries temporelles, calendrier, corrélations, pair plot…) |
| Nettoyage | Doublons, manquants, types, renommage, suppression, texte, filtres, outliers, nouvelles variables ; avant/après, historique, annulation, export |
| Rapports | Sections au choix ; PDF (couverture, sommaire, synthèse exécutive), HTML autonome, Excel, CSV (ZIP), JSON |
| Paramètres | Langue, limites, seuils statistiques, pondérations du score qualité |

La documentation détaillée est dans [docs/FONCTIONNALITES.md](docs/FONCTIONNALITES.md).
Un exemple de rapport est fourni : [PDF](docs/exemple_rapport.pdf), [HTML](docs/exemple_rapport.html),
[Excel](docs/exemple_rapport.xlsx), [JSON](docs/exemple_rapport.json).

## Formats pris en charge

| Famille | Extensions | Bibliothèque |
|---|---|---|
| Tabulaires | `.csv` `.tsv` `.txt` | pandas (+ Polars optionnel) |
| Bureautiques | `.xls` `.xlsx` `.xlsm` `.ods` | xlrd, openpyxl, odfpy |
| Analytiques | `.parquet` `.feather` `.orc` | pyarrow |
| Structurés | `.json` `.jsonl` `.xml` | pandas, lxml, defusedxml |
| Statistiques | `.sav` `.dta` `.sas7bdat` | pyreadstat, pandas |
| Bases de données | `.sqlite` `.db` | sqlite3 (lecture seule) |
| Archives | `.zip` | zipfile (contrôles de sécurité) |

Ajouter un format : écrire une fonction décorée par `@register_loader("monformat")` dans
[core/data_loader.py](core/data_loader.py) et déclarer l'extension dans `EXTENSION_FORMATS`
([core/file_detector.py](core/file_detector.py)) et `SUPPORTED_EXTENSIONS` ([config/settings.py](config/settings.py)).

---

## Architecture

```
app.py                     Point d'entrée : configuration, thème, navigation (st.navigation)
config/settings.py         Limites, seuils statistiques, pondérations (dataclasses)
i18n/                      Traductions (fr complet, en pour la navigation) et fonction t()
core/                      Logique métier, indépendante de Streamlit
  exceptions.py            Hiérarchie d'erreurs avec messages utilisateur
  file_detector.py         Format réel (signatures binaires), encodage, séparateur, décimale
  data_loader.py           Registre de chargeurs, ZIP / XML / SQLite sécurisés, lecture partielle
  schema_detector.py       Types statistiques, vue typée, corrections réversibles
  profiling_engine.py      Vue d'ensemble, manquants
  statistics_engine.py     Statistiques univariées
  multivariate_engine.py   Corrélations, contingence, comparaisons de groupes
  quality_engine.py        Audit, outliers, score transparent, règles de validité
  interpretation_engine.py Commentaires (constat / interprétation / hypothèse / recommandation), interface LLMProvider
  visualization_engine.py  Figures Plotly (UI, HTML) et matplotlib (PDF)
  cleaning_engine.py       Opérations pures, session avec historique et annulation, évaluateur sûr
  report_engine.py         Rapports PDF / HTML / Excel / CSV / JSON
  pipeline.py              Orchestration de l'analyse complète (AnalysisBundle)
views/                     Les 12 pages Streamlit
components/                Widgets réutilisables (KPI, tableaux, graphiques, erreurs, sidebar)
utils/                     État de session, cache, exports sécurisés, validateurs, formatage, jeux d'exemple
scripts/                   Génération des exemples, du logo et du rapport d'exemple
tests/                     Tests pytest (moteurs + smoke tests de l'interface via AppTest)
sample_data/               Jeux d'exemple multiformats
docs/                      Documentation et rapports d'exemple
```

Écart par rapport à l'arborescence proposée : les pages sont dans `views/` (et non `pages/`) car la
navigation utilise `st.navigation`, et Streamlit interprète automatiquement un dossier `pages/`
comme une navigation concurrente. Les modules `core/exceptions.py` et `core/pipeline.py` ont été
ajoutés pour centraliser les erreurs et l'orchestration.

### Principes

- **Le fichier d'origine n'est jamais modifié.** Les conversions de types s'appliquent à une *vue typée* ;
  les corrections manuelles sont des surcharges réversibles ; le nettoyage travaille sur une copie
  avec historique rejouable.
- **Aucun résultat inventé.** Chaque interprétation cite la mesure calculée dont elle découle.
- **Cache** : `st.cache_data` pour la lecture des fichiers ; l'analyse complète est mémorisée par session
  (clé : version des données + paramètres) et n'est recalculée qu'en cas de changement.

## Sécurité et confidentialité

- Aucune exécution de code issu des fichiers : macros XLSM ignorées, pas de pickle, expressions de
  nouvelles variables évaluées par un interpréteur `ast` restreint (jamais `eval`).
- XML : DTD et entités refusées (protection XXE). SQLite : ouverture en lecture seule, nom de table
  validé contre `sqlite_master`. ZIP : chemins suspects, archives chiffrées, nombre de fichiers, taille
  décompressée et taux de compression (zip bomb) contrôlés ; pas d'extraction sur disque.
- Exports CSV / Excel : neutralisation des cellules commençant par `= + - @` (injection de formules) ;
  noms de fichiers assainis ; HTML échappé.
- `gatherUsageStats = false` ; aucune connexion sortante. Le rapport HTML embarque Plotly (hors ligne).

## Choix statistiques documentés

- Variance et écart-type d'échantillon (n − 1) ; quartiles par interpolation linéaire ;
  asymétrie et excès d'aplatissement corrigés du biais.
- Corrélations uniquement entre variables quantitatives ; suppression des manquants **par paire**
  (n affiché pour chaque coefficient). Méthode recommandée selon asymétrie, valeurs extrêmes et taille d'échantillon.
- Score qualité = moyenne pondérée de 5 sous-scores (complétude 35 %, unicité 20 %, validité 20 %,
  cohérence 15 %, atypiques 10 %) : **indicateur heuristique**, pondérations modifiables, non universel.
- Valeurs atypiques : règle de Tukey (k = 1,5) et |z| > 3 — une valeur atypique n'est pas forcément une erreur.

## Limites connues

- **Très gros fichiers** : les données sont chargées en mémoire (pandas). Au-delà de ~1 million de lignes,
  la lecture partielle (N premières lignes) est proposée ; il n'y a pas de traitement « hors mémoire ».
  Mesure indicative : 500 000 lignes × 6 colonnes → chargement 0,1 s, analyse complète ~5 s, PDF ~6 s.
  Pour des volumes supérieurs, privilégier Parquet et la lecture partielle, ou un pré-échantillonnage.
- **Lecture partielle CSV** : ce sont les N *premières* lignes (pas un échantillon aléatoire).
- **ORC** dépend du support de pyarrow (absent sur certaines installations Windows : message explicite).
  **SAS7BDAT** : lecture seule. **XLS** : lecture seule (export en XLSX).
- **Détection des types** : heuristique. Les variables ordinales textuelles ne sont reconnues que si
  elles correspondent à une échelle connue (faible/moyen/fort, Likert…) ; sinon, correction manuelle.
  Une suite d'entiers consécutifs uniques est considérée comme un identifiant.
- **Traduction** : français complet ; l'anglais couvre la navigation (l'architecture `i18n/` permet
  d'étendre la traduction clé par clé).
- **Paramètres** : appliqués à l'instance locale (usage mono-utilisateur prévu) ; la limite de
  téléversement Streamlit se règle dans `.streamlit/config.toml` (`maxUploadSize`).
- **Thème** : clair et sombre via le menu ⋮ › Settings › Theme de Streamlit.
- **LLM** : non branché. Le protocole `LLMProvider` (core/interpretation_engine.py) permet d'ajouter un
  fournisseur ; le fournisseur par défaut n'effectue aucun appel.

## Tests

```bash
pytest
```

116 tests : reconnaissance des formats, chargement (tous formats, encodages, séparateurs, feuilles, tables,
ZIP malveillants, XXE, injection SQL), détection des types, exactitude des statistiques sur valeurs connues
(comparées à SciPy), corrélations, qualité, nettoyage (y compris le bac à sable d'expressions), interprétations,
exports et rapports, et smoke tests de chacune des 12 pages avec interactions (AppTest).
