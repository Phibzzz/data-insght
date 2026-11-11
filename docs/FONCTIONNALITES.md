# DATAINSIGHT — Documentation des fonctionnalités

Ce document décrit chaque rubrique de l'application, les calculs effectués et leurs règles.

## 1. Importation et reconnaissance des fichiers

1. **Validation** : extension acceptée, taille ≤ limite (500 Mo par défaut), fichier non vide.
2. **Format réel** : lecture des signatures binaires (SQLite, Parquet `PAR1`, Feather `ARROW1`, ORC,
   OLE2 pour XLS, ZIP pour XLSX/XLSM/ODS, SPSS `$FL2`, Stata, SAS). Un XLSX nommé `.csv` est reconnu
   comme XLSX, avec un avertissement.
3. **Fichiers texte** : encodage (BOM, UTF-8, puis charset-normalizer, repli cp1252/latin-1),
   séparateur (`csv.Sniffer` + régularité du nombre de séparateurs par ligne, hors guillemets),
   séparateur décimal (virgule détectée avec le point-virgule), JSON vs JSON Lines, XML.
4. **Choix utilisateur** : feuille (Excel/ODS), table (SQLite), fichier d'une archive ZIP ; options avancées
   (séparateur, encodage, décimale, ligne d'en-tête, lecture partielle).
5. **Normalisation** : noms de colonnes uniques et non vides, structures JSON imbriquées aplaties
   (`user.name`), listes sérialisées en texte, lignes entièrement vides ignorées (signalées).

Le résumé affiche : nom, format détecté, taille, lignes, colonnes, valeurs manquantes, doublons, aperçu.

## 2. Profil des variables

Chaque colonne reçoit un **type technique** (stockage) et un **type statistique** :

| Famille | Types |
|---|---|
| Quantitative | continue, discrète, latitude, longitude |
| Qualitative | nominale, ordinale, binaire, booléenne, mois |
| Temporelle | date, horodatage, année |
| Autre | texte libre, identifiant, constante, vide |

Règles principales : conversions testées sur un échantillon (≥ 95 % de valeurs convertibles),
virgule décimale et espaces de milliers gérés, dates multi-formats (format majoritaire retenu, pas
d'ambiguïté jour/mois), oui/non/vrai/faux, 0/1, échelles ordinales connues, ratio d'unicité
(≥ 98 % → identifiant), longueur moyenne (≥ 40 caractères → texte libre), indices sur le nom de colonne.
Chaque détection est accompagnée d'une justification et d'un indice de confiance.

**Corrections manuelles** : choisir un autre type ; les analyses sont recalculées. Le nombre de valeurs
non convertibles est indiqué. « Revenir au type détecté » annule la correction.

## 3. Exploration

Premières / dernières lignes, échantillon aléatoire (graine réglable), recherche plein texte,
filtres combinables (selon le type : intervalle numérique, période, liste de modalités, contient,
manquant…), sélection et tri des colonnes, informations (`dtypes`, non nuls, mémoire), export CSV/Excel.

## 4. Statistiques descriptives

**Quantitatives** : effectif, valides, manquants, infinis, moyenne, médiane, mode(s), min, max, étendue,
variance et écart-type (n − 1), Q1/Q2/Q3, IQR, coefficient de variation, asymétrie, excès d'aplatissement,
erreur-type, IC 95 % (Student), P5/P95, MAD, somme, zéros, négatifs, valeurs uniques, test de normalité
(Shapiro-Wilk si n ≤ 5 000, sinon D'Agostino-Pearson).

**Qualitatives** : effectif, valides, manquants, nombre de modalités, modalité dominante et sa part,
entropie normalisée, table de fréquences (absolue, relative, %), fréquences cumulées si l'ordre est
significatif (ordinale, discrète).

**Temporelles** : min, max, étendue, dates distinctes, fréquence estimée (écart médian), ruptures
(écart > 1,5 × écart médian), horodatages dupliqués, dates futures, dates < 1900, répartitions par
année / mois / jour de la semaine.

## 5. Analyse multivariée

- **Corrélations** Pearson, Spearman, Kendall (tau-b), p-values et effectif par paire ; seules les variables
  quantitatives sont croisées. Recommandation : Kendall si n < 30 ou nombreux ex aequo, Spearman si
  asymétrie |skew| > 1 ou valeurs extrêmes, sinon Pearson. Le choix s'applique aux rapports.
- **Covariance** d'échantillon.
- **Tableaux croisés** : effectifs, % ligne, % colonne, % total, χ² d'indépendance, V de Cramér,
  alerte si plus de 20 % d'effectifs théoriques < 5.
- **Quantitatif × qualitatif** : effectif, moyenne, médiane, écart-type, min, max par groupe ; η² ;
  ANOVA et Kruskal-Wallis.
- **Fortes associations** : |r| ≥ 0,7, V de Cramér ≥ 0,5 significatif, η² ≥ 0,26.

## 6. Qualité des données

Contrôles : valeurs manquantes, lignes dupliquées, colonnes vides, constantes, valeurs non convertibles
dans le type retenu, nombres/dates stockés en texte, modalités rares (< 1 %), valeurs atypiques (IQR et
z-score), valeurs infinies, espaces superflus, variantes de casse/accents, dates futures ou < 1900,
négatifs sur des grandeurs positives par nature (hypothèse fondée sur le nom), coordonnées hors bornes,
règles de validité utilisateur (min, max, valeurs autorisées, motif, obligatoire).

Indicateurs : taux de complétude, de manquants, de doublons, de validité, lignes complètes.

**Score global** (sur 100) = Σ pondération × sous-score :

| Sous-score | Définition | Poids |
|---|---|---|
| Complétude | 1 − cellules manquantes / cellules | 35 % |
| Unicité | 1 − lignes dupliquées / lignes | 20 % |
| Validité | 1 − cellules invalides / cellules renseignées | 20 % |
| Cohérence | 1 − cellules texte incohérentes / cellules texte | 15 % |
| Atypiques | 1 − valeurs hors bornes IQR / valeurs numériques | 10 % |

Appréciation : ≥ 90 excellente, ≥ 75 bonne, ≥ 60 moyenne, sinon faible. **Limite** : ce score est
heuristique ; il ne mesure pas l'exactitude métier des données. Chaque problème est accompagné d'une
recommandation ; une valeur atypique n'est pas nécessairement une erreur.

## 7. Visualisations

Suggestions automatiques selon les types, et constructeur : histogramme (classes, échelle log, couleur),
densité (KDE), boîte à moustaches, violon, ECDF, barres (tri, %, horizontal), camembert (≤ 8 modalités
conseillé), Pareto, série temporelle (pas de temps, agrégation, moyenne mobile), répartition temporelle,
heatmap calendaire, nuage de points (droite MCO, r), matrice de corrélation, pair plot (≤ 6 variables,
échantillon ≤ 2 000 lignes), comparaisons par groupe (IC 95 %). Export PNG (barre d'outils) et HTML.

## 8. Interprétations automatiques

Règles appliquées aux résultats : forte dispersion (CV > 1, variables positives), asymétrie (|skew| > 1),
queues épaisses, non-normalité, modalité dominante (≥ 80 %), forte cardinalité, couverture et
irrégularités temporelles, variables très incomplètes (≥ 30 %), atypiques, faible score qualité,
doublons, corrélations fortes. Les commentaires sont classés en **constats**, **interprétations**,
**hypothèses** et **recommandations** ; chacun indique la mesure source.
Le protocole `LLMProvider` permet d'ajouter un enrichissement par modèle de langage (non activé).

## 9. Nettoyage

Opérations sur une copie de travail : suppression des doublons (première/dernière/aucune occurrence),
manquants (suppression lignes/colonnes, moyenne, médiane, mode, valeur), conversion de type,
renommage, suppression de colonnes, normalisation du texte (espaces, casse, accents, chaînes vides),
remplacement de valeur, filtrage, outliers (écrêtage, mise à manquant, suppression ; IQR ou z-score),
nouvelles variables (formule sûre, classes par bornes ou quantiles, extraction de date).
Avant/après, historique horodaté, annulation (rejeu depuis l'original), réinitialisation,
export CSV / Excel / Parquet / JSON et historique JSON. Un interrupteur permet d'analyser les données
nettoyées ou d'origine.

## 10. Rapports

Sections : présentation, caractéristiques générales, variables, statistiques, valeurs manquantes,
qualité, corrélations, visualisations, interprétations, recommandations — toutes sélectionnables.

- **PDF** : couverture, sommaire paginé (et signets), synthèse exécutive, tableaux, graphiques, date.
- **HTML** : autonome et consultable hors ligne (Plotly embarqué), thème clair/sombre.
- **Excel** : une feuille par tableau de résultats. **CSV** : archive ZIP de tableaux. **JSON** : structure complète.
