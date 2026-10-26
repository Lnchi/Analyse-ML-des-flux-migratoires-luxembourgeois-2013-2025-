# DIMM_BI : Analyse ML des flux migratoires luxembourgeois (2013–2025)

Projet : **DIMM_BI** - Direction de l'Immigration du Luxembourg
Source : API REST EUROSTAT via package eurostat (Cazzaniga, 2022)
Séries collectées :

migr_resocc : Permis de séjour délivrés, par motif et nationalité, 
(elle couvre spécifiquement les permis à des fins professionnelles).
migr_resfas : regroupement familial
migr_resedu : éducation
migr_resoth : autres motifs

Méthodes : EDA avancée, Tests de stationnarité, ARIMA/ARIMAX, Prophet,
XGBoost, K-Means, CAH, Silhouette

## Remarque :
Saut de 2008 à 2013 car avant 2013 les données sont incomplètes
certaines séries n'existaient pas encore. On filtre sur 2013–2025.

- Cache local dans `data/cache/`, chemin configurable via `.env` 
- Fonctions organisées en modules (`src/`), appelées depuis `main.py` 
- `requirements.txt` + environnement virtuel 
- Figures affichées et sauvegardées dans `outputs/figures/` 

## Structure du projet

DIMM_BI_analyse/
    main.py                     # point d'entrée : orchestre tout le pipeline
    requirements.txt
    .env.example                # à copier en .env
    src/
        config.py                # chemins, palette, constantes, style matplotlib
        data/
            collect.py            # 1. collecte EUROSTAT (cache + retry)
            process.py            # 1.2 nettoyage / restructuration
        eda/
            temporal.py           # 2.1/2.2 évolution + décomposition STL
            heatmap.py            # 2.3 heatmap nationalités × années
            comparative.py        # 2.4 comparaison LU vs BE/FR/DE
            reason.py             # 2.5 distribution par motif de séjour
        forecasting/
            stationarity.py       # 3.1/3.2 ADF/KPSS + ACF/PACF
            arima_model.py        # 3.3 ARIMA + ARIMAX (variable exogène COVID)
            prophet_model.py      # 3.4 Prophet
            xgboost_model.py      # 3.5 XGBoost + feature engineering
            compare.py            # 3.6 comparaison des modèles + prévisions
        clustering/
            features.py           # 4.1 feature engineering par nationalité
            dimensionality.py     # 4.2 RobustScaler + PCA
            selection.py          # 4.3 sélection de k (Elbow/Silhouette/DB)
            kmeans.py             # 4.4 K-Means + silhouette détaillée
            cah.py                # 4.5 Classification Hiérarchique Ascendante
            interpretation.py     # 4.6 interprétation + profils radar
    data/cache/                  # cache local des séries EUROSTAT (.parquet)
    outputs/figures/              # figures générées (.png)
    notebooks/                    # notebook Colab, 


## Installation

```bash
# Créer un environnement virtuel
python -m venv .venv
source .venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

# Copier la configuration
cp .env.example .env
```
### Extensions 

- **Python** (Microsoft) : exécution, débogage
- **Pylance** : auto-complétion / vérification de types
- **Jupyter** : pour exécuter certaines cellules de façon interactive

## Utilisation

```bash
# Pipeline complet (EDA + prévision + clustering)
python main.py

# EDA + clustering uniquement (sans les modèles de prévision)
python main.py --skip-forecast

# EDA + prévision uniquement (sans le clustering)
python main.py --skip-cluster
```

Les figures sont affichées à l'écran et sauvegardées automatiquement
dans `outputs/figures/` (désactivable via `SAVE_FIGURES=false` dans `.env`).

Pour une exploration cellule par cellule comme sur Colab, chaque fonction
de `src/` peut être importée et appelée depuis une cellule interactive VS Code (`# %%`) :

```python
from src.data.collect import collect_raw_data
from src.data.process import build_df_all

collect_raw_data()
df_all = build_df_all()
```

## Contributions 

Indice de Profil Migratoire (IPM) : `src/clustering/migration_index.py` -> 
Score composite [0,1] par nationalité, pondération apprise via PCA (pas de poids arbitraires) 
Détection automatique de ruptures : `src/forecasting/breakpoints.py` -> 
CUSUM + test de Chow : détecte les ruptures structurelles au lieu de les imposer a priori (remplace 
l'hypothèse COVID=2020 de l'ARIMAX) 
Prévision hiérarchique par cluster : `src/forecasting/hierarchical.py` -> 
Prévoit chaque cluster séparément puis réconcilie (bottom-up) avec le total national 

## Perspectives

- **Validation bootstrap des clusters** et **ARIMA+XGBoost hybride** :
  Fragilité statistique accrue avec n=13 observations
  temporelles / clusters de petite taille 
- **Analyse des transitions de profils dans le temps** : nécessite un
  reclustering par fenêtre temporelle, ce qui réduit encore le nombre
  d'observations par période

## Interprétation

  ### Indice de Profil Migratoire (IPM) 

  La première composante principale explique 50,3 % de la variance totale, 
  ce qui indique qu'une dimension commune résume une part importante de l'information portée par 
  les quatre indicateurs retenus. L'utilisation de cette composante pour construire les pondérations de 
  l'Indice de Profil Migratoire est donc statistiquement justifiée.

  D'après PCA croissance ≈ 42 % est la variable la plus informative.
  La résilience Covid apporte peu d'information supplémentaire. 
  La plupart des nationalités qui ont une forte croissance ont également une bonne résilience. 
  Donc cette variable est assez redondante. La PCA lui donne un petit poids.

  ### Limites méthodologiques : les résultats de la prévision hiérarchique 

  La comparaison entre prévision directe du total national et prévision par sommation des clusters (bottom-up) 
  révèle un écart croissant sur l'horizon 2026-2028 (de -15% à -13%). 

  Une validation walk-forward (fenêtre glissante expansive, 5 replis sur les années 2021-2025) 
  a été menée pour arbitrer entre les deux approches : elle indique un léger avantage pour 
  l'approche bottom-up (RMSE = 6 925 contre 7 158 pour l'approche directe, soit -3,3%), 
  l'approche bottom-up l'emportant sur 3 des 5 replis testés. 
  
  Les erreurs les plus importantes apparaissent lors des premiers replis (2021-2022). 
  (respectivement 11 639 et 11 103 permis d'écart en 2021, soit environ 55% de la valeur observée)
  Cela s'explique par le fait que le modèle ne dispose alors que d'un historique très limité après 
  le choc provoqué par la pandémie de COVID-19. Dans ces conditions, les deux approches rencontrent des 
  difficultés similaires pour anticiper la reprise des flux migratoires. 

  À partir des replis les plus récents (2024-2025), les performances s'améliorent sensiblement, 
  le modèle bénéficiant d'un historique d'observations plus long. La méthode Bottom-Up devient alors légèrement 
  plus précise que la prévision directe.
  Compte tenu du faible nombre d'années disponibles (13 observations) et des 5 replis de validation seulement, 
  ces résultats ne démontrent pas de manière définitive la supériorité de l'approche Bottom-Up, 
  mais ils montrent qu'elle constitue une alternative pertinente lorsque les groupes présentent 
  des dynamiques migratoires différentes.