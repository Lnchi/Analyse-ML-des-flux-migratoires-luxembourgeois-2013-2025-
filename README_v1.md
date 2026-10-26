# DIMM_BI — Analyse ML des flux migratoires luxembourgeois (2013–2025)

Projet : **DIMM_BI** — Direction de l'Immigration du Luxembourg
Données : **EUROSTAT**
Méthodes : EDA avancée · Tests de stationnarité · ARIMA/ARIMAX · Prophet ·
XGBoost · K-Means · CAH · Silhouette

Ce dépôt est la version **VS Code / production** du notebook Google Colab
d'origine (`notebooks/DIMM_analyse_ML.ipynb`, conservé pour référence). Le
code est identique sur le fond (mêmes formules, mêmes hypothèses, mêmes
commentaires méthodologiques) ; il a simplement été réorganisé en modules
Python réutilisables, testables et versionnables avec Git.

## Ce qui a changé par rapport à Colab

| Colab | VS Code |
|---|---|
| `drive.mount('/content/drive')` | Cache local dans `data/cache/`, chemin configurable via `.env` |
| Cellules exécutées dans l'ordre, variables globales | Fonctions organisées en modules (`src/`), appelées depuis `main.py` |
| `!pip install ...` dans une cellule | `requirements.txt` + environnement virtuel |
| Figures affichées inline uniquement | Figures affichées **et** sauvegardées dans `outputs/figures/` |
| Un seul fichier `.ipynb` | Projet structuré, prêt pour Git / revue de code |

Le code métier (nettoyage EUROSTAT, tests statistiques, modèles ARIMA/Prophet/XGBoost,
clustering K-Means/CAH) n'a **pas été modifié fonctionnellement** — seule
l'organisation a changé.

## Structure du projet

```
DIMM_BI_analyse/
├── main.py                     # point d'entrée : orchestre tout le pipeline
├── requirements.txt
├── .env.example                # à copier en .env
├── src/
│   ├── config.py                # chemins, palette, constantes, style matplotlib
│   ├── data/
│   │   ├── collect.py            # 1. collecte EUROSTAT (cache + retry)
│   │   └── process.py            # 1.2 nettoyage / restructuration
│   ├── eda/
│   │   ├── temporal.py           # 2.1/2.2 évolution + décomposition STL
│   │   ├── heatmap.py            # 2.3 heatmap nationalités × années
│   │   ├── comparative.py        # 2.4 comparaison LU vs BE/FR/DE
│   │   └── reason.py             # 2.5 distribution par motif de séjour
│   ├── forecasting/
│   │   ├── stationarity.py       # 3.1/3.2 ADF/KPSS + ACF/PACF
│   │   ├── arima_model.py        # 3.3 ARIMA + ARIMAX (variable exogène COVID)
│   │   ├── prophet_model.py      # 3.4 Prophet
│   │   ├── xgboost_model.py      # 3.5 XGBoost + feature engineering
│   │   └── compare.py            # 3.6 comparaison des modèles + prévisions
│   └── clustering/
│       ├── features.py           # 4.1 feature engineering par nationalité
│       ├── dimensionality.py     # 4.2 RobustScaler + PCA
│       ├── selection.py          # 4.3 sélection de k (Elbow/Silhouette/DB)
│       ├── kmeans.py             # 4.4 K-Means final + silhouette détaillée
│       ├── cah.py                # 4.5 Classification Hiérarchique Ascendante
│       └── interpretation.py     # 4.6 interprétation + profils radar
├── data/cache/                  # cache local des séries EUROSTAT (.parquet)
├── outputs/figures/              # figures générées (.png)
└── notebooks/                    # notebook Colab original, conservé pour référence
```

## Installation

```bash
# 1. Cloner / ouvrir le dossier dans VS Code

# 2. Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Copier la configuration :Crée le fichier de config personnel à partir du modèle
cp .env.example .env
```

### Extensions VS Code recommandées

- **Python** (Microsoft) — exécution, débogage
- **Pylance** — auto-complétion / vérification de types
- **Jupyter** — si vous voulez exécuter certaines cellules de façon interactive
  (chaque fonction de `src/` peut être appelée depuis une cellule `# %%`)

## Utilisation

```bash
# Pipeline complet (EDA + prévision + clustering)
python main.py

# EDA + clustering uniquement (sans les modèles de prévision, plus rapide)
python main.py --skip-forecast

# EDA + prévision uniquement (sans le clustering)
python main.py --skip-cluster
```

Les figures sont affichées à l'écran **et** sauvegardées automatiquement
dans `outputs/figures/` (désactivable via `SAVE_FIGURES=false` dans `.env`).

Pour une exploration cellule par cellule comme sur Colab, chaque fonction
de `src/` peut aussi être importée et appelée depuis un notebook Jupyter
local ou une cellule interactive VS Code (`# %%`) :

```python
from src.data.collect import collect_raw_data
from src.data.process import build_df_all

collect_raw_data()
df_all = build_df_all()
```

## Prochaine étape

Une fois cette base en place, la **contribution méthodologique originale**
(indice de stabilité migratoire, prévision par cluster, prévision
hiérarchique, détection automatique de ruptures — cf. section finale du
notebook) pourra être ajoutée sous forme de nouveaux modules dans
`src/forecasting/` et `src/clustering/`, sans toucher au reste du pipeline.
