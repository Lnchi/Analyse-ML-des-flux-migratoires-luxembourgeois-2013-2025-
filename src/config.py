"""
Configuration du projet DIMM_BI : Analyse ML des flux migratoires

Regroupe :
- les chemins (cache, figures) 
- le style matplotlib 
- la palette de couleurs EUROSTAT / Luxembourg
- les constantes métier (pays, années, mapping des séries EUROSTAT)

"""

from pathlib import Path
import os

import matplotlib.pyplot as plt
from dotenv import load_dotenv


# Chargement des variables d'environnement (.env)

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

CACHE_DIR = ROOT_DIR / os.getenv("CACHE_DIR", "data/cache/")
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
# True = forcer le re-téléchargement (Si les données EUROSTAT sont mises à jour (annuellement))
FORCE_DOWNLOAD = os.getenv("FORCE_DOWNLOAD", "false").lower() == "true"
SAVE_FIGURES = os.getenv("SAVE_FIGURES", "true").lower() == "true"

# Constantes métier

COUNTRIES_NEIGHBORS = ["LU", "BE", "FR", "DE"] # (Luxembourg, Belgique, France et Allemagne)
YEARS = list(range(2013, 2025))

# Séries EUROSTAT collectées : catégorie métier (motif de séjour)
SERIES_MAP = {
    "migr_resocc": "WORK",   # Activités rémunérées
    "migr_resfas": "FAMI",   # Regroupement familial
    "migr_resedu": "EDUC",   # Éducation
    "migr_resoth": "OTHP",   # Autres motifs
}

REASON_LABELS = {
    "WORK": "Activités rémunérées",
    "FAMI": "Regroupement familial",
    "EDUC": "Éducation",
    "OTHP": "Autres motifs",
}

# Population de référence 2025 (en milliers d'habitants) : source EUROSTAT,
# utilisée pour normaliser les comparaisons LU vs BE/FR/DE.
POP_2025 = {
    "LU": 343.0,
    "BE": 11870.0,
    "FR": 68400.0,
    "DE": 83600.0,
}

# Pays en crise humanitaire documentée (UNHCR / Fragile States Index)
# utilisés comme feature binaire dans le clustering.
CRISIS_COUNTRIES = [
    "SY", "AF", "IQ", "YE", "ER", "SD", "SS", "SO",
    "CF", "CD", "ML", "NE", "TD", "ET", "MM",
]
# Nationalités exclues du calcul de l'Indice de Profil Migratoire (IPM) car
# leurs indicateurs (covid_resilience, acceleration) sont des artefacts
# réglementaires et non des dynamiques migratoires interprétables.
# UK : les ressortissants britanniques n'avaient besoin d'aucun permis de
# séjour avant le Brexit (effectif jusqu'en 2021) -> volumes ~nuls avant 2019,
# ce qui déclenche les valeurs de repli (1.0) dans build_clustering_features()
# plutôt qu'une vraie mesure de résilience ou d'accélération.
IPM_EXCLUDED_ARTIFACTS = ["UK"]

# Filtre à appliquer sur la dimension 'reason' interne à chaque série EUROSTAT
# (distincte de reason_cat = catégorie métier WORK/FAMI/EDUC/OTHP).
# Chaque série migr_res* contient une sous-dimension 'reason' avec des codes
# qui ne sont pas homogènes d'une série à l'autre. On ne garde que la
# modalité "parent" (agrégat), les sous-codes étant redondants avec elle.
#
# Valeurs confirmées par vérification sur le portail EUROSTAT (cases cochées par
# défaut pour chaque dataset) :
#   - migr_resocc : [EMP] coché, sous-codes EMP_BLUE/EMP_HSW/EMP_RSE/
#                   EMP_WRK_SEAS/EMP_OTH décochés
#   - migr_resfas : [TOTAL] + les 4 sous-codes (FAM/EDUC/EMP/OTH) tous cochés
#                   -> on ne garde que TOTAL pour éviter le double comptage
#   - migr_resedu : [EDUC] coché, sous-codes EDUC_STU/EDUC_OTH décochés
#   - migr_resoth : [OTH] coché, sous-codes RFG_SPROT_HUM/RFG_SPROT/HUM/RES/
#                   UAM/VICT_TRFF_HMN/NSP décochés
SERIES_REASON_FILTER = {
    "migr_resocc": "EMP",
    "migr_resfas": "TOTAL",
    "migr_resedu": "EDUC",
    "migr_resoth": "OTH",
}

# Codes EUROSTAT à exclure (agrégats, codes non nationaux)
AGG_CODES = [
    "TOTAL", "EXT_EU28", "EXT_EU27_2020", "EU28", "EU27_2020",
    "EEA", "EFTA", "UNK", "OTH", "RNC", "STLS", "UK_OCT",
]

# Palette couleurs EUROSTAT-inspired
PALETTE = {
    "primary":   "#003399",   # bleu EU
    "secondary": "#FFCC00",   # jaune EU
    "accent1":   "#E84142",
    "accent2":   "#2ECC71",
    "accent3":   "#F39C12",
    "neutral":   "#6C757D",
    "lux_red":   "#EF3340",   # rouge Luxembourg
    "lux_blue":  "#009FCA",
    "accent4":   "#9B59B6",   # violet pour XGBoost
}

# STYLE PUBLICATION-READY
def apply_publication_style() -> None:
    """Applique le style matplotlib 'publication-ready' utilisé dans tout le projet."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "axes.facecolor": "#F8F9FA",
        "axes.grid": True,
        "grid.alpha": 0.4,
        "grid.linestyle": "--",
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
    })


def savefig(fig, name: str, **kwargs) -> None:
    """
    Sauvegarde une figure dans outputs/figures/ si SAVE_FIGURES=true.
    
    """
    if not SAVE_FIGURES:
        return
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight", **kwargs)
    print(f"   Figure sauvegardée : {path}")
