"""
1-Collecte automatisée des données EUROSTAT

Source : API REST EUROSTAT via le package 'eurostat (Cazzaniga, 2022)
Séries collectées :
  - migr_resocc : permis de séjour délivrés pour motif professionnel
  - migr_resfas : regroupement familial
  - migr_resedu : éducation
  - migr_resoth : autres motifs

Remarque : migr_reslong (statuts de résident longue durée) mesure un stock
et non un flux annuel. Elle n'est donc pas concaténée avec les autres séries
dans df_all, pour ne pas fausser les analyses.

"""

import time

import eurostat
import pandas as pd

from src.config import CACHE_DIR, FORCE_DOWNLOAD, SERIES_MAP


def download_with_retry(series_code: str, max_retries: int = 3, wait_seconds: int = 30):
    """
    Télécharge une série EUROSTAT avec retry automatique en cas de timeout.

    Args:
        series_code: code de la série EUROSTAT (ex: 'migr_resocc')
        max_retries: nombre de tentatives maximum
        wait_seconds: attente entre chaque tentative
    """
    for attempt in range(1, max_retries + 1):
        try:
            print(f"   Tentative {attempt}/{max_retries}...")
            return eurostat.get_data_df(series_code)
        except Exception as e:
            print(f"   Échec tentative {attempt} : {type(e).__name__}")
            if attempt < max_retries:
                print(f"   Attente {wait_seconds}s avant retry...")
                time.sleep(wait_seconds)
            else:
                print(f"   {series_code} indisponible après {max_retries} tentatives")
                return None


def collect_raw_data() -> pd.DataFrame:
    """
    Télécharge (ou charge depuis le cache local) toutes les séries de
    SERIES_MAP, ajoute la catégorie métier (reason_cat) et les concatène.

    Returns:
        raw_all: DataFrame brut concaténé, toutes séries confondues.
    """
    raw_frames = []

    for series_code, reason_cat in SERIES_MAP.items():
        cache_path = CACHE_DIR / f"{series_code}.parquet"
        # Cache existant : chargement direct
        if cache_path.exists() and not FORCE_DOWNLOAD:
            print(f"{series_code} : chargé depuis cache")
            df_tmp = pd.read_parquet(cache_path)
        # Pas de cache : téléchargement avec retry
        else:
            print(f"{series_code} ({reason_cat}) : téléchargement...")
            df_tmp = download_with_retry(series_code, max_retries=3, wait_seconds=30)

            if df_tmp is None:
                print(f"{series_code} ignoré : continue sans cette série")
                continue
            # Sauvegarde en cache
            df_tmp.to_parquet(cache_path, index=False)
            print(f"{df_tmp.shape[0]} lignes sauvegardées : {cache_path}")

        df_tmp["reason_cat"] = reason_cat
        raw_frames.append(df_tmp)
        print(f"   shape : {df_tmp.shape}")

    raw_all = pd.concat(raw_frames, ignore_index=True) if raw_frames else None
    print(f"\n Total brut : {raw_all.shape[0]} lignes")
    return raw_all


def get_series_df(series_code: str) -> pd.DataFrame:
    """Raccourci pour inspecter une série EUROSTAT (en EDA)."""
    return eurostat.get_data_df(series_code)


    # migr_resocc : First permits issued for remunerated activities by reason, length of validity and citizenship
    # [EMP] Employment reasons COCHER : 
    # [EMP_BLUE] EU Blue card
    # [EMP_HSW] Highly skilled workers
    # [EMP_RSE] Researchers
    # [EMP_WRK_SEAS] Seasonal workers
    # [EMP_OTH] Other employment reasons

    # migr_resfas : First permits by reason, age, sex and citizenship
    # [TOTAL] Total COCHER
    # [FAM] Family reasons COCHER
    # [EDUC] Education reasons COCHER
    # [EMP] Employment reasons COCHER
    # [OTH] Other reason COCHER 

    # migr_resedu : First permits issued for education reasons by reason, length of validity and citizenship
    # [EDUC] Education reasons COCHER : 
    # [EDUC_STU] Study
    # [EDUC_OTH] Other educational reasons
    # Duration : 
    # [TOTAL] Total
    # [M3-5] From 3 to 5 months
    # [M6-11] From 6 to 11 months
    # [M_GE12] 12 months or over

    # migr_resoth : First permits issued for other reasons by reason, length of validity and citizenship
    # [OTH] Other reason COCHER : 
    # [RFG_SPROT_HUM] International protection (refugee and subsidiary protection status) and humanitarian status
    # [RFG_SPROT] International protection status (refugee and subsidiary protection status)
    # [HUM] Humanitarian status
    # [RES] Residence
    # [UAM] Unaccompanied minors
    # [VICT_TRFF_HMN] Victims of trafficking in human beings
    # [NSP] Not specified


def inspect_raw_data(raw_all: pd.DataFrame, n_rows: int = 10) -> None:
    """
    Exploration des données après collecte :
      - taille totale du DataFrame fusionné
      - aperçu tabulé des premières lignes de chaque série (WORK/FAMI/EDUC/OTHP)
      - modalités des dimensions 'age', 'sex', 'duration' et 'reason'

    La vérification de 'reason' : chaque série contient une
    sous-dimension 'reason' avec des sous-catégories qui somment à un code
    "parent" (ex: pour migr_resocc : EMP_BLUE + EMP_HSW + EMP_RSE +
    EMP_WRK_SEAS + EMP_OTH = EMP). process_eurostat() filtre désormais
    explicitement sur ce code parent (SERIES_REASON_FILTER dans
    config.py), afin d'éviter un double comptage lors de l'agrégation.

    Args:
        raw_all: DataFrame brut retourné par collect_raw_data()
        n_rows: nombre de lignes affichées par aperçu
    """
    from tabulate import tabulate

    print(f"\n Total brut : {raw_all.shape[0]:,} lignes")
    print(tabulate(raw_all.head(n_rows), headers="keys", tablefmt="grid", showindex=False))

    series_to_inspect = [
        ("WORK", "migr_resocc"),
        ("FAMI", "migr_resfas"),
        ("EDUC", "migr_resedu"),
        ("OTHP", "migr_resoth"),
    ]

    for label, series_code in series_to_inspect:
        print(f"\n Échantillon série {label} ({series_code}) :")
        df_series = get_series_df(series_code)
        print(tabulate(df_series.head(n_rows), headers="keys", tablefmt="grid", showindex=False))
        for dim in ["age", "sex", "duration", "reason"]:
            if dim in df_series.columns:
                print(f"   Modalités '{dim}' : {df_series[dim].unique()}")
            else:
                print(f"   (pas de dimension '{dim}' dans cette série)")

    print(f"\n raw_all : {raw_all.shape[0]:,} lignes x {raw_all.shape[1]} colonnes")


def check_reason_hierarchy(series_code: str, candidate_total: str, geo: str = "LU") -> None:
    """
    Vérifie si candidate_total est bien l'agrégat des autres
    modalités de la dimension 'reason' pour une série donnée, la vérification 
    est intégrée dans build_df_all(), voir
    verify_reason_aggregate() dans process.py)

    Exemple d'utilisation :
        check_reason_hierarchy("migr_resocc", candidate_total="EMP")
    """
    df = get_series_df(series_code)
    if "reason" not in df.columns or "geo" not in df.columns:
        print(f"{series_code} : pas de dimension 'reason' ou 'geo' à vérifier.")
        return

    year_cols = [c for c in df.columns if str(c).isdigit()]
    test_year = year_cols[-1]

    df_geo = df[df["geo"] == geo]
    other_codes = [c for c in df_geo["reason"].unique() if c != candidate_total]

    total_val = pd.to_numeric(
        df_geo[df_geo["reason"] == candidate_total][test_year], errors="coerce"
    ).sum()
    sum_others = pd.to_numeric(
        df_geo[df_geo["reason"].isin(other_codes)][test_year], errors="coerce"
    ).sum()

    print(f"\n Vérification {series_code} (geo={geo}, année={test_year}) :")
    print(f"   Valeur '{candidate_total}'          : {total_val:,.0f}")
    print(f"   Somme des autres modalités : {sum_others:,.0f}  {sorted(other_codes)}")

    if total_val == 0 and sum_others == 0:
        print(" Aucune donnée pour ce couple geo/année, réessayer avec un autre geo ou une autre année.")
    elif abs(total_val - sum_others) <= max(1, 0.02 * total_val):
        print(f"   '{candidate_total}' est bien l'agrégat (écart < 2%).")
    else:
        print(f"   Ecart IMPORTANT : '{candidate_total}' n'est PAS l'agrégat de ces modalités.")