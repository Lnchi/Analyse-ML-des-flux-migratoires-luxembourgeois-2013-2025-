"""
4.1 clustering des profils migratoires:
Objectif :
    Construire une matrice de caractéristiques permettant de regrouper
    les nationalités selon leurs profils migratoires au Luxembourg.

Hypothèse H2 :
    Les nationalités présentes au Luxembourg forment des groupes structurellement distincts en termes de :
        - volume de demandes 
        - motif de demande 
        - évolution temporelle 
        - réaction au choc COVID-19 
        - stabilité des flux 
        - dynamique récente

Une ligne de la matrice finale correspond à une nationalité.

Features construites par nationalité :

    - volume_total :nombre total de permis sur l'ensemble de la période.

    - log_volume :transformation logarithmique du volume total afin de réduire
        l'effet des très grandes nationalités.

    - cagr :taux de croissance annuel composé entre la première et la dernière
        observation disponible.

    - share_work / share_fami / share_educ / share_othp :part du volume total associée à chaque grand motif.

    - covid_resilience :ratio du volume 2021 sur le volume 2019.
        Une valeur proche de 1 indique un retour vers le niveau pré-COVID,
        tandis qu'une valeur inférieure à 1 indique que le niveau de 2021
        reste inférieur à celui de 2019.

    - cv_temporal :coefficient de variation des volumes annuels.
        Une valeur faible indique une évolution relativement stable,
        tandis qu'une valeur élevée indique une forte variabilité.

    - acceleration :rapport entre le volume annuel moyen récent (2019-2025)
        et le volume annuel moyen de la première partie de la période
        (2013-2017).

    - is_crisis :indicateur binaire indiquant si la nationalité appartient à la
        liste des pays identifiés comme étant concernés par une crise
        humanitaire documentée -> elle sert àinterpréter les clusters, plutôt qu'à démontrer qu'un cluster existe

Limitation :
    Les caractéristiques sont calculées sur une série relativement courte.
    Les résultats du clustering sont interprétés comme des
    profils structurels et non comme une classification définitive.
"""

import numpy as np
import pandas as pd

from src.config import CRISIS_COUNTRIES


def build_clustering_features(df_lu: pd.DataFrame, min_avg_permits: int = 10) -> pd.DataFrame:
    """
    Construit une matrice de caractéristiques avec une ligne par nationalité.
    Les données initiales contiennent plusieurs lignes par nationalité,
    année et motif. Elles sont agrégées afin d'obtenir un profil synthétique
    de chaque nationalité.

    Parameters:
    df_lu : pd.DataFrame -> Données du Luxembourg uniquement (geo == 'LU').

    min_avg_permits : int
        Seuil minimal de volume annuel moyen.
        Les nationalités dont le volume moyen est inférieur à ce seuil
        sont exclues afin d'éviter des indicateurs instables sur de très
        petits effectifs.

    Returns: pd.DataFrame -> Matrice de clustering :
            lignes    = nationalités
            colonnes  = caractéristiques migratoires
    """
    feats = []
    # Si la variable "duration" existe, on conserve uniquement TOTAL
    if "duration" in df_lu.columns:
        df_lu = df_lu[df_lu["duration"] == "TOTAL"]
    # Agrégation du nombre de permis par année.
    for citizen, grp in df_lu.groupby("citizen"):
        total_by_year = grp.groupby("year")["permits"].sum()
        # Les très petits effectifs peuvent produire :
        #   - des CAGR élevés 
        #   - des coefficients de variation instables 
        #   - des ratios COVID peu représentatifs
        # On exclut donc les nationalités dont le volume annuel moyen
        # est inférieur au seuil défini.
        if total_by_year.mean() < min_avg_permits:
            continue

        total = total_by_year.sum()

        # (CAGR)
        # On recherche la première observation strictement positive.
        # Cela évite un problème mathématique lorsque la première année
        # contient un volume nul.
        positive_values = total_by_year[total_by_year > 0]
        if len(positive_values) >= 2:
            first_year = positive_values.index[0]
            last_year = total_by_year.index[-1]

            first = positive_values.iloc[0]
            last = total_by_year.iloc[-1]

            # Nombre réel d'années entre les deux observations.
            n_years = last_year - first_year
            if n_years > 0 and first > 0:
                cagr_c = ((last / first) ** (1 / n_years) - 1)
            else:
                cagr_c = 0.0

        else:
            # Pas suffisamment d'observations positives pour calculer
            # un taux de croissance fiable.
            cagr_c = 0.0

        # Parts par motif
        # On calcule la part de chaque motif dans le volume total
        # de la nationalité sur l'ensemble de la période.
        total_reason = grp.groupby("reason_cat")["permits"].sum()
        share = {
            r: total_reason.get(r, 0) / total if total > 0 else 0
            for r in ["WORK", "FAMI", "EDUC", "OTHP"]
        }

        # Résilience COVID 
        # Comparaison du niveau de 2021 avec celui de 2019.
        # 2019 = dernière année pré-COVID utilisée comme référence.
        # 2021 = première année complète après le choc de 2020.
        #
        # Exemple :
        #   0.80 -> le volume 2021 représente 80 % du niveau 2019.
        #   1.00 -> retour au niveau de 2019.
        #   1.20 -> niveau 2021 supérieur de 20 % à 2019.
        v2019 = total_by_year.get(2019, np.nan)
        v2021 = total_by_year.get(2021, np.nan)

        if (pd.notna(v2019) and v2019 > 0 and pd.notna(v2021)):
            covid_res = v2021 / v2019
        else:
            # Valeur neutre lorsque le ratio ne peut pas être calculé.
            covid_res = 1.0

        # Concentration temporelle (coefficient de variation) : faible = évolution régulière
        # Il permet de mesurer la variabilité relative des volumes
        # indépendamment de leur niveau absolu.
        mean_volume = total_by_year.mean()

        if mean_volume > 0:
            cv = total_by_year.std() / mean_volume
        else:
            cv = 0.0

        # Accélération récente (2019-2025 vs 2013-2017)
        # Une valeur :
        #   > 1 -> volume récent supérieur au volume initial 
        #   = 1 -> niveau relativement stable 
        #   < 1 -> volume récent inférieur.
        early_values = total_by_year[total_by_year.index <= 2017]
        late_values = total_by_year[total_by_year.index >= 2019]
        v_early = early_values.mean()
        v_late = late_values.mean()
        if pd.notna(v_early) and v_early > 0:
            accel = v_late / v_early
        else:
            accel = 1.0
        # Construction du profil national 
        feats.append({
            "citizen": citizen,
            "volume_total": total,
            "log_volume": np.log1p(total),
            "cagr": cagr_c,
            "share_work": share["WORK"],
            "share_fami": share["FAMI"],
            "share_educ": share["EDUC"],
            "share_othp": share["OTHP"],
            "covid_resilience": covid_res,
            # Stabilité / volatilité temporelle.
            "cv_temporal": cv,
            # Dynamique récente par rapport au début de période.
            "acceleration": accel,
            # Contexte externe : pays associé à une crise.
            # is_crisis : variable contextuelle externe utilisée uniquement pour interpréter
            "is_crisis": 1 if citizen in CRISIS_COUNTRIES else 0,
        })

    df_feat_clust = pd.DataFrame(feats).set_index("citizen")

    print(f" Matrice de features : {df_feat_clust.shape}")
    print(f" Nationalités : {df_feat_clust.shape[0]}")
    print(f" Features     : {df_feat_clust.shape[1]}")
    print("\n Statistiques descriptives :")
    print(df_feat_clust.describe().round(3))

    return df_feat_clust
