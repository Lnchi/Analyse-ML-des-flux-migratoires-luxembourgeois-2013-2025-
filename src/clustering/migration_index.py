"""
4.7 Analyse quantitative : Indice composite de profils migratoires (IPM)

Le problème est que chaque nationalité est décrite par plusieurs variables (CAGR, covid_resilience, ...)
On doit regarder plusieurs chiffres pour dire laquelle des nationalités est la plus dynamique 
L'idée est de créer un seul nombre pour savoir qui présente les profils migratoires les plus dynamiques

Objectif : synthétiser en un score unique, par nationalité, la dynamique
migratoire observée (croissance, résilience aux chocs, régularité, poids
du motif familial vs travail). Contrairement à une lecture séparée des
variables, l'indice permet un classement, lisible pour un décideur, et une
pondération apprise (et non arbitraire) des variables.

Méthodologie :
1. Les variables candidates sont normalisées en [0, 1] (Min-Max), car
   l'indice doit rester interprétable (contrairement aux z-scores du
   clustering, on veut ici un score borné et positif).
2. Les poids sont dérivés des loadings de la première composante principale (PCA)
   calculée sur les mêmes variables : PC1 capture la plus grande part de
   variance commune entre indicateurs, donc les variables qui pèsent le
   plus dans PC1 reçoivent le poids le plus élevé dans l'indice.
3. L'indice final est une moyenne pondérée comprise entre 0 et 1.
   Une valeur proche de 0 correspond aux profils les moins dynamiques
   observés dans l'échantillon, tandis qu'une valeur proche de 1
   correspond aux profils les plus dynamiques.

Cet indice est distinct du clustering (K-Means/CAH) : le clustering
regroupe des profils qualitativement différents, l'indice les classe sur
un axe unique d'intensité/dynamisme, complémentaire du groupe d'appartenance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from src.config import PALETTE, IPM_EXCLUDED_ARTIFACTS, savefig

# Variables candidates à l'indice. On privilégie des variables déjà
# construites dans src/clustering/features.py (build_clustering_features) qui décrivent la dynamique,
# en excluant les variables de composition par motif (share_*) qui décrivent la nature de la migration pas son intensité.
INDEX_FEATURES = ["cagr", "covid_resilience", "acceleration", "cv_temporal"]

# Le coefficient de variation (cv_temporal) est un indicateur d'instabilité :
# plus il est élevé, moins le profil est stable. On l'inverse donc avant
# de le combiner avec les autres variables (toutes orientées -> plus = plus dynamique).
INVERTED_FEATURES = {"cv_temporal"}


def compute_migration_index(df_feat_clust: pd.DataFrame, index_features: list[str] | None = None) -> pd.DataFrame:
    """
    Calcule l'Indice composite de Profil Migratoire (IPM) pour chaque nationalité.

    Args:
        df_feat_clust: matrice de features issue de build_clustering_features()
        index_features: variables à combiner (par défaut INDEX_FEATURES)

    Returns:
        DataFrame indexé par nationalité avec les colonnes :
        - les variables normalisées (suffixe _norm)
        - 'ipm' : l'indice composite final, dans [0, 1]
        - 'ipm_rank' : rang (1 = profil le plus dynamique)
    """
    index_features = index_features or INDEX_FEATURES
    # Exclusion des nationalités dont le profil est un artefact réglementaire
    # (ex. UK/Brexit) et non une dynamique migratoire réelle (cf. config.py).
    excluded = [c for c in IPM_EXCLUDED_ARTIFACTS if c in df_feat_clust.index]
    if excluded:
        print(f" Nationalités exclues de l'IPM (artefacts réglementaires) : {excluded}")
    df_feat_clust = df_feat_clust.drop(index=excluded, errors="ignore")
    
    df = df_feat_clust[index_features].copy()

    # Inversion des variables d'instabilité avant normalisation
    for feat in INVERTED_FEATURES:
        if feat in df.columns:
            df[feat] = -df[feat]

    # Normalisation Min-Max en [0, 1] : nécessaire pour que l'indice soit
    # directement interprétable (0 = minimum observé, 1 = maximum observé) 
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(df)
    # Conversion en DataFrame afin de conserver
    # les noms des variables et les nationalités.
    df_norm = pd.DataFrame(X_norm, index=df.index, columns=[f"{c}_norm" for c in df.columns])

    # Pondération apprise : loadings absolus de PC1 (normalisés pour sommer à 1)
    # Une seule composante est retenue car l'objectif est d'identifier
    # un axe synthétique représentant la dynamique migratoire commune.
    pca = PCA(n_components=1) 
    pca.fit(X_norm)

    # Les loadings indiquent la contribution de chaque variable à PC1.
    # Comme toutes les variables ont été orientées dans le même sens
    # ("plus = plus dynamique"), on vérifie ici que les loadings ont
    # effectivement une orientation cohérente.
    loadings = pca.components_[0]

    # Si les variables contribuent dans des directions différentes,
    # l'interprétation d'un indice unique devient moins robuste.
    if np.any(loadings < 0) and np.any(loadings > 0):
        print(
            " Attention : les loadings de PC1 ont des signes opposés. "
            "L'interprétation de l'IPM comme axe unique de dynamisme "
            "doit être considérée avec prudence."
        )

    # On utilise les valeurs absolues des loadings pour transformer
    # leur importance en poids positifs.
    raw_weights = np.abs(loadings)
    # Normalisation des loadings afin que les poids soient positifs
    # et que leur somme soit égale à 1.
    # Ils peuvent ainsi être interprétés comme des contributions relatives
    # des différentes dimensions à l'indice.
    weights = raw_weights / raw_weights.sum() # La somme vaut exactement 1. Pour obtenir un pourcentage 

    print(" Pondération de l'indice (issue de PC1) :")
    # Affichage du poids de chaque variable.
    for feat, w in zip(index_features, weights):
        print(f"   {feat:<20} poids = {w:.3f}")
    # Pourcentage de variance expliqué par la première composante.
    # Plus cette valeur est élevée,
    # plus l'utilisation d'un indice unique est justifiée.
    print(f"   Variance expliquée par PC1 : {pca.explained_variance_ratio_[0] * 100:.1f}%")

    # On repart du DataFrame complet afin de conserver
    # toutes les autres variables calculées précédemment.
    df_result = df_feat_clust.copy()
    # Ajout des variables normalisées.
    for col in df_norm.columns:
        df_result[col] = df_norm[col]

    # Calcul de l'indice composite :
    # chaque nationalité obtient une moyenne pondérée de ses variables
    # normalisées.
    #
    # Le produit matriciel (@) permet de calculer automatiquement :
    # IPM = somme(variable_normalisée × poids)
    #
    # Comme les variables sont comprises entre 0 et 1 et que les poids
    # sont positifs et somment à 1, l'IPM est lui-même compris entre 0 et 1.
    df_result["ipm"] = X_norm @ weights 
    # Classement décroissant :
    # rang 1 = nationalité présentant le profil le plus dynamique
    # selon les dimensions retenues dans l'IPM.
    df_result["ipm_rank"] = (
        df_result["ipm"]
        .rank(ascending=False, method="min") # method="min" pour avoir un comportement explicite en cas d'égalité.
        .astype(int)
    )
    # Tri du meilleur vers le moins dynamique.
    return df_result.sort_values("ipm", ascending=False)


def plot_migration_index(df_ipm: pd.DataFrame, top_n: int = 20):
    """Visualise les top_n nationalités selon leur Indice de Profil Migratoire.

    Le graphique permet d'identifier les nationalités ayant
    les profils les plus dynamiques selon les dimensions retenues dans l'IPM.

    Les nationalités associées à un contexte de crise sont distinguées
    visuellement afin de faciliter l'interprétation, mais cet indicateur
    n'entre pas dans le calcul de l'IPM."""
    top = df_ipm.head(top_n).sort_values("ipm")

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.35)))
    # L'indicateur de crise n'est PAS utilisé pour calculer l'IPM.
    # Il sert uniquement ici à contextualiser visuellement les résultats.
    # Les nationalités associées à une crise sont affichées différemment.
    colors = [PALETTE["accent1"] if v == 1 else PALETTE["primary"] for v in top.get("is_crisis", pd.Series(0, index=top.index))]
    ax.barh(top.index, top["ipm"], color=colors, alpha=0.85)
    ax.set_xlabel("Indice de Profil Migratoire (IPM)")
    ax.set_title(f"Classement des {top_n} nationalités par IPM")
    ax.set_xlim(0, 1)

    plt.tight_layout()
    savefig(fig, "fig14_indice_profil_migratoire.png")
    plt.show()

    print("\n Top 10 nationalités (IPM) :")
    print(df_ipm[["ipm", "ipm_rank", "cagr", "covid_resilience", "acceleration"]].head(10).round(3).to_string())
