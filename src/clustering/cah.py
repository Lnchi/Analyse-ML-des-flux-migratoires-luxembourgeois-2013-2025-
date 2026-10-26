"""
4.5 Classification Hiérarchique Ascendante (CAH) - Validation et interprétation.
Objectif :Utiliser une seconde méthode de clustering afin de vérifier si les groupes
    identifiés par K-Means sont cohérents.

La CAH construit progressivement une hiérarchie de groupes en fusionnant
les nationalités les plus proches.

La méthode de Ward est utilisée car elle cherche à minimiser
l'augmentation de la variance intra-cluster à chaque fusion.

Deux éléments sont présentés :
    1. Le dendrogramme :
       permet de visualiser la hiérarchie des rapprochements entre
       nationalités et d'observer une coupure correspondant au nombre
       de clusters retenu avec K-Means.
    2. La heatmap des profils moyens :
       permet d'interpréter les caractéristiques des différents clusters.

    La CAH est utilisée ici comme méthode de validation complémentaire.
    Les clusters finaux restent ceux obtenus avec K-Means.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

from src.config import PALETTE, savefig


def plot_cah(df_feat_clust: pd.DataFrame, X_scaled, feat_cols: list, k_optimal: int):
    """    Réalise une CAH avec la méthode de Ward et produit :
        - un dendrogramme permettant d'observer la structure hiérarchique 
        - une heatmap présentant le profil moyen de chaque cluster.
    Parameters:
    df_feat_clust : pd.DataFrame ->Matrice des caractéristiques enrichie avec les clusters K-Means.

    X_scaled : array-like ->Variables normalisées utilisées précédemment pour le K-Means.
        La CAH est appliquée sur ces mêmes données afin que les deux
        méthodes soient comparables.

    feat_cols : list -> Liste des variables utilisées pour le clustering.

    k_optimal : int ->Nombre de clusters retenu lors de l'étape K-Means.

    Returns : 
 linkage_matrix : ndarray
        Matrice décrivant les fusions successives de la CAH.

    cluster_profiles : pd.DataFrame
        Moyennes des caractéristiques pour chaque cluster."""
    # La méthode de Ward fusionne progressivement les observations
    # en cherchant à minimiser l'augmentation de la variance
    # intra-cluster à chaque étape.
    #
    # X_scaled est utilisé ici exactement comme pour K-Means :
    # les variables sont donc sur des échelles comparables.
    linkage_matrix = linkage(X_scaled, method="ward", metric="euclidean")

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [1.2, 1]})
    # Pour obtenir k_optimal groupes, on coupe le dendrogramme
    # au niveau de la dernière fusion nécessaire pour passer
    # de k_optimal clusters à k_optimal - 1 clusters.
    #
    # Exemple :
    #   k = 3 -> seuil correspondant à la fusion qui ferait passer
    #            de 3 groupes à 2.
    threshold = linkage_matrix[-(k_optimal - 1), 2]
    dendrogram(
        linkage_matrix, labels=df_feat_clust.index.tolist(), ax=axes[0],
        color_threshold=threshold, above_threshold_color="gray",
        leaf_rotation=90, leaf_font_size=7,
    )
    # Ligne horizontale indiquant le niveau de coupure choisi.
    # Elle permet de visualiser les k groupes retenus.
    axes[0].axhline(threshold, color=PALETTE["accent1"], linestyle="--", linewidth=2,
                     label=f"Coupure k={k_optimal}")
    axes[0].set_title(f"Dendrogramme CAH (Ward)\nCoupure à {k_optimal} clusters",
                       fontsize=15, fontweight="bold", pad=10)
    axes[0].set_ylabel("Distance de Ward", fontsize=13)
    axes[0].tick_params(axis="x", labelsize=7)
    axes[0].legend()
    # On calcule la moyenne de chaque variable pour chaque cluster
    # K-Means.
    #
    # Cette étape permet de passer de la simple identification des groupes
    # à leur interprétation métier.
    cluster_profiles = df_feat_clust.groupby("cluster")[feat_cols].mean()
    # Transposition :
    #   lignes    = caractéristiques
    #   colonnes  = clusters
    sns.heatmap(cluster_profiles.T, ax=axes[1], annot=True, fmt=".2f")
    axes[1].set_title("Profil moyen des clusters\n", fontsize=15, fontweight="bold", pad=10)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="both", labelsize=11)

    fig.suptitle("Classification Hiérarchique Ascendante (CAH)",
                 fontsize=18, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    savefig(fig, "fig12_cah_dendrogramme.png")
    plt.show()

    return linkage_matrix, cluster_profiles
