"""
4.4 K-Means final + diagramme de silhouette détaillé.
Objectif :
    Appliquer le K-Means avec le nombre de clusters k sélectionné
    à l'étape précédente, puis évaluer la qualité de la partition obtenue.

Deux visualisations complémentaires sont produites :

    1. Diagramme de silhouette :
       permet d'évaluer la cohésion de chaque nationalité avec son cluster
       et sa séparation par rapport aux autres clusters.

    2. Projection PCA :
       permet de visualiser la répartition des nationalités dans l'espace
       réduit et d'identifier visuellement les différents groupes.

La silhouette moyenne est utilisée comme indicateur de la
qualité du clustering :
    - proche de 1  -> clusters bien séparés 
    - proche de 0  -> frontières entre clusters peu nettes 
    - négative    -> observation potentiellement mal affectée
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

from src.config import savefig


def fit_final_kmeans(df_feat_clust: pd.DataFrame, X_scaled, k_optimal: int) -> pd.DataFrame:
    """ Entraîne le K-Means définitif et ajoute les informations de clustering
    au DataFrame des caractéristiques:

    df_feat_clust : pd.DataFrame -> Matrice des caractéristiques par nationalité.
    X_scaled : array-like -> Variables standardisées/normalisées utilisées pour entraîner K-Means.
    k_optimal : int -> Nombre de clusters retenu lors de l'étape de sélection de k.

    Returns : 
    tuple
        - DataFrame avec le numéro de cluster et la silhouette
        - modèle K-Means entraîné
        - labels des clusters
        - valeurs individuelles de silhouette"""
    # K-Means++ améliore l'initialisation des centroïdes.
    # n_init=50 permet de tester plusieurs initialisations et de retenir
    # celle donnant la meilleure solution selon l'inertie.
    # max_iter=1000 laisse suffisamment d'itérations pour converger.
    km_final = KMeans(n_clusters=k_optimal, init="k-means++", n_init=50, max_iter=1000, random_state=42)
    # Attribution de chaque nationalité à son cluster.
    cluster_labels = km_final.fit_predict(X_scaled)
    df_feat_clust = df_feat_clust.copy()
    # Ajout du numéro du cluster à chaque nationalité.
    df_feat_clust["cluster"] = cluster_labels
    # Pour chaque nationalité, silhouette_samples mesure :
    #   - sa proximité avec les autres nationalités de son cluster 
    #   - sa distance par rapport au cluster voisin le plus proche.
    #
    # Cela permet d'identifier les nationalités bien intégrées
    # à leur cluster et celles dont l'affectation est moins claire.
    sil_vals = silhouette_samples(X_scaled, cluster_labels)
    df_feat_clust["silhouette"] = sil_vals

    return df_feat_clust, km_final, cluster_labels, sil_vals


def plot_silhouette_and_pca(df_feat_clust: pd.DataFrame, X_scaled, X_pca, km_final,
                             cluster_labels, sil_vals, k_optimal: int, pca):
    """Produit deux visualisations complémentaires du clustering final :
        1. Diagramme de silhouette par cluster.
        2. Projection PCA des nationalités colorées selon leur cluster.
    Returns: Silhouette moyenne du clustering."""

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    # Position verticale de départ du premier cluster.
    y_lower = 10
    # Palette utilisée pour différencier visuellement les clusters.
    cluster_colors = [plt.cm.Set2(i / k_optimal) for i in range(k_optimal)]

    for i in range(k_optimal):
        # Récupération des valeurs de silhouette des nationalités
        # appartenant au cluster i.
        #
        # Le tri permet d'obtenir la forme classique du diagramme
        # de silhouette.
        cluster_sil = np.sort(sil_vals[cluster_labels == i])
        size = len(cluster_sil)
        # Position verticale de fin du cluster courant.
        y_upper = y_lower + size
        axes[0].fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil,
                               facecolor=cluster_colors[i], alpha=0.85, edgecolor="white", linewidth=0.5)
        axes[0].text(-0.05, y_lower + size / 2, f"C{i}", fontsize=9, color=cluster_colors[i], fontweight="bold")
        # Espace entre deux clusters afin de faciliter la lecture
        y_lower = y_upper + 5
    # Silhouette moyenne
    mean_sil = silhouette_score(X_scaled, cluster_labels)
    # Ligne verticale représentant la silhouette moyenne.
    axes[0].axvline(mean_sil, color="red", linestyle="--", linewidth=1.5,
                     label=f"Silhouette moyen = {mean_sil:.3f}")
    axes[0].set_xlabel("Valeur de silhouette")
    axes[0].set_ylabel("Nationalités (groupées par cluster)")
    axes[0].set_title(f"Diagramme de silhouette : k={k_optimal}")
    axes[0].legend(loc="lower right")
    # Une silhouette peut théoriquement être comprise entre -1 et 1.
    axes[0].set_xlim([-0.15, 1.0])
    # Chaque point représente une nationalité.
    # La couleur indique le cluster auquel elle appartient.
    axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap="Set2",
                     s=80, alpha=0.9, edgecolors="black", linewidths=0.5)
    # Projection des centroïdes dans l'espace PCA
    # Les centroïdes K-Means sont exprimés dans l'espace X_scaled.
    # On les transforme donc dans l'espace PCA afin de pouvoir
    # les afficher sur la même figure que les nationalités.
    centroids_pca = pca.transform(km_final.cluster_centers_)
    axes[1].scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker="*", s=400, c="black",
                     zorder=10, label="Centroïdes")
    for nat in df_feat_clust["volume_total"].nlargest(10).index:
        idx = df_feat_clust.index.get_loc(nat)
        axes[1].annotate(nat, (X_pca[idx, 0], X_pca[idx, 1]), fontsize=8,
                          xytext=(3, 3), textcoords="offset points")
    # on indique également la variance expliquée par chaque axe.
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    axes[1].set_title("Projection PCA par cluster")
    axes[1].legend()

    plt.suptitle("K-Means final : validation par silhouette", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "fig11_silhouette_kmeans.png")
    plt.show()

    return mean_sil
