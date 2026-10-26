"""
4.3 Sélection du nombre optimal de clusters k.
Objectif :
    Déterminer le nombre de groupes permettant de représenter au mieux
    les profils migratoires des nationalités.

2 critères complémentaires sont utilisés :
    - Elbow / méthode du coude :
        mesure l'inertie intra-cluster.
        Plus l'inertie est faible, plus les observations d'un même cluster
        sont proches.
        On recherche cependant le "coude" de la courbe plutôt que le minimum
        absolu, car l'inertie diminue mécaniquement lorsque k augmente.

    - Silhouette Score :
        mesure simultanément la cohésion des observations au sein de leur
        cluster et leur séparation vis-à-vis des autres clusters.
        Plus le score est élevé, meilleure est la séparation des groupes.

Les 2 critères sont utilisés de manière complémentaire afin d'éviter
de choisir le nombre de clusters sur un seul indicateur.

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

from src.config import PALETTE, savefig


def select_optimal_k(X_scaled, k_max: int = 11):
    """Calcule les indicateurs de qualité du clustering pour différents nombres
    de clusters k.

    Parameters:X_scaled :
        Matrice des observations après normalisation.
        Une ligne correspond à une nationalité.

    k_max : int
        Nombre maximal de clusters testé.

    Returns:dict
        Contient les valeurs de k testées ainsi que :
            - inertias : inertie intra-cluster
            - sil_scores : scores de silhouette
            """
    # On commence à k=2 car un seul cluster ne permet pas d'évaluer
    # la séparation entre groupes.
    # On limite également k au nombre d'observations disponibles.
    # Il faut au minimum une observation dans chaque cluster.
    k_range = range(2, min(k_max, X_scaled.shape[0] - 1))
    inertias, sil_scores, db_scores = [], [], []

    for k in k_range:
        # KMeans cherche à minimiser la distance entre les observations
        # et le centre de leur cluster.
        #
        # k-means++ améliore l'initialisation des centroïdes par rapport
        # à une initialisation aléatoire simple.
        #
        # n_init=20 signifie que l'algorithme est exécuté 20 fois avec
        # différentes initialisations et conserve la meilleure solution.
        km = KMeans(n_clusters=k, init="k-means++", n_init=20, max_iter=500, random_state=42)
        # Apprentissage du modèle et attribution d'un cluster à chaque
        # nationalité.
        labels = km.fit_predict(X_scaled)
        # L'inertie correspond à la somme des distances quadratiques entre
        # chaque observation et le centroïde de son cluster.
        # Elle diminue nécessairement lorsque le nombre de clusters augmente.
        # On recherche donc le point à partir duquel l'ajout de clusters
        # apporte relativement peu d'amélioration : le "coude".
        inertias.append(km.inertia_)
        # Le score compare :
        #   - la proximité d'une observation avec son propre cluster ;
        #   - sa distance avec les clusters voisins.
        #
        # Une valeur proche de 1 indique une bonne séparation.
        # Une valeur proche de 0 indique des clusters qui se chevauchent.
        # Une valeur négative peut signaler une mauvaise affectation.
        #
        # On cherche donc à MAXIMISER ce score.
        sil_scores.append(silhouette_score(X_scaled, labels))

    return {"k_range": list(k_range), "inertias": inertias, "sil_scores": sil_scores}


def plot_k_selection(selection: dict):
    """Trace les 2 courbes de sélection de k (Elbow, Silhouette, Davies)."""
    k_range = selection["k_range"]
    # Récupération des résultats calculés précédemment.
    inertias, sil_scores = selection["inertias"], selection["sil_scores"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    # Graphique 1 : méthode du coude
    axes[0].plot(k_range, inertias, "o-", color=PALETTE["primary"], linewidth=2, markersize=8)
    axes[0].set_xlabel("Nombre de clusters k")
    axes[0].set_ylabel("Inertie intra-cluster")
    axes[0].set_title("Méthode du coude (Elbow)")
    # Graphique 2 : Silhouette Score
    # Pour la silhouette, on recherche le score maximal.
    best_k_sil = k_range[int(np.argmax(sil_scores))]
    axes[1].plot(k_range, sil_scores, "s-", color=PALETTE["accent2"], linewidth=2, markersize=8)
    axes[1].axvline(best_k_sil, color=PALETTE["accent1"], linestyle="--", linewidth=1.5,
                     label=f"k optimal = {best_k_sil}")
    axes[1].set_xlabel("Nombre de clusters k")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Coefficient de silhouette")
    axes[1].legend()

    

    plt.suptitle("Sélection du nombre optimal de clusters k", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "fig10_selection_k.png")
    plt.show()

    print(f"\n   k optimal (Silhouette) : {best_k_sil}  [score = {max(sil_scores):.3f}]")
    return best_k_sil
