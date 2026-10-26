"""
4.2 Réduction dimensionnelle : normalisation RobustScaler + PCA.

Objectif :
    Préparer les caractéristiques construites à l'étape 4.1 avant le clustering.
Deux étapes sont réalisées :
    1. Normalisation avec RobustScaler :Les variables n'ont pas toutes la même échelle et certaines présentent
       des valeurs extrêmes. RobustScaler utilise la médiane et l'écart
       interquartile, ce qui le rend moins sensible aux outliers que
       StandardScaler.

    2. Réduction dimensionnelle avec PCA :La PCA transforme les variables initiales en composantes principales
       qui résument une grande partie de la variabilité des profils
       migratoires.

RobustScaler est préféré à StandardScaler car les variables présentent
des valeurs extrêmes (accélération jusqu'à x24, covid_resilience jusqu'à 4.5)
et une distribution asymétrique de log_volume. (Branco et al., 2016).

La PCA permet ensuite de représenter les nationalités dans un espace
de dimension réduite et de faciliter l'interprétation des profils.

Limitation :
    La PCA est une méthode de réduction dimensionnelle linéaire.
    Elle est utilisée ici pour visualiser et résumer les profils avant le clustering.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

from src.config import PALETTE, savefig

# Variables exclues de la PCA
# Certaines variables ne doivent pas participer directement à la réduction dimensionnelle :
# - volume_total :mesure la taille des flux.
#   Cette information est déjà représentée par log_volume.
#   Conserver les deux variables donnerait un poids supplémentaire à la dimension "taille du flux".
# On le retire pour éviter que le clustering sépare seulement "grands volumes" vs "petits volumes".
# - share_othp :les quatre parts par motif vérifient :share_work + share_fami + share_educ + share_othp = 1
#   Une des quatre variables est donc déductible des trois
#   autres. share_othp est retirée afin d'éviter cette redondance.
# - is_crisis :variable contextuelle utilisée pour interpréter les clusters.
#   Elle ne participe pas à leur construction afin d'éviter que les groupes
#   soient artificiellement déterminés par le statut "pays en crise".


EXCLUDED_FEATURES = {"volume_total", "share_othp", "is_crisis"}


def scale_and_reduce(df_feat_clust: pd.DataFrame):
    """
    Normalise les caractéristiques avec RobustScaler puis applique une PCA.

    Parameters: df_feat_clust : pd.DataFrame
        Matrice construite à l'étape 4.1.
        Une ligne correspond à une nationalité et une colonne à une
        caractéristique du profil migratoire.

    Returns: dict Contient :
            - feat_cols : variables utilisées
            - scaler : objet RobustScaler 
            - X_scaled : matrice normalisée
            - pca : modèle PCA 
            - X_pca : coordonnées des nationalités dans l'espace PCA
            - cum_var : variance expliquée cumulée
            - n_comp_90 : nombre de composantes nécessaires pour atteindre 90 %
    """
    feat_cols = [c for c in df_feat_clust.columns if c not in EXCLUDED_FEATURES]
    # Sans normalisation, les variables présentant les plus grandes
    # amplitudes pourraient dominer la PCA.
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df_feat_clust[feat_cols])
    # Le nombre maximal de composantes est limité par :
    #   - le nombre de variables utilisées 
    #   - le nombre de nationalités - 1
    # La seconde contrainte vient du fait qu'avec n observations,
    # la matrice centrée ne peut avoir qu'au maximum n-1 dimensions indépendantes.
    pca = PCA(n_components=min(len(feat_cols), len(df_feat_clust) - 1))
    # Projection des nationalités dans l'espace des composantes principales.
    X_pca = pca.fit_transform(X_scaled)
    # explained_variance_ratio_ indique la proportion de variance expliquée
    # par chaque composante.
    # La somme cumulée permet de déterminer combien de composantes sont
    # nécessaires pour conserver au moins 90 % de l'information.
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_comp_90 = np.argmax(cum_var >= 0.90) + 1

    return {
        "feat_cols": feat_cols, "scaler": scaler, "X_scaled": X_scaled,
        "pca": pca, "X_pca": X_pca, "cum_var": cum_var, "n_comp_90": n_comp_90,
    }


def plot_pca(df_feat_clust: pd.DataFrame, reduction: dict):
    """
    Visualise les résultats de la PCA.
    Deux graphiques sont produits :
        1. Scree plot :
           montre la variance expliquée par chaque composante ainsi
           que la variance cumulée.
        2. Projection des nationalités :
           représente les nationalités selon les deux premières
           composantes principales.
    Les huit nationalités ayant les plus gros volumes sont annotées
    afin de faciliter l'interprétation du graphique.
    """
    # Récupération des résultats produits par scale_and_reduce().
    pca = reduction["pca"]
    X_pca = reduction["X_pca"]
    cum_var = reduction["cum_var"]
    n_comp_90 = reduction["n_comp_90"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # Variance expliquée individuellement par chaque composante.
    axes[0].bar(range(1, len(pca.explained_variance_ratio_) + 1),
                pca.explained_variance_ratio_ * 100,
                color=PALETTE["primary"], alpha=0.8, label="Variance individuelle")
    # Deuxième axe permettant d'afficher la variance cumulée.
    ax0b = axes[0].twinx()
    ax0b.plot(range(1, len(cum_var) + 1), cum_var * 100, color=PALETTE["accent1"],
              linewidth=2.5, marker="o", markersize=6, label="Variance cumulée")
    ax0b.axhline(90, color="gray", linestyle="--", linewidth=1, label="Seuil 90%")
    ax0b.set_ylabel("Variance cumulée (%)")
    axes[0].set_xlabel("Composante principale")
    axes[0].set_ylabel("Variance expliquée (%)")
    axes[0].set_title("Analyse en Composantes Principales")
    axes[0].annotate(f"{n_comp_90} CP\nexpliquent 90%",
                      xy=(n_comp_90, pca.explained_variance_ratio_[n_comp_90 - 1] * 100),
                      xytext=(n_comp_90 + 1.5, 12),
                      arrowprops=dict(arrowstyle="->", color="black"), fontsize=10)
    lines1, labels1 = axes[0].get_legend_handles_labels()
    lines2, labels2 = ax0b.get_legend_handles_labels()
    axes[0].legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)
    # log_volume est utilisé ici uniquement pour la visualisation.
    # Il a déjà participé à la PCA comme variable explicative.
    axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=df_feat_clust["log_volume"], cmap="Blues",
                     s=60, alpha=0.8, edgecolors="navy", linewidths=0.5)
    # Identification des huit nationalités ayant les volumes totaux
    # les plus importants.
    top_nations = df_feat_clust["volume_total"].nlargest(8).index
    for nat in top_nations:
        idx = df_feat_clust.index.get_loc(nat)
        axes[1].annotate(nat, (X_pca[idx, 0], X_pca[idx, 1]), fontsize=8, color="darkblue",
                          xytext=(3, 3), textcoords="offset points")
    # Les axes correspondent aux deux premières composantes.
    # Le pourcentage indique leur contribution individuelle à la variance.
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    axes[1].set_title("Projection PCA : Espace des nationalités")

    plt.suptitle("Réduction dimensionnelle : PCA sur features de clustering", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "fig9_pca.png")
    plt.show()
    # Part de variance expliquée par les deux premières composantes.
    print(f"\n   PC1 + PC2 expliquent {(pca.explained_variance_ratio_[:2].sum() * 100):.1f}% de la variance")
    print(f"   {n_comp_90} composantes suffisent pour 90% de la variance")
