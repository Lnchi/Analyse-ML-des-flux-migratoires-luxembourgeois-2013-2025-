"""
4.6 Interprétation des clusters de profils migratoires + visualisation radar.
Objectif :
    Décrire les caractéristiques moyennes de chaque cluster afin de
    faciliter leur interprétation d'un point de vue statistique et métier.

Le radar permet de comparer visuellement les profils relatifs des clusters.
Les variables sont normalisées uniquement pour la représentation graphique.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import savefig

RADAR_FEATURES = ["share_work", "share_fami", "share_educ", "cagr", "covid_resilience"]
RADAR_LABELS = ["Travail", "Famille", "Éducation", "CAGR", "Résilience\nCOVID"]


def print_cluster_interpretation(df_feat_clust: pd.DataFrame):
    """Affiche un résumé statistique de chaque cluster (nationalités, volume, CAGR, parts...)."""
    print("  INTERPRÉTATION DES CLUSTERS DE PROFILS MIGRATOIRES\n")
    for c in sorted(df_feat_clust["cluster"].unique()):
        grp = df_feat_clust[df_feat_clust["cluster"] == c]
        print(f"  CLUSTER {c}  :  {len(grp)} nationalités\n")
        print(f"  Nationalités : {', '.join(grp.index.tolist())}")
        print(f"  Volume moyen total   : {grp['volume_total'].mean():,.0f} permis")
        print(f"  CAGR moyen           : {grp['cagr'].mean() * 100:.1f}%/an")
        print(f"  Part travail         : {grp['share_work'].mean() * 100:.1f}%")
        print(f"  Part famille         : {grp['share_fami'].mean() * 100:.1f}%")
        print(f"  Part éducation       : {grp['share_educ'].mean() * 100:.1f}%")
        print(f"  Résilience COVID     : {grp['covid_resilience'].mean():.2f}")
        print(f"  Silhouette moyen     : {grp['silhouette'].mean():.3f}\n")


def plot_radar_profiles(df_feat_clust: pd.DataFrame, k_optimal: int):
    """Dessine le profil radar moyen de chaque cluster.

    Les variables sont normalisées entre 0 et 1 uniquement pour la
    visualisation afin de pouvoir comparer des indicateurs exprimés
    dans des échelles différentes. Les données originales utilisées
    pour le clustering ne sont pas modifiées."""
     # Calcul du profil moyen de chaque cluster.
    # Chaque ligne représente un cluster et chaque colonne
    # une caractéristique migratoire.
    cluster_profiles_radar = df_feat_clust.groupby("cluster")[RADAR_FEATURES].mean()
    n = len(RADAR_FEATURES)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, k_optimal, figsize=(4 * k_optimal, 5), subplot_kw=dict(polar=True))
    if k_optimal == 1:
        axes = [axes]

    for idx, (c, row) in enumerate(cluster_profiles_radar.iterrows()):
        values = []
        for feat in RADAR_FEATURES:
            # Normalisation min-max de chaque variable uniquement pour le radar.
            # Cela permet de ramener les indicateurs sur une échelle commune [0,1].
            # cette normalisation est uniquement graphique.
            vmin = df_feat_clust[feat].min()
            vmax = df_feat_clust[feat].max()
            values.append((row[feat] - vmin) / (vmax - vmin + 1e-9))
        # On répète la première valeur afin de fermer le polygone du radar.
        values += values[:1]

        ax = axes[idx]
        ax.fill(angles, values, alpha=0.25, color=plt.cm.Set2(idx / k_optimal))
        ax.plot(angles, values, linewidth=2, color=plt.cm.Set2(idx / k_optimal))
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(RADAR_LABELS, size=9)
        ax.set_yticklabels([])
        # Nombre de nationalités appartenant au cluster.
        n_members = (df_feat_clust["cluster"] == c).sum()
        ax.set_title(f"Cluster {c}\n({n_members} nationalités)", size=11, fontweight="bold", pad=15)

    plt.suptitle("Profils radar des clusters migratoires pour Luxembourg",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    savefig(fig, "fig13_radar_clusters.png")
    plt.show()
