"""
2.3 Heatmap nationalités x années (Top 20)

Deux heatmaps complémentaires :
  - volumes absolus : permet d'identifier les nationalités qui représentent les
     plus grands volumes de permis de séjour délivrés au Luxembourg.
  - Z-score par nationalité : quelles années sont anormalement fortes/faibles
    relativement à l'historique propre à chaque nationalité (neutralise les
    différences d'échelle entre populations).
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from src.config import savefig


def plot_nationality_heatmap(df_lu: pd.DataFrame, top_n: int = 20):
    """Trace les heatmaps volumes absolus + Z-score pour le top_n des nationalités."""

    # On additionne les permis sur l'ensemble de la période
    # pour chaque nationalité.
    top_nationalities = (
        df_lu.groupby("citizen")["permits"].sum().nlargest(top_n).index.tolist()
    )

    # on agrège les permis par :
    #     nationalité + année
    # Les lignes correspondent aux nationalités.
    # Les colonnes correspondent aux années.

    pivot_heat = (
        df_lu[df_lu["citizen"].isin(top_nationalities)]
        .groupby(["citizen", "year"])["permits"]
        .sum()
        .unstack("year")
        .fillna(0)
    )
    # Le Z-score est calculé séparément pour chaque nationalité,
    # sur l'ensemble de ses années.
    scaler_heat = StandardScaler()
    # on transpose pivot_heat :
    #
    #     pivot_heat.T
    #
    # pour que les nationalités deviennent les variables.
    #
    # Après la standardisation, on transpose à nouveau (.T)
    # afin de retrouver la même structure :
    #
    #     lignes    = nationalités
    #     colonnes  = années
    pivot_zscore = pd.DataFrame(
        scaler_heat.fit_transform(pivot_heat.T).T,
        index=pivot_heat.index,
        columns=pivot_heat.columns,
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Cette première heatmap montre le nombre réel de permis
    # délivrés pour chaque nationalité et chaque année.
    #
    # Elle permet d'identifier :
    #     - les nationalités dominantes en volume 
    #     - les années de forte activité 
    #     - les éventuelles baisses importantes
    sns.heatmap(pivot_heat, ax=axes[0], cmap="Blues", fmt=".0f", annot=True,
                annot_kws={"size": 8}, linewidths=0.3, cbar_kws={"label": "Nombre de permis"})
    axes[0].set_title(f"Top {top_n} nationalités")
    axes[0].set_xlabel("Année")
    axes[0].set_ylabel("Nationalité")
    # Elle montre l'écart de chaque année par rapport à la moyenne
    # historique de la nationalité concernée.
    sns.heatmap(pivot_zscore, ax=axes[1], cmap="RdYlGn", center=0, fmt=".1f", annot=True,
                annot_kws={"size": 8}, linewidths=0.3,
                cbar_kws={"label": "Z-score (écart à la moyenne)"})
    axes[1].set_title("Tendances relatives (Z-score) - Pic/Creux par nationalité")
    axes[1].set_xlabel("Année")
    axes[1].set_ylabel("")

    plt.suptitle("Analyse matricielle des flux migratoires au Luxembourg 2013-2025",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    savefig(fig, "fig3_heatmap_nationalites.png")
    plt.show()

    return top_nationalities, pivot_heat, pivot_zscore
