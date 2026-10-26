"""
2.4 Analyse comparative : LU vs BE vs FR vs DE

Cette analyse permet de comparer l'évolution du nombre de permis
délivrés au Luxembourg avec celle des pays voisins :
    - LU : Luxembourg
    - BE : Belgique
    - FR : France
    - DE : Allemagne

Deux indicateurs sont utilisés :

1. Volumes absolus
   → nombre total de permis délivrés par pays et par année.
   → Le Luxembourg (double axe Y car LU est 10x-40x inférieur aux autres pays) ayant une population beaucoup plus faible que
     ses voisins, ses volumes sont naturellement beaucoup plus faibles.

2. Taux pour 1 000 habitants
   → permet de comparer les pays indépendamment de leur taille.
   → Un pays plus grand ne sera donc pas automatiquement favorisé
     simplement parce qu'il possède une population plus importante.
"""

import pandas as pd
import matplotlib.pyplot as plt

from src.config import COUNTRIES_NEIGHBORS, PALETTE, POP_2025, savefig


def build_comparison_data(df_all: pd.DataFrame) -> pd.DataFrame:
    """Agrège les permis par pays/année et calcule le taux pour 1 000 habitants."""
    # COUNTRIES_NEIGHBORS contient les quatre pays utilisés
    # dans l'analyse comparative : LU, BE, FR et DE.
    ts_compare = (
        df_all[df_all["geo"].isin(COUNTRIES_NEIGHBORS)]
        .groupby(["geo", "year"])["permits"]
        .sum()
        .reset_index()
    )
    # POP_2025 contient la population de référence pour chaque pays.
    # .map() utilise le code du pays présent dans la colonne "geo"
    ts_compare["pop"] = ts_compare["geo"].map(POP_2025)
    # la population est exprimée en milliers
    ts_compare["permits_per_1000"] = ts_compare["permits"] / ts_compare["pop"]
    return ts_compare


def plot_country_comparison(ts_compare: pd.DataFrame):
    """Trace la comparaison LU vs BE/FR/DE (volumes absolus + taux normalisé)."""
    colors_geo = {"LU": PALETTE["lux_red"], "BE": "#1E4D8C", "FR": "#003082", "DE": "#000000"}
    linewidth_geo = {"LU": 3.0, "BE": 1.8, "FR": 1.8, "DE": 1.8}
    # Le premier graphique présente les volumes absolus
    # Le second présente les taux normalisés pour 1 000 habitants
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Graphe gauche : volumes absolus, double axe Y (LU à droite)
    ax_left = axes[0]
    ax_right = axes[0].twinx()
    for geo, grp in ts_compare.groupby("geo"):
        # On trie les observations par année 
        grp = grp.sort_values("year")
        ax = ax_right if geo == "LU" else ax_left
        ax.plot(grp["year"], grp["permits"], label=geo, color=colors_geo[geo],
                linewidth=linewidth_geo[geo], marker="o", markersize=5)

    ax_left.set_ylabel("Nombre de permis (BE, FR, DE)")
    ax_right.set_ylabel("Nombre de permis (LU)", color=PALETTE["lux_red"])
    ax_right.tick_params(axis="y", labelcolor=PALETTE["lux_red"])
    # Formatage des nombres avec séparateurs de milliers.
    # Exemple : 12500 devient 12,500
    ax_left.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax_right.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    # La zone 2019.5 - 2020.5 permet de mettre visuellement
    # en évidence l'année 2020 et le choc lié à la pandémie
    ax_left.axvspan(2019.5, 2020.5, alpha=0.08, color="red")
    ax_left.set_xlabel("Année")
    ax_left.set_title("Permis délivrés")

    lines_l, labels_l = ax_left.get_legend_handles_labels()
    lines_r, labels_r = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_l + lines_r, labels_l + labels_r, title="Pays")

    # Graphe droit : taux normalisé pour 1 000 habitants
    for geo, grp in ts_compare.groupby("geo"):
        grp = grp.sort_values("year")
        # Tracé du nombre de permis pour 1 000 habitants
        axes[1].plot(grp["year"], grp["permits_per_1000"], label=geo, color=colors_geo[geo],
                     linewidth=linewidth_geo[geo], marker="o", markersize=5)

    axes[1].set_title("Permis pour 1 000 habitants - Taux relatif")
    axes[1].set_ylabel("Permis / 1 000 habitants")
    axes[1].legend(title="Pays")
    axes[1].set_xlabel("Année")
    axes[1].axvspan(2019.5, 2020.5, alpha=0.08, color="red")

    plt.suptitle(
        "Comparaison Luxembourg vs pays voisins\n",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    savefig(fig, "fig4_comparaison_pays.png")
    plt.show()
