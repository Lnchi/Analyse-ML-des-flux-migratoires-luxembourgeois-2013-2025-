"""
2.5 Distribution par motif de séjour (WORK / FAMI / EDUC / OTHP).
Objectif :
    - mesurer le nombre de permis délivrés au Luxembourg pour chaque motif ;
    - suivre l'évolution de ces motifs au cours du temps ;
    - calculer la part (%) de chaque motif dans le total annuel.
"""

import pandas as pd
import matplotlib.pyplot as plt

from src.config import REASON_LABELS, savefig


def build_reason_data(df_all: pd.DataFrame) -> pd.DataFrame:
    """Agrège les permis LU par année et motif, calcule les parts annuelles (%)

        La fonction retourne un DataFrame contenant :
        - year        : année
        - reason_cat  : catégorie du motif
        - permits     : nombre de permis pour le motif
        - reason_label: libellé du motif
        - total_year  : nombre total de permis tous motifs confondus
                        pour l'année
        - share       : part du motif dans le total annuel (%)

    """
    df_reason = (
        df_all[df_all["geo"] == "LU"]
        .groupby(["year", "reason_cat"])["permits"]
        .sum()
        .reset_index()
    )
    # REASON_LABELS est défini dans src.config et permet
    # d'obtenir un libellé
    df_reason["reason_label"] = df_reason["reason_cat"].map(REASON_LABELS)
    # Calcul du total de permis pour chaque année
    # transform("sum") conserve le même nombre total sur
    # toutes les lignes appartenant à une même année
    df_reason["total_year"] = df_reason.groupby("year")["permits"].transform("sum")
    # Calcul de la part annuelle de chaque motif
    df_reason["share"] = df_reason["permits"] / df_reason["total_year"] * 100
    return df_reason


def plot_reason_distribution(df_reason: pd.DataFrame):
    """Trace la répartition par motif (parts % en barres empilées + volumes en aires)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # Transformation du DataFrame pour avoir :
    # - les années en lignes
    # - les motifs en colonnes
    # - la part (%) comme valeurs
    #
    # Les valeurs manquantes sont remplacées par 0.
    pivot_share = df_reason.pivot(index="year", columns="reason_label", values="share").fillna(0)
    # Création d'un graphique en barres empilées
    # Chaque couleur représente un motif de séjour
    pivot_share.plot(kind="bar", stacked=True, ax=axes[0], colormap="tab10", alpha=0.85, width=0.75)
    axes[0].set_title("Répartition par motif de séjour (% annuel)")
    axes[0].set_ylabel("Part (%)")
    axes[0].set_xlabel("Année")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].tick_params(axis="x", rotation=45)

    pivot_abs = df_reason.pivot(index="year", columns="reason_label", values="permits").fillna(0)
    # Création d'un graphique en aires empilées pour visualiser l'évolution du volume total
    # et la contribution de chaque motif.
    pivot_abs.plot(kind="area", stacked=True, ax=axes[1], colormap="tab10", alpha=0.7)
    axes[1].set_title("Volume absolu par motif de séjour")
    axes[1].set_ylabel("Nombre de permis")
    axes[1].set_xlabel("Année")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    axes[1].legend(loc="upper left", fontsize=8)

    plt.suptitle(
        "Analyse par motif de séjour - Luxembourg 2013-2025\n",
        fontsize=13, fontweight="bold",
    )
    # Ajustement automatique de la disposition des graphiques
    # afin d'éviter les chevauchements
    plt.tight_layout()
    savefig(fig, "fig5_repartition_motifs.png")
    plt.show()

    print("\nPart moyenne par motif sur 2013-2025 :")
    print(df_reason.groupby("reason_label")["share"].mean().round(1).sort_values(ascending=False).to_string())
