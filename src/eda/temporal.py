"""
2.1 Évolution temporelle - Luxembourg total
2.2 Décomposition additive classique

 1. Évolution temporelle du nombre total de permis de séjour
       délivrés au Luxembourg entre 2013 et 2025 :
       - évolution annuelle des volumes 
       - taux de croissance annuel (YoY) 
       - moyenne mobile sur 3 ans 
       - CAGR entre 2016 et 2024 
       - identification du pic de permis 
       - mesure de l'impact du COVID-19 en 2020

       2. Décomposition de la série temporelle :
       - série originale 
       - tendance 
       - composante cyclique/saisonnière 
       - résidu 
       - calcul d'un ratio signal/bruit

La période 2013-2025 correspond à la période retenue dans
l'analyse du projet. Les années 2009-2012 sont exclues en raison
d'une rupture méthodologique identifiée dans les données EUROSTAT

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

from src.config import PALETTE, savefig


def build_lu_timeseries(df_all: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construit la série annuelle totale de permis pour le Luxembourg."""
    df_lu = df_all[df_all["geo"] == "LU"].copy()
    # le nombre TOTAL de permis délivrés au Luxembourg.
    # le volume total de permis délivrés au Luxembourg
    # pour chaque année.
    ts_lu = df_lu.groupby("year")["permits"].sum().reset_index()
    ts_lu.columns = ["year", "total_permits"]

    # Taux de croissance annuel (YoY)
    # # 'yoy_growth' indique si le nombre de permis a augmenté ou diminué d'une année sur l'autre.
    
    # pct_change() compare la valeur d'une année avec celle
    # de l'année précédente 
    # Le YoY permet donc de mesurer la progression ou la baisse
    # du nombre de permis d'une année sur l'autre.
    ts_lu["yoy_growth"] = ts_lu["total_permits"].pct_change() * 100

    # Moyenne mobile 3 ans (lisse la tendance générale)
    # calcul de la moyenne mobile (rolling_mean) permet de voir la tendance générale du nombre de permis.
    # rolling(3) : crée une "fenêtre glissante" de 3 périodes. Elle regarde la ligne actuelle, la précédente et la suivante.
    # mean() : On calcule la moyenne de ces 3 années.
    ts_lu["rolling_mean_3"] = ts_lu["total_permits"].rolling(3, center=True).mean()

    return df_lu, ts_lu


def plot_evolution(ts_lu: pd.DataFrame, annee_debut: int = 2016, annee_fin: int = 2024) -> float:
    """
    Trace l'évolution des permis (volume + moyenne mobile) et le taux de
    croissance annuel. Calcule le CAGR (Taux de Croissance Annuel Composé)
    entre annee_debut et annee_fin.

    2013 est exclu du calcul du CAGR car année de transition (séries
    incomplètes) -> on prend 2016 comme première année stable.
    """

    # On récupère le nombre total de permis correspondant
    # aux deux années utilisées pour calculer le CAGR.
    v_debut = ts_lu[ts_lu["year"] == annee_debut]["total_permits"].values[0]
    v_fin = ts_lu[ts_lu["year"] == annee_fin]["total_permits"].values[0]
    n_years = annee_fin - annee_debut

    # Il représente le taux de croissance annuel moyen
    # nécessaire pour passer de la valeur initiale à la
    # valeur finale sur la période considérée.
    cagr = (v_fin / v_debut) ** (1 / n_years) - 1
    print(f"CAGR {annee_debut}-{annee_fin} : {cagr:.2%}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Panel 1 : évolution du nombre de permis
    ax1 = axes[0]
    ax1.fill_between(ts_lu["year"], ts_lu["total_permits"], alpha=0.15, color=PALETTE["primary"])
    ax1.plot(ts_lu["year"], ts_lu["total_permits"], color=PALETTE["primary"], linewidth=2.5,
              marker="o", markersize=7, label="Permis délivrés")

    ax1.plot(ts_lu["year"], ts_lu["rolling_mean_3"], color=PALETTE["accent3"], linewidth=2,
              linestyle="--", label="Moyenne mobile 3 ans")
    # Cette zone verticale permet d'identifier visuellement
    # l'année 2020, correspondant au choc lié à la pandémie
    # de COVID-19.
    ax1.axvspan(2019.5, 2020.5, alpha=0.12, color="red", label="Choc COVID-19")
    ax1.annotate(f"CAGR = {cagr:.1%}\n({annee_debut}-{annee_fin})",
                 xy=(2022, ts_lu["total_permits"].max() * 0.92),
                 fontsize=10, color=PALETTE["primary"],
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8EFF7", alpha=0.8))
    ax1.set_ylabel("Nombre de permis délivrés")
    ax1.set_title("Évolution des permis de séjour au Luxembourg (2013-2025)")
    ax1.legend(loc="upper left")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Panel 2 : taux de croissance YoY
    ax2 = axes[1]
    colors = [PALETTE["accent2"] if v >= 0 else PALETTE["accent1"] for v in ts_lu["yoy_growth"].fillna(0)]
    ax2.bar(ts_lu["year"], ts_lu["yoy_growth"], color=colors, alpha=0.8, width=0.6)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Variation annuelle (%)")
    ax2.set_title("Taux de croissance annuel (YoY)")
    ax2.set_xlabel("Année")

    plt.tight_layout()
    savefig(fig, "fig1_evolution_permis.png")
    plt.show()

    print(f"\n CAGR {annee_debut}-{annee_fin} : {cagr:.2%}")
    # Recherche de l'année correspondant au nombre maximal
    # de permis délivrés.
    print(f"   Pic maximal : {ts_lu.loc[ts_lu['total_permits'].idxmax(), 'year']}")
    # Récupération du taux de croissance de 2020 afin de
    # quantifier l'impact du COVID-19 sur les volumes.
    print(f"   Impact COVID : {ts_lu[ts_lu['year'] == 2020]['yoy_growth'].values[0]:.1f}%")
    return cagr


def plot_stl_decomposition(ts_lu: pd.DataFrame, period: int = 3) -> float:
    """
    Décomposition additive de la série (tendance / composante cyclique / résidu).

    period=3 : compromis méthodologique pour estimer une composante cyclique
    malgré la faible longueur de la série (les titres de séjour ont des
    durées de validité variables : 1 à 5 ans). Cette composante ne doit donc PAS être interprétée comme
    la preuve d'un véritable cycle migratoire de trois ans.
    """

    # seasonal_decompose() attend une série dont l'index
    # représente le temps.
    #
    # L'année devient donc l'index de la série.
    ts_indexed = ts_lu.set_index("year")["total_permits"]
    # Le modèle additif suppose :
    #
    #     Série = Tendance
    #           + Saison/Cycle
    #           + Résidu
    #
    # period=3 signifie qu'on recherche une structure
    # répétitive associée à une période de 3 observations.
    #
    # extrapolate_trend="freq" permet d'estimer la tendance
    # jusqu'aux extrémités de la série.
    decomp = seasonal_decompose(ts_indexed, model="additive", period=period, extrapolate_trend="freq")

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    components = [
        (ts_indexed, "Série originale", PALETTE["primary"]),
        (decomp.trend, "Tendance", PALETTE["accent3"]),
        (decomp.seasonal, "Composante cyclique", PALETTE["accent2"]),
        (decomp.resid, "Résidu", PALETTE["accent1"]),
    ]
    for ax, (series, title, color) in zip(axes, components):
        ax.plot(series.index, series.values, color=color, linewidth=2)
        if "Résidu" in title:
            ax.bar(series.index, series.values, color=color, alpha=0.5, width=0.5)
            # Ligne zéro :
            # elle permet d'identifier les années où le résidu
            # est positif ou négatif.
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_ylabel(title, fontsize=9)
        ax.fill_between(series.index, series.values, alpha=0.1, color=color)

    axes[0].set_title("Décomposition de la série temporelle",
                       fontsize=13, fontweight="bold")
    axes[-1].set_xlabel("Année")

    # Ratio Signal/Bruit : std écart-type(tendance) / std(résidu). >1 = série stable et prévisible.
    snr = decomp.trend.std() / decomp.resid.dropna().std()
    fig.text(0.99, 0.02, f"Signal/Noise ratio = {snr:.2f}", ha="right", fontsize=9, color=PALETTE["neutral"])

    plt.tight_layout()
    savefig(fig, "fig2_decomposition_STL.png")
    plt.show()
    print(f"   Ratio Signal/Bruit : {snr:.2f}")
    return snr
    # Ce ratio ne signifie pas à lui seul que la série est
    # "stable et prévisible".
