"""
3.6 Comparaison des modèles ARIMA / ARIMAX / Prophet / XGBoost
et prévisions finales 2026-2028.

Les métriques utilisées pour la comparaison finale sont calculées
sur le TEST FINAL : 2023-2025.

Les années 2021-2022 ont uniquement servi à la validation
walk-forward et à la sélection des paramètres.

L'objectif est de comparer les performances des modèles sur les mêmes années de test.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import PALETTE, savefig


def build_comparison_table(
    mae_arima,
    mae_arimax,
    
    mae_xgb,
    rmse_arima,
    rmse_arimax,
    
    rmse_xgb,
    mape_arima,
    mape_arimax,
    
    mape_xgb,
) -> pd.DataFrame:
    """
    Construit le tableau comparatif des quatre modèles.

    Les métriques sont calculées sur le TEST FINAL 2023-2025 :

        - MAE  : erreur absolue moyenne
        - RMSE : racine de l'erreur quadratique moyenne
        - MAPE : erreur moyenne en pourcentage

    Plus les valeurs sont faibles, meilleur est le modèle.

    Le classement est effectué selon la MAPE.
    """

    # Tableau contenant les performances des quatre modèles.
    metrics_df = pd.DataFrame({
        "Modèle": [
            "ARIMA(auto)",
            "ARIMAX",
            "XGBoost",
        ],
        "MAE": [
            mae_arima,
            mae_arimax,
            
            mae_xgb,
        ],
        "RMSE": [
            rmse_arima,
            rmse_arimax,
            
            rmse_xgb,
        ],
        "MAPE(%)": [
            mape_arima,
            mape_arimax,
            
            mape_xgb,
        ],
    })

    # Classement des modèles selon leur MAPE.
    # Le modèle ayant la MAPE la plus faible obtient le rang 1.
    metrics_df["Rang (MAPE)"] = (metrics_df["MAPE(%)"].rank(method="min").astype(int))
    print("\n===== TABLEAU COMPARATIF DES MODÈLES =====")
    print(metrics_df.to_string(index=False))
    # Identification du modèle ayant la plus faible MAPE.
    best_model = metrics_df.loc[metrics_df["MAPE(%)"].idxmin(),"Modèle"]
    print(f"\nMeilleur modèle selon la MAPE : {best_model}")

    return metrics_df


def plot_final_forecast(
    series: pd.Series,
    model_arima,
    model_arimax,
    results_xgb: dict,
):
    """
    Visualisation des prévisions des 4 modèles :

        - ARIMA
        - ARIMAX
        
        - XGBoost

    La figure présente :
        - les observations réelles 2013-2025 ;
        - les prévisions sur le test 2023-2025 ;
        - les prévisions futures 2026-2028.

    Les années 2023-2025 permettent de comparer les modèles
    aux valeurs réellement observées.

    Les années 2026-2028 correspondent à l'horizon de prévision final.
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    series_years = np.array(series.index)
    ax.plot(
        series_years,
        series.values,
        "o-",
        color=PALETTE["primary"],
        linewidth=2.5,
        markersize=7,
        label="Observations réelles",
        zorder=5,
    )
    # Le modèle ARIMA est entraîné sur 2013-2022.
    # prévisions pour 2023-2028.
    fc_arima, _ = model_arima.predict(
        n_periods=6,
        return_conf_int=True,
        alpha=0.05,
    )
    years_forecast = np.array([2023, 2024, 2025, 2026, 2027, 2028])

    fc_arima = pd.Series(fc_arima.values, index=years_forecast) 

    ax.plot(
        years_forecast,
        fc_arima.values,
        "s--",
        color=PALETTE["accent1"],
        linewidth=2,
        markersize=7,
        label=f"ARIMA{model_arima.order}",
    )
    # Récupération des prévisions ARIMAX déjà calculées.
    fc_arimax = model_arimax["fc_future"]

    ax.plot(
        fc_arimax.index,
        fc_arimax.values,
        "D--",
        color=PALETTE["accent2"],
        linewidth=2,
        markersize=7,
        label="ARIMAX",
    )
    """ # On sélectionne uniquement les années 2023-2028 pour la prévision de Prophet
    fc_prophet = forecast_p[forecast_p["ds"].dt.year.isin(years_forecast)]

    ax.plot(
        fc_prophet["ds"].dt.year,
        fc_prophet["yhat"],
        "^--",
        color=PALETTE["accent3"],
        linewidth=2,
        markersize=7,
        label="Prophet",
    ) """
    # Prévisions XGBoost sur l'ensemble de la période 2023-2028 :
    # - 2023-2025 : prévisions du jeu de test
    # - 2026-2028 : prévisions futures récursives
    fc_xgb = results_xgb["fc_full"]
    ax.plot(
        fc_xgb.index,
        fc_xgb.values,
        "P--",
        color=PALETTE["accent4"],
        linewidth=2,
        markersize=7,
        label="XGBoost",
    )
    # Le modèle a été entraîné jusqu'à 2022.
    ax.axvline(
        2022.5,
        color="gray",
        linestyle=":",
        linewidth=1.5,
        label="Fin entraînement (2022)",
    )

    # Zone de test : 2023-2025.
    ax.axvspan(
        2022.5,
        2025.5,
        alpha=0.05,
        color="blue",
        label="Période test (2023-2025)",
    )

    # Zone de prévision future : 2026-2028.
    ax.axvspan(
        2025.5,
        2028.5,
        alpha=0.05,
        color="green",
        label="Horizon de prévision (2026-2028)",
    )
    ax.axvspan(
        2019.5,
        2020.5,
        alpha=0.08,
        color="red",
    )

    ax.annotate(
        "COVID-19",
        xy=(2020, series.min() * 0.95),
        fontsize=9,
        color="red",
        ha="center",
    )
    ax.set_xlabel("Année")
    ax.set_ylabel("Nombre de permis de séjour")

    ax.set_title(
        "Comparaison des prévisions des modèles "
        "ARIMA, ARIMAX et XGBoost\n"
        "(test 2023-2025 et prévisions 2026-2028)",
        fontsize=12,
    )

    # Formatage de l'axe Y avec séparateur de milliers.
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda x, _: f"{int(x):,}"
        )
    )

    ax.set_xlim(2012.5,2028.5,)

    ax.legend(
        loc="upper left",
        fontsize=9,
        ncol=2,
    )

    plt.tight_layout()

    # Sauvegarde de la figure.
    savefig(fig,"fig8_comparaison_3_modeles.png",)

    plt.show()
    future_years = [2026, 2027, 2028]
    prev_df = pd.DataFrame({
        "Année": future_years,

        "ARIMA": [
            fc_arima.loc[year]
            for year in future_years
        ],

        "ARIMAX": [
            fc_arimax.loc[year]
            for year in future_years
        ],

        "XGBoost": [
            fc_xgb.loc[year]
            for year in future_years
        ],
    })

    print("\n===== Prévision 2026-2028 =====")
    print(
        prev_df.to_string(
            index=False,
            formatters={
                "ARIMA": "{:,.0f}".format,
                "ARIMAX": "{:,.0f}".format,
                
                "XGBoost": "{:,.0f}".format,
            },
        )
    )

    return prev_df