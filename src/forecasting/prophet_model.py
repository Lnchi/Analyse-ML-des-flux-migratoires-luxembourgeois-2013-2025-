"""
3.4 Modèle Prophet (Taylor & Letham, 2018).

Prophet modélise : y(t) = g(t) + s(t) + h(t) + ε
  g(t) = tendance (linéaire)
  s(t) = saisonnalité (désactivée : données annuelles)
  h(t) = effets des événements ponctuels (choc COVID-19 modélisé explicitement)
  ε(t) = erreur
"""

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error


def build_prophet_dataframe(series: pd.Series) -> pd.DataFrame:
    """Formate la série au format attendu par Prophet : colonnes 'ds' et 'y'."""
    return pd.DataFrame({
        "ds": pd.to_datetime([f"{y}-01-01" for y in series.index]), # 'ds' : date de l'observation et Conversion des années en dates
        "y": series.values.astype(float), # 'y'  : valeur observée 
    })  


def select_changepoint_prior_scale(
    series: pd.Series,
    values: list = [0.01, 0.05, 0.1, 0.3],
    validation_years: list = [2021, 2022],
):
    """
    Sélectionne changepoint_prior_scale par validation temporelle
    walk-forward.

    Schéma :

        2013-2020 -> prédit 2021
        2013-2021 -> prédit 2022

    Les années 2023-2025 ne sont jamais utilisées pour choisir
    le paramètre.

    La meilleure valeur est celle qui minimise la MAE moyenne
    sur les années de validation.
    """

    df_prophet = build_prophet_dataframe(series)

    results = []

    # Test de chaque valeur du paramètre.
    for cps in values:

        predictions = []
        real_values = []

        print(
            f"\n===== changepoint_prior_scale = {cps} ====="
        )

        # Validation walk-forward.
        for year in validation_years:

            # Toutes les observations précédant l'année
            # de validation servent à l'entraînement.
            df_train = df_prophet[
                df_prophet["ds"].dt.year < year
            ]

            # Valeur réelle de l'année à prédire.
            real_value = series.loc[year]

            # Choc COVID connu.
            covid_shock = pd.DataFrame({
                "holiday": "COVID-19 Shock",
                "ds": pd.to_datetime(["2020-01-01"]),
                "lower_window": 0,
                "upper_window": 1,
            })

            # Modèle Prophet avec la valeur de CPS testée.
            model = Prophet(
                growth="linear",

                # Paramètre dont on cherche la meilleure valeur.
                changepoint_prior_scale=cps,

                holidays=covid_shock,

                # Données annuelles :
                # aucune saisonnalité infra-annuelle.
                yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=False,

                interval_width=0.95,
            )

            # Entraînement uniquement sur le passé.
            model.fit(df_train)

            # Une seule période à prévoir.
            future = model.make_future_dataframe(
                periods=1,
                freq="YS"
            )

            forecast = model.predict(future)

            # Dernière prévision = année de validation.
            prediction = forecast.iloc[-1]["yhat"]

            predictions.append(float(prediction))
            real_values.append(float(real_value))

            print(
                f"{year} : "
                f"prévu={prediction:,.0f} | "
                f"réel={real_value:,.0f}"
            )

        # Évaluation globale sur 2021-2022.
        mae = mean_absolute_error(
            real_values,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                real_values,
                predictions
            )
        )

        mape = (
            np.mean(
                np.abs(
                    (
                        np.array(real_values)
                        - np.array(predictions)
                    )
                    / np.array(real_values)
                )
            )
            * 100
        )

        results.append({
            "changepoint_prior_scale": cps,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
        })

        print(
            f"MAE validation = {mae:,.0f}"
        )

    results_df = pd.DataFrame(results)

    # Sélection de la valeur donnant la plus faible MAE.
    best_cps = results_df.loc[
        results_df["MAE"].idxmin(),
        "changepoint_prior_scale"
    ]

    print(
        "\n===== Sélection du paramètre Prophet ====="
    )

    print(
        f"Années de validation : {validation_years}"
    )

    print(
        results_df.to_string(index=False)
    )

    print(
        f"\nValeur retenue de "
        f"changepoint_prior_scale : {best_cps}"
    )

    return best_cps, results_df

def run_prophet_pipeline(series: pd.Series, test_years: list, test: pd.Series, train_cutoff: int = 2022, changepoint_prior_scale: float = 0.1):
    """
    Entraîne Prophet sur df_train_p (<= train_cutoff), prévoit test + horizon futur,
    et calcule les métriques sur le jeu de test.
    """
    # Préparation des données au format nécessaire pour Prophet
    df_prophet = build_prophet_dataframe(series)
    # Séparation des données d'entraînement
    # seules les années jusqu'à 2022 sont utilisées
    # pour entraîner le modèle
    df_train_p = df_prophet[df_prophet["ds"].dt.year <= train_cutoff]

    # On définit le choc COVID-19 comme un événement particulier.
    # Il commence en 2020 et est prolongé d'une année
    covid_shock = pd.DataFrame({
        "holiday": "COVID-19 Shock",
        "ds": pd.to_datetime(["2020-01-01"]),
        "lower_window": 0, "upper_window": 1, # upper_window = 1 : l'effet est également pris en compte en 2021
    })

    model = Prophet(
        growth="linear", # Tendance générale supposée linéaire
        changepoint_prior_scale=changepoint_prior_scale,     # Pour controler la flexibilité de la tendance
        holidays=covid_shock, # Ajout de l'événement COVID-19 au modèle
        yearly_seasonality=False,        # données annuelles : pas de saisonnalité infra-annuelle
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.95,    # Intervalle de confiance de 95 % pour les prévisions 
    )
    model.fit(df_train_p) # # Entraînement du modèle uniquement sur les données

    future = model.make_future_dataframe(periods=6, freq="YS") # Création de 6 périodes futures
    # Prévision 
    forecast = model.predict(future)
    # On récupère uniquement les prévisions correspondant
    # aux années du jeu de test (2023-2025)
    fc_test_p = forecast[forecast["ds"].dt.year.isin(test_years)]["yhat"].values
    # Mean Absolute Error :
    # erreur absolue moyenne entre les valeurs réelles
    # et les valeurs prédites
    mae = mean_absolute_error(test.values, fc_test_p)
    # Root Mean Squared Error :
    # pénalise davantage les grandes erreurs
    rmse = np.sqrt(mean_squared_error(test.values, fc_test_p))
    # Mean Absolute Percentage Error :
    # erreur moyenne exprimée en pourcentage
    mape = np.mean(np.abs((test.values - fc_test_p) / test.values)) * 100

    print(f"Années test        : {test_years}")
    print(f"Prévisions Prophet : {fc_test_p}")
    print(f"Valeurs réelles    : {test.values}")
    print("   Métriques Prophet (test) :\n")
    print(f"   MAE  = {mae:,.0f} permis")
    print(f"   RMSE = {rmse:,.0f} permis")
    print(f"   MAPE = {mape:.1f}%")

    return {
        "model": model, "forecast": forecast, "fc_test": fc_test_p,
        "metrics": {"mae": mae, "rmse": rmse, "mape": mape},
    }


def plot_prophet_components(model: Prophet, forecast: pd.DataFrame):
    """Visualise les composantes du modèle Prophet (tendance, changepoints, effet COVID)."""

    import matplotlib.pyplot as plt

    from src.config import savefig
    # Génération du graphique des composantes du modèle
    fig = model.plot_components(forecast)
    fig.suptitle("Composantes du modèle Prophet", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "fig7_prophet_composantes.png")
    plt.show()
