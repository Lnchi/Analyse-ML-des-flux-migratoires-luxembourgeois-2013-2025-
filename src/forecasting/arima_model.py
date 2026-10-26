"""
3.3 Modèle ARIMA : sélection automatique AIC/BIC (pmdarima.auto_arima).
3.3a Modèle ARIMAX : ARIMA + variable exogène binaire COVID-19 (Bijak, 2010).

split temporel train/test : 2013-2022 (train) et 2023-2025 (test).
Le modèle est entraîné uniquement sur les données d'entraînement.
Les années 2023-2025 sont conservées pour évaluer la qualité
des prévisions hors échantillon.
"""

import numpy as np
import pandas as pd
import pmdarima as pm
# Métriques permettant de mesurer l'erreur des prévisions.
from sklearn.metrics import mean_absolute_error, mean_squared_error


def train_test_split(series: pd.Series, n_test: int = 3):
    """
    Sépare la série en deux parties :

    - train : observations utilisées pour entraîner le modèle 
    - test  : observations utilisées pour évaluer les prévisions

        train = 2013-2022
        test  = 2023-2025

    L'index est réinitialisé afin d'obtenir un index entier
    commençant à 0 pour l'entraînement du modèle.
    """
    # Sauvegarde des années originales pour les réutiliser
    # ensuite pour donner les bonnes années aux prévisions
    series_years = series.index.tolist()
    # 2013, 2014, ... deviennent 0, 1, ...
    series_reset = series.reset_index(drop=True)
    # Toutes les observations sauf les n_test dernières
    train = series_reset.iloc[:-n_test]
    test = series_reset.iloc[-n_test:]
    return train, test, series_years


def fit_auto_arima(train: pd.Series, exog=None):
    """
    Recherche automatiquement le meilleur modèle ARIMA

    Le choix de l'ordre (p,d,q) est effectué avec auto_arima
    en utilisant le critère d'information AIC

    Paramètres :
    - train : série utilisée pour entraîner le modèle 
    - exog  : éventuelle variable exogène pour l'ARIMAX
    """
    # start_p / max_p :
    # recherche de p entre 0 et 3.
    # start_q / max_q :
    # recherche de q entre 0 et 3.
    # d=None :
    # auto_arima détermine automatiquement le nombre
    # de différenciations nécessaire.
    # seasonal=False :
    # on utilise ici un modèle ARIMA non saisonnier,
    # et non un modèle SARIMA.
    # information_criterion="aic" :
    # le modèle est sélectionné selon le plus faible AIC.
    # stepwise=True :
    # permet de réduire le nombre de modèles testés
    # grâce à une recherche progressive.
    model = pm.auto_arima(
        train, X=exog,
        start_p=0, max_p=3, start_q=0, max_q=3, d=None,
        seasonal=False,
        information_criterion="aic",
        stepwise=True, suppress_warnings=True, error_action="ignore", trace=True,
    )
    print(f"\n Meilleur modèle : ARIMA{model.order}")
    # AIC : Akaike Information Criterion.
    # Plus l'AIC est faible, meilleur est le compromis
    # entre qualité d'ajustement et complexité du modèle.
    print(f"   AIC = {model.aic():.2f}")
    # BIC : Bayesian Information Criterion.
    # Comme l'AIC, il pénalise les modèles trop complexes.
    print(f"   BIC = {model.bic():.2f}")
    print("\n Résumé du modèle :")
    print(model.summary())
    return model


def evaluate_forecast(test_values: np.ndarray, forecast_values: np.ndarray, name: str = "Modèle") -> dict:
    """
    Calcule les principales métriques d'erreur sur le jeu de test.

    - MAE  : Mean Absolute Error
             erreur absolue moyenne.

    - RMSE : Root Mean Squared Error
             racine de l'erreur quadratique moyenne.

    - MAPE : Mean Absolute Percentage Error
             erreur absolue moyenne en pourcentage.
    """
    mae = mean_absolute_error(test_values, forecast_values)
    rmse = np.sqrt(mean_squared_error(test_values, forecast_values))
    # test_values correspond aux valeurs réelles
    mape = np.mean(np.abs((test_values - forecast_values) / test_values)) * 100

    print(f"\n Métriques {name} (test) :")
    print(f"   MAE  = {mae:,.0f} permis")
    print(f"   RMSE = {rmse:,.0f} permis")
    print(f"   MAPE = {mape:.1f}%")
    return {"mae": mae, "rmse": rmse, "mape": mape}


def run_arima_pipeline(series: pd.Series, n_test: int = 3):
    """
    Pipeline ARIMA complet :

    1. séparation train/test 
    2. recherche automatique du meilleur modèle ARIMA 
    3. prévisions sur le jeu de test 
    4. calcul des métriques 
    5. prévisions futures 2026-2028

    Retourne un dictionnaire contenant les résultats nécessaires
    pour la comparaison des modèles et les visualisations finales.
    """
    train, test, series_years = train_test_split(series, n_test)
    # auto_arima recherche automatiquement le meilleur
    # triplet (p,d,q) selon l'AIC.
    model_arima = fit_auto_arima(train)
    # Les années correspondant au jeu de test.
    test_years = series_years[-n_test:]
    # Prévision des n_test observations futures.
    #
    # return_conf_int=True demande également les intervalles
    # de confiance à 95 %
    fc_test, confint_test = model_arima.predict(n_periods=n_test, return_conf_int=True, alpha=0.05)

    # Transformation des prévisions en Series pandas
    # avec les années correspondantes comme index.
    fc_test = pd.Series(fc_test.values, index=test_years)
    # Transformation des valeurs réelles du jeu de test
    # en Series avec les mêmes années.
    test_renamed = pd.Series(test.values, index=test_years)

    #  Prévisions futures : 2026-2028
    # - 3 prévisions pour 2023-2025
    # - 3 prévisions supplémentaires pour 2026-2028
    #
    # Total = 6 périodes

    fc_future, confint_future = model_arima.predict(n_periods=n_test + 3, return_conf_int=True, alpha=0.05)  # noqa: RUF059

    # Création des années correspondant aux prévisions.
    future_years = test_years + [2026, 2027, 2028]
    fc_future = pd.Series(fc_future.values, index=future_years)
    # Comparaison des valeurs réellement observées
    # avec les valeurs prédites par ARIMA.
    metrics = evaluate_forecast(test_renamed.values, fc_test.values, name="ARIMA")

    return {
        "model": model_arima, "train": train, "test": test, "series_years": series_years,
        "test_years": test_years, "fc_test": fc_test, "confint_test": confint_test,
        "test_renamed": test_renamed, "fc_future": fc_future, "metrics": metrics,
    }

def walk_forward_arima(
    series: pd.Series,
    validation_years: list = [2021, 2022]
):
    """
    Validation temporelle walk-forward pour ARIMA.

    Schéma :

        2013-2020 -> prédit 2021
        2013-2021 -> prédit 2022

    Les années 2023-2025 restent totalement réservées
    au test final.
    """

    years = series.index.tolist()

    predictions = []
    real_values = []

    for test_year in validation_years:

        # Position de l'année à prédire
        i = years.index(test_year)

        # Toutes les observations avant l'année test
        train = series.iloc[:i]

        # Valeur réelle
        real_value = series.iloc[i]

        # Recherche du meilleur ARIMA uniquement
        # avec les données disponibles avant l'année test
        model = fit_auto_arima(train)

        # Prévision d'une seule année
        forecast = model.predict(
            n_periods=1
        )

        prediction = float(forecast.iloc[0])

        predictions.append(prediction)
        real_values.append(float(real_value))

        print(
            f"{test_year} : "
            f"prévu={prediction:,.0f} | "
            f"réel={real_value:,.0f} | "
            f"écart={real_value - prediction:+,.0f}"
        )

    metrics = evaluate_forecast(
        np.array(real_values),
        np.array(predictions),
        name="ARIMA validation walk-forward"
    )

    return {
        "metrics": metrics,
        "predictions": predictions,
        "real_values": real_values,
        "years": validation_years,
    }
def run_arimax_pipeline(train: pd.Series, test: pd.Series, test_years: list,
                         test_renamed: pd.Series, covid_year_index: int = 7):
    """
    Pipeline ARIMAX : ajoute une variable exogène binaire COVID (1 en 2020, 0 sinon).
    Recommandé par Bijak (2010) pour modéliser les ruptures structurelles.
    """
    n_train = len(train)
    # Création d'une variable exogène contenant initialement uniquement
    # des zéros.
    #
    # Forme :
    #     nombre d'observations x 1 variable
    exog_train = np.zeros((n_train, 1))
    # On affecte la valeur 1 à l'année correspondant à 2020.
    exog_train[covid_year_index, 0] = 1  # 2020 = index 7 dans 2013..2022

    # Variable COVID pour le jeu de test
    n_test = len(test)
    # Pour 2023-2025, on considère que la variable COVID vaut 0.
    exog_test = np.zeros((n_test, 1))
    # Variable COVID pour les prévisions futures
    exog_future = np.zeros((3, 1))

    print("\n Recherche automatique ARIMAX (auto_arima + exog COVID)")
    model_arimax = fit_auto_arima(train, exog=exog_train)
    # Prévision des années du jeu de test.
    #
    # X=exog_test fournit au modèle la valeur de la variable COVID
    # correspondant à chaque année à prévoir
    fc_arimax_raw, confint_arimax = model_arimax.predict(
        n_periods=n_test, X=exog_test, return_conf_int=True, alpha=0.05
    )
    # Transformation des prévisions en Series pandas
    # avec les années comme index.
    fc_arimax_test = pd.Series(fc_arimax_raw.values, index=test_years)
    # Comparaison année par année entre :
    # - la prévision ARIMAX ;
    # - la valeur réellement observée.
    for year, pred, real in zip(test_years, fc_arimax_test.values, test_renamed.values):
        print(f"   {year} : prédit={pred:,.0f} | réel={real:,.0f} | écart={real - pred:+,.0f}")

    # Pour les 6 périodes :
    # - 2023-2025 : exog_test
    # - 2026-2028 : exog_future
    #
    # np.vstack() permet de concaténer verticalement
    # les deux blocs de variables exogènes.
    fc_arimax_fut_raw, confint_arimax_fut = model_arimax.predict(
        n_periods=n_test + 3, X=np.vstack([exog_test, exog_future]), return_conf_int=True, alpha=0.05
    )
    # Années correspondant aux prévisions.
    future_years_arimax = test_years + [2026, 2027, 2028]
    fc_arimax_future = pd.Series(fc_arimax_fut_raw.values, index=future_years_arimax)
    # Comparaison entre les valeurs réelles et les prévisions
    # pour les années 2023-2025.
    metrics = evaluate_forecast(test_renamed.values, fc_arimax_test.values, name="ARIMAX")

    return {
        "model": model_arimax, "fc_test": fc_arimax_test, "confint_test": confint_arimax,
        "fc_future": fc_arimax_future, "confint_future": confint_arimax_fut, "metrics": metrics,
    }

# Si le walk-forward donne aussi des performances raisonnables, donc absence d'overfitting.
def walk_forward_arimax(
    series: pd.Series,
    validation_years: list = [2021, 2022]
):
    """
    Validation temporelle walk-forward pour ARIMAX.

    Cette validation est utilisée uniquement pour évaluer la
    capacité de généralisation du modèle et, si nécessaire,
    sélectionner les paramètres.

    Schéma utilisé :

        2013-2020 -> prédit 2021
        2013-2021 -> prédit 2022

    Les années 2023-2025 ne sont jamais utilisées pendant cette
    validation. Elles restent réservées au TEST FINAL.

    Cela permet de comparer ARIMAX avec les autres modèles
    sur exactement les mêmes années de validation.
    """

    years = series.index.tolist()

    predictions = []
    real_values = []

    # On parcourt uniquement les années définies comme validation.
    for test_year in validation_years:

        # Recherche de la position de l'année à prédire.
        i = years.index(test_year)

        # Toutes les observations précédant l'année test
        # sont utilisées pour entraîner le modèle.
        train = series.iloc[:i]

        # Valeur réellement observée pour l'année test.
        real_value = series.iloc[i]

        # Variable COVID pour les observations d'entraînement.
        # 1 en 2020, 0 sinon.
        exog_train = np.array([
            [1 if year == 2020 else 0]
            for year in train.index
        ])

        # Variable COVID connue pour l'année à prévoir.
        #
        # Exemple :
        # 2021 -> 0
        # 2022 -> 0
        exog_test = np.array([
            [1 if test_year == 2020 else 0]
        ])

        # Recherche du meilleur ARIMA uniquement
        # avec les données disponibles avant l'année test.
        model = fit_auto_arima(
            train,
            exog=exog_train
        )

        # Prévision d'une seule année.
        forecast = model.predict(
            n_periods=1,
            X=exog_test
        )

        prediction = float(forecast.iloc[0])

        predictions.append(prediction)
        real_values.append(float(real_value))

        print(
            f"{test_year} : "
            f"prévu={prediction:,.0f} | "
            f"réel={real_value:,.0f} | "
            f"écart={real_value - prediction:+,.0f}"
        )

    # Calcul des métriques sur les mêmes années
    # que celles utilisées pour les autres modèles.
    metrics = evaluate_forecast(
        np.array(real_values),
        np.array(predictions),
        name="ARIMAX validation walk-forward"
    )

    return {
        "metrics": metrics,
        "predictions": predictions,
        "real_values": real_values,
        "years": validation_years,
    }


