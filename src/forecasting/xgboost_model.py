"""
3.5 Modèle XGBoost : feature engineering temporel.

Transforme la série en variables explicatives :

    - lag1       : valeur de l'année précédente
    - lag2       : valeur d'il y a deux ans
    - trend      : tendance linéaire
    - trend_sq   : tendance quadratique
    - covid_flag : indicateur du choc COVID-19 en 2020
    - post_covid : indicateur des années après 2020

Les variables lag1 et lag2 sont justifiées par l'analyse ACF/PACF,
qui montre les dépendances temporelles principales.

Les hyperparamètres du modèle XGBoost sont testés sur plusieurs valeurs
et comparés à l'aide d'une validation temporelle afin de limiter
le risque d'overfitting.

Justification méthodologique :
Hyndman & Athanasopoulos (2021) montrent l'intérêt des méthodes
d'apprentissage automatique lorsque les variables temporelles
sont correctement construites.

La série est courte. Les paramètres du modèle sont simples pour limiter le risque de sur-apprentissage.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Variables utilisées comme entrées du modèle XGBoost.
FEAT_COLS = ["lag1", "lag2", "trend", "trend_sq", "covid_flag", "post_covid"]
TARGET = "y" # Variable que le modèle doit prédire.


def build_features(s: pd.Series) -> pd.DataFrame:
    """Construit les variables explicatives à partir de la série temporelle."""
    df = pd.DataFrame({"year": s.index, "y": s.values})  # DataFrame contenant l'année et la valeur observée
    df["lag1"] = df["y"].shift(1) # Valeur observée l'année précédente
    df["lag2"] = df["y"].shift(2) # Valeur observée deux années auparavant
    df["trend"] = np.arange(len(df)) # Tendance linéaire : 0 pour la première année, 1 pour la deuxième, etc
    df["trend_sq"] = df["trend"] ** 2 # Elle permet au modèle de représenter une évolution qui n'est pas strictement linéaire
    df["covid_flag"] = (df["year"] == 2020).astype(int) # Vaut 1 uniquement pour l'année 2020, et 0 pour toutes les autres années
    df["post_covid"] = (df["year"] > 2020).astype(int) # Vaut 1 après 2020 et 0 avant ou pendant 2020. La variable permet de représenter une modification de tendance après le choc COVID-19
    
    # Les deux premières observations ne peuvent pas avoir de lag2. Elles sont donc supprimées
    return df.dropna()

def select_xgboost_params(
    df_feat: pd.DataFrame,
    validation_years: list = [2021, 2022],
):
    """
    Sélection des hyperparamètres XGBoost par validation temporelle
    walk-forward.

    Schéma :

        2013-2020 -> prédit 2021
        2013-2021 -> prédit 2022

    Les années 2023-2025 ne sont jamais utilisées pour sélectionner
    les hyperparamètres.

    La MAE moyenne sur les années de validation est utilisée
    pour sélectionner la meilleure configuration.
    """

    max_depth_values = [2, 3]
    n_estimators_values = [50, 100, 200]
    learning_rate_values = [0.03, 0.05, 0.1]

    results = []

    # Test de toutes les combinaisons.
    for max_depth in max_depth_values:
        for n_estimators in n_estimators_values:
            for learning_rate in learning_rate_values:

                predictions = []
                real_values = []

                # Validation walk-forward :
                # chaque année est prédite uniquement avec
                # les observations précédentes.
                for year in validation_years:

                    train_df = df_feat[
                        df_feat["year"] < year
                    ]

                    test_df = df_feat[
                        df_feat["year"] == year
                    ]

                    X_train = train_df[FEAT_COLS].values
                    y_train = train_df[TARGET].values

                    X_val = test_df[FEAT_COLS].values
                    y_val = test_df[TARGET].values

                    model = xgb.XGBRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,

                        # Régularisation par sous-échantillonnage.
                        subsample=0.8,

                        # Utilisation d'une partie des variables
                        # à chaque arbre.
                        colsample_bytree=0.8,

                        # Régularisation L2.
                        reg_lambda=2.0,

                        random_state=42,
                        verbosity=0,
                    )

                    # Entraînement uniquement sur le passé.
                    model.fit(
                        X_train,
                        y_train,
                        verbose=False
                    )

                    # Prévision de l'année de validation.
                    pred = model.predict(X_val)

                    predictions.append(float(pred[0]))
                    real_values.append(float(y_val[0]))

                # MAE moyenne sur 2021-2022.
                mae = mean_absolute_error(
                    real_values,
                    predictions
                )

                results.append({
                    "max_depth": max_depth,
                    "n_estimators": n_estimators,
                    "learning_rate": learning_rate,
                    "MAE": mae,
                })

    results_df = (
        pd.DataFrame(results)
        .sort_values("MAE")
    )

    # La meilleure configuration est celle qui minimise
    # l'erreur de validation.
    best_params = (
        results_df.iloc[0]
        .drop("MAE")
        .to_dict()
    )

    # Conversion des valeurs en int/float.
    best_params["max_depth"] = int(
        best_params["max_depth"]
    )
    best_params["n_estimators"] = int(
        best_params["n_estimators"]
    )
    best_params["learning_rate"] = float(
        best_params["learning_rate"]
    )

    print("\n===== Validation walk-forward XGBoost =====")
    print(
        "Années de validation : "
        f"{validation_years}"
    )

    print("\nComparaison des hyperparamètres :")
    print(
        results_df.to_string(index=False)
    )

    print("\nMeilleurs paramètres :")
    print(best_params)

    print(
        f"MAE validation : "
        f"{results_df.iloc[0]['MAE']:,.0f} permis"
    )

    return best_params

def forecast_xgboost_test_recursive(
    model,
    series: pd.Series,
    test_years: list = [2023, 2024, 2025],
) -> pd.Series:
    """
    Génère les prévisions XGBoost récursives sur le jeu de test

    Le modèle est entraîné uniquement sur les données jusqu'à 2022

    Pour éviter de donner au modèle les vraies valeurs du test :
    
        2023 utilise les valeurs observées 2021-2022
        2024 utilise la prédiction 2023
        2025 utilise les prédictions 2023-2024

    Cette méthode rend l'évaluation XGBoost comparable
    aux prévisions des modèles ARIMA, ARIMAX et Prophet
    """

    # Historique contenant uniquement les observations disponibles
    # avant le début du jeu de test.
    history = series.loc[series.index < test_years[0]].copy()

    # Dernière valeur de tendance observée.
    last_trend = len(series.loc[series.index <= 2022]) - 1

    predictions = []

    for year in test_years:

        # On avance d'une année dans la tendance.
        trend = last_trend + 1

        # Les lags utilisent l'historique disponible.
        # Après 2023, cet historique contient les prédictions précédentes.
        lag1 = history.iloc[-1]
        lag2 = history.iloc[-2]

        # Tendance quadratique.
        trend_sq = trend ** 2

        # Variable COVID.
        covid_flag = int(year == 2020)

        # Indicateur post-COVID.
        post_covid = int(year > 2020)

        # Construction des variables explicatives.
        X_test_future = pd.DataFrame([{
            "lag1": lag1,
            "lag2": lag2,
            "trend": trend,
            "trend_sq": trend_sq,
            "covid_flag": covid_flag,
            "post_covid": post_covid,
        }])

        # Prévision de l'année.
        prediction = float(
            model.predict(X_test_future[FEAT_COLS])[0]
        )

        # IMPORTANT :
        # on ajoute la prédiction à l'historique.
        # Elle sera donc utilisée comme lag pour l'année suivante.
        history.loc[year] = prediction

        predictions.append(prediction)

        last_trend = trend

    return pd.Series(
        predictions,
        index=test_years,
        name="XGBoost"
    )

def forecast_xgboost_future(
    model,
    series: pd.Series,
    future_years: list = [2026, 2027, 2028],
) -> pd.Series:
    """
    Génère les prévisions XGBoost pour les années futures.

    Les prévisions sont réalisées de manière récursive :
    - la prévision d'une année devient une variable lag
      pour l'année suivante.

    Exemple :
        2026 utilise les valeurs observées 2024-2025
        2027 utilise la valeur prédite en 2026
        2028 utilise les valeurs prédites en 2026-2027
    """

    # Copie de la série pour conserver les valeurs observées.
    history = series.copy()

    # Dernière valeur de tendance observée.
    last_trend = len(series) - 1

    predictions = []

    for year in future_years:

        # Valeur de tendance pour l'année future.
        trend = last_trend + 1

        # Construction des variables explicatives.
        lag1 = history.iloc[-1]
        lag2 = history.iloc[-2]

        trend_sq = trend ** 2

        # Variable COVID :
        # 1 uniquement en 2020, donc 0 pour les années futures.
        covid_flag = int(year == 2020)

        # Variable indiquant la période après COVID.
        post_covid = int(year > 2020)

        # Création du vecteur de variables explicatives.
        X_future = pd.DataFrame([{
            "lag1": lag1,
            "lag2": lag2,
            "trend": trend,
            "trend_sq": trend_sq,
            "covid_flag": covid_flag,
            "post_covid": post_covid,
        }])

        # Prévision de l'année.
        prediction = float(model.predict(X_future[FEAT_COLS])[0])

        # On ajoute la prévision à l'historique.
        # Elle pourra donc être utilisée comme lag
        # pour l'année suivante.
        history.loc[year] = prediction

        predictions.append(prediction)

        # On avance la tendance d'une année.
        last_trend = trend

    return pd.Series(
        predictions,
        index=future_years,
        name="XGBoost",
    )

def run_xgboost_pipeline(series: pd.Series, test_years: list, test_renamed: pd.Series, train_cutoff: int = 2022):
    """
    Entraîne le modèle XGBoost sur les données d'entraînement,
    Étapes :
    1. création des variables temporelles ;
    2. séparation temporelle train/test ;
    3. sélection des hyperparamètres sur le train ;
    4. entraînement du modèle final ;
    5. prévision sur le jeu de test ;
    6. calcul des métriques ;
    7. calcul de l'importance des variables.
    réalise les prévisions sur le jeu de test et calcule les métriques.

        TRAIN : jusqu'à 2022
        TEST  : 2023-2025
    """
    # Construction des variables explicatives
    df_feat = build_features(series)

    X = df_feat[FEAT_COLS].values # Sélection des variables utilisées comme entrées du modèle
    y = df_feat[TARGET].values # Variable cible à prédire

    train_mask = df_feat["year"] <= train_cutoff
    test_mask = df_feat["year"] > train_cutoff

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # Sélection des hyperparamètres uniquement sur la période
    # de validation 2021-2022.
    #
    # Les années 2023-2025 restent totalement indépendantes
    # et servent uniquement au test final.
    best_params = select_xgboost_params(
        df_feat,
        validation_years=[2021, 2022]
    )

    model = xgb.XGBRegressor(
        n_estimators=best_params["n_estimators"], # Nombre d'arbres construits
        max_depth=best_params["max_depth"], # Profondeur maximale de chaque arbre. Une profondeur faible limite la complexité du modèle. 
        learning_rate=best_params["learning_rate"], # Vitesse d'apprentissage. Une valeur faible rend l'apprentissage plus progressif.
        subsample=0.8, # Fraction des observations utilisées pour chaque arbre pour réduire le risque d'overfitting.
        colsample_bytree=0.8, # Fraction des variables utilisées pour chaque arbre pour ajouter une forme de régularisation.
        reg_lambda=2.0,       # L2 régularisation, évite l'overfitting sur série courte
        random_state=42, # Pour obtenir les mêmes résultats à chaque exécution.
        verbosity=0, # Pas de sortie supplémentaire pendant l'entraînement
    )
    # Une fois les hyperparamètres sélectionnés,
    # le modèle final est réentraîné sur TOUT le train,
    # c'est-à-dire 2013-2022.
    model.fit(
        X_train,
        y_train,
        verbose=False
    )

    # Prévisions récursives sur le test 2023-2025.
    #
    # Le modèle ne voit jamais les vraies valeurs de 2023-2025
    # lorsqu'il prédit les années suivantes.
    y_pred_series = forecast_xgboost_test_recursive(
        model=model,
        series=series,
        test_years=test_years,
    )

    # Conversion en tableau NumPy pour les métriques.
    y_pred = y_pred_series.values



    # Prévisions futures 2026-2028.
    # Les prévisions sont générées récursivement :
    # une prévision future peut être utilisée comme lag
    # pour l'année suivante.
    fc_future = forecast_xgboost_future(
        model=model,
        series=series,
        future_years=[2026, 2027, 2028],
    )

    # Prévisions XGBoost sur toute la période de comparaison :
    # 2023-2025 = prévisions du jeu de test
    # 2026-2028 = prévisions futures récursives
    fc_test_series = pd.Series(
    y_pred,
    index=test_years,
    name="XGBoost"
)
    # Historique 2013-2025 + prévisions futures 2026-2028
    fc_full = pd.concat([fc_test_series,fc_future])

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    # XGBoost fournit une mesure de l'importance
    # de chaque variable utilisée par le modèle
    feature_importance = pd.DataFrame({
        "feature": FEAT_COLS, "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print(f"Années test        : {test_years}")
    print(f"Prévisions XGBoost : {y_pred}")
    print(f"Valeurs réelles    : {test_renamed.values}")
    print(f" Métriques XGBoost (test) :")
    print(f"   MAE  = {mae:,.0f} | RMSE = {rmse:,.0f} | MAPE = {mape:.1f}%")
    print("\n   Feature importance :")
    print(feature_importance.to_string(index=False))

    return {
    "model": model,

    # Prévisions sur le jeu de test : 2023-2025.
    "fc_test": y_pred,

    # Prévisions futures : 2026-2028.
    "fc_future": fc_future,

    # Importance des variables utilisées par XGBoost.
    "feature_importance": feature_importance,

    # Hyperparamètres sélectionnés par validation.
    "best_params": best_params,

    # Métriques calculées uniquement sur le test 2023-2025.
    "metrics": {
        "mae": mae,
        "rmse": rmse,
        "mape": mape
    },
    "fc_full": fc_full,   # utiliser dans plot_final_forecast
}

