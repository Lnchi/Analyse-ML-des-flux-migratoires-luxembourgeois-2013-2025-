"""
Exploration des données brutes EUROSTAT 

Ce script n'est pas appelé par main.py : c'est un outil d'exploration, à lancer manuellement quand on veut inspecter les données brutes
(avant nettoyage), pas à chaque exécution du pipeline.

Deux façons de l'utiliser dans VS Code :

1. En terminal :
       python scripts/explore_raw_data.py.

2. En mode interactif Jupyter (équivalent de display() sur Colab) : les blocs "# %%" ci-dessous sont reconnus par
   l'extension Jupyter de VS Code comme des cellules. Ouvre ce fichier,
   clique sur "Run Cell" au-dessus de chaque bloc, et raw_all s'affichera
   dans un tableau interactif (Data Viewer), comme display()
   sur Colab.
"""

# %%
import sys
from pathlib import Path

# Permet d'exécuter ce script depuis scripts/ tout en important src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.collect import collect_raw_data, inspect_raw_data, get_series_df

# %%
raw_all = collect_raw_data()

# %%
inspect_raw_data(raw_all, n_rows=10)

# %%
# Affichage interactif (fonctionne uniquement en cellule Jupyter VS Code,
# équivalent de display(raw_all) sur Colab). En script classique, cette
# ligne est silencieuse.
raw_all

# %%
# Inspection libre, à adapter selon les besoins :
# raw_all[raw_all["geo"] == "LU"].head(20)
# raw_all["geo"].value_counts()

# %%
# Affichage Aprés nettoyage (df_all) 
from src.data.process import build_df_all

df_all = build_df_all()

# %%
print(f"   Nationalités : {sorted(df_all['citizen'].unique())}")
print(f"\n df_all :\n {df_all}") # affichage interactif en cellule Jupyter, équivalent de display(df_all)

# %%
# Analyse Exploratoire des Données (EDA)
# Evolution temporelle - Luxembourg total 
# %%
# Construction de la série temporelle pour le Luxembourg
from src.eda.temporal import build_lu_timeseries

# %%
# Construction de la série temporelle annuelle du Luxembourg
_, ts_lu = build_lu_timeseries(df_all)

print("\nSérie temporelle annuelle du Luxembourg :\n")
print(ts_lu)



# %%
# Top 20 nationalités par volume total sur la période
from src.eda.heatmap import plot_nationality_heatmap
from src.eda.temporal import build_lu_timeseries

df_lu, ts_lu = build_lu_timeseries(df_all)
# On additionne les permis pour chaque nationalité sur l'ensemble
# de la période 2013-2025.
top_nationalities = (
    df_lu.groupby("citizen")["permits"]
    .sum()
    .nlargest(20)
    .index
    .tolist()
)
print("\nTop 20 nationalités :")
print(top_nationalities)


# %%
from src.forecasting.stationarity import get_series
series = get_series(ts_lu)
print("\nLa série temporelle des permis luxembourgeois :")
print(series)


# %%
from src.forecasting.arima_model import train_test_split
train, test, series_years = train_test_split(series, 3)
print(f"Longueur série : {len(series)}")
print(f"Index : {series.index.tolist()}")
print(f"Train : {len(train)} obs")
print(f"Test  : {len(test)} obs")



# %% 
from src.forecasting.arima_model import (
    run_arima_pipeline,
    run_arimax_pipeline,
    walk_forward_arima,
    walk_forward_arimax,
)


# ARIMA
results_arima = run_arima_pipeline(series, 3)
print("\n===== ARIMA =====")
# Comparaison prévisions / valeurs réelles sur le TEST FINAL
# Test final : 2023-2025
for year in results_arima["test_years"]:
    pred = results_arima["fc_test"].loc[year]
    real = results_arima["test_renamed"].loc[year]
    print(
        f"{year} : "
        f"prévu={pred:,.0f} | "
        f"réel={real:,.0f} | "
        f"écart={real - pred:+,.0f}"
    )
print("\nPrévisions test ARIMA :")
print(results_arima["fc_test"])
print("\nPrévisions futures ARIMA :")
print(results_arima["fc_future"])
# VALIDATION WALK-FORWARD ARIMA
print("\n===== VALIDATION WALK-FORWARD ARIMA =====")
metrics_walk_forward_arima = walk_forward_arima(
    series,
    validation_years=[2021, 2022]
)
print("\nRésultats validation ARIMA :")
print(metrics_walk_forward_arima)



# ARIMAX
# Récupération des éléments nécessaires au pipeline ARIMAX
train = results_arima["train"]
test = results_arima["test"]
test_years = results_arima["test_years"]
test_renamed = results_arima["test_renamed"]
results_arimax = run_arimax_pipeline(
    train,
    test,
    test_years,
    test_renamed
)
print("\n===== ARIMAX =====")
print("\nPrévisions test ARIMAX :")
print(results_arimax["fc_test"])
print("\nPrévisions futures ARIMAX :")
print(results_arimax["fc_future"])



# VALIDATION WALK-FORWARD ARIMAX
print("\n===== VALIDATION WALK-FORWARD ARIMAX =====")
metrics_walk_forward_arimax = walk_forward_arimax(
    series,
    validation_years=[2021, 2022]
)
print("\nRésultats validation ARIMAX :")
print(metrics_walk_forward_arimax)


# %% 
""" from src.forecasting.prophet_model import (
    run_prophet_pipeline,
    select_changepoint_prior_scale,
)


# VALIDATION WALK-FORWARD PROPHET
print("\n===== VALIDATION WALK-FORWARD PROPHET =====")
best_cps, results_cps = select_changepoint_prior_scale(
    series,
    values=[0.01, 0.05, 0.1, 0.3],
    validation_years=[2021, 2022]
)
print(
    f"\nValeur retenue pour Prophet : {best_cps}"
)

# PROPHET FINAL
results_prophet = run_prophet_pipeline(
    series=series,
    test_years=[2023, 2024, 2025],
    test=test,
    train_cutoff=2022,
    changepoint_prior_scale=best_cps
)

print("\n===== PROPHET =====")
print("\nPrévisions test Prophet :")
print(results_prophet["fc_test"])
print("\nMétriques Prophet :")
print(results_prophet["metrics"]) """


# %% 
import importlib
import src.forecasting.xgboost_model as xgb_model
from src.forecasting.xgboost_model import run_xgboost_pipeline

importlib.reload(xgb_model)

# XGBOOST
results_xgb = run_xgboost_pipeline(
    series=series,
    test_years=test_years,
    test_renamed=test_renamed,
    train_cutoff=2022,
)

print("\n===== XGBOOST =====")
print("\nPrévisions test XGBoost :")
print(results_xgb["fc_test"])
print("\nValeurs réelles :")
print(test_renamed)

print("\nPrévisions futures 2026-2028 :")
print(results_xgb["fc_future"])

print("\nPrévisions XGBoost complètes :")
print(results_xgb["fc_full"]) # fc_full contient 2013-2025+ 2026-2028 prédites

print("\nComparaison prévisions / valeurs réelles :")

for year, pred, real in zip(
    test_years,
    results_xgb["fc_test"],
    test_renamed.values
):
    print(
        f"{year} : "
        f"prévu={pred:,.0f} | "
        f"réel={real:,.0f} | "
        f"écart={real - pred:+,.0f}"
    )

print("\nMétriques XGBoost :")
print(
    f"MAE  = {results_xgb['metrics']['mae']:,.0f} permis"
)
print(
    f"RMSE = {results_xgb['metrics']['rmse']:,.0f} permis"
)
print(
    f"MAPE = {results_xgb['metrics']['mape']:.1f}%"
)

print("\nMeilleurs hyperparamètres XGBoost :")
print(results_xgb["best_params"])

print("\nImportance des variables :")
print(
    results_xgb["feature_importance"].to_string(index=False)
)


# %%
