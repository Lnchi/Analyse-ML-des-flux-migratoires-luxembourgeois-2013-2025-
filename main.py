"""
DIMM_BI - Analyse ML des flux migratoires luxembourgeois (2013-2025)

Usage :
    python main.py                  # exécute le pipeline complet
    python main.py --skip-cluster   # EDA + prévision uniquement
    python main.py --skip-forecast  # EDA + clustering uniquement

Chaque étape correspond à :
    1. Collecte + nettoyage des données EUROSTAT
    2. Analyse exploratoire (EDA)
    3. Modélisation prédictive (ARIMA / ARIMAX / Prophet / XGBoost)
    4. Clustering des profils migratoires (K-Means + CAH)
"""

import argparse
import warnings

warnings.filterwarnings("ignore")

from src.config import apply_publication_style
from src.data.collect import collect_raw_data
from src.data.process import build_df_all
from src.eda.temporal import build_lu_timeseries, plot_evolution, plot_stl_decomposition
from src.eda.heatmap import plot_nationality_heatmap
from src.eda.comparative import build_comparison_data, plot_country_comparison
from src.eda.reason import build_reason_data, plot_reason_distribution
from src.forecasting.stationarity import get_series, run_stationarity_tests, plot_acf_pacf
from src.forecasting.arima_model import run_arima_pipeline, run_arimax_pipeline
from src.forecasting.prophet_model import run_prophet_pipeline, plot_prophet_components
from src.forecasting.xgboost_model import run_xgboost_pipeline
from src.forecasting.compare import build_comparison_table, plot_final_forecast
from src.clustering.features import build_clustering_features
from src.clustering.dimensionality import scale_and_reduce, plot_pca
from src.clustering.selection import select_optimal_k, plot_k_selection
from src.clustering.kmeans import fit_final_kmeans, plot_silhouette_and_pca
from src.clustering.cah import plot_cah
from src.clustering.interpretation import print_cluster_interpretation, plot_radar_profiles
from src.clustering.migration_index import compute_migration_index, plot_migration_index
from src.forecasting.breakpoints import cusum_detection, scan_breakpoints, plot_breakpoint_analysis


from src.forecasting.hierarchical import (
    build_cluster_timeseries, run_hierarchical_forecast, plot_hierarchical_forecast,
    walk_forward_validation, plot_walk_forward_validation,
)

from src.forecasting.arima_model import run_arima_pipeline, run_arimax_pipeline, walk_forward_arima, walk_forward_arimax
from src.forecasting.prophet_model import run_prophet_pipeline, plot_prophet_components, select_changepoint_prior_scale

def run_data_pipeline():
    """Étape 1 : collecte + nettoyage EUROSTAT."""
    print("\n" + "=" * 70)
    print("1. COLLECTE + NETTOYAGE DES DONNÉES EUROSTAT")
    print("=" * 70)
    collect_raw_data()          # peuple/valide le cache local (voir data/cache/)
    df_all = build_df_all()     # DataFrame long nettoyé et normalisé
    return df_all


def run_eda(df_all):
    """Étape 2 : analyse exploratoire des données."""
    print("\n" + "=" * 70)
    print("2. ANALYSE EXPLORATOIRE DES DONNÉES (EDA)")
    print("=" * 70)

    df_lu, ts_lu = build_lu_timeseries(df_all)
    plot_evolution(ts_lu)
    plot_stl_decomposition(ts_lu)
    plot_nationality_heatmap(df_lu, top_n=20)

    ts_compare = build_comparison_data(df_all)
    plot_country_comparison(ts_compare)

    df_reason = build_reason_data(df_all)
    plot_reason_distribution(df_reason)

    return df_lu, ts_lu


def run_forecasting(ts_lu):
    """Étape 3 : modélisation prédictive (ARIMA / ARIMAX / XGBoost)."""
    print("\n" + "=" * 70)
    print("3. MODÉLISATION PRÉDICTIVE : SÉRIES TEMPORELLES")
    print("=" * 70)

    series = get_series(ts_lu)
    _, _, series_diff = run_stationarity_tests(series)
    plot_acf_pacf(series, series_diff)

    print("\n" + "=" * 70)
    print("\n Détection automatique de ruptures ")
    print("=" * 70)
    cusum_res = cusum_detection(series)
    chow_res = scan_breakpoints(series)
    plot_breakpoint_analysis(series, cusum_res, chow_res)

    # validation walk-forward ARIMA (diagnostic) 
    print("\n ============= Validation walk-forward ARIMA (2021-2022) ============= ")
    walk_forward_arima(series)

    arima_res = run_arima_pipeline(series)

    arimax_res = run_arimax_pipeline(
        arima_res["train"],
        arima_res["test"],
        arima_res["test_years"],
        arima_res["test_renamed"],
    )

    # validation walk-forward ARIMAX (diagnostic) 
    print("\n ============= Validation walk-forward ARIMAX (2021-2022) ============= ")
    walk_forward_arimax(series)

    """ # sélection du changepoint_prior_scale AVANT d'appeler Prophet 
    print("\n ============= Sélection changepoint_prior_scale Prophet (walk-forward) ============= ")
    best_cps, cps_results = select_changepoint_prior_scale(series)

    prophet_res = run_prophet_pipeline(
        series, arima_res["test_years"], arima_res["test"],
        changepoint_prior_scale=best_cps,  
    )
    plot_prophet_components(prophet_res["model"], prophet_res["forecast"]) """

    xgb_res = run_xgboost_pipeline(series, arima_res["test_years"], arima_res["test_renamed"])

    build_comparison_table(
    # MAE
    arima_res["metrics"]["mae"],
    arimax_res["metrics"]["mae"],
    
    xgb_res["metrics"]["mae"],

    # RMSE
    arima_res["metrics"]["rmse"],
    arimax_res["metrics"]["rmse"],
    
    xgb_res["metrics"]["rmse"],

    # MAPE
    arima_res["metrics"]["mape"],
    arimax_res["metrics"]["mape"],
    
    xgb_res["metrics"]["mape"],
)
    plot_final_forecast(
    series,
    arima_res["model"],
    arimax_res,
    
    xgb_res,
)


def run_clustering(df_lu):
    """Étape 4 : clustering des profils migratoires (K-Means + CAH)."""
    print("\n" + "=" * 70)
    print("4. CLUSTERING DES PROFILS MIGRATOIRES")
    print("=" * 70)

    df_feat_clust = build_clustering_features(df_lu)
    reduction = scale_and_reduce(df_feat_clust)
    plot_pca(df_feat_clust, reduction)

    selection = select_optimal_k(reduction["X_scaled"])
    plot_k_selection(selection)

    # k=2 est retenu à partir de l'analyse conjointe de la méthode du coude
    # et du coefficient de silhouette.
    # Le choix final est fixé ici conformément à l'interprétation présentée
    # dans le rapport / README.

    k_optimal = 2

    df_feat_clust, km_final, cluster_labels, sil_vals = fit_final_kmeans(
        df_feat_clust, reduction["X_scaled"], k_optimal
    )
    plot_silhouette_and_pca(
        df_feat_clust, reduction["X_scaled"], reduction["X_pca"], km_final,
        cluster_labels, sil_vals, k_optimal, reduction["pca"],
    )

    plot_cah(df_feat_clust, reduction["X_scaled"], reduction["feat_cols"], k_optimal)

    print_cluster_interpretation(df_feat_clust)
    plot_radar_profiles(df_feat_clust, k_optimal)

    print("\n Indice de profil migratoire ")
    df_ipm = compute_migration_index(df_feat_clust)
    plot_migration_index(df_ipm, top_n=20)

    print("\n Prévision hiérarchique par cluster ")
    # Prévision hiérarchique :
    # on construit d'abord une série temporelle annuelle pour chaque cluster.
    # Les nationalités exclues du clustering sont conservées dans le groupe
    # résiduel "OTHER", afin que la somme des clusters corresponde toujours
    # au total national.
    ts_by_cluster = build_cluster_timeseries(df_lu, df_feat_clust)
    # Prévision bottom-up :
    # chaque cluster est prévu séparément, puis les prévisions sont additionnées.
    # Cette somme est comparée à une prévision directe du total national.
    hier_result = run_hierarchical_forecast(ts_by_cluster)
    # Visualisation de l'historique des clusters et comparaison
    # entre prévision directe et prévision bottom-up.
    plot_hierarchical_forecast(ts_by_cluster, hier_result)
    # Validation walk-forward :
    # comparaison des deux approches sur plusieurs années historiques
    # afin de vérifier laquelle donne les erreurs de prévision les plus faibles.
    df_val = walk_forward_validation(ts_by_cluster)
    plot_walk_forward_validation(df_val)

    return df_feat_clust, df_ipm, hier_result, df_val


def main():
    parser = argparse.ArgumentParser(description="Pipeline ML - Flux migratoires luxembourgeois")
    parser.add_argument("--skip-forecast", action="store_true", help="Ne pas exécuter la modélisation prédictive")
    parser.add_argument("--skip-cluster", action="store_true", help="Ne pas exécuter le clustering")
    args = parser.parse_args()

    apply_publication_style()

    df_all = run_data_pipeline()
    df_lu, ts_lu = run_eda(df_all)

    if not args.skip_forecast:
        run_forecasting(ts_lu)

    if not args.skip_cluster:
        run_clustering(df_lu)

    print("\nPipeline terminé. Figures sauvegardées dans outputs/figures/.")


if __name__ == "__main__":
    main()
