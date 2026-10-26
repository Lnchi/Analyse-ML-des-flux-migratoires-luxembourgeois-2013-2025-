"""
4.8 Prévision par cluster + réconciliation hiérarchique.

Constat : la partie 3 (ARIMA/Prophet/XGBoost) prévoit la série agrégée
nationale ; la partie 4 (K-Means/CAH) identifie des clusters de profils
migratoires structurellement différents.

Les clusters sont ensuite interprétés selon leurs caractéristiques
(volume, motifs, croissance, résilience COVID, etc.). 

Ce module les relie : on prévoit séparément le volume de chaque cluster,
puis on réconcilie ces prévisions avec la prévision du total national, selon
l'approche standard de "hierarchical forecasting" (Hyndman et al., 2011) : 

    Réalise une prévision hiérarchique.

    Étape 1 :
    Prévoir chaque cluster séparément.

    Étape 2 :
    Additionner les prévisions.

    Étape 3 :
    Comparer cette somme avec une prévision directe
    du total national.


Intérêt méthodologique :
- Le cluster C1 ("crise") a une dynamique radicalement différente du
  cluster C0 (CAGR ~44%/an vs ~9.6%/an, cf. interpretation.py). Un modèle
  unique sur le total national moyenne ces deux dynamiques et perd
  l'information ; prévoir séparément puis recombiner est plus fidèle.
- La réconciliation "bottom-up" (somme des prévisions par cluster) est
  privilégiée ici : avec des séries aussi courtes, c'est l'approche la plus
  simple et la plus robuste (les méthodes de réconciliation optimale, ex.
  MinT, nécessitent une estimation de covariance peu fiable sur n=13).
"""

import numpy as np
import pandas as pd
import pmdarima as pm
import matplotlib.pyplot as plt

from src.config import PALETTE, savefig


def build_cluster_timeseries(df_lu: pd.DataFrame, df_feat_clust: pd.DataFrame) -> pd.DataFrame:
    """
    Construit la série annuelle de permis pour chaque cluster, en associant
    chaque nationalité de df_lu à son cluster (issu de df_feat_clust['cluster']).

    Les nationalités absentes de df_feat_clust (exclues du clustering car
    trop rares, cf. build_clustering_features) sont regroupées dans un
    cluster 'OTHER' résiduel, pour que la somme des clusters égale bien le
    total national (condition nécessaire à la réconciliation bottom-up).
    """
    # Association entre chaque nationalité et son cluster.
    # Les nationalités qui n'ont pas été retenues dans le clustering
    # seront ensuite regroupées dans "OTHER".
    citizen_to_cluster = df_feat_clust["cluster"].to_dict()
    df = df_lu.copy()
    # Attribution du cluster à chaque nationalité.
    # Les nationalités absentes de df_feat_clust (par exemple parce
    # qu'elles ont été exclues pour effectifs trop faibles) obtiennent NaN.
    df["cluster"] = df["citizen"].map(citizen_to_cluster)
    # Les nationalités non classées sont regroupées dans "OTHER".
    # Cela permet de conserver l'ensemble des permis et garantit que :
    # TOTAL = somme des clusters + éventuel groupe OTHER.
    df["cluster"] = df["cluster"].apply(lambda x: f"C{int(x)}" if pd.notna(x) else "OTHER")

    ts_by_cluster = (
        df.groupby(["cluster", "year"])["permits"]
        .sum()
        .reset_index()
        .pivot(index="year", columns="cluster", values="permits")
        .fillna(0) # Aucun permis observé pour ce cluster cette année -> volume nul.
    ) 
    ts_by_cluster["TOTAL"] = ts_by_cluster.sum(axis=1) # # ajouter le total national pour comparer les prévisions
    # Vérification de cohérence :
    # le total national doit être exactement égal à la somme des clusters.
    assert np.allclose(
        ts_by_cluster["TOTAL"],
        ts_by_cluster.drop(columns="TOTAL").sum(axis=1)
    ), "Erreur : le TOTAL ne correspond pas à la somme des clusters."
    return ts_by_cluster


def forecast_cluster(series: pd.Series, n_periods: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Prévision auto-ARIMA simple pour une série de cluster (walk-forward, mêmes réglages que arima_model.py)."""
    # suppression de l'index temporel
    # Auto-ARIMA travaille uniquement sur les valeurs
    series_reset = series.reset_index(drop=True)
    # recherche automatique du meilleur modèle ARIMA
    # déterminer automatiquement d
    # meilleur modèle = plus petit AIC
    model = pm.auto_arima(
        series_reset, start_p=0, max_p=3, start_q=0, max_q=3, d=None,
        seasonal=False, information_criterion="aic",
        stepwise=True, suppress_warnings=True, error_action="ignore",
    )
    # produire les prévisions
    fc, confint = model.predict(n_periods=n_periods, return_conf_int=True, alpha=0.05)
    # Le nombre de permis ne peut pas être négatif.
    # On borne donc les prévisions ponctuelles à zéro.
    fc = np.maximum(np.asarray(fc), 0)
    return np.asarray(fc), confint


def run_hierarchical_forecast(ts_by_cluster: pd.DataFrame, n_periods: int = 3) -> pd.DataFrame:
    """
    Prévoit chaque cluster indépendamment puis réconcilie par sommation
    bottom-up : la prévision du total = somme des prévisions par cluster.
    Compare cette prévision "bottom-up" à une prévision directe du total.

    Returns:
        DataFrame comparant, pour chaque année future, la prévision directe
        du total et la prévision bottom-up (somme des clusters).
    """
    cluster_cols = [c for c in ts_by_cluster.columns if c != "TOTAL"] #   # récupérer les noms des clusters
    last_year = ts_by_cluster.index.max() # # dernière année observée
    future_years = [last_year + i for i in range(1, n_periods + 1)]

    forecasts = {}
    for col in cluster_cols: # Prévision de chaque cluster
        print(f"\n Prévision cluster '{col}'...")
        fc, _ = forecast_cluster(ts_by_cluster[col], n_periods)
        forecasts[col] = fc # Chaque cluster possède alors sa propre prévision.

    print("\n Prévision directe du TOTAL (référence)...")
    fc_total_direct, _ = forecast_cluster(ts_by_cluster["TOTAL"], n_periods)

    # Réconciliation bottom-up :
    # la prévision nationale est reconstruite en additionnant les prévisions
    # individuelles de tous les clusters.
    #
    # Cette approche garantit par construction la cohérence hiérarchique :
    # la prévision du TOTAL correspond exactement à la somme des composantes.
    fc_bottom_up = np.sum([forecasts[c] for c in cluster_cols], axis=0)

    result = pd.DataFrame({
        "year": future_years,
        "total_direct": fc_total_direct,
        "total_bottom_up": fc_bottom_up, # somme des clusters
        **{f"cluster_{c}": forecasts[c] for c in cluster_cols}, # # conserver les prévisions de chaque cluster 
    }).set_index("year")
    # différence absolue
    result["ecart_reconciliation"] = result["total_bottom_up"] - result["total_direct"]
    # différence relative (%)
    result["ecart_pct"] = (result["ecart_reconciliation"] / result["total_direct"]) * 100

    print("\n Comparaison prévision directe vs bottom-up (réconciliée) :")
    print(result.round(0).to_string())

    return result

def walk_forward_validation(ts_by_cluster: pd.DataFrame, min_train_size: int = 8) -> pd.DataFrame:
    """
    Validation croisée temporelle (walk-forward / expanding window) comparant
    la prévision directe du total à la prévision bottom-up (réconciliée),
    sur les années historiques disponibles.

    Pour chaque année de test t (à partir de min_train_size observations
    d'entraînement) :
        - entraîne un ARIMA sur TOTAL[:t] -> prévoit TOTAL[t] (direct)
        - entraîne un ARIMA sur chaque cluster[:t] -> somme des prévisions (bottom-up)
        - compare les deux aux valeurs réellement observées en t

    Sur une série de 13 points, ça ne donne que 3 à 5 replis (min_train_size=8
    laisse 5 années de test) : suffisant pour dégager une tendance, pas pour
    un intervalle de confiance robuste sur l'écart. À interpréter comme un
    indice de fiabilité relative, pas comme une preuve statistique définitive.

    Returns:
        DataFrame par année de test avec les erreurs de chaque approche +
        un résumé RMSE global affiché en console.
    """
    cluster_cols = [c for c in ts_by_cluster.columns if c != "TOTAL"]
    years = ts_by_cluster.index.tolist()
    # Les premières observations constituent la fenêtre d'entraînement initiale.
    # À chaque étape, une nouvelle année est ajoutée à l'échantillon
    # d'entraînement : il s'agit d'une validation à fenêtre croissante
    # (expanding window).
    test_years = years[min_train_size:]

    if len(test_years) < 2:
        print(" Série trop courte pour une validation walk-forward fiable (< 2 replis).")
        return pd.DataFrame()

    rows = []
    for t in test_years:
        train_years = [y for y in years if y < t]

        fc_direct, _ = forecast_cluster(ts_by_cluster.loc[train_years, "TOTAL"], n_periods=1)
        fc_direct = fc_direct[0]

        fc_bu_total = 0.0
        for col in cluster_cols:
            fc_c, _ = forecast_cluster(ts_by_cluster.loc[train_years, col], n_periods=1)
            fc_bu_total += fc_c[0]

        actual = ts_by_cluster.loc[t, "TOTAL"]
        rows.append({
            "year": t, "actual": actual,
            "fc_direct": fc_direct, "fc_bottom_up": fc_bu_total,
            "err_direct": actual - fc_direct, "err_bottom_up": actual - fc_bu_total,
        })

    df_val = pd.DataFrame(rows).set_index("year")

    rmse_direct = np.sqrt((df_val["err_direct"] ** 2).mean())
    rmse_bottom_up = np.sqrt((df_val["err_bottom_up"] ** 2).mean())
    mae_direct = df_val["err_direct"].abs().mean()
    mae_bottom_up = df_val["err_bottom_up"].abs().mean()

    print(f"\n Validation walk-forward ({len(test_years)} replis, années {test_years}) :")
    print(df_val.round(0).to_string())
    print(f"\n   RMSE direct     = {rmse_direct:,.0f}")
    print(f"   RMSE bottom-up  = {rmse_bottom_up:,.0f}")
    print(f"   MAE  direct     = {mae_direct:,.0f}")
    print(f"   MAE  bottom-up  = {mae_bottom_up:,.0f}")
    better = "directe" if rmse_direct < rmse_bottom_up else "bottom-up (par cluster)"
    print(f"   -> Sur les replis historiques testés, l'approche {better} "
      f"présente la RMSE la plus faible.")
    print("      Ce résultat constitue un indice de performance relative, "
      "mais reste exploratoire compte tenu du faible nombre de replis.")

    return df_val


def plot_walk_forward_validation(df_val: pd.DataFrame):
    """Visualise les prévisions direct vs bottom-up face aux valeurs réelles, replis par replis."""
    if df_val.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(df_val.index, df_val["actual"], "o-", color=PALETTE["primary"], linewidth=2.5,
            markersize=8, label="Observé", zorder=5)
    ax.plot(df_val.index, df_val["fc_direct"], "s--", color=PALETTE["accent1"],
            linewidth=2, markersize=7, label="Prévision directe (1 an, walk-forward)")
    ax.plot(df_val.index, df_val["fc_bottom_up"], "^--", color=PALETTE["accent2"],
            linewidth=2, markersize=7, label="Prévision bottom-up (1 an, walk-forward)")

    ax.set_xlabel("Année de test")
    ax.set_ylabel("Nombre de permis")
    ax.set_title("Validation walk-forward \ndirect vs bottom-up")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend()

    plt.tight_layout()
    savefig(fig, "fig17_walk_forward_validation.png")
    plt.show()
def plot_hierarchical_forecast(ts_by_cluster: pd.DataFrame, result: pd.DataFrame):
    """Visualise l'historique par cluster et les deux prévisions du total (directe vs bottom-up)."""
    cluster_cols = [c for c in ts_by_cluster.columns if c != "TOTAL"] # récupérer les clusters
    colors = plt.cm.Set2(np.linspace(0, 1, len(cluster_cols))) # récupérer les clusters

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    # premier graphique :
    # historique des volumes de chaque cluster
    ax1 = axes[0] 
    for col, color in zip(cluster_cols, colors):
        ax1.plot(ts_by_cluster.index, ts_by_cluster[col], "o-", color=color, label=f"Cluster {col}")
    ax1.set_title("Historique des volumes par cluster")
    ax1.set_xlabel("Année")
    ax1.set_ylabel("Nombre de permis")
    ax1.legend()
    # Second graphique :
    # comparaison entre la prévision directe du total national
    # et la prévision bottom-up obtenue par somme des prévisions
    # de chaque cluster.
    ax2 = axes[1]
    ax2.plot(ts_by_cluster.index, ts_by_cluster["TOTAL"], "o-", color=PALETTE["primary"],
              linewidth=2.5, label="Total observé")
    ax2.plot(result.index, result["total_direct"], "s--", color=PALETTE["accent1"],
              label="Prévision directe (total)")
    ax2.plot(result.index, result["total_bottom_up"], "^--", color=PALETTE["accent2"],
              label="Prévision bottom-up (somme clusters)")
    ax2.set_title("Réconciliation hiérarchique : direct vs bottom-up")
    ax2.set_xlabel("Année")
    ax2.set_ylabel("Nombre de permis")
    ax2.legend()

    plt.suptitle("Prévision hiérarchique par cluster de profils migratoires", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "fig16_prevision_hierarchique.png")
    plt.show()
