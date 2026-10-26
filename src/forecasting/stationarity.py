"""
3.1 Tests de stationnarité (ADF - Augmented Dickey-Fuller, KPSS).
3.2 Analyse ACF / PACF (détermination visuelle des ordres p et q).

Hypothèse H1 : la série des permis luxembourgeois présente une tendance
croissante non-stationnaire avec rupture structurelle en 2020 (choc COVID-19).
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from src.config import PALETTE, savefig


def get_series(ts_lu: pd.DataFrame) -> pd.Series:
    """
    Extrait la série temporelle des permis luxembourgeois.

    Le DataFrame contient:
    - 'year' : année
    - 'total_permits' : nombre total de permis

    L'année devient l'index de la série et les valeurs sont converties
    en float pour être compatibles avec les tests statistiques.
    """
    # On utilise l'année comme index et on récupère uniquement
    # la colonne contenant le nombre total de permis.
    return ts_lu.set_index("year")["total_permits"].astype(float)


def stationarity_report(series: pd.Series, name: str = "Série") -> dict:
    """
    Applique les tests ADF et KPSS (complémentaires) :
      - H0 ADF  : présence d'une racine unitaire (non-stationnaire)
      - H0 KPSS : stationnarité (autour d'une tendance) lorsque regression="ct" 
    """
    # Dictionnaire qui contiendra les résultats des deux tests
    results = {}
    # Test ADF : Augmented Dickey-Fuller

    # autolag="AIC" permet de sélectionner automatiquement le nombre
    # de retards (lags) utilisé par le test selon le critère AIC.
    adf_stat, adf_p, adf_lags, adf_nobs, adf_cvals, _ = adfuller(series, autolag="AIC")
    results["ADF"] = {
        "statistic": adf_stat, "p-value": adf_p,
        # Si p < 5 %, on rejette H0.
        # On considère alors la série comme stationnaire
        # selon le test ADF.
        "critical_5pct": adf_cvals["5%"], "stationary_H0rejected": adf_p < 0.05,
    }

    # Test KPSS
    # regression="ct" cad le test vérifie la stationnarité
    # autour d'une constante et d'une tendance.
     # nlags="auto" laisse statsmodels déterminer le nombre de retards 

    kpss_stat, kpss_p, kpss_lags, kpss_cvals = kpss(series, regression="ct", nlags="auto")
    results["KPSS"] = {
        "statistic": kpss_stat, "p-value": kpss_p,
        # Pour le KPSS, H0 correspond à la stationnarité.
        # Si p > 5 %, on ne rejette donc pas H0.
        "critical_5pct": kpss_cvals["5%"], "trend_stationary_H0notrejected": kpss_p > 0.05,
    }
    # Affichage des résultats
    print("_______________________________")
    print(f"Tests de stationnarité : {name}")
    print("_______________________________")
    # Interprétation du test ADF
    # H0 : présence d'une racine unitaire.
    print(f" ADF  / stat={adf_stat:.4f} / p={adf_p:.4f} / "
          f"{'Non-stationnaire' if adf_p > 0.05 else 'Stationnaire'} (H0: racine unitaire)")

    # Interprétation du test KPSS.
    # H0 : stationnarité autour d'une tendance.
    print(f"  KPSS / stat={kpss_stat:.4f} / p={kpss_p:.4f} / "
          f"{'Trend-stationnaire' if kpss_p > 0.05 else 'Non-stationnaire'} (H0: stationnarité)")

    
    # Diagnostic combiné ADF + KPSS
    # ADF non significatif + KPSS significatif :
    if adf_p > 0.05 and kpss_p < 0.05:
        print("  DIAGNOSTIC : Série I(1), différenciation d'ordre 1 requise")
    # ADF non significatif + KPSS non significatif :
    elif adf_p > 0.05 and kpss_p > 0.05:
        print("  DIAGNOSTIC : Série DS (difference-stationary)")
    # ADF significatif + KPSS non significatif :
    elif adf_p < 0.05 and kpss_p > 0.05:
        print("  DIAGNOSTIC : Série stationnaire")

    return results


def run_stationarity_tests(series: pd.Series) -> tuple[dict, dict, pd.Series]:
    """Teste la série originale puis différenciée (d=1) et retourne les deux séries.
    Retourne :
    - les résultats ADF/KPSS sur la série originale ;
    - les résultats ADF/KPSS sur la série différenciée ;
    - la série différenciée elle-même."""

    # Tests sur la série originale
    res_original = stationarity_report(series, "Série originale")
    # Différenciation d'ordre 1
    # La différenciation calcule :
    #
    #     y(t) - y(t-1)
    #
    # Elle permet généralement de supprimer une tendance
    # et de rendre une série non-stationnaire stationnaire.
    # La première observation devient NaN puisqu'il n'existe
    # pas d'observation précédente.
    series_diff = series.diff().dropna()
    # Tests sur la série différenciée
    res_diff = stationarity_report(series_diff, "Série différenciée (d=1)")
    return res_original, res_diff, series_diff


def plot_acf_pacf(series: pd.Series, series_diff: pd.Series, max_lags: int | None = None):
    """Trace ACF/PACF pour la série originale et différenciée (détermine p, q pour ARIMA)."""

    # Détermination du nombre de retards (lags)
    # Si max_lags n'est pas fourni, on utilise au maximum 5 retards.
    if max_lags is None:
        max_lags = min(5, len(series) // 2 - 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    # ACF de la série originale
    plot_acf(series, lags=max_lags, ax=axes[0, 0], alpha=0.05,
             title="ACF - Série originale", color=PALETTE["primary"])
    # PACF de la série originale
    plot_pacf(series, lags=max_lags, ax=axes[0, 1], alpha=0.05,
              title="PACF - Série originale", color=PALETTE["primary"])
    # ACF de la série différenciée
    plot_acf(series_diff, lags=max_lags, ax=axes[1, 0], alpha=0.05,
             title="ACF - Série différenciée (d=1)", color=PALETTE["accent2"])
    # PACF de la série différenciée
    plot_pacf(series_diff, lags=max_lags, ax=axes[1, 1], alpha=0.05,
              title="PACF - Série différenciée (d=1)", color=PALETTE["accent2"])

    plt.tight_layout()
    savefig(fig, "fig6_acf_pacf.png")
    plt.show()
