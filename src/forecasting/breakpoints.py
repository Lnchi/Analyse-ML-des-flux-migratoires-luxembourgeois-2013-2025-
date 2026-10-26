"""
3.7 Détection automatique de ruptures structurelles

Limitation :
    Avec uniquement 13 observations, les résultats sont interprétés avec
    l'analyse graphique et le contexte métier, notamment le choc COVID

Objectif :
    Identifier automatiquement des ruptures dans l'évolution
    annuelle des permis de séjour au Luxembourg.

Dans la version initiale du modèle ARIMAX, le choc COVID-19 était introduit
a priori à l'aide d'une variable binaire :
    - 1 en 2020 ;
    - 0 pour les autres années.

Cette approche suppose donc avant l'analyse que 2020 constitue une rupture.
La présente étape adopte une approche plus exploratoire : les ruptures sont
recherchées directement dans la série temporelle.

Deux méthodes complémentaires sont utilisées :

1. CUSUM
   La méthode CUSUM (Cumulative Sum) calcule la somme cumulée des écarts
   entre les observations et la moyenne de la série.
   Elle permet d'identifier le point où les écarts cumulés sont les plus
   importants.

2. Test de Chow
   Le test de Chow compare :
       - un modèle de tendance unique sur toute la période 
       - deux modèles de tendance séparés avant et après une année candidate.

   Hypothèse nulle H0 : absence de rupture structurelle.

   Si la p-value est inférieure à 5 %, H0 est rejetée et la rupture peut être
   considérée comme significative.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from src.config import PALETTE, savefig

def cusum_detection(series: pd.Series, threshold_sigma: float = 1.5) -> pd.DataFrame:
    """
    Détection exploratoire d'une rupture par CUSUM.
    Dans une série sans changement majeur de niveau, le CUSUM tend à
    osciller autour de zéro.

    Dans notre cas, la série présente une tendance importante.
    Le CUSUM peut donc rester positif ou négatif.
    Nous retenons alors comme point de rupture potentiel l'année
    correspondant à l'extremum absolu du CUSUM.

    Parameters :
    series : pd.Series : Série temporelle annuelle des permis.
    
    threshold_sigma : float : Paramètre pour compatibilité avec la version initiale.
        Il n'est pas utilisé pour déterminer l'extremum dans cette version.

    Returns : pd.DataFrame :
        - cusum : somme cumulée des écarts à la moyenne 
        - is_extremum : True pour l'année correspondant à l'extremum.
    """
    values = series.values.astype(float)# transformer la série en tableau numérique
    mean = values.mean() # calculer la moyenne
    # Calcul des écarts de chaque observation à la moyenne.
    deviations = values - mean
    # Somme cumulée des écarts.
    # Cette quantité permet d'identifier les périodes où les écarts
    # par rapport à la moyenne deviennent importants
    cusum = np.cumsum(deviations)# calculer la somme cumulée des écarts
    # Recherche de l'année où la valeur absolue du CUSUM est maximale
    # Cette année constitue notre indicateur de rupture potentielle
    extremum_idx = np.argmax(np.abs(cusum))
    # Création d'un vecteur booléen permettant d'identifier
    # uniquement l'extremum.
    is_extremum = np.zeros(len(cusum), dtype=bool)
    is_extremum[extremum_idx] = True

    result = pd.DataFrame({"cusum": cusum, "is_extremum": is_extremum}, index=series.index)
    print(f" CUSUM : point de rupture potentiel = {series.index[extremum_idx]} "
          f"(|CUSUM| = {abs(cusum[extremum_idx]):,.0f})")
    print("   NB : sur une série courte "
          "le CUSUM peut représenter la tendance générale plutôt qu'une rupture.")
    return result

def chow_test(series: pd.Series, break_year: int) -> dict:
    """
    Test de Chow pour une année de rupture candidate.
    Parameters :
    series : pd.Series
        Série temporelle étudiée.

    break_year : int
        Année candidate pour la rupture.

    Returns :
    dict
        Année testée, statistique F, p-value et indicateur de significativité.
    """
    # Années et valeurs sous forme de tableaux numériques
    years = series.index.values.astype(float)
    y = series.values.astype(float)
    # Séparation des observations avant et après l'année candidate.
    before = years <= break_year
    after = years > break_year

    n = len(y) # Nombre total d'observations.
    k = 2  # Deux paramètres du modèle linéaire : intercept + pente

    def rss_linear(x, yv):
        """
        Ajuste une tendance linéaire et calcule sa RSS.

        RSS = somme des carrés des résidus.

        Une RSS faible signifie que la droite reproduit relativement
        bien les observations.
        """
        # Une régression linéaire nécessite suffisamment d'observations
        if len(x) < k + 1: 
            return None, len(x) # S'il n'y a pas assez de données, on abandonne. Impossible d'ajuster une droite.
        slope, intercept, _, _, _ = stats.linregress(x, yv) # Calcule automatiquement intercept de la droite.
        pred = intercept + slope * x # Calcule les valeurs prédites.
        return np.sum((yv - pred) ** 2), len(x) # Calcule la RSS.

    rss_pooled, _ = rss_linear(years, y) # Premier modèle : Une seule droite pour toute la série.
    rss_before, n1 = rss_linear(years[before], y[before]) # Deuxième modèle : Une droite avant et 
    rss_after, n2 = rss_linear(years[after], y[after]) # Une droite après la rupture
    # Vérification qu'il existe suffisamment d'observations
    # dans les deux sous-périodes.
    if rss_before is None or rss_after is None or (n1 + n2 - 2 * k) <= 0:
        return {"break_year": break_year, "f_stat": np.nan, "p_value": np.nan, "significant": False} # Si un morceau contient trop peu de données le test est impossible.

    # RSS du modèle avec deux tendances.
    rss_unrestricted = rss_before + rss_after

    # Statistique de Chow.
    #
    # Si le modèle avec deux tendances améliore fortement
    # l'ajustement par rapport au modèle unique, la statistique F
    # devient élevée.
    f_stat = ((rss_pooled - rss_unrestricted) / k) / (rss_unrestricted / (n - 2 * k)) # Calcul de la statistique de Chow.
    # Si F élevé alors les deux modèles expliquent beaucoup mieux la série qu'un seul. Donc rupture probable.
    p_value = 1 - stats.f.cdf(f_stat, k, n - 2 * k) # Transformation du F en p-value

    return {
        "break_year": break_year, "f_stat": f_stat, "p_value": p_value,
        "significant": p_value < 0.05, 
    }


def scan_breakpoints(series: pd.Series, min_segment: int = 3) -> pd.DataFrame:
    """
    Teste automatiquement plusieurs années candidates.

    Afin d'éviter les ruptures situées trop près des extrémités de la série,
    on impose au minimum `min_segment` observations de chaque côté.

    Exemple avec min_segment=3 : Avant la rupture : au moins 3 observations & Après la rupture : au moins 3 observations

    Returns :
    pd.DataFrame
        Résultats du test de Chow pour chaque année candidate,
        triés par p-value croissante.
    """
    years = series.index.tolist() # Liste des années.
    candidates = years[min_segment:-min_segment] # On ne teste pas les extrémités. Il faut suffisamment d'observations avant et aprés.

    results = [chow_test(series, y) for y in candidates] # Application du test de Chow à chaque année candidate.
    df_results = pd.DataFrame(results).sort_values("p_value") # On rassemble tous les résultats. La meilleure rupture est celle ayant la plus petite p-value.
    return df_results


def plot_breakpoint_analysis(series: pd.Series, cusum_result: pd.DataFrame, chow_results: pd.DataFrame):
    """
    Visualise les résultats de la détection de ruptures.

    Le graphique contient deux parties :

    1. Série temporelle observée
       Les ruptures statistiquement significatives détectées par
       le test de Chow sont représentées par des lignes verticales.

    2. CUSUM
       L'année correspondant à l'extremum du CUSUM est mise en évidence.

    La visualisation permet de confronter :
        - l'évolution réelle de la série 
        - le résultat du CUSUM 
        - les résultats statistiques du test de Chow.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1 = axes[0]
    ax1.plot(series.index, series.values, "o-", color=PALETTE["primary"], linewidth=2.5, markersize=7) # Affiche la série temporelle.
    significant_breaks = chow_results[chow_results["significant"]]["break_year"].tolist() # On récupère uniquement les ruptures significatives
    for by in significant_breaks:
        ax1.axvline(by + 0.5, color=PALETTE["accent1"], linestyle="--", linewidth=1.5, alpha=0.8) # Trace une ligne verticale à chaque rupture.
    ax1.set_ylabel("Permis délivrés")
    ax1.set_title("Série observée et ruptures structurelles détectées par le test de Chow")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    ax2 = axes[1]
    colors = [PALETTE["accent1"] if b else PALETTE["neutral"] for b in cusum_result["is_extremum"]] # Si rupture couleur rouge. Sinon gris.
    ax2.bar(cusum_result.index, cusum_result["cusum"], color=colors, alpha=0.85) # Histogramme du CUSUM.
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("CUSUM")
    ax2.set_xlabel("Année")
    ax2.set_title("Somme cumulée des écarts à la moyenne (CUSUM)\n")
                   

    plt.tight_layout() # Évite que les éléments du graphique se chevauchent.
    savefig(fig, "fig15_detection_ruptures.png")
    plt.show()
    print("\n" + "=" * 70)
    print("\n Ruptures significatives (test de Chow, p<0.05) :")
    print("=" * 70)
    if significant_breaks: 
        print(chow_results[chow_results["significant"]].round(4).to_string(index=False)) # S'il existe une rupture significative, on affiche F et p-value
    else:
        print("   Aucune rupture statistiquement significative détectée à p<0.05. La taille très faible de l'échantillon (13 observations) limite la puissance du test.")
        print("   Résultats les plus proches du seuil :") # on affiche les trois années les plus proches du seuil de significativité.
        print(chow_results.head(3).round(4).to_string(index=False))
