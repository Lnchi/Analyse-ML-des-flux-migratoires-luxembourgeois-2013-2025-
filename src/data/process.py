"""
1.2-Nettoyage et restructuration des séries EUROSTAT.

process_eurostat() transforme n'importe quelle série EUROSTAT (format large,
une colonne par année) en un DataFrame long normalisé :
    [geo, citizen, reason_cat, year, permits]

Étapes : identification des colonnes années, harmonisation des noms de
colonnes (EUROSTAT varie selon la série), filtrage géographique (LU + pays
voisins), filtrage des dimensions d'agrégation (durée/âge/sexe/reason = TOTAL),
reshape wide -> long, filtrage 2013-2025, suppression des agrégats
statistiques EUROSTAT et des valeurs manquantes.
"""

import numpy as np
import pandas as pd

from src.config import AGG_CODES, COUNTRIES_NEIGHBORS, SERIES_MAP, SERIES_REASON_FILTER
from src.data.collect import get_series_df


def process_eurostat(df: pd.DataFrame, reason_cat: str, reason_filter: str | None = None) -> pd.DataFrame:
    """
    Nettoie et restructure une série EUROSTAT et filtre immédiatement
    sur LU + pays voisins pour réduire la mémoire.

    Args:
        df: DataFrame brut EUROSTAT (format large)
        reason_cat: catégorie métier associée à la série (WORK/FAMI/EDUC/OTHP)
        reason_filter: modalité de la dimension 'reason' à conserver
            None = ne filtre pas, toutes les
            modalités sont sommées lors de l'agrégation (cas d'une
            série sans ligne d'agrégat). Voir SERIES_REASON_FILTER dans
            config.py pour la valeur par série et sa justification.

    Returns:
        DataFrame long avec colonnes [geo, citizen, reason_cat, year, permits]
    """
    geo_filter = COUNTRIES_NEIGHBORS

    # Identifier les colonnes années
    year_cols = [c for c in df.columns if str(c).isdigit() and 2000 <= int(c) <= 2030]
    id_cols = [c for c in df.columns if c not in year_cols]

    # Normalisation des noms de colonnes (EUROSTAT peut varier selon la série)
    col_map = {}
    for c in id_cols:
        cl = c.lower()
        if "geo" in cl:
            col_map[c] = "geo"
        elif "citizen" in cl:
            col_map[c] = "citizen"
        elif "sex" in cl:
            col_map[c] = "sex"
        elif "age" in cl:
            col_map[c] = "age"
        elif "duration" in cl:
            col_map[c] = "duration"
        elif "reason" in cl:
            col_map[c] = "reason"
    df = df.rename(columns=col_map)

    # Filtrer sur les pays avant le melt (gain mémoire majeur)
    if "geo" in df.columns:
        df = df[df["geo"].isin(geo_filter)]

    # Filtrer les dimensions d'agrégation EUROSTAT : on garde les modalités "TOTAL"
    if "duration" in df.columns:
        df = df[df["duration"] == "TOTAL"]
    if "age" in df.columns:
        df = df[df["age"] == "TOTAL"]
    if "sex" in df.columns:
        df = df[df["sex"].isin(["T", "TOTAL"])]
    if "reason" in df.columns:  # noqa: SIM102
        # Sans ce filtre,
        # le melt() produit une ligne par sous-catégorie, et le groupby()
        # .sum() en aval double-compte les permis (sous-catégories + leur
        # agrégat additionnés ensemble).
        if reason_filter is not None:
            df = df[df["reason"] == reason_filter]
        # sinon (reason_filter is None) : pas de ligne d'agrégat identifiée
        # pour cette série -> on garde toutes les modalités, qui sont
        # exclusives et se somment correctement.

    # Reshape wide -> long
    df_long = df.melt(
        id_vars=[c for c in df.columns if c not in year_cols],
        value_vars=year_cols,
        var_name="year",
        value_name="permits",
    )
    df_long["year"] = df_long["year"].astype(int)
    df_long["permits"] = pd.to_numeric(df_long["permits"], errors="coerce")
    df_long["reason_cat"] = reason_cat

    # Filtrer 2013-2025
    df_long = df_long[df_long["year"].between(2013, 2025)]

    # Supprimer les agrégats EUROSTAT + codes non nationaux/spécifiques
    if "citizen" in df_long.columns:
        df_long = df_long[~df_long["citizen"].isin(AGG_CODES)]

    # On ne conserve pas ces dimensions dans la base finale
    for col in ["duration", "age", "sex", "reason"]:
        if col in df_long.columns:
            df_long = df_long.drop(columns=col)

    return df_long.dropna(subset=["permits"])


"""def verify_reason_aggregate(df_tmp: pd.DataFrame, series_code: str, reason_filter: str | None) -> None:
    
    Vérifie automatiquement, à chaque exécution du pipeline, que la modalité
    reason_filter retenue pour une série est bien l'agrégat (somme) des
    autres modalités de la dimension 'reason'.

    la comparaison est restreinte aux mêmes modalités
    (age/sex/duration = TOTAL) que celles réellement appliquées par
    process_eurostat(). Sans cette restriction, sommer sur toutes les
    ventilations démographiques confondues peut créer un écart artificiel
    même quand le filtre 'reason' est correct.
    

    # Si aucun filtre "reason" n'est défini pour cette série,
    # il n'y a rien à vérifier
    if reason_filter is None:
        return

    df = df_tmp.copy()
    # Normalisation minimale des noms de colonnes (même logique que process_eurostat)
    col_map = {}
    for c in df.columns:
        cl = str(c).lower()
        if "age" in cl:
            col_map[c] = "age"
        elif "sex" in cl:
            col_map[c] = "sex"
        elif "duration" in cl:
            col_map[c] = "duration"
        elif "reason" in cl:
            col_map[c] = "reason"
    df = df.rename(columns=col_map)

    if "reason" not in df.columns:
        return

    # Restreindre aux mêmes modalités que process_eurostat avant de comparer
    if "age" in df.columns:
        df = df[df["age"] == "TOTAL"]
    if "sex" in df.columns:
        df = df[df["sex"].isin(["T", "TOTAL"])]
    if "duration" in df.columns:
        df = df[df["duration"] == "TOTAL"]

    # Identification des colonnes correspondant aux années
    year_cols = [c for c in df.columns if str(c).isdigit() and 2000 <= int(c) <= 2030]
    # Liste de tous les autres codes "reason"
    # (les sous-catégories du code parent)
    other_codes = [c for c in df["reason"].unique() if c != reason_filter]
    # Sélection des lignes correspondant au code parent,
    # puis empilement de toutes les années dans une seule colonne
    # grâce à stack().
    total_val = pd.to_numeric(
        df.loc[df["reason"] == reason_filter, year_cols].stack(), errors="coerce"
    ).sum()
    # Même principe mais pour tous les autres codes.
    sum_others = pd.to_numeric(
        df.loc[df["reason"].isin(other_codes), year_cols].stack(), errors="coerce"
    ).sum()
    # Calcul de l'écart relatif entre les deux volumes
    # Si le code parent vaut zéro,
    # on évite une division par zéro.
    ecart_pct = abs(total_val - sum_others) / total_val * 100 if total_val else float("nan")
    # Si l'écart est inférieur ou égal à 2 %,
    # on considère que le filtre est cohérent.
    status = "Ok" if ecart_pct <= 2 else "ATTENTION"
    print(f"   Vérif. reason='{reason_filter}' vs somme({other_codes}) : "
          f"{total_val:,.0f} vs {sum_others:,.0f} (écart {ecart_pct:.1f}%) {status}")
    if ecart_pct > 2:
        print(f"   -> Le filtre '{reason_filter}' pour {series_code} ne correspond peut-être pas "
              f"à un véritable agrégat. Vérifier sur le portail EUROSTAT.")

"""

def verify_reason_aggregate(
    df_tmp: pd.DataFrame,
    series_code: str,
    reason_filter: str | None,
    geo_of_interest: list[str] | None = None,
) -> None:
    """
    Vérifie que la modalité parent (reason_filter) correspond bien à la
    somme de ses sous-modalités, ligne par ligne (geo + citizen + année),
    avec les mêmes filtres que process_eurostat (age/sex/duration = TOTAL).

    Rapporte deux statistiques distinctes :
      - "TOUS PAYS" : sur l'ensemble des géographies EUROSTAT (informatif)
      - "LU/BE/FR/DE uniquement" : sur les seuls pays utilisés
        par le pipeline (COUNTRIES_NEIGHBORS) pour juger si le filtre est fiable pour le projet.

    Certains pays déclarants (souvent hors zone UE, ou petits
    contributeurs) ne publient pas systématiquement la ventilation détaillée
    par sous-motif pour toutes leurs combinaisons citoyenneté/année, seul
    l'agrégat est transmis à EUROSTAT. Ça crée des écarts globaux qui
    n'affectent pas forcément LU/BE/FR/DE 
    """
    # Si aucun filtre de motif n'est fourni, il n'y a rien à comparer.
    if reason_filter is None:
        return
    # On travaille sur une copie afin de ne pas modifier le DataFrame
    df = df_tmp.copy()
    # Harmonisation des noms de colonnes
    col_map = {}
    for c in df.columns:
        cl = str(c).lower()
        if "geo" in cl:
            col_map[c] = "geo"
        elif "citizen" in cl:
            col_map[c] = "citizen"
        elif "age" in cl:
            col_map[c] = "age"
        elif "sex" in cl:
            col_map[c] = "sex"
        elif "duration" in cl:
            col_map[c] = "duration"
        elif "reason" in cl:
            col_map[c] = "reason"
    df = df.rename(columns=col_map)
    #  Vérification des colonnes indispensables
    required = ["geo", "citizen", "reason"]
     # Identification des colonnes manquantes.
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"   Vérification impossible pour {series_code} : colonnes manquantes {missing}")
        return
    # Application des mêmes filtres que process_eurostat()
    if "age" in df.columns:
        df = df[df["age"] == "TOTAL"]
    if "sex" in df.columns:
        df = df[df["sex"].isin(["T", "TOTAL"])]
    if "duration" in df.columns:
        df = df[df["duration"] == "TOTAL"]
    # Identification des colonnes correspondant aux année
    year_cols = [c for c in df.columns if str(c).isdigit() and 2000 <= int(c) <= 2030]
    if not year_cols:
        print(f"   Aucune colonne année trouvée pour {series_code}.")
        return
    # Passage du format large au format long
    df_long = df.melt(
        id_vars=["geo", "citizen", "reason"],
        value_vars=year_cols,
        var_name="year",
        value_name="permits",
    )
    df_long["year"] = df_long["year"].astype(int)
    # Conversion des valeurs de permis en valeurs numériques.
    # Les valeurs impossibles à convertir deviennent NaN.
    df_long["permits"] = pd.to_numeric(df_long["permits"], errors="coerce")
    # Séparation du motif parent et des sous-motifs
    parent = df_long[df_long["reason"] == reason_filter].copy()
    # "children" contient les autres modalités de reason
    children = df_long[df_long["reason"] != reason_filter].copy()
    # Si aucune ligne correspondant au motif parent n'existe,
    # la vérification ne peut pas être réalisée.
    if parent.empty:
        print(f"   ATTENTION : aucune ligne '{reason_filter}' trouvée pour {series_code}.")
        return
    # Calcul de la somme des sous-modalités
    # Pour chaque combinaison :
    #
    #     geo + citizen + year
    #
    # on additionne les valeurs des différentes modalités présentes
    # dans "children".
    children_sum = (
        children.groupby(["geo", "citizen", "year"], as_index=False)["permits"]
        .sum(min_count=1)
        .rename(columns={"permits": "sum_children"})
    )
    # Calcul de la valeur du motif parent
    parent_sum = (
        parent.groupby(["geo", "citizen", "year"], as_index=False)["permits"]
        .sum(min_count=1)
        .rename(columns={"permits": "parent"})
    )

    # Mise en correspondance parent / sous-modalités
    # On associe la valeur du parent à la somme de ses sous-modalités
    # pour chaque :
    #
    #     geo + citizen + year
    #
    # Le "inner" conserve uniquement les combinaisons pour lesquelles
    # le parent ET les sous-modalités existent.
    check = parent_sum.merge(children_sum, on=["geo", "citizen", "year"], how="inner")
    # Suppression des lignes pour lesquelles l'une des deux valeurs
    # à comparer est absente.
    check = check.dropna(subset=["parent", "sum_children"])
    # Si aucune combinaison comparable n'existe, on arrête la vérification.
    if check.empty:
        print(f"   ATTENTION : aucune combinaison comparable pour {series_code}.")
        return
    
    # Calcul de l'écart absolu
    # On calcule la différence absolue entre :
    #
    #     motif parent
    #     somme des sous-motifs
    check["diff"] = (check["parent"] - check["sum_children"]).abs()
    # np.nan (pas pd.NA) : évite le FutureWarning de downcasting sur .fillna()
    # Calcul de l'écart relatif en pourcentage
    # Formule :
    #
    #     écart (%) =
    #     |parent - somme enfants| / parent × 100
    #
    # Les parents égaux à zéro sont remplacés par NaN afin d'éviter
    # une division par zéro.
    check["diff_pct"] = (check["diff"] / check["parent"].replace(0, np.nan) * 100).astype(float)
    # Fonction interne pour afficher les statistiques
    # Cette fonction évite de dupliquer le même calcul pour :
    #   - tous les pays
    #   - les pays d'intérêt uniquement
    def _report(subset: pd.DataFrame, label: str) -> None:
        # Si aucune combinaison n'est disponible pour le sous-ensemble,
        # on affiche un message
        if subset.empty:
            print(f"      [{label}] Aucune combinaison à vérifier.")
            return
        max_diff = subset["diff_pct"].dropna().max()
        mean_diff = subset["diff_pct"].dropna().mean()
        # Nombre de combinaisons considérées comme conformes.
        #
        # Le seuil retenu ici est de ±2 %
        nb_ok = (subset["diff_pct"].fillna(0) <= 2).sum()
        # Nombre total de combinaisons comparables.
        nb_total = len(subset)
        print(f"      [{label}] Conforme à ±2% : {nb_ok:,}/{nb_total:,} "
              f"| écart moyen {mean_diff:.2f}% | écart max {max_diff:.2f}%")

    print(f"   Vérif. reason='{reason_filter}' pour {series_code} :")
    # Cette première vérification est informative :
    # elle prend en compte toutes les géographies présentes
    # dans le dataset EUROSTAT.
    _report(check, "TOUS PAYS")
    # Rapport limité aux pays utilisés par le pipeline
    if geo_of_interest:
        check_relevant = check[check["geo"].isin(geo_of_interest)]
        _report(check_relevant, "LU/BE/FR/DE uniquement")
        # Identification des anomalies sur les pays d'intérêt
        # Une combinaison est considérée comme anomalie lorsque
        # l'écart entre le parent et les sous-modalités dépasse 2 %.
        anomalies = check_relevant[check_relevant["diff_pct"].fillna(0) > 2].sort_values("diff_pct", ascending=False)
        if anomalies.empty:
            print(f"      -> '{reason_filter}' est cohérent sur LU/BE/FR/DE : filtre VALIDE pour ce pipeline.")
        # Au moins une combinaison présente un écart > 2 %.
        # On signale donc qu'une analyse supplémentaire est nécessaire.
        else:
            print(f"      -> ATTENTION : {len(anomalies)} anomalies sur LU/BE/FR/DE, à examiner :")
            print(anomalies[["geo", "citizen", "year", "parent", "sum_children", "diff_pct"]]
                  .head(10).to_string(index=False))
        
def build_df_all() -> pd.DataFrame:
    """
    Traite série par série (WORK/FAMI/EDUC/OTHP) et concatène le résultat
    dans un DataFrame unique, normalisé et prêt pour l'EDA / le ML.

    Returns:
        df_all: DataFrame long [geo, citizen, reason_cat, year, permits]
    """
    processed_frames = []
    for series_code, reason_cat in SERIES_MAP.items():
        try:
            print(f"{series_code}...")
            df_tmp = get_series_df(series_code)
            print(f"   brut : {df_tmp.shape} / ", end="")
            reason_filter = SERIES_REASON_FILTER.get(series_code)
            verify_reason_aggregate(df_tmp, series_code, reason_filter, geo_of_interest=COUNTRIES_NEIGHBORS,)
            df_clean = process_eurostat(df_tmp, reason_cat, reason_filter=reason_filter)
            print(f"nettoyé : {df_clean.shape}")
            processed_frames.append(df_clean)
            del df_tmp
        except Exception as e:
            print(f"{series_code} indisponible : {e}")

    df_all = pd.concat(processed_frames, ignore_index=True)
    print(f"\n df_all final : {df_all.shape}")
    print(f"   Pays     : {sorted(df_all['geo'].unique())}")
    print(f"   Années   : {sorted(df_all['year'].unique())}")
    print(f"   Motifs   : {sorted(df_all['reason_cat'].unique())}")
    print(f"   Nationalités (LU) : {df_all[df_all['geo'] == 'LU']['citizen'].nunique()}")

    return df_all