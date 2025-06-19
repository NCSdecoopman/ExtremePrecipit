import argparse
import os
from pathlib import Path
from tqdm.auto import tqdm

from src.utils.logger import get_logger
from src.utils.config_tools import load_config
from src.utils.data_utils import load_data

import numpy as np
import polars as pl
from numba import njit
from scipy.stats import chi2


def build_x_ttilde(df: pl.DataFrame, best_model: pl.DataFrame, break_year: int | None = None) -> pl.DataFrame:
    # Renommer pour harmoniser
    df = df.rename({"max_mm_j": "x"})

    # Calcul tmin et tmax par NUM_POSTE
    tminmax = (
        df.group_by("NUM_POSTE")
        .agg([
            pl.col("year").min().alias("tmin"),
            pl.col("year").max().alias("tmax")
        ])
    )
    df = df.join(tminmax, on="NUM_POSTE", how="left")

    # Ajoute colonne `has_break` selon le modèle
    break_info = best_model.select([
        pl.col("NUM_POSTE"),
        pl.col("model").str.contains("_break_year").alias("has_break")
    ])
    df = df.join(break_info, on="NUM_POSTE", how="left")

    # Ajoute colonne `t_plus` si rupture, sinon null
    df = df.with_columns([
        pl.when(pl.col("has_break"))
          .then(pl.lit(break_year))
          .otherwise(None)
          .alias("t_plus")
    ])

    # Calcul de t_tilde selon présence ou non de la rupture
    df = df.with_columns([
        pl.when(~pl.col("has_break"))
          .then((pl.col("year") - pl.col("tmin")) / (pl.col("tmax") - pl.col("tmin")))
          .otherwise(
              pl.when(pl.col("year") < pl.col("t_plus"))
                .then(0.0)
                .otherwise((pl.col("year") - pl.col("t_plus")) / (pl.col("tmax") - pl.col("t_plus")))
          )
          .alias("t_tilde")
    ])

    return df.select([
        "NUM_POSTE", "x", "t_tilde",
        "tmin", "tmax",
        "has_break",
        "t_plus"
    ])




def calculate_z_T1(T:int, mu1: float, sigma1: float, xi0: float) -> float:
    """
    Calcul z_T,1 dans z_T (t) = z_T,0 + z_T,1 * t
    """
    CT = (-np.log(1 - 1/T))**(-xi0) - 1
    return mu1 + (sigma1 / xi0) * CT

def compute_calculate_zT1(T, df):
    return df.select([
        pl.col("NUM_POSTE", "model", "mu0", "mu1", "sigma0", "sigma1", "xi"),
        pl.struct(["mu1", "sigma1", "xi"])
        .map_elements(lambda s: calculate_z_T1(T, s["mu1"], s["sigma1"], s["xi"]),
                        return_dtype=pl.Float64)
        .alias("z_T1")
    ])


# Fonctions de mu et sigma en fonction de z
@njit
def mu1_zT1_func(T, z_T1, hat_sigma1, hat_xi0):
    CT = (-np.log(1 - 1/T)) ** (-hat_xi0) - 1
    return z_T1 - (hat_sigma1 / hat_xi0) * CT

@njit
def sigma1_zT1_func(T, z_T1, hat_mu1, hat_xi0):
    CT = (-np.log(1 - 1/T)) ** (-hat_xi0) - 1
    return hat_xi0 * (z_T1 - hat_mu1) / CT



# Fonctions de vraisemblance profilées
@njit
def log_likelihood_M1(T, z_T1, x, t_tilde, mu0, sigma0, mu1, sigma1, xi0):
    """
    Effet temporel sur mu
    """
    mu1 = mu1_zT1_func(T, z_T1, sigma1, xi0)
    z = (x - (mu0 + mu1 * t_tilde)) / sigma0
    term = 1 + xi0 * z
    if np.any(term <= 0):
        return -np.inf  # hors du domaine de définition
    return -np.sum(
        np.log(sigma0)
        + (1 + 1/xi0) * np.log(term)
        + term ** (-1 / xi0)
    )

@njit
def log_likelihood_M2(T, z_T1, x, t_tilde, mu0, sigma0, mu1, sigma1, xi0):
    """
    Effet temporel sur sigma
    """
    sigma1 = sigma1_zT1_func(T, z_T1, mu1, xi0)
    sigma = sigma0 + sigma1 * t_tilde
    z = (x - mu0) / sigma
    term = 1 + xi0 * z
    if np.any(sigma <= 0) or np.any(term <= 0):
        return -np.inf
    return -np.sum(
        np.log(sigma)
        + (1 + 1/xi0) * np.log(term)
        + term ** (-1 / xi0)
    )

@njit
def log_likelihood_M3(T, z_T1, x, t_tilde, mu0, sigma0, mu1, sigma1, xi0):
    """
    Effet temporel sur mu et sigma
    """
    mu1 = mu1_zT1_func(T, z_T1, sigma1, xi0)
    sigma = sigma0 + sigma1 * t_tilde
    mu = mu0 + mu1 * t_tilde
    z = (x - mu) / sigma
    term = 1 + xi0 * z
    if np.any(sigma <= 0) or np.any(term <= 0):
        return -np.inf
    return -np.sum(
        np.log(sigma)
        + (1 + 1/xi0) * np.log(term)
        + term ** (-1 / xi0)
    )

def select_log_likelihood(model: str):
    """Renvoi la fonction de vraisemblance du modèle selectionné"""
    model = model.lower()          # pour tolérer « M1 », « m1 », etc.
    if "m1" in model:              
        return log_likelihood_M1
    elif "m2" in model:
        return log_likelihood_M2
    elif "m3" in model:
        return log_likelihood_M3
    else:
        raise ValueError(f"Modèle non reconnu : {model}")




def profile_loglikelihood_per_station(
    T: int,
    df_series: pl.DataFrame,
    df_zT1: pl.DataFrame,
    threshold: float = 0.10,
    span: float = 100,           # intervalle autour de z_T1   
    precision: float = 0.05     # résolution désirée
) -> pl.DataFrame:
    """
    Calcule la log-vraisemblance profilée pour chaque station autour de z_T1.
    
    Returns un DataFrame Polars avec :
    NUM_POSTE | ic_lower | ic_upper | significant
    """
    n_points = int(2 * span / precision) + 1 # Nombre de points nécessaire
    # longueur = 2*span et nb_intervalle = longueur/précision

    # Merge des deux tables
    merged = df_series.join(df_zT1, on="NUM_POSTE", how="inner")
    rows = []

    for group_key, group in tqdm(list(merged.group_by("NUM_POSTE")), desc="Profiling stations"):
        # Forcer clé de groupe en string
        num_poste = str(group_key) if isinstance(group_key, str) else str(group_key[0])

        # Données
        x = group["x"].to_numpy()
        t_tilde = group["t_tilde"].to_numpy()

        # Paramètres du modèle
        params = group[["model", "mu0", "mu1", "sigma0", "sigma1", "xi", "z_T1"]].unique().row(0)
        model, mu0, mu1, sigma0, sigma1, xi0, z_T1_hat = params
        
        # Grille autour de z_T1_hat
        z_grid = np.linspace(z_T1_hat - span, z_T1_hat + span, n_points)

        # Fonction de vraisemblance
        loglik_func = select_log_likelihood(model)
        loglik_values = np.array([
            loglik_func(T, z, x, t_tilde, mu0, sigma0, mu1, sigma1, xi0)
            for z in z_grid
        ])

        # Vraisemblance exacte au point z_T1
        loglik_hat = loglik_func(T, z_T1_hat, x, t_tilde, mu0, sigma0, mu1, sigma1, xi0)

        # Déviance
        deviance = 2 * (loglik_hat - loglik_values)
        chi2_threshold = chi2.ppf(1 - threshold, df=1)

        # IC
        valid = deviance <= chi2_threshold
        if np.any(valid):
            z_valid = z_grid[valid]
            ic_lower, ic_upper = z_valid[0], z_valid[-1]
        else:
            ic_lower = ic_upper = np.nan

        # Significativité
        significant = False if np.isnan(ic_lower) or np.isnan(ic_upper) else not (ic_lower <= 0 <= ic_upper)

        rows.append({
            "NUM_POSTE": num_poste,
            "ic_lower": ic_lower,
            "ic_upper": ic_upper,
            "significant": significant
        })

    return pl.DataFrame(rows).with_columns(pl.col("NUM_POSTE").cast(pl.Utf8))




def main(config, args, T: int = 10):
    global logger
    logger = get_logger(__name__)

    echelles = config.get("echelles", "quotidien")
    season = config.get("season", "hydro")
    model_path = config.get("config", "config/observed_settings.yaml")
    gev_dir = config["gev"]["path"]["outputdir"]
    break_year = config.get("gev", {}).get("break_year", 1985)
    
    for echelle in echelles:
        logger.info(f"--- Traitement échelle: {echelle.upper()} saison: {season}---")

        # ETAPE 1 
        # Ouverture de la table NUM_POSTE ┆ model ┆ mu0 ┆ mu1 ┆ sigma0 ┆ sigma1 ┆ xi ┆ log_likelihood
        path_dir = Path(gev_dir) / echelle / args.season
        best_model_path = path_dir / "gev_param_best_model.parquet"
        best_model = pl.read_parquet(best_model_path)

        # ETAPE 2
        # Ouverture des séries de maxima
        # Choix du répertoire de lecture
        model_path_name = Path(model_path).name

        if model_path_name == "observed_settings.yaml":
            input_dir = Path(config["statistics"]["path"]["outputdir"]) / echelle
            logger.info(f"Source STATION détectée → lecture dans : {input_dir}")

        elif model_path_name == "modelised_settings.yaml":
            input_dir = Path(config["statistics"]["path"]["outputdir"]) / "horaire"
            logger.info(f"Source AROME détectée → lecture dans : {input_dir}")

        else:
            logger.error(f"Nom de fichier de configuration non reconnu : {model_path_name}")
            sys.exit(1)

        # Fixation de l'échelle pour le choix des colonnes à lire
        mesure = "max_mm_h" if echelle == "horaire" else "max_mm_j"
        cols = ["NUM_POSTE", mesure]

        # Liste des années disponibles
        years = [
            int(name) for name in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, name)) and name.isdigit() and len(name) == 4
        ]

        if years:
            min_year = min(years) if echelle == "quotidien" else 1990 # Année minimale
            max_year = 2022 # max(years)
        else:
            logger.error("Aucune année valide trouvée.")

        if season in ["hydro", "djf"]:
            min_year+=1 # On commence en 1960

        logger.info(f"Chargement des données de {min_year} à {max_year} : {input_dir}")
        df = load_data(input_dir, season, echelle, cols, min_year, max_year)

        # Filtrer les stations avec un résultat de GEV
        df = df.filter(pl.col("NUM_POSTE").is_in(best_model["NUM_POSTE"].to_list()))
        assert df["NUM_POSTE"].n_unique() == best_model["NUM_POSTE"].n_unique(), \
            "Les deux DataFrames n'ont pas le même nombre de NUM_POSTE uniques"

        # Filtrer les lignes avec des valeurs non NaN
        df = df.drop_nulls()

        # Normaliser t en t_tilde ou t_tilde* suivant la présence ou non d'un break_point
        df_series = build_x_ttilde(df, best_model, break_year)

        # Ajoute une colonne "z_T1" au DataFrame best_model
        df_zT1 = compute_calculate_zT1(T, best_model)

        # log-vraisemblance profilée pour chaque station autour de z_T1
        table_ic = profile_loglikelihood_per_station(
            T,
            df_series,
            df_zT1,
            threshold=args.threshold  # passage du seuil pour chi²
        )

        # Récupère model et z_T1 depuis df_zT1
        model_info = df_zT1.select(["NUM_POSTE", "model", "z_T1"])

        # Jointure avec les paramètres du modèle
        table_ic = table_ic.with_columns(pl.col("NUM_POSTE").cast(pl.Utf8))     # correspondance sur NUM_POSTE
        model_info = model_info.with_columns(pl.col("NUM_POSTE").cast(pl.Utf8)) # correspondance sur NUM_POSTE
        final_table = table_ic.join(model_info, on="NUM_POSTE", how="left")

        # Réorganisation des colonnes
        final_table = final_table.select(["NUM_POSTE", "model", "z_T1", "ic_lower", "ic_upper", "significant"])

        # DONNER LES RESULTATS AVEC UNE PENTE PAR 10 ANS
        # 1) durée par station Δtᵢ = tmaxᵢ – tminᵢ (ou t+ lors d'un break_point)
        delta_years = (
            df_series
            .group_by("NUM_POSTE")
            .agg(
                (
                    pl.when(pl.max("has_break"))               # booléen UNIQUE par station
                    .then(pl.max("tmax") - pl.max("t_plus")) # tmax – t_plus si rupture
                    .otherwise(pl.max("tmax") - pl.min("tmin"))  # tmax – tmin sinon
                ).alias("delta_year")
            )
            .with_columns(pl.col("NUM_POSTE").cast(pl.Utf8)) # correspondance sur NUM_POSTE
        )

        # 2) jointure avec le tableau final
        final_table = (
            final_table
            .join(delta_years, on="NUM_POSTE")
            .with_columns([
                (10 / pl.col("delta_year")).alias("k10")
            ])
            .with_columns([
                (pl.col("z_T1")     * pl.col("k10")).alias("z_T1"),
                (pl.col("ic_lower") * pl.col("k10")).alias("ic_lower"),
                (pl.col("ic_upper") * pl.col("k10")).alias("ic_upper"),
            ])
            .drop(["delta_year", "k10"])
        )

        # Sauvegarde
        out_path = path_dir / "niveau_retour.parquet"
        final_table.write_parquet(out_path)
        logger.info(f"Tableau final enregistré: {out_path}")
        logger.info(final_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de calcul du niveau de retour et son IC")
    parser.add_argument("--config", type=str, default="config/observed_settings.yaml")
    parser.add_argument("--echelle", choices=["horaire", "quotidien"], nargs='+', default=["quotidien"])
    parser.add_argument("--season", type=str, default="son")
    parser.add_argument("--threshold", type=float, default=0.10, help="Seuil pour IC")
    args = parser.parse_args()

    config = load_config(args.config)
    config["config"] = args.config
    config["echelles"] = args.echelle
    config["season"] = args.season
    main(config, args)
