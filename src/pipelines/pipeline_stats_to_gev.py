import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from typing import Tuple, Union

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from src.utils.config_tools import load_config
from src.utils.logger import get_logger
from src.utils.data_utils import years_to_load, load_data, cleaning_data_observed

# from hades_stats import sp_dist # GEV stationnaire
from hades_stats import ns_gev_m1, ns_gev_m2, ns_gev_m3 # GEV non stationnaire
from hades_stats import NsDistribution, ObsWithCovar, FitNsDistribution
from scipy.special import gamma # gamma d'Euler Γ(x)

import contextlib
import os
import sys

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# Global parameters for bounds and x0
PARAM_DEFAULTS = {
    # mu0 is defined by likelihood obtained in the stationary case (cf init_gev_params_from_moments)
    "mu0": {
        "bounds": (-np.inf, np.inf) #(0, 500)  # large bound to adapt to different regions
    },

    # mu1: temporal slope, initialized at 0 (no initial trend)
    "mu1": {
        "bounds": (-np.inf, np.inf) #(-100, 100)
    },

    # sigma0: dynamically estimated initial standard deviation (cf. init_gev_params_from_moments)
    "sigma0": {
        "bounds": (-np.inf, np.inf) #(0.1, 100)
    },

    # sigma1: temporal variation of standard deviation, fixed at 0 initially
    "sigma1": {
        "bounds": (-np.inf, np.inf) #(-50, 50)
    },

    # xi: shape parameter, fixed at 0.1 initially
    "xi": {
        "bounds": (-0.5, 0.5) # xi is defined as constant with xi_init = 0.1 in the code
    }
}



MODEL_REGISTRY = {
    # --------------------------------------------------------------------------------------------- STATIONARY
    # mu(t) = mu0 ; sigma(t) = sigma0 ; xi(t) = xi
    "s_gev": (ns_gev_m1, ["mu0", "mu1", "sigma0", "xi"]), # Stationary via mu1 fixed at 0

    # --------------------------------------------------------------------------------------------- NON STATIONNAIRE
    # μ(t) = μ₀ + μ₁·t ; σ(t) = σ₀ ; ξ(t) = ξ
    "ns_gev_m1":  (ns_gev_m1,  ["mu0", "mu1", "sigma0", "xi"]),

    # μ(t) = μ₀ ; σ(t) = σ₀ + σ₁·t ; ξ(t) = ξ
    "ns_gev_m2":  (ns_gev_m2,  ["mu0", "sigma0", "sigma1", "xi"]),

    # μ(t) = μ₀ + μ₁·t ; σ(t) = r × μ(t) ; ξ(t) = ξ
    # "ns_gev_m12": (ns_gev_m12, ["mu0", "mu1", "sigma_ratio", "xi"]),

    # μ(t) = μ₀ + μ₁·t ; σ(t) = σ₀ + σ₁·t ; ξ(t) = ξ
    "ns_gev_m3":  (ns_gev_m3,  ["mu0", "mu1", "sigma0", "sigma1", "xi"]),

    # --------------------------------------------------------------------------------------------- NON STATIONNAIRE AVEC POINT DE RUPTURE
    "ns_gev_m1_break_year":  (ns_gev_m1,  ["mu0", "mu1", "sigma0", "xi"]),
    "ns_gev_m2_break_year":  (ns_gev_m2,  ["mu0", "sigma0", "sigma1", "xi"]),
    "ns_gev_m3_break_year":  (ns_gev_m3,  ["mu0", "mu1", "sigma0", "sigma1", "xi"])
}

# Model initialization:
# 
# - s_gev (stationary):
#     mu0, sigma0, xi initialized from empirical moments (mean and standard deviation).
#
# - ns_gev_m1 (mu effect):
#     mu0, sigma0, xi initialized from s_gev; mu1 initialized at 0.
#
# - ns_gev_m2 (sigma effect):
#     mu0, sigma0, xi initialized from s_gev; sigma1 initialized at 0.
#
# - ns_gev_m3 (mu and sigma effect):
#     Hybrid initialization based on ns_gev_m1 and ns_gev_m2 performance:
#       • If log-likelihood(ns_gev_m1) > log-likelihood(ns_gev_m2):
#           mu0, mu1, sigma0, xi initialized from ns_gev_m1; sigma1 initialized at 0.
#       • Else:
#           mu0, sigma0, sigma1, xi initialized from ns_gev_m2; mu1 initialized at 0.




# Let Gamma be the gamma function, g_k = Gamma(1 - k*xi) for k in {1, 2, 3, 4}
# --- GEV Expected Value ---
# E(X) = mu + (sigma / xi) * (g1 - 1)
# ⇒ mu = E(X) - (sigma / xi) * (g1 - 1)
# --- GEV Standard Deviation ---
# V(X) = (sigma^2 / xi^2) * (g2 - g1^2)
# sigma = Standard Deviation * xi / sqrt(g2 - g1^2)
def init_gev_params_from_moments(mean_emp: float, std_emp: float, xi: float) -> Tuple[float, float]:
    """
    mean_emp = empirical mean
    std_emp = empirical standard deviation
    xi = shape parameter (fixed at 0.1)
    Returns: (mu, sigma) - estimated initial parameters
    """
    gamma1 = gamma(1 - xi)                                  # g1 = Gamma(1 - xi)
    gamma2 = gamma(1 - 2*xi)                                # g2 = Gamma(1 - 2*xi)
    sigma = std_emp * xi / np.sqrt(gamma2 - gamma1**2)      # sigma = STD * xi / sqrt(g2 - g1^2)
    mu = mean_emp - sigma / xi * (gamma1 - 1)               # mu = E(X) - (sigma / xi) * (g1 - 1)
    return mu, sigma                                        # mu and sigma
     

# Sécurise le cas non stationnaire
def postprocess_s_gev_results(results: list[Tuple]) -> Tuple[list[str], list[Tuple]]:
    """Verify mu1 is zero for stationary GEV and return filtered parameters."""
    for res in results:
        if len(res) != 6:  # NUM_POSTE + 4 parameters + log-likelihood
            raise ValueError(f"Unexpected result length in s_gev: {len(res)}")

    mu1_values = [res[2] for res in results]  # mu1 is the 3rd element
    if any(not (np.isnan(v) or v == 0) for v in mu1_values):
        raise ValueError("Error: 's_gev' model must have mu1 = 0.")
    
    param_names = ["mu0", "sigma0", "xi"] 
    filtered_results = [(res[0], res[1], res[3], res[4], res[5]) for res in results]  
    # keep NUM_POSTE, mu0, sigma0, xi, loglik (remove mu1)
    
    return param_names, filtered_results


def gev_non_stationnaire(
    df: pd.DataFrame,
    col_val: str,
    model_name: str,
    init_params: dict[str, float] = None
) -> Tuple:
    global logger
    df = df.dropna(subset=[col_val])
    values = pd.Series(df[col_val].values, index=df.index)


    # ###############################################################################
    # # MODIF TEMP
    # # --- retirer l'observation du maximum ---
    # if not values.empty:
    #     max_val = values.max()
    #     idx_max = values[values == max_val].index
    #     df = df.drop(index=idx_max)  # supprime aussi la covariable correspondante
    #     values = pd.Series(df[col_val].values, index=df.index)
    # ###############################################################################






    model_struct, param_names = MODEL_REGISTRY[model_name]

    # === Custom initialization based on empirical moments of a stationary GEV ===
    # Step 1: Calculate empirical moments (mean and standard deviation) of observed maxima
    mean_emp = values.mean()
    std_emp = values.std()

    # Step 2: Fix xi at 0.1 to avoid extreme tail behavior
    xi_init = 0.1 

    # Step 3: Use inverted GEV formulas for expected value and variance
    # to find mu and sigma such that the initial GEV matches mean and standard deviation
    mu_init, sigma_init = init_gev_params_from_moments(mean_emp, std_emp, xi=xi_init)

    # Temporal covariate: normalized year
    covar = pd.DataFrame({"x": df["year_norm"]}, index=df.index)

    obs = ObsWithCovar(values, covar)
    ns_dist = NsDistribution("gev", model_struct) # model_struct can be None (stationary)
    bounds = [PARAM_DEFAULTS[param]["bounds"] for param in param_names]

    # Retrieve initial parameters from simpler models if available
    if init_params:
        # mu0, sigma0 and xi come from empirical moments or previous stationary model
        mu_init = init_params.get("mu0", mu_init)
        sigma_init = init_params.get("sigma0", sigma_init)
        xi_init = init_params.get("xi", xi_init)

    custom_x0 = {
        "mu0": mu_init,         # mu0
        "mu1": 0,               # mu1 starts at 0
        "sigma0": sigma_init,   # sigma0
        "sigma1": 0,            # sigma1 starts at 0
        # NOTE: inversion because to_params_ts() returns -xi
        "xi": -xi_init          # initial fixed xi
    }

    if model_name in ["ns_gev_m3", "ns_gev_m3_break_year"]: # Effet temporel sur μ et σ
        # On récupère les paramètres déjà estimés dans les modèles plus simples
        if init_params:
            custom_x0["mu1"] = init_params.get("mu1", 0)
            custom_x0["sigma1"] = init_params.get("sigma1", 0)
        else:
            custom_x0["mu1"] = 0
            custom_x0["sigma1"] = 0

    # Cela revient à repartir de la vraisemblance obtenue dans le cas stationnaire, 
    # sauf que dans le cas non stationnaire on autorise plus de paramètres donc la vraisemblance va augmenter.

    x0 = [custom_x0[param] for param in param_names]

    # OPTIMIZATION METHODS
    # BFGS for non-stationary models, L-BFGS-B with bounds for others (to keep mu1 = 0)
    
    optim_methods = []

    if model_name == "s_gev":
        # Fix mu1 to 0 via bounds
        bounds = [
            (0, 0) if param == "mu1" else PARAM_DEFAULTS[param]["bounds"]
            for param in param_names
        ]
    else:
        bounds = [PARAM_DEFAULTS[param]["bounds"] for param in param_names]
        # Unbounded method for non-stationary models
        optim_methods.append({"method": "BFGS", "x0": x0}) 

    # Ajoute d'autres méthodes d’optimisation avec bornes si échecs
    BOUND_COMPATIBLE_METHODS = ["L-BFGS-B", "TNC", "SLSQP", "Powell", "Nelder-Mead"]
    for method in BOUND_COMPATIBLE_METHODS:
        optim_methods.append({"method": method, "x0": x0, "bounds": bounds})

    best_params = None
    best_log_likelihood = -np.inf

    for optim_kwargs in optim_methods:        
        try:
            fit = FitNsDistribution(obs, ns_distribution=ns_dist, fit_kwargs=[optim_kwargs])
            with suppress_stdout():
                fit.fit()

            log_likelihood = -fit.nllh() # nllh() returns negative log-likelihood
            p = fit.ns_distribution.to_params_ts()
            param_names = fit.ns_distribution.par_names
            param_values = list(p)

            # Fix xi: to_params_ts() returns -xi, we want xi
            # xi is always the last parameter in the param_names list generated by NsDistribution
            param_values[-1] = -param_values[-1]

            if all(np.isfinite(param_values)):
                if log_likelihood > best_log_likelihood:
                    best_log_likelihood = log_likelihood
                    best_params = tuple(param_values) + (log_likelihood,)
            else:
                logger.warning(f"Fit non fini pour NUM_POSTE={df['NUM_POSTE'].iloc[0]} : {param_values}")
        except Exception as e:
            # Tente de récupérer le message de l'optimiseur s’il existe
            msg = None
            if "fit" in locals() and hasattr(fit, "_result") and fit._result is not None:
                msg = fit._result.message
            else:
                msg = str(e)

            logger.debug(f"[DEBUG] Échec avec méthode {optim_kwargs['method']} pour NUM_POSTE={df['NUM_POSTE'].iloc[0]} : {msg}")
            logger.debug(f"[DEBUG] x0 utilisés : {optim_kwargs.get('x0')}")
            logger.debug(f"[DEBUG] Bornes utilisées : {optim_kwargs.get('bounds')}")

    if best_params is not None:
        return best_params

    # Si tous les essais échouent
    poste = df["NUM_POSTE"].iloc[0] if "NUM_POSTE" in df.columns else "INCONNU"
    logger.debug(f"[DEBUG] Échec total du fit pour NUM_POSTE={poste} (toutes les méthodes)")
    return (np.nan,) * len(param_names) + (np.nan,) # toujours param + log_likelihood

def fit_ns_gev_for_point(
    key: int,
    group: pd.DataFrame,
    col_val: str,
    len_serie: int,
    model_name: str,
    init_params: dict[str, float] = None
):

    num_poste = key
    serie_valide = group.dropna(subset=[col_val])
    if len(serie_valide) < len_serie:
        return (num_poste,) + (np.nan,) * len(MODEL_REGISTRY[model_name][1]) + (np.nan,) # toujours param + log_likelihood
    try:
        result = gev_non_stationnaire(
            serie_valide, col_val, model_name,
            init_params=init_params
        )
        return (num_poste,) + result
    except Exception as e:
        logger.error(f"Error for NUM_POSTE={num_poste}: {type(e).__name__} - {e}")
        return (num_poste,) + (np.nan,) * len(MODEL_REGISTRY[model_name][1]) + (np.nan,) 

def fit_gev_par_point(
    df_pl: pl.DataFrame,
    col_val: str,
    len_serie: int,
    model_name: str,
    break_year: Union[int, None] = None,
    max_workers: int = 48,
    output_dir: Union[str, Path] = None,
    init_params_by_poste: dict = None
) -> pd.DataFrame:

    
    df = df_pl.to_pandas()
    if "year" not in df.columns:
        logger.error("La colonne 'year' est absente du DataFrame. Elle est requise pour le modèle NS-GEV.")
        raise ValueError("Colonne 'year' manquante dans les données.")

    # Normalisation globale entre min_year et max_year
    min_year = df["year"].min()
    max_year = df["year"].max()

    if break_year is not None:
        if max_year == break_year:
            raise ValueError("`break_year` cannot be equal to `max_year` (division by zero).")

        df["year_norm"] = np.where(
            df["year"] < break_year,
            0,
            (df["year"] - break_year) / (max_year - break_year) # Piecewise linear trend with breakpoint
        )
        logger.info(f"Temporal covariate created with breakpoint at {break_year}")
    else:
        df["year_norm"] = (df["year"] - min_year) / (max_year - min_year) # t_norm = (t - min_year) / (max_year - min_year)


    # def norm_1delta_0centred_pandas(series): normalisation supplémentaire faire dans hades avec ObsWithCovar
    #     res0 = series.astype(float) / (series.max() - series.min())
    #     dx = res0.min() + 0.5
    #     return res0 - dx
    # df["year_norm"] = norm_1delta_0centred_pandas(df["year_norm"])

    logger.debug(f"[DEBUG] Années normalisées : min={df['year_norm'].min()}, max={df['year_norm'].max()}")
    grouped = list(df.groupby('NUM_POSTE'))
   

    if model_name != "s_gev" and output_dir is not None:
        if init_params_by_poste is None:           # on crée le conteneur, vide
                init_params_by_poste = {}
                
        try:
            df_init = pd.read_parquet(Path(output_dir) / "gev_param_s_gev.parquet")
            init_params_by_poste = {
                row["NUM_POSTE"]: {
                    "mu0": row["mu0"],
                    "sigma0": row["sigma0"],
                    "xi": row["xi"],
                    "mu1": row.get("mu1", 0),
                    "sigma1": row.get("sigma1", 0),
                }
                for _, row in df_init.iterrows()
            }

            if model_name in ["ns_gev_m3", "ns_gev_m3_break_year"]:
                try:
                    # Utilise le suffixe "_break_year" si nécessaire
                    suffix = "_break_year" if "break_year" in model_name else ""

                    path_m1 = Path(output_dir) / f"gev_param_ns_gev_m1{suffix}.parquet"
                    path_m2 = Path(output_dir) / f"gev_param_ns_gev_m2{suffix}.parquet"

                    if path_m1.exists() and path_m2.exists():
                        df_m1 = pd.read_parquet(path_m1).set_index("NUM_POSTE")
                        df_m2 = pd.read_parquet(path_m2).set_index("NUM_POSTE")

                        for poste in init_params_by_poste:
                            row_m1 = df_m1.loc[poste] if poste in df_m1.index else None
                            row_m2 = df_m2.loc[poste] if poste in df_m2.index else None

                            if row_m1 is not None and row_m2 is not None:
                                if row_m1["log_likelihood"] >= row_m2["log_likelihood"]:
                                    init_params_by_poste[poste]["mu0"] = row_m1["mu0"]
                                    init_params_by_poste[poste]["mu1"] = row_m1["mu1"]
                                    init_params_by_poste[poste]["sigma0"] = row_m1["sigma0"]
                                    init_params_by_poste[poste]["xi"] = row_m1["xi"]
                                    init_params_by_poste[poste]["sigma1"] = 0.0
                                else:
                                    init_params_by_poste[poste]["mu0"] = row_m2["mu0"]
                                    init_params_by_poste[poste]["mu1"] = 0.0
                                    init_params_by_poste[poste]["sigma0"] = row_m2["sigma0"]
                                    init_params_by_poste[poste]["sigma1"] = row_m2["sigma1"]
                                    init_params_by_poste[poste]["xi"] = row_m2["xi"]

                            elif row_m1 is not None:
                                init_params_by_poste[poste]["mu0"] = row_m1["mu0"]
                                init_params_by_poste[poste]["mu1"] = row_m1["mu1"]
                                init_params_by_poste[poste]["sigma0"] = row_m1["sigma0"]
                                init_params_by_poste[poste]["xi"] = row_m1["xi"]
                                init_params_by_poste[poste]["sigma1"] = 0.0

                            elif row_m2 is not None:
                                init_params_by_poste[poste]["mu0"] = row_m2["mu0"]
                                init_params_by_poste[poste]["mu1"] = 0.0
                                init_params_by_poste[poste]["sigma0"] = row_m2["sigma0"]
                                init_params_by_poste[poste]["sigma1"] = row_m2["sigma1"]
                                init_params_by_poste[poste]["xi"] = row_m2["xi"]

                    else:
                        logger.warning(f"Fichiers d'initialisation manquants pour {model_name} : {path_m1} ou {path_m2}")

                except Exception as e:
                    logger.warning(f"Impossible de charger ns_gev_m1/m2 pour initialisation {model_name} : {e}")



        except Exception as e:
            logger.warning(f"Impossible de charger les paramètres stationnaires pour initialiser {model_name}: {e}")


    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                fit_ns_gev_for_point,
                key,
                group,
                col_val,
                len_serie,
                model_name,
                init_params_by_poste.get(key) if init_params_by_poste else None
            )
            for key, group in grouped
        }

        results = []
        progress_bar = tqdm(total=len(futures), desc="Fitting NS-GEV")
        counter = 0

        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                results.append(res)
            counter += 1
            if counter % 5000 == 0 or counter == len(futures):
                progress_bar.update(5000 if counter % 5000 == 0 else len(futures) % 5000)

        progress_bar.close()

    if model_name == "s_gev":
        param_names, filtered_results = postprocess_s_gev_results(results)
    else:
        param_names = MODEL_REGISTRY[model_name][1]
        filtered_results = results

    df_result = pd.DataFrame(filtered_results, columns=["NUM_POSTE"] + param_names + ["log_likelihood"])

    n_total = len(df_result)
    n_failed = df_result[param_names].isna().all(axis=1).sum()
    logger.info(f"NS-GEV fitting completed: {n_total - n_failed} successes, {n_failed} failures.")
    return df_result

def pipeline_gev_from_statisticals(config, max_workers: int=48, n_bootstrap: int=100):
    global logger
    logger = get_logger(__name__)
    
    echelles = config.get("echelles", "quotidien")
    season = config.get("season", "hydro")
    model_path = config.get("config", "config/observed_settings.yaml")
    model_name = config.get("model", "s_gev")
    reduce_activate = config.get("reduce_activate", False)

    for echelle in echelles:
        logger.info(f"--- Traitement de {model_path} \nEchelle : {echelle.upper()} \n Modèle : {model_name} ---")

        # Application d'un point de rupture
        if model_name in ["ns_gev_m1_break_year", "ns_gev_m2_break_year", "ns_gev_m3_break_year"]:
            break_year = config.get("gev", {}).get("break_year", 1985)
            logger.info(f"--- Application d'un point de rupture en {break_year} ---")
        else:
            break_year = None
            logger.info(f"--- Pas de point de rupture ---")

        # Choix du répertoire de lecture
        model_path_name = Path(model_path).name

        if model_path_name == "observed_settings.yaml":
            input_dir = Path(config["statistics"]["path"]["outputdir"]) / echelle
            logger.info(f"STATION source detected → reading from: {input_dir}")

        elif model_path_name == "modelised_settings.yaml":
            input_dir = Path(config["statistics"]["path"]["outputdir"]) / "horaire"
            logger.info(f"AROME source detected → reading from: {input_dir}")

        else:
            logger.error(f"Nom de fichier de configuration non reconnu : {model_path_name}")
            sys.exit(1)

        # Paramètre de chargement des données
        if reduce_activate and echelle == "quotidien":
            mesure, min_year, max_year, len_serie = years_to_load("reduce", season, input_dir)
            suffix_save = "_reduce"
        elif reduce_activate and echelle == "horaire":
            mesure, min_year, max_year, len_serie = years_to_load("horaire_reduce", season, input_dir)
            suffix_save = "_reduce"
        else:
            mesure, min_year, max_year, len_serie = years_to_load(echelle, season, input_dir)
            suffix_save = ""

        # Création du répertoire de sortie
        output_dir = Path(config["gev"]["path"]["outputdir"]) / f"{echelle}{suffix_save}" / season
        output_dir.mkdir(parents=True, exist_ok=True)

        cols = ["NUM_POSTE", mesure, "nan_ratio"]

        logger.info(f"Chargement des données de {min_year} à {max_year} : {input_dir}")
        df = load_data(input_dir, season, echelle, cols, min_year, max_year)

        # Selection des stations suivant le NaN max
        df = cleaning_data_observed(df, echelle, len_serie)

        logger.info(f"Application de la GEV pour la saison {season}")
        df_gev_param = fit_gev_par_point(
            df, 
            mesure, 
            len_serie=len_serie, 
            model_name=model_name, 
            break_year=break_year,
            max_workers=max_workers,
            output_dir=output_dir
        )

        df_gev_param.to_parquet(f"{output_dir}/gev_param_{model_name}.parquet")

        logger.info(f"Enregistré sous {output_dir}/gev_param_{model_name}.parquet")
        logger.info(df_gev_param)


def str2bool(v):
    if v == "True":
        return True
    else:
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline application de la GEV sur les maximas.")
    parser.add_argument("--config", type=str, default="config/observed_settings.yaml")
    parser.add_argument("--echelle", type=str, choices=["horaire", "quotidien"], nargs="+", default=["horaire", "quotidien"])
    parser.add_argument("--season", type=str, default="hydro")
    parser.add_argument("--model", type=str, choices=list(MODEL_REGISTRY.keys()), default="")
    parser.add_argument("--reduce_activate", type=str2bool, default=False)

    args = parser.parse_args()

    config = load_config(args.config)
    config["config"] = args.config
    config["echelles"] = args.echelle
    config["season"] = args.season
    config["model"] = args.model
    config["reduce_activate"] = args.reduce_activate

    pipeline_gev_from_statisticals(config, max_workers=96)
