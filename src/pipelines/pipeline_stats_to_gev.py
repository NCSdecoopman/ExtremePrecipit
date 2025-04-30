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
from src.utils.data_utils import load_data

from hades_stats import sp_dist # GEV stationnaire
from hades_stats import ns_gev_m1, ns_gev_m2, ns_gev_m3 # GEV non stationnaire
from hades_stats import NsDistribution, ObsWithCovar, FitNsDistribution
from hades_stats.fit import FitDist

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

# Paramètres globaux de bounds et x0
PARAM_DEFAULTS = {
    # μ0 est défini par vraisemblance obtenue dans le cas stationnaire (cf init_gev_params_from_moments)
    "mu0": {
        "bounds": (0, 500)  # borne large pour s'adapter à différentes régions
    },

    # μ1 : pente temporelle, initialisée à 0 (pas de tendance au départ)
    "mu1": {
        "bounds": (-100, 100)
    },

    # σ0 : écart-type initial estimé dynamiquement (cf. init_gev_params_from_moments)
    "sigma0": {
        "bounds": (0.1, 100)
    },

    # σ1 : variation temporelle de l’écart-type, fixée à 0 au départ
    "sigma1": {
        "bounds": (-50, 50)
    },

    # ξ : paramètre de forme, fixé initialement à 0.1
    "xi": {
        "bounds": (-0.5, 0.5) # ξ est défini comme constante avec xi_init = 0.1 dans le code
    }
}



MODEL_REGISTRY = {
    # μ(t) = μ₀ ; σ(t) = σ₀ ; ξ(t) = ξ
    "s_gev": (ns_gev_m1, ["mu0", "mu1", "sigma0", "xi"]), # Stationnaire via μ₁ fixée toujours à 0

    # μ(t) = μ₀ + μ₁·t ; σ(t) = σ₀ ; ξ(t) = ξ
    "ns_gev_m1":  (ns_gev_m1,  ["mu0", "mu1", "sigma0", "xi"]),

    # μ(t) = μ₀ ; σ(t) = σ₀ + σ₁·t ; ξ(t) = ξ
    "ns_gev_m2":  (ns_gev_m2,  ["mu0", "sigma0", "sigma1", "xi"]),

    # μ(t) = μ₀ + μ₁·t ; σ(t) = r × μ(t) ; ξ(t) = ξ
    # "ns_gev_m12": (ns_gev_m12, ["mu0", "mu1", "sigma_ratio", "xi"]),

    # μ(t) = μ₀ + μ₁·t ; σ(t) = σ₀ + σ₁·t ; ξ(t) = ξ
    "ns_gev_m3":  (ns_gev_m3,  ["mu0", "mu1", "sigma0", "sigma1", "xi"])
}


# Soit Γ une loi gamma, notons gₖ = Γ(1 - kξ) avec k dans {1, 2, 3, 4}
# --- Calcul de l'espérance de la GEV ---
# E(X) = μ + (σ / ξ) * (g₁ - 1)
# ⇒ μ = E(X) − (σ / ξ) × (g₁ - 1)
# --- Calcul de l'écart-type de la GEV ---
# V(X) = (σ² / ξ²) * (g₂ - g₁²)
# Ecart-type = = √V(X) = (σ / ξ) * √(g₂ - g₁²)
# ⇒ σ = Écart-type × ξ / √(g₂ - g₁²)
def init_gev_params_from_moments(mean_emp: float, std_emp: float, xi: float) -> Tuple[float, float]:
    """
    mean_emp = moyenne empirique (calculée à partir des données)
    std_emp = écart-type empirique
    xi = paramètre de forme (on le fixe ici à 0.1)
    Retourne : (mu, sigma) — les paramètres initiaux estimés
    """
    gamma1 = gamma(1 - xi)                                  # g₁ = Γ(1 - ξ)
    gamma2 = gamma(1 - 2*xi)                                # g₂ = Γ(1 - 2ξ)
    sigma = std_emp * xi / np.sqrt(gamma2 - gamma1**2)      # σ = Écart-type × ξ / √(g₂ - g₁²)
    mu = mean_emp - sigma / xi * (gamma1 - 1)               # μ = E(X) − (σ / ξ) × (g₁ - 1)
    return mu, sigma                                        # μ et σ
     

# Sécurise le cas non stationnaire
def postprocess_s_gev_results(results: list[Tuple]) -> Tuple[list[str], list[Tuple]]:
    """Vérifie que mu1 est bien nul pour le modèle s_gev, et retourne les paramètres sans suppression."""
    for res in results:
        if len(res) != 6:  # NUM_POSTE + 4 paramètres + loglikehood
            raise ValueError(f"Résultat inattendu dans s_gev : longueur différent de 6")

    mu1_values = [res[2] for res in results]  # mu1 = 3e élément (res = NUM_POSTE, mu0, mu1, sigma0, xi, loglik)
    if any(not (np.isnan(v) or v == 0) for v in mu1_values):
        raise ValueError("Erreur : le modèle 's_gev' est censé fixer mu1 = 0. Une valeur estimée est différente de 0.")
    
    param_names = ["mu0", "sigma0", "xi"] # pour ne pas enregistrer mu1
    filtered_results = [(res[0], res[1], res[3], res[4], res[5]) for res in results]  
    # garde NUM_POSTE, mu0, sigma0, xi, loglik (supprime mu1)
    
    return param_names, filtered_results


def gev_non_stationnaire(df: pd.DataFrame, col_val: str, model_name: str) -> Tuple:
    global logger
    df = df.dropna(subset=[col_val])
    values = pd.Series(df[col_val].values, index=df.index)

    model_struct, param_names = MODEL_REGISTRY[model_name]

    # === Initialisation personnalisée basée sur les moments empiriques d'un GEV stationnaire ===
    # Étape 1 : calcul des moments empiriques (moyenne et écart-type) des maxima observés
    mean_emp = values.mean()
    std_emp = values.std()

    # Étape 2 : on fixe xi à 0.1 pour éviter les comportements extrêmes (queue trop courte ou trop lourde)
    xi_init = 0.1 # ξ physique initial voulu

    # Étape 3 : on utilise les formules inversées de l'espérance et de la variance de la GEV
    # pour retrouver mu et sigma de manière à ce que la GEV initialisée ait la même moyenne et écart-type
    mu_init, sigma_init = init_gev_params_from_moments(mean_emp, std_emp, xi=xi_init)

    # Covariable temporelle : l'année normalisée
    covar = pd.DataFrame({"x": df["year_norm"]}, index=df.index)

    obs = ObsWithCovar(values, covar)
    ns_dist = NsDistribution("gev", model_struct) # model_struct peut être None (stationnaire)
    bounds = [PARAM_DEFAULTS[param]["bounds"] for param in param_names]

    # Initialisation personnalisée basée sur une GEV stationnaire
    custom_x0 = {
        "mu0": mu_init,             # μ₀ = μ
        "mu1": 0,                   # μ₁ = 0
        "sigma0": sigma_init,       # σ₀ = σ
        "sigma1": 0,                # σ₁ = 0 
        # Attention ! inversion car to_params_ts() retourne -xi
        "xi": -xi_init              # ξ = constante choisie dès le début
    }
    # Cela revient à repartir de la vraisemblance obtenue dans le cas stationnaire, 
    # sauf que dans le cas non stationnaire on autorise plus de paramètres donc la vraisemblance va augmenter.

    x0 = [custom_x0[param] for param in param_names]

    optim_methods = []

    if model_name == "s_gev":
        # Fixe mu1 à 0 : conversion en stationnaire
        bounds = [
            (v, v) if param == "mu1" else PARAM_DEFAULTS[param]["bounds"]
            for param, v in zip(param_names, x0)
        ]

    # Ajoute les méthodes d’optimisation
    optim_methods.append({"method": "L-BFGS-B", "x0": x0, "bounds": bounds})
    optim_methods.append({"method": "Powell", "x0": x0, "bounds": bounds})

    for optim_kwargs in optim_methods:
        try:
            fit = FitNsDistribution(obs, ns_distribution=ns_dist, fit_kwargs=[optim_kwargs])
            with suppress_stdout():
                fit.fit()

            log_likelihood = -fit.nllh() # nllh() retourne la négative log-vraisemblance
            p = fit.ns_distribution.to_params_ts()
            param_names = fit.ns_distribution.par_names
            param_values = list(p)

            # Corrige xi : to_params_ts() retourne -xi, nous on veut xi
            # xi est toujours le dernier paramètre de la liste param_names générée par NsDistribution
            param_values[-1] = -param_values[-1]

            if all(np.isfinite(param_values)):
                return tuple(param_values) + (log_likelihood,)
            else:
                logger.warning(f"Fit non fini pour NUM_POSTE={df['NUM_POSTE'].iloc[0]} : {param_values}")
        except Exception as e:
            logger.debug(f"Échec avec méthode {optim_kwargs['method']} pour NUM_POSTE={df['NUM_POSTE'].iloc[0]}")

    # Si tous les essais échouent
    poste = df["NUM_POSTE"].iloc[0] if "NUM_POSTE" in df.columns else "INCONNU"
    logger.warning(f"Échec total du fit pour NUM_POSTE={poste} (toutes les méthodes)")
    return (np.nan,) * len(param_names) + (np.nan,) # toujours param + log_likelihood

def fit_ns_gev_for_point(key: int, group: pd.DataFrame, col_val: str, len_serie: int, model_name: str) -> Tuple:
    num_poste = key
    serie_valide = group.dropna(subset=[col_val])
    if len(serie_valide) < len_serie:
        return (num_poste,) + (np.nan,) * len(MODEL_REGISTRY[model_name][1]) + (np.nan,) # toujours param + log_likelihood
    try:
        result = gev_non_stationnaire(serie_valide, col_val, model_name)
        return (num_poste,) + result
    except Exception as e:
        logger.error(f"Erreur pour NUM_POSTE={num_poste} : {type(e).__name__} - {e}")
        return (num_poste,) + (np.nan,) * len(MODEL_REGISTRY[model_name][1]) + (np.nan,) # toujours param + log_likelihood

def fit_gev_par_point(df_pl: pl.DataFrame, col_val: str, len_serie: int, model_name: str, max_workers: int = 48) -> pd.DataFrame:
    df = df_pl.to_pandas()
    if "year" not in df.columns:
        logger.error("La colonne 'year' est absente du DataFrame. Elle est requise pour le modèle NS-GEV.")
        raise ValueError("Colonne 'year' manquante dans les données.")

    # Normalisation globale entre min_year et max_year
    min_year = df["year"].min()
    max_year = df["year"].max()
    df["year_norm"] = (df["year"] - min_year) / (max_year - min_year)

    grouped = list(df.groupby('NUM_POSTE'))

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fit_ns_gev_for_point, key, group, col_val, len_serie, model_name): key
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
            if counter % 1000 == 0 or counter == len(futures):
                progress_bar.update(1000 if counter % 1000 == 0 else len(futures) % 1000)

        progress_bar.close()

    if model_name == "s_gev":
        param_names, filtered_results = postprocess_s_gev_results(results)
    else:
        param_names = MODEL_REGISTRY[model_name][1]
        filtered_results = results

    df_result = pd.DataFrame(filtered_results, columns=["NUM_POSTE"] + param_names + ["log_likelihood"])

    n_total = len(df_result)
    n_failed = df_result[param_names[0]].isna().sum()
    logger.info(f"NS-GEV fitting terminé : {n_total - n_failed} réussites, {n_failed} échecs.")
    return df_result

def pipeline_gev_from_statisticals(config, model_name: str, max_workers: int = 48):
    global logger
    logger = get_logger(__name__)
    
    echelles = config.get("echelles")
    model_path = config.get("model")

    for echelle in echelles:
        logger.info(f"--- Traitement de {model_path} - échelle : {echelle.upper()} - modèle : {model_name} ---")

        if model_path == "config/observed_settings.yaml":
            input_dir = Path(config["statistics"]["path"]["outputdir"]) / echelle 
        else:
            input_dir = Path(config["statistics"]["path"]["outputdir"]) / "horaire"

        output_dir = Path(config["gev"]["path"]["outputdir"]) / echelle
        output_dir.mkdir(parents=True, exist_ok=True)

        mesure = "max_mm_h" if echelle == "horaire" else "max_mm_j"
        cols = ["NUM_POSTE", mesure]

        min_year = 1960
        max_year = 2015
        logger.info(f"Chargement des données de {min_year} à {max_year}")
        df = load_data(input_dir, "hydro", echelle, cols, min_year, max_year)

        logger.info("Application de la GEV")
        len_serie = 48 if echelle=="quotidien" else 20
        df_gev_param = fit_gev_par_point(df, mesure, len_serie=len_serie, model_name=model_name, max_workers=max_workers)
        df_gev_param.to_parquet(f"{output_dir}/gev_param_{model_name}.parquet")
        logger.info(f"Enregistré sous {output_dir}/gev_param_{model_name}.parquet")
        logger.info(df_gev_param)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline application de la GEV sur les maximas.")
    parser.add_argument("--config", type=str, default="config/observed_settings.yaml")
    parser.add_argument("--echelle", type=str, choices=["horaire", "quotidien"], nargs="+", default=["horaire", "quotidien"])
    parser.add_argument("--model_structure", type=str, choices=list(MODEL_REGISTRY.keys()), default="")
    args = parser.parse_args()

    config = load_config(args.config)
    config["echelles"] = args.echelle
    config["model"] = args.config

    pipeline_gev_from_statisticals(config, model_name=args.model_structure, max_workers=96)