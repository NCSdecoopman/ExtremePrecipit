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
from src.utils.data_utils import load_data, cleaning_data_observed

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

# Paramètres globaux de bounds et x0
PARAM_DEFAULTS = {
    # μ0 est défini par vraisemblance obtenue dans le cas stationnaire (cf init_gev_params_from_moments)
    "mu0": {
        "bounds": (-np.inf, np.inf) #(0, 500)  # borne large pour s'adapter à différentes régions
    },

    # μ1 : pente temporelle, initialisée à 0 (pas de tendance au départ)
    "mu1": {
        "bounds": (-np.inf, np.inf) #(-100, 100)
    },

    # σ0 : écart-type initial estimé dynamiquement (cf. init_gev_params_from_moments)
    "sigma0": {
        "bounds": (-np.inf, np.inf) #(0.1, 100)
    },

    # σ1 : variation temporelle de l’écart-type, fixée à 0 au départ
    "sigma1": {
        "bounds": (-np.inf, np.inf) #(-50, 50)
    },

    # ξ : paramètre de forme, fixé initialement à 0.1
    "xi": {
        "bounds": (-0.5, 0.5) # ξ est défini comme constante avec xi_init = 0.1 dans le code
    }
}



MODEL_REGISTRY = {
    # --------------------------------------------------------------------------------------------- STATIONNAIRE
    # μ(t) = μ₀ ; σ(t) = σ₀ ; ξ(t) = ξ
    "s_gev": (ns_gev_m1, ["mu0", "mu1", "sigma0", "xi"]), # Stationnaire via μ₁ fixée toujours à 0

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

# Initialisation des modèles :
# 
# - s_gev (stationnaire) :
#     μ₀, σ₀, ξ initialisés par les moments empiriques (moyenne et écart-type).
#
# - ns_gev_m1 (effet sur μ) :
#     μ₀, σ₀, ξ initialisés par s_gev ; μ₁ initialisé à 0.
#     → Hypothèse : pas de tendance sur σ.
#
# - ns_gev_m2 (effet sur σ) :
#     μ₀, σ₀, ξ initialisés par s_gev ; σ₁ initialisé à 0.
#     → Hypothèse : pas de tendance sur μ.
#
# - ns_gev_m3 (effet sur μ et σ) :
#     Initialisation hybride basée sur la performance de ns_gev_m1 et ns_gev_m2 :
#       • Si log-vraisemblance(ns_gev_m1) > log-vraisemblance(ns_gev_m2) :
#           μ₀, μ₁, σ₀, ξ initialisés depuis ns_gev_m1 ; σ₁ initialisé à 0.
#       • Sinon :
#           μ₀, σ₀, σ₁, ξ initialisés depuis ns_gev_m2 ; μ₁ initialisé à 0.
#     → On repart du meilleur modèle unidimensionnel pour estimer l’autre effet.




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


def gev_non_stationnaire(
    df: pd.DataFrame,
    col_val: str,
    model_name: str,
    init_params: dict[str, float] = None
) -> Tuple:
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

    # On récupère les initiations déjà trouvées issues de modèles plus simples
    if init_params:
        # μ₀, σ₀ et ξ viennent soit des moments empiriques, soit d’un modèle stationnaire précédent
        mu_init = init_params.get("mu0", mu_init)
        sigma_init = init_params.get("sigma0", sigma_init)
        xi_init = init_params.get("xi", xi_init)

    custom_x0 = {
        "mu0": mu_init,         # μ₀
        "mu1": 0,               # μ₁ = 0 pour commencer et le définir
        "sigma0": sigma_init,   # σ₀
        "sigma1": 0,            # σ₁ = 0 pour commencer et le définir
        # Attention ! inversion car to_params_ts() retourne -xi
        "xi": -xi_init          # ξ = constante choisie dès le début
    }

    if model_name in ["ns_gev_m3", "ns_gev_m3_break_year"]: # Effet temporel sur μ et σ
        # On récupère les paramètres déjà estimés dans les modèles plus simples
        custom_x0["mu1"] = init_params.get("mu1", 0)
        custom_x0["sigma1"] = init_params.get("sigma1", 0)

    # Cela revient à repartir de la vraisemblance obtenue dans le cas stationnaire, 
    # sauf que dans le cas non stationnaire on autorise plus de paramètres donc la vraisemblance va augmenter.

    x0 = [custom_x0[param] for param in param_names]

    # METHODES D'OPTIMISATION
    # BFGS libre uniquement pour les modèles non-stationnaires, L-BFGS-B sinon (pour fixer mu1 = 0 permanent)
    # L-BFGS-B (et co) avec bornes pour tous les modèles, y compris s_gev
    
    optim_methods = []

    if model_name == "s_gev":
        # Fixe mu1 à 0 via les bornes et laisse les autres comme paramétrées
        bounds = [
            (0, 0) if param == "mu1" else PARAM_DEFAULTS[param]["bounds"]
            for param in param_names
        ]
    else:
        bounds = [PARAM_DEFAULTS[param]["bounds"] for param in param_names] # comme paramétrées
        # Méthode sans borne pour les modèles non stationnaires
        optim_methods.append({"method": "BFGS", "x0": x0}) # pas de 'bounds' ici

    # Ajoute d'autres méthodes d’optimisation avec bornes si échecs
    BOUND_COMPATIBLE_METHODS = ["L-BFGS-B", "TNC", "SLSQP", "Powell", "Nelder-Mead"]
    for method in BOUND_COMPATIBLE_METHODS:
        optim_methods.append({"method": method, "x0": x0, "bounds": bounds})

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
            # Tente de récupérer le message de l'optimiseur s’il existe
            msg = None
            if "fit" in locals() and hasattr(fit, "_result") and fit._result is not None:
                msg = fit._result.message
            else:
                msg = str(e)

            logger.debug(f"[DEBUG] Échec avec méthode {optim_kwargs['method']} pour NUM_POSTE={df['NUM_POSTE'].iloc[0]} : {msg}")
            logger.debug(f"[DEBUG] x0 utilisés : {optim_kwargs.get('x0')}")
            logger.debug(f"[DEBUG] Bornes utilisées : {optim_kwargs.get('bounds')}")

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
        logger.error(f"Erreur pour NUM_POSTE={num_poste} : {type(e).__name__} - {e}")
        return (num_poste,) + (np.nan,) * len(MODEL_REGISTRY[model_name][1]) + (np.nan,) # toujours param + log_likelihood

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
            raise ValueError("`break_year` ne peut pas être égal à `max_year` (division par zéro).")

        df["year_norm"] = np.where(
            df["year"] < break_year,
            0,
            (df["year"] - break_year) / (max_year - break_year) # t₊ = (t - 1985) / (max_year - 1985) = (t − t₀) ⋅ 𝟙{t > t₀}
        )
        logger.info(f"Covariable temporelle créée avec rupture à {break_year}")
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
    logger.info(f"NS-GEV fitting terminé : {n_total - n_failed} réussites, {n_failed} échecs.")
    return df_result

def pipeline_gev_from_statisticals(config, max_workers: int=48, n_bootstrap: int=100):
    global logger
    logger = get_logger(__name__)
    
    echelles = config.get("echelles", "quotidien")
    season = config.get("season", "hydro")
    model_path = config.get("config", "config/observed_settings.yaml")
    model_name = config.get("model", "s_gev")

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
            logger.info(f"Source STATION détectée → lecture dans : {input_dir}")

        elif model_path_name == "modelised_settings.yaml":
            input_dir = Path(config["statistics"]["path"]["outputdir"]) / "horaire"
            logger.info(f"Source AROME détectée → lecture dans : {input_dir}")

        else:
            logger.error(f"Nom de fichier de configuration non reconnu : {model_path_name}")
            sys.exit(1)

        # Création du répertoire de sortie
        output_dir = Path(config["gev"]["path"]["outputdir"]) / echelle / season
        output_dir.mkdir(parents=True, exist_ok=True)

        # Fixation de l'échelle pour le choix des colonnes à lire
        mesure = "max_mm_h" if echelle == "horaire" else "max_mm_j"
        cols = ["NUM_POSTE", mesure, "nan_ratio"]

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

        # Selection des stations suivant le NaN max
        df = cleaning_data_observed(df, 0.15)

        logger.info(f"Application de la GEV pour la saison {season}")
        len_serie = 50 if echelle=="quotidien" else 20 # Longueur minimale d'une série valide

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline application de la GEV sur les maximas.")
    parser.add_argument("--config", type=str, default="config/observed_settings.yaml")
    parser.add_argument("--echelle", type=str, choices=["horaire", "quotidien"], nargs="+", default=["horaire", "quotidien"])
    parser.add_argument("--season", type=str, default="hydro")
    parser.add_argument("--model", type=str, choices=list(MODEL_REGISTRY.keys()), default="")

    args = parser.parse_args()

    config = load_config(args.config)
    config["config"] = args.config
    config["echelles"] = args.echelle
    config["season"] = args.season
    config["model"] = args.model

    pipeline_gev_from_statisticals(config, max_workers=96)