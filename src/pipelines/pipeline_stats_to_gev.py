import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from typing import Tuple, Optional, List, Dict

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from hades_stats import sp_dist
from hades_stats import Distribution

from src.utils.config_tools import load_config
from src.utils.logger import get_logger
from src.utils.data_utils import load_data

def gev_stationnaire(serie: pd.Series) -> Tuple[float, float, float]:
    """Renvoie xi, mu, sigma pour une GEV stationnaire (avec xi = -c)."""
    c, loc, scale = sp_dist('genextreme').fit(serie)
    xi = -c
    return xi, loc, scale

def fit_gev_for_point(key: int, group: pd.DataFrame, col_val: str) -> Tuple[int, Optional[float], Optional[float], Optional[float]]:
    num_poste = key
    try:
        xi, loc, scale = gev_stationnaire(group[col_val])
        return num_poste, xi, loc, scale
    except Exception:
        return num_poste, np.nan, np.nan, np.nan

def fit_gev_par_point(df_pl: pl.DataFrame, col_val: str, max_workers: int = 48) -> pd.DataFrame:
    """
    Applique une GEV stationnaire à chaque point (lat, lon) à partir des maxima annuels.
    Parallélise l'ajustement avec `ProcessPoolExecutor`.
    """
    df = df_pl.to_pandas()
    grouped = list(df.groupby('NUM_POSTE'))

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fit_gev_for_point, key, group, col_val): key
            for key, group in grouped
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fitting GEV"):
            res = future.result()
            if res is not None:
                results.append(res)

    df_result = pd.DataFrame(results, columns=["NUM_POSTE", "xi", "mu", "sigma"])
    n_total = len(df_result)
    n_failed = df_result["xi"].isna().sum()
    logger.info(f"GEV fitting terminé : {n_total - n_failed} réussites, {n_failed} échecs.")
    return df_result


def bootstrap_gev_return_levels(
        group: pd.DataFrame, 
        col_val: str, 
        return_periods: List[int], 
        n_bootstrap: int = 10) -> Dict[int, Tuple[float, float]]:
    
    quantile_bounds = {T: [] for T in return_periods}

    data = group[col_val].dropna().values
    if len(data) < 10:
        return {T: (np.nan, np.nan) for T in return_periods}

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        try:
            c, loc, scale = sp_dist('genextreme').fit(sample)
            xi = -c
            dist = Distribution("gev", c=c, loc=loc, scale=scale)
            for T in return_periods:
                try:
                    q = dist.return_level(T)
                    quantile_bounds[T].append(q)
                except Exception:
                    quantile_bounds[T].append(np.nan)
        except Exception:
            for T in return_periods:
                quantile_bounds[T].append(np.nan)

    return {
        T: (np.nanpercentile(quantile_bounds[T], 2.5),
            np.nanpercentile(quantile_bounds[T], 97.5))
        for T in return_periods
    }


def periode_retour(c: int, loc: int, scale: int, value: int):
    d = Distribution("gev", c, loc, scale)
    return d.return_event(value)  # Calcule la période de retour associée à la valeur value


def compute_return_levels_for_point(
        row: pd.Series,
        return_periods: List[int],
        group: Optional[pd.DataFrame] = None,
        col_val: Optional[str] = None,
        with_ic: bool = False) -> List[Dict]:

    num_poste, xi, mu, sigma = row["NUM_POSTE"], row["xi"], row["mu"], row["sigma"]
    levels = []

    if np.isnan(xi) or np.isnan(mu) or np.isnan(sigma):
        for T in return_periods:
            levels.append({
                "NUM_POSTE": num_poste,
                "return_period": T,
                "return_level": np.nan,
                "lower": np.nan,
                "upper": np.nan
            })
        return levels

    try:
        c = -xi
        d = Distribution("gev", c=c, loc=mu, scale=sigma)
        for T in return_periods:
            try:
                q = d.return_level(T)
            except Exception:
                q = np.nan

            levels.append({
                "NUM_POSTE": num_poste,
                "xi": xi,
                "mu": mu,
                "sigma": sigma,
                "return_period": T,
                "return_level": q,
                "lower": np.nan,
                "upper": np.nan
            })

        if with_ic and group is not None and col_val is not None:
            ic_dict = bootstrap_gev_return_levels(group, col_val, return_periods)
            for i, T in enumerate(return_periods):
                lower, upper = ic_dict.get(T, (np.nan, np.nan))
                levels[i]["lower"] = lower
                levels[i]["upper"] = upper

    except Exception:
        for T in return_periods:
            levels.append({
                "NUM_POSTE": num_poste,
                "xi": xi,
                "mu": mu,
                "sigma": sigma,
                "return_period": T,
                "return_level": np.nan,
                "lower": np.nan,
                "upper": np.nan
            })

    return levels



def compute_return_levels(
        result: pd.DataFrame,
        return_periods: List[int],
        full_df: pd.DataFrame,
        col_val: str,
        max_workers: int = 48,
        with_ic: bool = False) -> pd.DataFrame:

    all_results = []
    groups = dict(tuple(full_df.groupby("NUM_POSTE")))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                compute_return_levels_for_point,
                row,
                return_periods,
                groups.get(row["NUM_POSTE"]),
                col_val,
                with_ic
            )
            for _, row in result.iterrows()
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Return levels + IC"):
            res = future.result()
            if res:
                all_results.extend(res)

    return pd.DataFrame(all_results)



def pipeline_gev_from_statisticals(config, max_workers: int = 48):
    global logger
    logger = get_logger(__name__)

    echelles = config.get("echelles")
    model = config.get("model")
    input_dir = config["statistics"]["path"]["outputdir"]

    T = [2, 5, 10, 20, 30, 40, 50, 80, 100]

    for echelle in echelles:
        logger.info(f"--- Traitement pour l’échelle : {echelle.upper()} avec comme model : {model}---")

        input_dir = Path(input_dir) / echelle if model == "config/observed_settings.yaml" else Path(input_dir) / "horaire"

        output_dir = Path(config["gev"]["path"]["outputdir"]) / echelle
        output_dir.mkdir(parents=True, exist_ok=True)

        mesure = "max_mm_h" if echelle == "horaire" else "max_mm_j"
        cols = ["NUM_POSTE", mesure]

        logger.info(f"Chargement des données de 1960 à 2010")
        df = load_data(input_dir, "hydro", echelle, cols, min_year=1960, max_year=2010)
        
        logger.info("Application de la GEV")
        df_gev_param = fit_gev_par_point(df, mesure, max_workers=max_workers)
        df_gev_param.to_parquet(f"{output_dir}/gev_param.parquet")
        logger.info(f"Enregistré sous {output_dir}/gev_param.parquet")
        
        df_gev_param = pd.read_parquet(f"{output_dir}/gev_param.parquet")
        print(df_gev_param)

        logger.info("Calcul des périodes de retours")
        df_return_levels = compute_return_levels(
            df_gev_param, T, df.to_pandas(), mesure, max_workers=max_workers, with_ic=True
        ) 
        df_return_levels.to_parquet(f"{output_dir}/gev_retour.parquet")    
        logger.info(f"Enregistré sous {output_dir}/gev_retour.parquet") 
        print(df_return_levels)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline application de la GEV sur les maximas.")
    parser.add_argument("--config", type=str, default="config/modelised_settings.yaml")
    parser.add_argument("--echelle", type=str, choices=["horaire", "quotidien"], nargs="+", default=["horaire", "quotidien"])
    args = parser.parse_args()

    config = load_config(args.config)
    config["echelles"] = args.echelle
    config["model"] = args.config

    pipeline_gev_from_statisticals(config, max_workers=96)
