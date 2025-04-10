import argparse
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import as_completed
import tempfile
import dask
from dask.diagnostics import ProgressBar
from dask import delayed, compute
from tqdm import tqdm
import numcodecs

from src.utils.config_tools import load_config
from src.utils.logger import get_logger

def find_exact_point_index(lat_target: float, lon_target: float, da: xr.DataArray) -> int:
    """
    Retourne l'indice dans la dimension 'points' du DataArray correspondant exactement à (lat, lon).
    Lève une erreur si aucune correspondance n'est trouvée.
    
    Paramètres :
    - lat_target : latitude exacte recherchée
    - lon_target : longitude exacte recherchée
    - da : DataArray ou Dataset xarray avec les coords 'lat' et 'lon' sur la dimension 'points'
    
    Retour :
    - idx (int) : indice dans la dimension 'points'
    """
    # Convertit tout en float32 pour comparaison exacte
    lat_target = np.float32(lat_target)
    lon_target = np.float32(lon_target)
    lat = da["lat"].values.astype(np.float32)
    lon = da["lon"].values.astype(np.float32)

    matches = np.where((lat == lat_target) & (lon == lon_target))[0]
    if len(matches) == 0:
        logger.warning(f"Aucun point trouvé pour lat={lat_target}, lon={lon_target}")
    elif len(matches) > 1:
        logger.warning(f"Plusieurs points trouvés pour lat={lat_target}, lon={lon_target}")
    
    return int(matches[0])


def find_matching_point(df, lat_obs, lon_obs):
    # Calcul de la distance euclidienne
    distances = np.sqrt((df["lat"] - lat_obs)**2 + (df["lon"] - lon_obs)**2)
    idx_min = distances.idxmin()

    lat_mod = df.loc[idx_min, "lat"]
    lon_mod = df.loc[idx_min, "lon"]

    return lat_mod, lon_mod


def load_zarr_years(path: Path, years: list[int], echelle: str = "horaire", type: str = "obs", var: str = "pr") -> xr.DataArray:
    """Charge et concatène les fichiers Zarr sur plusieurs années pour une variable donnée.
    Si échelle est 'quotidien', effectue une conversion par somme de 6h à 6h avec timestamp à 00:30.
    """
    datasets = [xr.open_zarr(path / f"{year}.zarr") for year in years]
    # ds_merged = xr.concat(datasets, dim="time")

    # On concatène un par un avec une barre de progression
    ds_merged = xr.open_zarr(path / f"{years[0]}.zarr")
    for year in tqdm(years[1:], desc="Concaténation progressive"):
        ds = xr.open_zarr(path / f"{year}.zarr")
        ds_merged = xr.concat([ds_merged, ds], dim="time")
    logger.info(f"Chargement {path} et concaténation")

    da = ds_merged[var]  # DataArray de la variable demandée

    if echelle == "quotidien" and type == "mod":
        logger.info("Décalage temporelle pour l'échelle quotidien")
        # Étape 1 : décalage de -6h
        da_shifted = da.copy()
        da_shifted["time"] = da_shifted["time"] - pd.Timedelta(hours=6)

        # Étape 2 : somme journalière
        da_daily = da_shifted.resample(time="1D").sum(min_count=1)

        # Étape 3 : replacer les timestamps à 00:30 du jour J
        time_new = pd.date_range(
            start=da_daily["time"].values[0],
            periods=da_daily.sizes["time"],
            freq="D"
        ) + pd.Timedelta(minutes=30)

        da_daily["time"] = time_new
        da = da_daily

    return da, len(datasets)


# Fonction de traitement d'une ligne
def process_row(lat, lon, df_match, da_obs, da_mod, scale_factor_obs, scale_factor_mod, fill_value = -9999):
    lat_mod, lon_mod = find_matching_point(df_match, lat, lon)

    idx_obs = find_exact_point_index(lat, lon, da_obs)
    idx_mod = find_exact_point_index(lat_mod, lon_mod, da_mod)

    times_obs = da_obs["time"].values
    pr_obs = da_obs["pr"].isel(points=idx_obs).values
    df_obs = pd.DataFrame({"time": times_obs, "pr_obs": pr_obs})
    df_obs["pr_obs"] = df_obs["pr_obs"].replace(fill_value, np.nan) / scale_factor_obs

    times_mod = da_mod["time"].values
    pr_mod = da_mod["pr"].isel(points=idx_mod).values
    df_mod = pd.DataFrame({"time": times_mod, "pr_mod": pr_mod})
    df_mod["pr_mod"] = df_mod["pr_mod"].replace(fill_value, np.nan) / scale_factor_mod

    df_merged = pd.merge(df_obs, df_mod, on="time", how="inner", validate="one_to_one")
    df_merged = df_merged.dropna(subset=["pr_obs"])
    df_merged["lat"] = lat
    df_merged["lon"] = lon

    return df_merged
    
    
def wrapper_process_row_with_reload(
    row,
    path_zarr_obs,
    path_zarr_mod,
    df_match,
    scale_factor_obs,
    scale_factor_mod,
    output_path,
    fill_value=-9999
):
    logger = get_logger(__name__)
    lat, lon = row
    try:
        # Rechargement local (picklable)
        da_obs = xr.open_zarr(path_zarr_obs)
        da_mod = xr.open_zarr(path_zarr_mod)

        df_result = process_row(lat, lon, df_match, da_obs, da_mod, scale_factor_obs, scale_factor_mod, fill_value)

        if df_result is not None and not df_result.empty:
            lat = df_result["lat"].iloc[0]
            lon = df_result["lon"].iloc[0]

            df_result["me"] = df_result["pr_mod"] - df_result["pr_obs"]
            mean_error = df_result["me"].mean()

            lat_str = f"{lat}"
            lon_str = f"{lon}"
            me_str = "nan" if np.isnan(mean_error) else f"{mean_error:.2f}"
            filename = f"lat_{lat_str}_lon_{lon_str}_me_{me_str}.parquet"

            df_result.drop(columns="me").to_parquet(Path(output_path) / filename)

            logger.info(f"{len(df_result)} comparaisons dans {filename}")
            return True
        else:
            try:
                idx_obs = find_exact_point_index(lat, lon, da_obs)
                pr_obs = da_obs["pr"].isel(points=idx_obs).values
                pr_obs_clean = np.where(pr_obs == fill_value, np.nan, pr_obs) / scale_factor_obs
                n_valid_obs = np.count_nonzero(~np.isnan(pr_obs_clean))
                if n_valid_obs != 0:
                    logger.error("[ERROR] Comptage anormal")
            except Exception as e:
                n_valid_obs = -1
                logger.warning(f"Impossible de compter les valeurs valides pour ({lat}, {lon}) : {e}")

            logger.info(f"[SKIP] 0 comparaisons pour ({lat},{lon})")
            return False
    except Exception as e:
        print(f"Erreur sur point ({lat}, {lon}) : {e}")
        return False
    

@delayed
def dask_wrapper_process_row(
    row,
    path_zarr_obs,
    path_zarr_mod,
    df_match,
    scale_factor_obs,
    scale_factor_mod,
    output_path,
    fill_value=-9999
):
    return wrapper_process_row_with_reload(
        row,
        path_zarr_obs,
        path_zarr_mod,
        df_match,
        scale_factor_obs,
        scale_factor_mod,
        output_path,
        fill_value
    )


def write_temp_zarr(da: xr.DataArray, path: Path):
    for coord in ["lat", "lon"]:
        da.coords[coord].encoding.pop("chunks", None)

    encoding = {
        "pr": {
            "compressor": numcodecs.Blosc(cname="zstd", clevel=9, shuffle=2),
            "dtype": "float32"
        }
    }
    
    da.load()
    da = da.chunk({"time": -1, "points": -1})

    with ProgressBar():
        da.to_zarr(path, mode="w", encoding=encoding)



def pipeline_obs_vs_mod(config_obs, config_mod):
    global logger
    logger = get_logger(__name__)
    echelles = config_obs.get("echelles", ["horaire", "quotidien"])
    PATH_ZARR_OBS = Path(config_obs["zarr"]["path"]["outputdir"])
    PATH_ZARR_MOD = Path(config_mod["zarr"]["path"]["outputdir"])
    PATH_METADATA_OBS = Path(config_obs["metadata"]["path"]["outputdir"])
    PATH_METADATA_MOD = Path(config_mod["metadata"]["path"]["outputdir"])

    scale_factor_obs = config_obs["zarr"]["variables"]["pr"]["scale_factor"]
    scale_factor_mod = config_mod["zarr"]["variables"]["pr"]["scale_factor"]

    ANNEES = list(range(1959, 2011))

    OUTPUT_PATH = Path(config_obs["obs_vs_mod"]["path"]["outputdir"])
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        #for echelle in echelles:
        for echelle in ["quotidien"]:
            logger.info(f"Traitement {echelle} de {min(ANNEES)} à {max(ANNEES)}")
            output_path_echelle = OUTPUT_PATH / echelle
            output_path_echelle.mkdir(parents=True, exist_ok=True)  # Pour être sûr que le dossier existe

            # === CHARGEMENT DES DONNÉES OBS & MOD (CENTRALISÉ) ===
            # métadonnées
            df_obs = pd.read_csv(f"{PATH_METADATA_OBS}/postes_{echelle}.csv")  # contient lat, lon observés
            df_match = pd.read_csv(f"{PATH_METADATA_MOD}/arome_horaire.csv")  # contient lat, lon observés
            
            # xarray
            with dask.config.set(scheduler="threads"):
                da_obs, nb_dataset_obs = load_zarr_years(PATH_ZARR_OBS / echelle, ANNEES, echelle, type="obs")
                logger.info(f"{len(df_obs)} stations et {nb_dataset_obs} zarr chargés pour les observations")

                da_mod, nb_dataset_mod = load_zarr_years(PATH_ZARR_MOD / "horaire", ANNEES, echelle, type="mod")            
                logger.info(f"{len(df_match)} stations et {nb_dataset_mod} zarr chargés pour les modélisations")

            # Préparation
            rows = list(df_obs[["lat", "lon"]].itertuples(index=False, name=None))
            logger.info(f"{len(rows)} stations à traiter")

            # Sauvegarde des DATA (non picklables)
            path_zarr_obs_temp = Path(tmpdir) / "da_obs.zarr"
            path_zarr_mod_temp = Path(tmpdir) / "da_mod.zarr"
            
            with dask.config.set(scheduler="threads"):
                write_temp_zarr(da_obs, path_zarr_obs_temp)
                write_temp_zarr(da_mod, path_zarr_mod_temp)

            logger.info("Fichiers rendus pickable")

            tasks = [
                dask_wrapper_process_row(
                    row,
                    str(path_zarr_obs_temp),
                    str(path_zarr_mod_temp),
                    df_match,
                    scale_factor_obs,
                    scale_factor_mod,
                    str(output_path_echelle)
                )
                for row in rows
            ]

            results = compute(*tasks)
            n_true = sum(results)
            n_false = len(results) - n_true

            logger.info(f"{n_true} enregistrements / {n_false} vides sur {len(results)} points")


if __name__ == "__main__":
    dask.config.set(scheduler="processes")

    parser = argparse.ArgumentParser(description="Pipeline obs vs mod")
    parser.add_argument("--config_obs", type=str, default="config/observed_settings.yaml")
    parser.add_argument("--config_mod", type=str, default="config/modelised_settings.yaml")
    parser.add_argument("--echelles", nargs="+", default=["horaire", "quotidien"])
    args = parser.parse_args()

    config_obs = load_config(args.config_obs)
    config_mod = load_config(args.config_mod)

    pipeline_obs_vs_mod(config_obs, config_mod)
