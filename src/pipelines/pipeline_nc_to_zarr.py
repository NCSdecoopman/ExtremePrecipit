import os
import argparse
import re
import numcodecs
import xarray as xr
xr.set_options(file_cache_maxsize=1)  # évite l'accumulation de cache de fichier
import dask
from dask import delayed
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm.dask import TqdmCallback
from typing import Dict, Union, Optional
from shapely.geometry import Polygon, MultiPolygon
from shapely import vectorized

from tqdm import tqdm

from src.utils.config_tools import load_config
from src.utils.logger import get_logger

# ---------------------------------------------------------------------------
# 1) Liste des fichiers .nc

def list_nc_files(directory: str) -> list:
    nc_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(directory)
        for file in files if file.endswith('.nc')
    ]
    logger.info(f"{len(nc_files)} fichiers .nc trouvés dans {directory}")
    nc_files = sorted(nc_files)
    return nc_files

# ---------------------------------------------------------------------------
# 2) Extraction année depuis nom de fichier

def extract_year_from_nc(filename: str) -> Optional[str]:
    basename = os.path.basename(filename)
    match = re.search(r'_(\d{4})\d{8}-\d{12}\.nc$', basename)
    if match:
        year = match.group(1)
        logger.debug(f"Année {year} extraite depuis {basename}")
        return year
    else:
        logger.warning(f"Aucune année trouvée dans le fichier : {basename}")
        return None
    
# ---------------------------------------------------------------------------
def generate_metadata(nc_files: list, output_path: str, geom_france) -> None:
    logger.info("Début de la génération du fichier de métadonnées...")

    all_coords = []

    for nc_file in tqdm(nc_files, desc="Lecture des fichiers .nc"):
        ds = xr.open_dataset(nc_file)

        # --- Récupération des coordonnées
        lat = ds["lat"]
        lon = ds["lon"]

        # Cas 2D
        if lat.ndim == 2 and lon.ndim == 2:
            lat2d, lon2d = lat.values, lon.values
        elif lat.ndim == 1 and lon.ndim == 1:
            lat2d, lon2d = np.meshgrid(lat.values, lon.values, indexing="ij")
        else:
            raise ValueError("lat/lon ne sont pas dans un format reconnu")

        ny, nx = lat2d.shape
        lat_flat = lat2d.ravel()
        lon_flat = lon2d.ravel()

        inside_mask_flat = vectorized.contains(geom_france, lon_flat, lat_flat)
        inside_indices = np.where(inside_mask_flat)[0]

        lat_inside = lat_flat[inside_indices].astype(np.float32)
        lon_inside = lon_flat[inside_indices].astype(np.float32)

        # --- Bornes (si présentes)
        lat_bnds = ds["lat_bnds"].values if "lat_bnds" in ds else np.full((lat.size, 2), np.nan)
        lon_bnds = ds["lon_bnds"].values if "lon_bnds" in ds else np.full((lon.size, 2), np.nan)

        if lat.ndim == 2:
            lat_bnds = lat_bnds.reshape(-1, 2)
            lon_bnds = lon_bnds.reshape(-1, 2)

        lat_bnds_inside = lat_bnds[inside_indices].astype(np.float32)
        lon_bnds_inside = lon_bnds[inside_indices].astype(np.float32)

        for i in range(lat_inside.shape[0]):
            all_coords.append({
                "lat": lat_inside[i],
                "lon": lon_inside[i],
                "lat_bnd_min": lat_bnds_inside[i][0],
                "lat_bnd_max": lat_bnds_inside[i][1],
                "lon_bnd_min": lon_bnds_inside[i][0],
                "lon_bnd_max": lon_bnds_inside[i][1]
            })

        ds.close()

    # --- DataFrame final
    df = pd.DataFrame(all_coords).drop_duplicates(subset=["lat", "lon"]).reset_index(drop=True)
    df = df.astype({
        "lat": np.float32,
        "lon": np.float32,
        "lat_bnd_min": np.float32,
        "lat_bnd_max": np.float32,
        "lon_bnd_min": np.float32,
        "lon_bnd_max": np.float32
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"Métadonnées filtrées et enregistrées sous {output_path}")



# ---------------------------------------------------------------------------
# 3) Construction des chunks adaptés (gère "points")

def build_xarray_chunks(ds: xr.Dataset, chunk_config: Dict[str, Union[int, float]]) -> Dict[str, int]:
    computed_chunks = {}
    for dim in ds.dims:
        config_key = dim
        if dim == "points" and "points" in chunk_config:
            config_key = "points"
        elif dim == "points" and "y" in chunk_config:
            config_key = "y"
        conf_val = chunk_config.get(config_key, None)
        if conf_val in [".inf", "inf", float("inf")]:
            computed_chunks[dim] = ds.sizes[dim]
        elif isinstance(conf_val, int):
            computed_chunks[dim] = conf_val
        else:
            computed_chunks[dim] = 1
    return computed_chunks

# ---------------------------------------------------------------------------
# 4) Filtrage spatial France 

def load_geom_france(ne_directory: str):
    shapefile_path = os.path.join(ne_directory, "ne_10m_admin_0_countries.shp")
    gdf = gpd.read_file(shapefile_path, engine="pyogrio")
    gdf_france = gdf[gdf["ADMIN"] == "France"].to_crs(epsg=4326)
    geom_france = gdf_france.geometry.iloc[0]
    return geom_france

def filter_ds_to_france(ds: xr.Dataset, geom_france) -> xr.Dataset:
    """
    Filtre un dataset pour ne conserver que les points strictement à l'intérieur de la France
    et remplace (y, x) par une dimension unique 'points'.
    """
    if not isinstance(geom_france, (Polygon, MultiPolygon)):
        raise TypeError("La géométrie France n’est pas un objet Polygon/MultiPolygon Shapely.")
    
    # 🔍 Gestion de la grille lat/lon
    lat = ds["lat"]
    lon = ds["lon"]

    # Cas 2D
    if lat.ndim == 2 and lon.ndim == 2:
        lat2d, lon2d = lat.values, lon.values
    # Cas 1D (lat(y), lon(x))
    elif lat.ndim == 1 and lon.ndim == 1:
        lat2d, lon2d = np.meshgrid(lat.values, lon.values, indexing="ij")
    else:
        raise ValueError("lat/lon ne sont pas dans un format reconnu")

    ny, nx = lat2d.shape
    lat_flat = lat2d.ravel()
    lon_flat = lon2d.ravel()

    inside_mask_flat = vectorized.contains(geom_france, lon_flat, lat_flat)
    logger.info(f"{inside_mask_flat.sum()} points conservés à l’intérieur de la France sur {len(lat_flat)}.")

    inside_indices = np.where(inside_mask_flat)[0]

    lat_inside = lat_flat[inside_indices].astype(np.float32)
    lon_inside = lon_flat[inside_indices].astype(np.float32)
    y_indices, x_indices = np.unravel_index(inside_indices, (ny, nx))

    ds_france = ds.isel(
        y=("points", y_indices),
        x=("points", x_indices)
    )

    ds_france = ds_france.drop_vars(["y", "x"], errors="ignore")
    ds_france = ds_france.assign_coords({
        "lat": (("points",), lat_inside),
        "lon": (("points",), lon_inside)
    })

    return ds_france


# ---------------------------------------------------------------------------
# 5) Ouverture NetCDF + filtrage spatial

def load_nc_file(
    nc_path: str,
    variables_config: Dict[str, Dict[str, Union[str, int]]],
    chunk_config: Dict[str, Union[int, float]],
    geom_france
) -> xr.Dataset:
    logger.info(f"--- Traitement du fichier NetCDF : {nc_path} ---")
    vars_to_keep = list(variables_config.keys())

    # Ouverture du dataset avec uniquement les variables demandées
    ds = xr.open_dataset(nc_path, chunks="auto")[vars_to_keep]

    # Arrondir les timestamps au :30:00 le plus proche
    if "time" in ds.coords:
        original_time = ds["time"].values

        # Convertir en pandas datetime index pour pouvoir arrondir facilement
        rounded_time = xr.DataArray(
            pd.to_datetime(original_time).round("30min"),
            dims="time"
        )

        ds = ds.assign_coords(time=rounded_time)
        logger.info("Les timestamps ont été arrondis à la demi-heure la plus proche (:00 ou :30).")

    logger.info(f"Dataset ouvert avec chunks natifs. Dimensions : {dict(ds.sizes)}")

    # Filtrage spatial
    ds = filter_ds_to_france(ds, geom_france)

    # Chunking
    computed_chunks = build_xarray_chunks(ds, chunk_config)
    ds = ds.chunk(computed_chunks)
    logger.info(f"Dataset rechunké")

    # On traite uniquement les data_vars (évite de toucher aux coordonnées !)
    for var in ds.data_vars:
        var_conf = variables_config.get(var, {})
        desired_dtype = var_conf.get("dtype", None)
        scale_factor = var_conf.get("scale_factor", None)
        unit_conversion = var_conf.get("unit_conversion", 1.0)
        fill_value = var_conf.get("fill_value", None)

        if desired_dtype is not None:
            logger.info(f"Conversion de {var} vers {desired_dtype}")
            if unit_conversion != 1.0:
                logger.info(f"Conversion d’unité de {var} avec un facteur {unit_conversion}")
                ds[var] = ds[var] * unit_conversion

            if scale_factor is not None:
                logger.info(f"Application d’un facteur d’encodage {scale_factor} pour {var}")
                ds[var] = ds[var] * scale_factor

            if fill_value is not None:
                logger.info(f"Remplacement des NaN de {var} par la sentinelle {fill_value}")
                ds[var] = ds[var].fillna(fill_value)

            ds[var] = ds[var].round().astype(desired_dtype)
        else:
            logger.info(f"Aucun dtype spécifié pour {var}")

    logger.info(f"Chargement finalisé pour {nc_path}")
    return ds


# ---------------------------------------------------------------------------
# 6) Sauvegarde Zarr adaptée à "points"

def save_to_zarr(
    ds: xr.Dataset,
    output_path: str,
    chunk_config: Dict[str, Union[int, float]],
    compressor_config: Dict
) -> None:
    
    codec = numcodecs.Blosc(
        cname=compressor_config["blosc"]["cname"],
        clevel=compressor_config["blosc"]["clevel"],
        shuffle=compressor_config["blosc"]["shuffle"]
    )

    encoding = {}
    for var in ds.data_vars:
        chunk_sizes = []
        for dim in ds[var].dims:
            config_key = "points" if dim == "points" else dim
            val = chunk_config.get(config_key, 1)
            if val in [".inf", "inf", float("inf")]:
                chunk_sizes.append(ds.sizes[dim])
            else:
                chunk_sizes.append(val)
        encoding[var] = {
            "chunks": tuple(chunk_sizes),
            "compressor": codec
        }

    for coord in ["lat", "lon"]:
        if coord in ds.coords:
            chunk_sizes = []
            for dim in ds[coord].dims:
                val = chunk_config.get("points", 1)
                if val in [".inf", "inf", float("inf")]:
                    chunk_sizes.append(ds.sizes[dim])
                else:
                    chunk_sizes.append(val)
            encoding[coord] = {
                "chunks": tuple(chunk_sizes),
                "compressor": codec
            }

    logger.info(f"Encodage appliqué")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    ds.to_zarr(output_path, mode='w', encoding=encoding)

    logger.info(f"Fichier Zarr sauvegardé sous {output_path}")


# ---------------------------------------------------------------------------
# 7) Pipeline principal

def pipeline_nc_to_zarr(config):
    global logger
    logger = get_logger(__name__)
    
    nc_dir = config["nc"]["path"]["inputdir"]
    metadata_output_path = os.path.join(config["metadata"]["path"]["outputdir"], "arome_horaire.csv")
    zarr_dir = config["zarr"]["path"]["outputdir"]
    chunk_config = config["zarr"]["chunks"]
    compressor_config = config["zarr"]["compressor"]
    variables_config = config["zarr"]["variables"]
    ne_directory = config["spatial_filter"]["ne_directory"]
    log_dir = config.get("log", {}).get("directory", "logs")

    nc_files = list_nc_files(nc_dir)
    
    geom_france = load_geom_france(ne_directory)

    generate_metadata(nc_files, metadata_output_path, geom_france)

    summary = []

    for nc_file in nc_files:
        year = extract_year_from_nc(nc_file)
        entry = {
            "Année": year or "inconnue",
            ".nc trouvé": "oui",
            ".zarr généré": "non",
            "Nombre de NaN": "N/A",
            "Structure du .zarr": "N/A",
            "Configuration des chunks": str(chunk_config),
            "Facteur d'échelle pr": "",
            "Sentinelle pr": ""
        }

        if year is None:
            logger.warning(f"Fichier ignoré (année manquante) : {nc_file}")
            summary.append(entry)
            continue

        output_path = os.path.join(zarr_dir, "horaire", f"{year}.zarr")
        overwrite = config["zarr"].get("overwrite", False)

        if os.path.exists(output_path) and not overwrite:
            logger.info(f"Le fichier Zarr pour l'année {year} existe déjà et overwrite=False : skipping.")
            entry[".zarr généré"] = "non (déjà existant)"
            summary.append(entry)
            continue

        ds = load_nc_file(nc_file, variables_config, chunk_config, geom_france)
        ds = ds.persist()

        nan_total = 0

        # Étape 1 : créer les tâches Dask
        compute_tasks = []
        var_names = []
        fill_infos = []

        for var in ds.data_vars:
            var_conf = variables_config.get(var, {})
            fill_value = var_conf.get("fill_value", None)

            if fill_value is not None:
                task = ds[var].isin([fill_value]).sum()
                fill_infos.append(f"fill ({fill_value})")
            else:
                task = ds[var].isnull().sum()
                fill_infos.append("NaN")

            compute_tasks.append(task)
            var_names.append(var)

        # Étape 2 : compute direct
        results = dask.compute(*compute_tasks)

        # Étape 3 : journalisation + total
        for var, kind, count in zip(var_names, fill_infos, results):
            count = int(count)  # Assure qu’on a bien un int natif
            if count > 0:
                logger.warning(f"{count} {kind} détectés pour '{var}'")
            else:
                logger.info(f"Aucun {kind} détecté pour '{var}'")
            nan_total += count


        coord_tasks = []
        coord_names = []

        for coord in ["lat", "lon"]:
            if coord in ds.coords:
                task = ds[coord].isnull().sum()
                coord_tasks.append(task)
                coord_names.append(coord)

        # Exécution unique et parallèle
        coord_results = dask.compute(*coord_tasks)

        # Analyse et log
        for coord, count in zip(coord_names, coord_results):
            count = int(count)
            if count > 0:
                logger.warning(f"{count} NaN détectés pour la coordonnée '{coord}'")
            else:
                logger.info(f"Aucun NaN détecté pour la coordonnée '{coord}'")
            nan_total += count


        entry["Nombre de NaN"] = nan_total

        structure = {var: ds[var].dims for var in ds.data_vars}
        entry["Structure du .zarr"] = str(structure)

        pr_conf = variables_config.get("pr", {})
        entry["Facteur d'échelle pr"] = pr_conf.get("scale_factor", "1 (aucun)")
        entry["Sentinelle pr"] = pr_conf.get("fill_value", "N/A")

        save_to_zarr(ds, output_path, chunk_config, compressor_config)
        ds.close()

        entry[".zarr généré"] = "oui"
        summary.append(entry)

    # Log final
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "pipeline_nc_to_zarr_resume.log")

    with open(log_path, "a") as f:
        f.write("Résumé du pipeline NetCDF -> Zarr\n")
        f.write("="*100 + "\n\n")
        f.write("{:<8} | {:<11} | {:<14} | {:<14} | {:<30} | {:<25} | {:<20} | {:<15}\n".format(
            "Année", ".nc trouvé", ".zarr généré", "Nombre de NaN", "Structure du .zarr", "Chunks config", "Scale factor pr", "Sentinelle pr"
        ))
        f.write("-"*100 + "\n")

        for entry in summary:
            f.write("{:<8} | {:<11} | {:<14} | {:<14} | {:<30} | {:<25} | {:<20} | {:<15}\n".format(
                entry["Année"],
                entry[".nc trouvé"],
                entry[".zarr généré"],
                str(entry["Nombre de NaN"]),
                entry["Structure du .zarr"][:28] + "..." if len(entry["Structure du .zarr"]) > 30 else entry["Structure du .zarr"],
                str(entry["Configuration des chunks"])[:23] + "..." if len(entry["Configuration des chunks"]) > 25 else entry["Configuration des chunks"],
                entry.get("Facteur d'échelle pr", "1 (aucun)"),
                entry.get("Sentinelle pr", "N/A")
            ))

        f.write("\nPipeline terminé avec succès.\n")

    logger.info(f"Log résumé enregistré sous {log_path}")


# ---------------------------------------------------------------------------
# 8) Entrypoint

if __name__ == "__main__":

    # from dask.distributed import Client, LocalCluster

    # cluster = LocalCluster(
    #     n_workers=2,           # un par cœur physique
    #     threads_per_worker=1,   # ajustable, à tester
    #     memory_limit='16GB'     # ~50% de la RAM totale divisée par 20 workers
    # )
    # client = Client(cluster)

    dask.config.set(scheduler="threads")

    parser = argparse.ArgumentParser(description="Pipeline .nc vers .zarr")
    parser.add_argument("--config", type=str, default="config/modelised_settings.yaml")
    parser.add_argument("--echelle", type=str, choices=["horaire"])
    args = parser.parse_args()

    config = load_config(args.config)
    config["echelles"] = [args.echelle]  # Ne traiter qu'une seule échelle
    
    pipeline_nc_to_zarr(config)
