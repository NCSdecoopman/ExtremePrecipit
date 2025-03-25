import sys
import os
import argparse
import re
import numcodecs
import xarray as xr
import numpy as np
import geopandas as gpd
from tqdm.dask import TqdmCallback
from typing import Dict, Union, Optional
from datetime import datetime

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

def filter_ds_to_france(ds: xr.Dataset, ne_directory: str) -> xr.Dataset:
    """
    Filtre un dataset pour ne conserver que les points strictement à l'intérieur de la France
    et remplace (y, x) par une dimension unique 'points'.
    """
    from shapely.geometry import Point

    shapefile_path = os.path.join(ne_directory, "ne_10m_admin_0_countries.shp")
    gdf = gpd.read_file(shapefile_path, engine="pyogrio")
    gdf_france = gdf[gdf["ADMIN"] == "France"].to_crs(epsg=4326)
    geom_france = gdf_france.geometry.iloc[0]

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

    points = [Point(lon, lat) for lon, lat in zip(lon_flat, lat_flat)]
    inside_mask_flat = np.array([geom_france.contains(pt) for pt in points])
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
    ne_directory: str
) -> xr.Dataset:
    logger.info(f"--- Traitement du fichier NetCDF : {nc_path} ---")
    vars_to_keep = list(variables_config.keys())
    logger.info(f"Variables demandées : {vars_to_keep}")

    ds = xr.open_dataset(nc_path, chunks="auto")[vars_to_keep]
    logger.info(f"Dataset ouvert avec chunks natifs. Dimensions : {dict(ds.sizes)}")

    logger.info("Application du filtre spatial pur France (points)")
    ds = filter_ds_to_france(ds, ne_directory=ne_directory)
    logger.info(f"Dataset réduit aux points France. Dimensions : {dict(ds.sizes)}")

    computed_chunks = build_xarray_chunks(ds, chunk_config)
    ds = ds.chunk(computed_chunks)
    logger.info(f"Dataset rechunké : { {k: v.chunks for k, v in ds.data_vars.items()} }")

def load_nc_file(
    nc_path: str,
    variables_config: Dict[str, Dict[str, Union[str, int]]],
    chunk_config: Dict[str, Union[int, float]],
    ne_directory: str
) -> xr.Dataset:
    logger.info(f"--- Traitement du fichier NetCDF : {nc_path} ---")
    vars_to_keep = list(variables_config.keys())
    logger.info(f"Variables demandées : {vars_to_keep}")

    # Ouverture du dataset avec uniquement les variables demandées
    ds = xr.open_dataset(nc_path, chunks="auto")[vars_to_keep]
    logger.info(f"Dataset ouvert avec chunks natifs. Dimensions : {dict(ds.sizes)}")

    # Filtrage spatial
    logger.info("Application du filtre spatial pur France (points)")
    ds = filter_ds_to_france(ds, ne_directory=ne_directory)
    logger.info(f"Dataset réduit aux points France. Dimensions : {dict(ds.sizes)}")

    # Chunking
    computed_chunks = build_xarray_chunks(ds, chunk_config)
    ds = ds.chunk(computed_chunks)
    logger.info(f"Dataset rechunké : { {k: v.chunks for k, v in ds.data_vars.items()} }")

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

    logger.info(f"Encodage appliqué pour {output_path} : {encoding}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with TqdmCallback(desc="Écriture Zarr", unit="chunk"):
        ds.to_zarr(output_path, mode='w', encoding=encoding)

    logger.info(f"Fichier Zarr sauvegardé sous {output_path}")


# ---------------------------------------------------------------------------
# 7) Pipeline principal

def pipeline_nc_to_zarr(config):
    nc_dir = config["nc"]["path"]["inputdir"]
    zarr_dir = config["zarr"]["path"]["outputdir"]
    chunk_config = config["zarr"]["chunks"]
    compressor_config = config["zarr"]["compressor"]
    variables_config = config["zarr"]["variables"]
    ne_directory = config["spatial_filter"]["ne_directory"]
    log_dir = config.get("log", {}).get("directory", "logs")

    nc_files = list_nc_files(nc_dir)

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

        output_path = os.path.join(zarr_dir, f"{year}.zarr")
        overwrite = config["zarr"].get("overwrite", False)

        if os.path.exists(output_path) and not overwrite:
            logger.info(f"Le fichier Zarr pour l'année {year} existe déjà et overwrite=False : skipping.")
            entry[".zarr généré"] = "non (déjà existant)"
            summary.append(entry)
            continue

        ds = load_nc_file(nc_file, variables_config, chunk_config, ne_directory)

        nan_total = 0

        # Analyse des data_vars (pr, autres)
        for var in ds.data_vars:
            var_conf = variables_config.get(var, {})
            fill_value = var_conf.get("fill_value", None)
            if fill_value is not None:
                count = ds[var].isin([fill_value]).sum().compute().item()
                if count > 0:
                    logger.warning(f"{count} valeurs sentinelles détectées pour '{var}'")
                else:
                    logger.info(f"Aucune sentinelle détectée pour '{var}'")
            else:
                count = ds[var].isnull().sum().compute().item()
                if count > 0:
                    logger.warning(f"{count} NaN détectés pour '{var}'")
                else:
                    logger.info(f"Aucun NaN détecté pour '{var}'")
            nan_total += count

        # Analyse des coords (lat, lon)
        for coord in ["lat", "lon"]:
            if coord in ds.coords:
                count = ds[coord].isnull().sum().compute().item()
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
    # Parser la config
    parser = argparse.ArgumentParser(description="Pipeline NetCDF vers Zarr avec filtrage spatial et encodage.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/modelised_settings.yaml",
        help="Chemin vers le fichier de configuration YAML (par défaut : config/modelised_settings.yaml)"
    )
    args = parser.parse_args()
    config_path = args.config
    config = load_config(config_path)
    
    log_dir = config.get("log", {}).get("directory", "logs")
    logger = get_logger(__name__, log_to_file=True, log_dir=log_dir)

    logger.info(f"Démarrage du pipeline NetCDF → Zarr avec la config : {config_path}")
    pipeline_nc_to_zarr(config)
