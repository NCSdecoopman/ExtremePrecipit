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
from typing import Dict, Union, Optional
from shapely.geometry import Polygon, MultiPolygon
from shapely import vectorized

from tqdm import tqdm

from src.utils.config_tools import load_config
from src.utils.logger import get_logger


# ---------------------------------------------------------------------------
# 1) List .nc files

def list_nc_files(directory: str) -> list:
    nc_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(directory)
        for file in files if file.endswith('.nc')
    ]
    logger.info(f"{len(nc_files)} .nc files found in {directory}")
    nc_files = sorted(nc_files)
    return nc_files

# ---------------------------------------------------------------------------
# 2) Extraction année depuis nom de fichier

def extract_year_from_nc(filename: str) -> Optional[str]:
    basename = os.path.basename(filename)
    match = re.search(r'_(\d{4})\d{8}-\d{12}\.nc$', basename)
    if match:
        year = match.group(1)
        logger.debug(f"Year {year} extracted from {basename}")
        return year
    else:
        logger.warning(f"No year found in file: {basename}")
        return None
    
# ---------------------------------------------------------------------------
def generate_metadata(nc_files: list, output_path: str, geom_france, alti_path: str) -> None:
    logger.info("Generating metadata file...")

    all_coords = []

    for nc_file in tqdm(nc_files, desc="Reading .nc files"):
        ds = xr.open_dataset(nc_file)

        # --- Coordinates retrieval
        lat = ds["lat"]
        lon = ds["lon"]

        # 2D case
        if lat.ndim == 2 and lon.ndim == 2:
            lat2d, lon2d = lat.values, lon.values
        elif lat.ndim == 1 and lon.ndim == 1:
            lat2d, lon2d = np.meshgrid(lat.values, lon.values, indexing="ij")
        else:
            raise ValueError("lat/lon are not in a recognized format")

        ny, nx = lat2d.shape
        lat_flat = lat2d.ravel()
        lon_flat = lon2d.ravel()

        inside_mask_flat = vectorized.contains(geom_france, lon_flat, lat_flat)
        inside_indices = np.where(inside_mask_flat)[0]

        lat_inside = lat_flat[inside_indices].astype(np.float32)
        lon_inside = lon_flat[inside_indices].astype(np.float32)

        for i in range(lat_inside.shape[0]):
            all_coords.append({
                "lat": lat_inside[i],
                "lon": lon_inside[i]
            })

        ds.close()

    # --- Final DataFrame
    df = pd.DataFrame(all_coords).drop_duplicates(subset=["lat", "lon"]).reset_index(drop=True)
    df["NUM_POSTE"] = np.arange(1, len(df) + 1)
    df = df.astype({
        "lat": np.float32,
        "lon": np.float32
    })


    # Altitude addition
    alti_file = xr.open_dataset(alti_path)
    lat_orog = alti_file["lat"].values
    lon_orog = alti_file["lon"].values
    orog = alti_file["orog"].values  # altitudes en mètres

    # Verification: assume lat/lon/orog are all 2D
    assert lat_orog.ndim == 2 and lon_orog.ndim == 2 and orog.ndim == 2

    # Matching DataFrame construction
    df_orog = pd.DataFrame({
        "lat": lat_orog.ravel().astype(np.float32),
        "lon": lon_orog.ravel().astype(np.float32),
        "altitude": np.rint(orog.ravel()).astype(np.int16)
    })

    # Merge on exact coordinates
    df = df.merge(df_orog, on=["lat", "lon"], how="left")

    os.makedirs(output_path, exist_ok=True)
    df.to_csv(f"{output_path}/postes_horaire.csv", index=False)
    df.to_csv(f"{output_path}/postes_quotidien.csv", index=False)

    logger.info(f"Metadata filtered and saved to {output_path}")
    
    return df



# ---------------------------------------------------------------------------
# 3) Construction des chunks adaptés (gère "points")

def build_xarray_chunks(ds: xr.Dataset, chunk_config: Dict[str, Union[int, float]]) -> Dict[str, int]:
    computed_chunks = {}
    for dim in ds.dims:
        config_key = dim  # default

        # Specific NUM_POSTE handling
        if dim == "NUM_POSTE" and "NUM_POSTE" in chunk_config:
            config_key = "NUM_POSTE"

        conf_val = chunk_config.get(config_key, None)

        if conf_val in [".inf", "inf", float("inf")]:
            computed_chunks[dim] = ds.sizes[dim]
        elif isinstance(conf_val, int):
            computed_chunks[dim] = conf_val
        else:
            computed_chunks[dim] = 1

    return computed_chunks


# ---------------------------------------------------------------------------
# 4) France spatial filtering 

def load_geom_france(ne_directory: str):
    shapefile_path = os.path.join(ne_directory, "ne_10m_admin_0_countries.shp")
    gdf = gpd.read_file(shapefile_path, engine="pyogrio")
    gdf_france = gdf[gdf["ADMIN"] == "France"].to_crs(epsg=4326)
    geom_france = gdf_france.geometry.iloc[0]
    return geom_france

def filter_ds_to_france(ds: xr.Dataset, geom_france) -> xr.Dataset:
    """
    Filters a dataset to strictly keep points inside France
    and replaces (y, x) with a unique dimension 'points'.
    """
    if not isinstance(geom_france, (Polygon, MultiPolygon)):
        raise TypeError("France geometry is not a Shapely Polygon/MultiPolygon.")
    
    # lat/lon grid handling
    lat = ds["lat"]
    lon = ds["lon"]

    # 2D case
    if lat.ndim == 2 and lon.ndim == 2:
        lat2d, lon2d = lat.values, lon.values
    # 1D case (lat(y), lon(x))
    elif lat.ndim == 1 and lon.ndim == 1:
        lat2d, lon2d = np.meshgrid(lat.values, lon.values, indexing="ij")
    else:
        raise ValueError("lat/lon are not in a recognized format")

    ny, nx = lat2d.shape
    lat_flat = lat2d.ravel()
    lon_flat = lon2d.ravel()

    inside_mask_flat = vectorized.contains(geom_france, lon_flat, lat_flat)
    logger.info(f"{inside_mask_flat.sum()} points kept inside France out of {len(lat_flat)}.")

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
# 5) NetCDF opening + spatial filtering

def load_nc_file(
    nc_path: str,
    variables_config: Dict[str, Dict[str, Union[str, int]]],
    chunk_config: Dict[str, Union[int, float]],
    geom_france
) -> xr.Dataset:
    logger.info(f"--- Processing NetCDF file: {nc_path} ---")
    vars_to_keep = list(variables_config.keys())

    # Open dataset with requested variables only
    ds = xr.open_dataset(nc_path, chunks="auto")[vars_to_keep]

    # Round timestamps to nearest :30:00
    if "time" in ds.coords:
        original_time = ds["time"].values

        # Convert to pandas datetime index for easier rounding
        rounded_time = xr.DataArray(
            pd.to_datetime(original_time).round("30min"),
            dims="time"
        )

        ds = ds.assign_coords(time=rounded_time)
        logger.info("Timestamps rounded to nearest half-hour (:00 or :30).")

    logger.info(f"Dataset opened with native chunks. Dimensions: {dict(ds.sizes)}")

    # Spatial filtering
    ds = filter_ds_to_france(ds, geom_france)

    # Chunking
    computed_chunks = build_xarray_chunks(ds, chunk_config)
    ds = ds.chunk(computed_chunks)
    logger.info(f"Dataset rechunked")

    # Processing only data_vars
    for var in ds.data_vars:
        var_conf = variables_config.get(var, {})
        desired_dtype = var_conf.get("dtype", None)
        scale_factor = var_conf.get("scale_factor", None)
        unit_conversion = var_conf.get("unit_conversion", 1.0)
        fill_value = var_conf.get("fill_value", None)

        if desired_dtype is not None:
            logger.info(f"Converting {var} to {desired_dtype}")
            if unit_conversion != 1.0:
                logger.info(f"Unit conversion for {var} with factor {unit_conversion}")
                ds[var] = ds[var] * unit_conversion

            if scale_factor is not None:
                logger.info(f"Encoding factor {scale_factor} applied to {var}")
                ds[var] = ds[var] * scale_factor

            if fill_value is not None:
                logger.info(f"Replacing NaNs in {var} with sentinel {fill_value}")
                ds[var] = ds[var].fillna(fill_value)

            ds[var] = ds[var].round().astype(desired_dtype)
        else:
            logger.info(f"No dtype specified for {var}")

    logger.info(f"Loading finalized for {nc_path}")
    return ds


# ---------------------------------------------------------------------------
# 6) Sauvegarde Zarr adaptée à "points"

def add_num_poste_to_dataset(ds: xr.Dataset, postes_df: pd.DataFrame) -> xr.Dataset:
    lat = ds["lat"].values.astype(np.float32)
    lon = ds["lon"].values.astype(np.float32)
    coords_df = pd.DataFrame({"lat": lat, "lon": lon})

    # Merge to get NUM_POSTE
    merged = coords_df.merge(postes_df[["lat", "lon", "NUM_POSTE"]], on=["lat", "lon"], how="left")

    if merged["NUM_POSTE"].isnull().any():
        raise ValueError("Some points have no match in postes_df")

    # Récupération de NUM_POSTE uniquement
    num_poste_values = merged["NUM_POSTE"].values.astype(np.int32)

    # Assign coordinate
    ds = ds.assign_coords(NUM_POSTE=("points", num_poste_values))

    # Remove all other unnecessary coordinates
    ds = ds.drop_vars(["lat", "lon", "points"], errors="ignore")

    # Change main dimension
    ds = ds.swap_dims({"points": "NUM_POSTE"})  # replace 'points' with 'NUM_POSTE'
    ds = ds.sortby("NUM_POSTE")  # optional but neat

    return ds


def save_to_zarr(
    ds: xr.Dataset,
    output_path: str,
    chunk_config: Dict[str, Union[int, float]],
    compressor_config: Dict
) -> None:
    
    # --- Zarr compression with Blosc
    codec = numcodecs.Blosc(
        cname=compressor_config["blosc"]["cname"],
        clevel=compressor_config["blosc"]["clevel"],
        shuffle=compressor_config["blosc"]["shuffle"]
    )

    # --- Encoding dictionary construction
    encoding = {}

    def get_chunk_size(dim: str) -> int:
        val = chunk_config.get(dim, 1)
        return ds.sizes[dim] if val in [".inf", "inf", float("inf")] else int(val)

    for var in ds.data_vars:
        chunk_sizes = tuple(get_chunk_size(dim) for dim in ds[var].dims)
        encoding[var] = {"chunks": chunk_sizes, "compressor": codec}

    for coord in ["lat", "lon", "NUM_POSTE"]:
        if coord in ds.coords:
            chunk_sizes = tuple(get_chunk_size(dim) for dim in ds[coord].dims)
            encoding[coord] = {"chunks": chunk_sizes, "compressor": codec}

    logger.info("Encodage appliqué")

    # --- Explicit Dataset rechunk before export
    all_chunks = {}
    for var, enc in encoding.items():
        for dim, size in zip(ds[var].dims, enc["chunks"]):
            all_chunks[dim] = size  # overwrite accepted

    ds = ds.chunk(all_chunks)

    # --- Création dossier + export
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ds.to_zarr(output_path, mode='w', encoding=encoding)

    logger.info(f"Fichier Zarr sauvegardé sous {output_path}")


# Conversion function to generate hourly data
def convert_to_daily_precip(
    ds: xr.Dataset,
    var_name: str = "pr",
    scale_factor: float = 10.0,
    final_dtype: str = "int16",
    fill_value: int = -9999
) -> xr.Dataset:
    """
    Converts hourly (encoded) precipitation to daily precipitation
    (sum from 6h to 6h), with timestamp at 00h30 of day J+1.
    """
    pr = ds[var_name].astype("float32")

    # --- Cancel scale
    pr = pr / scale_factor # returns to mm/h (float32)

    # Aggregation by hydrological day with resample
    pr_daily = pr.resample(time="1D", offset="6h").sum()

    # Timestamp adjustment to 00:30 of J+1
    pr_daily["time"] = pr_daily["time"] - np.timedelta64(6, "h") + np.timedelta64(30, "m")

    # Re-scale
    pr_daily = pr_daily * scale_factor
    pr_daily = pr_daily.round().fillna(fill_value).astype(final_dtype)

    return pr_daily.to_dataset(name=var_name)



# ---------------------------------------------------------------------------
# 7) Pipeline principal

def pipeline_nc_to_zarr(config):
    global logger
    logger = get_logger(__name__)
    
    nc_dir = config["nc"]["path"]["inputdir"]
    alti_dir = config["altitude"]["path"]
    metadata_output_path = config["metadata"]["path"]["outputdir"]
    zarr_dir = config["zarr"]["path"]["outputdir"]
    chunk_config = config["zarr"]["chunks"]
    compressor_config = config["zarr"]["compressor"]
    variables_config = config["zarr"]["variables"]
    ne_directory = config["spatial_filter"]["ne_directory"]
    log_dir = config.get("log", {}).get("directory", "logs")

    nc_files = list_nc_files(nc_dir)
    
    geom_france = load_geom_france(ne_directory)

    postes_df = generate_metadata(nc_files, metadata_output_path, geom_france, alti_dir)

    for nc_file in nc_files:
        year = extract_year_from_nc(nc_file)
        if year is None:
            logger.warning(f"File skipped (missing year): {nc_file}")
            continue

        overwrite = config["zarr"].get("overwrite", False)
        # var_name = list(variables_config.keys())[0]
        # var_conf = variables_config[var_name]

        output_path_horaire = os.path.join(zarr_dir, "horaire", f"{year}.zarr")
        # output_path_quotidien = os.path.join(zarr_dir, "quotidien", f"{year}.zarr")

        horaire_exists = os.path.exists(output_path_horaire)
        # quotidien_exists = os.path.exists(output_path_quotidien)

        # Don't load unnecessarily
        if horaire_exists and not overwrite: # and quotidien_exists
            logger.info(f"[{year}] HOURLY already exists, skipping save.")
            # Load from Zarr to allow aggregation
            ds = xr.open_zarr(output_path_horaire)

        else:
            # Initial loading
            ds = load_nc_file(nc_file, variables_config, chunk_config, geom_france)
            ds = add_num_poste_to_dataset(ds, postes_df)
            save_to_zarr(ds, output_path_horaire, chunk_config, compressor_config)
            logger.info(f"[{year}] HOURLY saved")


        # # Quotidien
        # if not quotidien_exists or overwrite:
        #     logger.info(f"Conversion vers précipitations quotidiennes pour {year}")
        #     ds_daily = convert_to_daily_precip(
        #         ds,
        #         var_name=var_name,
        #         scale_factor=var_conf.get("scale_factor", 1.0),
        #         final_dtype=var_conf.get("dtype", "int16"),
        #         fill_value=var_conf.get("fill_value", -9999)
        #     )
        #     save_to_zarr(ds_daily, output_path_quotidien, chunk_config, compressor_config)
        #     logger.info(f"[{year}] QUOTIDIEN sauvegardé")
        #     ds_daily.close()
        # else:
        #     logger.info(f"Zarr QUOTIDIEN déjà existant pour {year}, skip.")

        ds.close()




# ---------------------------------------------------------------------------
# 8) Entrypoint

if __name__ == "__main__":
    dask.config.set(scheduler="threads")
    config = load_config("config/modelised_settings.yaml")
    pipeline_nc_to_zarr(config)
