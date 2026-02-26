import gzip
import argparse
from pathlib import Path
import os
import requests
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
import zarr
import numcodecs
from typing import Dict, List

from src.utils.config_tools import load_config
from src.utils.logger import get_logger

# ================================================================
# Download functions
# ================================================================

def download_utils(dep: int, echelle: str, output_file: Path, full_url: str, logger) -> str | None:
    if output_file.exists():
        logger.info(f"Scale {echelle} file already present for department {dep:02d}")
        try:
            with gzip.open(output_file, 'rb') as f:
                f.read()
        except Exception:
            logger.warning(f"Corrupted file detected: {output_file.name}, removing and re-downloading.")
            output_file.unlink()
        else:
            return None  # everything is fine

    try:
        r = requests.get(full_url, stream=True)
        if r.status_code == 200:
            with open(output_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Download completed for {output_file.name}")
        else:
            logger.warning(f"Could not download {full_url} (status={r.status_code})")
            return full_url
    except Exception as e:
        logger.warning(f"Error while downloading {full_url}: {e}")
        return full_url

    try:
        with gzip.open(output_file, 'rb') as f:
            f.read()
    except Exception:
        logger.error(f"Downloaded file corrupted: {output_file.name} — deleting.")
        output_file.unlink()
        return full_url

    logger.info(f"File validated: {output_file.name}")
    return None  # all OK


def download_quotidien_zip(dep: int, output_dir: Path, logger) -> str | None:
    """
    Downloads daily files for a given department according to:
    - https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/BASE/QUOT/Q_XX_previous-1950-2023_RR-T-Vent.csv.gz
    """
    file_name = f"Q_{dep:02d}_previous-1950-2023_RR-T-Vent.csv.gz"
    full_url = f"https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/BASE/QUOT/{file_name}"
    output_file = output_dir / file_name
    return download_utils(dep, 'quotidien', output_file, full_url, logger)


def download_horaire_zip(dep: int, output_dir: Path, logger) -> None:
    """
    Downloads hourly files for a given department according to:
    1950-1959, ..., 2010-2019 and 2020-2023
    """
    base_url = "https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/BASE/HOR"
    dep_str = f"{dep:02d}"
    failed = []

    for year in range(1950, 2011, 10):
        end_year = year + 9
        file_name = f"H_{dep_str}_{year}-{end_year}.csv.gz"
        full_url = f"{base_url}/{file_name}"
        output_file = output_dir / file_name
        result = download_utils(dep, f'horaire_{year}-{end_year}', output_file, full_url, logger)
        if result:
            failed.append(result)

    # 2020-2023
    file_name = f"H_{dep_str}_previous-2020-2023.csv.gz"
    full_url = f"{base_url}/{file_name}"
    output_file = output_dir / file_name
    result = download_utils(dep, 'horaire_2020-2023', output_file, full_url, logger)
    if result:
        failed.append(result)

    return failed


def download_all_zips(activate: bool, echelle: str, logger, max_workers: int = 48) -> dict:
    """
    Parallel download of hourly/daily files for all departments.
    Returns a dictionary with directory paths per scale.
    """
    dirs = {}
    if echelle == "horaire":
        horaire_dir = Path("data/temp/horaire_zip")
        horaire_dir.mkdir(parents=True, exist_ok=True)
        dirs["horaire"] = horaire_dir
    elif echelle == "quotidien":
        quotidien_dir = Path("data/temp/quotidien_zip")
        quotidien_dir.mkdir(parents=True, exist_ok=True)
        dirs["quotidien"] = quotidien_dir

    if activate:
        logger.info("[STEP 1] Downloading files from Météo-France")
        tasks = []

        for dep in range(1, 96):
            if echelle == "horaire":
                tasks.append(("horaire", dep, dirs["horaire"]))
            elif echelle == "quotidien":
                tasks.append(("quotidien", dep, dirs["quotidien"]))

        failed_all = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for echelle, dep, output_dir in tasks:
                if echelle == "horaire":
                    futures.append(executor.submit(download_horaire_zip, dep, output_dir, logger))
                elif echelle == "quotidien":
                    futures.append(executor.submit(download_quotidien_zip, dep, output_dir, logger))

            for future in tqdm(as_completed(futures), total=len(futures), desc="Téléchargements"):
                try:
                    result = future.result()
                    if result:
                        if isinstance(result, list):
                            failed_all.extend(result)
                        else:
                            failed_all.append(result)
                except Exception as e:
                    logger.warning(f"Error during a download: {e}")

        logger.info("File download completed")

        if failed_all:
            failed_file = Path(f"logs/observed/failed_downloads.txt")
            failed_file.parent.mkdir(parents=True, exist_ok=True)  # <- crée le dossier si nécessaire

            with open(failed_file, "w") as f:
                for url in failed_all:
                    f.write(url + "\n")
            logger.warning(f"[FINISH] {len(failed_all)} files missing or corrupted:")
            for url in failed_all:
                logger.warning(f"  - {url}")
        else:
            logger.info("[FINISH] All files downloaded and validated.")

    return dirs


# ================================================================
# Dataset formation functions
# ================================================================

def read_csv_file_polars(f, 
                         cols, 
                         var_col: str,
                         date_col: str,
                         id_col: str = "NUM_POSTE", 
                         lat_col: str = "lat", 
                         lon_col: str = "lon"):
    try:
        df = pl.read_csv(
            f,
            separator=";",
            has_header=True,
            columns=cols,
            encoding="utf8-lossy",
            null_values=["", "NaN", "NA"],  # extend as needed
        )
        if "LAT" in df.columns and "LON" in df.columns and "ALTI" in df.columns:
            df = df.rename({"LAT": "lat", "LON": "lon", "ALTI": "altitude"})
        else:
            logger.warning(f"[WARN] File {f.name} does not contain LAT/LON/ALTI")

        # Nettoyage RR : remplacer ',' par '.' si RR est de type str
        if df.schema.get(var_col) == pl.Utf8:
            bad_values = (
                df
                .filter(pl.col(var_col).str.contains(",|NaN|mq|tr"))
                .select(var_col)
                .unique()
                .to_series()
                .to_list()
            )
            if bad_values:
                logger.warning(f"[COL VAL] File {f.name} → {var_col} contains non-standard values: {bad_values}")
            
            df = df.with_columns(
                pl.col(var_col)
                .str.replace(",", ".")
                .cast(pl.Float64, strict=False)
                .alias(var_col)
            )

        # Global type enforcement for all columns
        df = df.with_columns([
            pl.col(id_col).cast(pl.Utf8),
            pl.col(lat_col).cast(pl.Float64, strict=False),
            pl.col(lon_col).cast(pl.Float64, strict=False),
            pl.col(date_col).cast(pl.Int64, strict=False),
        ])

        if date_col == "AAAAMMJJ":
            df = df.with_columns([
                (pl.col(date_col).cast(pl.Utf8)
                .str.strptime(pl.Datetime, format="%Y%m%d")
                .dt.offset_by("30m")  # Add +30 min
                .alias("time"))
            ])

        elif date_col == "AAAAMMJJHH":
            df = df.with_columns([
                (
                    (pl.col(date_col).cast(pl.Utf8) + "00")  # Ex: "2024010117" + "00" → "202401011700"
                    .str.strptime(pl.Datetime, format="%Y%m%d%H%M")
                    .dt.offset_by("-30m")  # Shift to HH-0.5h → ex: 17:00 → 16:30
                    .alias("time")
                )
            ])

        return df

    except Exception as e:
        print(f"Erreur avec {f}: {e}")
        return None
    
def delete_artefact_coord(df, 
                          id_col: str = "NUM_POSTE", 
                          lat_col: str = "lat", 
                          lon_col: str = "lon",
                          alti_col: str = "altitude"):
    # Calculate frequency of each (id, lat, lon)
    coord_counts = (
        df
        .select([id_col, lat_col, lon_col, alti_col])
        .group_by([id_col, lat_col, lon_col, alti_col])
        .len()
        .sort(by=[id_col, "len"], descending=True)
    )

    # Keep the most frequent coordinate par identifier
    coord_mode = (
        coord_counts
        .group_by(id_col)
        .first()
        .drop("len")
    )

    # Merge with original data (after removing columns to correct)
    df_correct = (
        df.drop([lat_col, lon_col, alti_col])
        .join(coord_mode, on=id_col, how="left")
    )

    return df_correct

def concat_csv_temp(csv_dir: str, cols: str, var_col: str, date_col: str, max_workers: int = 48):
    path_files = Path(csv_dir)
    files = sorted(list(path_files.glob("*.csv.gz")))
    df_list = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(read_csv_file_polars, f, cols, var_col, date_col): f for f in files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Reading files (.csv.gz)"):
            result = future.result()
            if result is not None:
                df_list.append(result)

    if df_list:
        df_all = pl.concat(df_list, how="vertical", rechunk=True)
        # Study period selection
        df_all = df_all.filter(
            df_all[date_col]
            .cast(str)
            .str.slice(0, 4)
            .cast(int)
            .is_between(1959, 2022)
        )
        df_all = delete_artefact_coord(df_all)
        return df_all
    else:
        return pl.DataFrame()

def group_df_by_year(df_all: pl.DataFrame) -> dict[int, pl.DataFrame]:
    df_all = df_all.with_columns(pl.col("time").dt.year().alias("year"))

    years = df_all.select("year").unique().sort("year")["year"].to_list()
    
    grouped_dict = {
        year: df_all.filter(pl.col("year") == year).drop("year")
        for year in years
    }

    return grouped_dict


# ================================================================
# Utility functions
# ================================================================

def get_time_axis(annee: int, echelle: str) -> pd.DatetimeIndex:
    """
    Generates a time axis aligned with .nc files (00:30 origin)
    """
    if echelle == "horaire":
        # Alignment at 00:30 as in CNRM-AROME files
        start = f"{annee}-01-01 00:30:00"
        end = f"{annee}-12-31 23:30:00"
        freq = "h"
    elif echelle == "quotidien":
        # Alignment at 00:30 for consistency
        start = f"{annee}-01-01 00:30:00"
        end = f"{annee}-12-31 00:30:00"
        freq = "D"
    else:
        raise ValueError(f"Unknown scale: {echelle}")

    return pd.date_range(start=start, end=end, freq=freq)



def generate_zarr_structure(zarr_dir, echelle, coords_df, config):
    """
    Creates an empty Zarr structure for each year from 1959 to 2022
    and stores a Dataset (time, points):
    - pr: precipitation
    """
    zarr_dir.mkdir(parents=True, exist_ok=True)

    lat = coords_df["lat"]
    lon = coords_df["lon"]
    if lat.len() != lon.len():
        logger.warning(f"LAT/LON size differs: lat {len(lat)} and lon {len(lon)}")

    fill_value = config["zarr"]["variables"]["pr"]["fill_value"]
    dtype_str = config["zarr"]["variables"]["pr"]["dtype"]
    dtype = np.dtype(dtype_str)

    # Create for each year
    for year in range(1959, 2023):
        # Generate time axis
        time = get_time_axis(year, echelle)

        ds = xr.Dataset(
            data_vars={
                "pr": (("time", "points"), np.full((len(time), len(lat)), fill_value, dtype=dtype))
            },
            coords={
                "time": time,
                "points": np.arange(len(lat))  # station index
            }
        )
        num_poste = coords_df["NUM_POSTE"]
        ds = ds.assign_coords(NUM_POSTE=("points", num_poste))
        ds = ds.swap_dims({"points": "NUM_POSTE"})
        ds = ds.drop_vars("points")

        # Compressor configuration
        codec = numcodecs.Blosc(**config["zarr"]["compressor"]["blosc"])
        encoding = {
            "pr": {"chunks": (len(time), 100), "compressor": codec, "dtype": dtype},
            "NUM_POSTE": {"chunks": (len(num_poste),), "compressor": codec}
        }

        output_path = zarr_dir / echelle / f"{year}.zarr"
        output_path.parent.mkdir(exist_ok=True, parents=True)
        ds.to_zarr(output_path, mode="w", encoding=encoding)

        logger.info(f"[EMPTY DATASET GENERATED] Year {year}")

        logger.debug(f"[DATASET STRUCTURE] Année {year}")
        logger.debug(f"  Dimensions : {dict(ds.sizes)}")
        logger.debug(f"  Coordonnées : {list(ds.coords)}")
        logger.debug(f"  Variables : {list(ds.data_vars)}")
        logger.debug(f"  time[0]: {ds.time.values[0]}, time[-1]: {ds.time.values[-1]}")
        logger.debug(f"  NUM_POSTE[0:3]: {ds.NUM_POSTE.values[:3]}")


def generate_metadata(meta_dir: str, echelle: str, df: pl.DataFrame):    
    meta_dir.mkdir(parents=True, exist_ok=True)
    postes_path = meta_dir / f"postes_{echelle}.csv"

    unique_coords_df = (
        df
        .select(["NUM_POSTE", "lat", "lon", "altitude"])
        .unique()
        .sort("NUM_POSTE")
    )

    unique_coords_df.write_csv(postes_path)
    logger.info(f"{unique_coords_df['NUM_POSTE'].n_unique()} stations generated")
    return unique_coords_df



# ================================================================
# Zarr filling functions
# ================================================================

# NUM_POSTE - points matching
def get_poste_to_point_map(postes_df: pl.DataFrame, logger) -> dict:
    # Mapping NUM_POSTE -> index in Zarr NUM_POSTE dimension (ds.NUM_POSTE.values)
    return {v: i for i, v in enumerate(postes_df["NUM_POSTE"])}

# Mapping time - index
def get_time_to_index_map(zarr_path: Path) -> dict:
    ds = xr.open_zarr(zarr_path)
    time = pd.to_datetime(ds['time'].values)
    return {t: i for i, t in enumerate(time)}

def fill_zarr_for_year(year: int, 
                       zarr_path: Path, 
                       postes_df: pl.DataFrame,
                       df_year: pl.DataFrame, 
                       fill_col: str,
                       config):
    """
    Fills a Zarr file for a given year with precipitation from a Polars DataFrame.
    Uses direct NumPy indexing to bypass Dask limitations.
    """
    # Load mappings
    poste_map = get_poste_to_point_map(postes_df, logger)
    time_map = get_time_to_index_map(zarr_path)

    # Open Zarr disabling Dask
    store = zarr.DirectoryStore(str(zarr_path))
    ds = xr.open_zarr(store, chunks=None)

    # Necessary columns extraction
    values = df_year[fill_col].to_numpy()
    times = df_year["time"].to_numpy()
    postes = df_year["NUM_POSTE"].to_numpy()

    # Time and matching conversions
    time_idx = np.array([time_map.get(pd.Timestamp(t)) for t in times])
    point_idx = np.array([poste_map.get(p) for p in postes])

    # Valid lines filtering
    valid_mask = (
        (time_idx != None) &
        (point_idx != None) &
        (~np.isnan(values))
    )

    time_idx = time_idx[valid_mask].astype(int)
    point_idx = point_idx[valid_mask].astype(int)
    values = values[valid_mask]

    # Config parameters retrieval
    scale_factor = config["zarr"]["variables"]["pr"]["scale_factor"]
    unit_conversion = config["zarr"]["variables"]["pr"]["unit_conversion"]
    fill_value = config["zarr"]["variables"]["pr"]["fill_value"]
    dtype_str = config["zarr"]["variables"]["pr"]["dtype"]
    dtype = np.dtype(dtype_str)

    # Unit conversion + scaling
    scaled_values = np.round(values * unit_conversion * scale_factor)

    # NaN to fill_value replacement and cast
    scaled_values = np.where(np.isnan(scaled_values), fill_value, scaled_values).astype(dtype)

    # Direct write to Zarr variable (NumPy)
    ds["pr"].data[time_idx, point_idx] = scaled_values
    ds.to_zarr(zarr_path, mode="a", append_dim=None)

    logger.info(f"[FILLING] Year {year} — {len(values)} values inserted out of {len(df_year)} lines")



# ================================================================
# Main functions
# ================================================================

def fill_zarr_task(year_df_tuple, zarr_base_path, postes_df, val_col, config):
    year, df_year = year_df_tuple
    zarr_path = zarr_base_path / f"{year}.zarr"
    
    if not zarr_path.exists():
        logger.warning(f"[SKIP] {year} (Zarr not found)")
        return f"[SKIPPED] {year}"

    fill_zarr_for_year(year, zarr_path, postes_df, df_year, val_col, config)
    return f"[DONE] {year}"


def pipeline_csv_to_zarr(config, max_workers: int = 48):    
    global logger
    logger = get_logger(__name__)
    echelles = config.get("echelles", ["horaire", "quotidien"])
    zarr_dir = Path(config["zarr"]["path"]["outputdir"])
    meta_dir = Path(config["metadata"]["path"]["outputdir"])

    # Étape 1 : Téléchargement
    activate_download_data = config["data"]["download"]

    # Étape 2 : Formation des datasets
    for echelle in echelles:
        dirs = download_all_zips(activate_download_data, echelle, logger)
        input_dir = dirs[echelle] # temp_data directory
        val_col = "RR1" if echelle == "horaire" else "RR"
        date_col = "AAAAMMJJHH" if echelle == "horaire" else "AAAAMMJJ"
        all_cols = ["NUM_POSTE", "LAT", "LON", "ALTI", date_col, val_col]

        logger.info(f"[CONCATENATION] Uploading datasets from {input_dir}")
        df_all = concat_csv_temp(input_dir, all_cols, val_col, date_col, max_workers)
        logger.info(df_all)
        df_all_years = group_df_by_year(df_all)

        # Duplicate check
        dupes = (
            df_all
            .select(["NUM_POSTE", "lat", "lon"])
            .unique()
            .group_by("NUM_POSTE")
            .len() 
            .filter(pl.col("len") > 1)
        )
        n_stations_diff_latlon = dupes.height
        if n_stations_diff_latlon == 0:
            logger.info(f"[OK] All stations have only one LAT/LON")
        else:
            logger.warning(f"Number of stations with multiple LAT/LON: {n_stations_diff_latlon}")


        overwrite = config["zarr"].get("overwrite", False)
        if overwrite:
            # Step 2: Empty zarr generation
            logger.info("[GENERATION] Generating associated metadata")
            postes_df = generate_metadata(meta_dir, echelle, df_all)


            logger.info(f"[GENERATION] Generating empty zarr files")
            generate_zarr_structure(zarr_dir, echelle, postes_df, config)

            n_missing_alt = postes_df.filter(pl.col("altitude").is_null()).height
            if n_missing_alt > 0:
                    logger.warning(f"{n_missing_alt} stations without height info.")

            # Step 3: Zarr filling
            logger.info("[FILLING] Beginning Zarr filling")
            zarr_base_path = zarr_dir / echelle

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(fill_zarr_task, item, zarr_base_path, postes_df, val_col, config): item[0]
                    for item in df_all_years.items()
                }
                for future in as_completed(futures):
                    msg = future.result()
                    if msg is not None:
                        logger.info(msg)
        else:
            logger.info(["[SKIP] No zarr rewrite"])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline CSV.gz vers .zarr")
    parser.add_argument("--config", type=str, default="config/observed_settings.yaml")
    parser.add_argument("--echelle", type=str, choices=["horaire", "quotidien"], nargs="+", default=["horaire", "quotidien"])
    args = parser.parse_args()

    config = load_config(args.config)
    config["echelles"] = args.echelle
    
    pipeline_csv_to_zarr(config, max_workers=96)
