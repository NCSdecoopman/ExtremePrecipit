import os
import zipfile
import argparse
from pathlib import Path
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from netCDF4 import Dataset

from src.utils.config_tools import load_config
from src.utils.logger import get_logger

# ================================================================
# Fonctions utilitaires
# ================================================================

def download_utils(dep: int, echelle: str, output_file: Path, full_url: str, logger) -> None:
    if output_file.exists():
        logger.info(f"Fichier échelle {echelle} déjà présent pour le département {dep:02d}")
        return

    try:
        r = requests.get(full_url, stream=True)
        if r.status_code == 200:
            with open(output_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Téléchargement terminé pour le département {dep:02d} à l'échelle horaire")
        else:
            logger.warning(f"Impossible de DL {full_url} (status={r.status_code})")
    except Exception as e:
        logger.warning(f"Erreur lors du téléchargement de {full_url} : {e}")


def download_quotidien_zip(dep: int, output_dir: Path, logger) -> None:
    """
    Télécharge les fichiers quotidiens pour un département donné selon :
    - https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/BASE/QUOT/Q_XX_previous-1950-2023_RR-T-Vent.csv.gz
    """
    base_url = "https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/BASE/QUOT"
    file_name = f"Q_{dep:02d}_previous-1950-2023_RR-T-Vent.csv.gz"
    full_url = f"{base_url}/{file_name}"
    output_file = output_dir / file_name

    download_utils(dep, 'quotidienne', output_file, full_url, logger)


def download_horaire_zip(dep: int, output_dir: Path, logger) -> None:
    """
    Télécharge les fichiers horaires pour un département donné selon :
    1950-1959, ..., 2010-2019 ainsi que 2020-2023
    """
    base_url = "https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/BASE/HOR"
    dep_str = f"{dep:02d}"
    # Périodes décennales de 1950 à 2010
    for year in range(1950, 2011, 10):
        end_year = year + 9
        file_name = f"H_{dep_str}_{year}-{end_year}.csv.gz"
        full_url = f"{base_url}/{file_name}"
        output_file = output_dir / file_name
        download_utils(dep, f'horaire_{year}-{end_year}', output_file, full_url, logger)

    # Période spéciale 2020-2023
    file_name = f"H_{dep_str}_previous-2020-2023.csv.gz"
    full_url = f"{base_url}/{file_name}"
    output_file = output_dir / file_name
    download_utils(dep, 'horaire_2020-2023', output_file, full_url, logger)


def download_all_zips(echelles: list, logger, max_workers: int = 8) -> None:
    """
    Téléchargement parallèle des fichiers horaires/quotidiens pour tous les départements.
    """
    horaire_dir = Path("data/temp/horaire_zip")
    quotidien_dir = Path("data/temp/quotidien_zip")
    horaire_dir.mkdir(parents=True, exist_ok=True)
    quotidien_dir.mkdir(parents=True, exist_ok=True)

    tasks = []

    # Préparation des tâches
    for dep in range(1, 96):
        if "quotidienne" in echelles:
            tasks.append(("quotidienne", dep, quotidien_dir))
        if "horaire" in echelles:
            tasks.append(("horaire", dep, horaire_dir))

    # Téléchargement en parallèle
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for echelle, dep, output_dir in tasks:
            if echelle == "horaire":
                futures.append(executor.submit(download_horaire_zip, dep, output_dir, logger))
            elif echelle == "quotidienne":
                futures.append(executor.submit(download_quotidien_zip, dep, output_dir, logger))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Téléchargements"):
            try:
                future.result()
            except Exception as e:
                logger.warning(f"Erreur pendant un téléchargement : {e}")



def get_time_axis(annee: int, echelle: str) -> pd.DatetimeIndex:
    """
    Génère un axe de temps en fonction de l'échelle :
      - horaire : freq="H"
      - quotidienne : freq="D"

    De {annee}-01-01 00:00:00 à {annee}-12-31 23:59:59
    """
    if echelle == "horaire":
        freq = "H"
        start = f"{annee}-01-01 00:00:00"
        end = f"{annee}-12-31 23:59:59"
    elif echelle == "quotidienne":
        freq = "D"
        start = f"{annee}-01-01"
        end = f"{annee}-12-31"
    else:
        raise ValueError(f"Échelle inconnue: {echelle}")

    time_axis = pd.date_range(start=start, end=end, freq=freq, tz="UTC")
    return time_axis


def parse_csv(echelle: str, csv_path: Path, station_index: dict) -> pd.DataFrame:
    """
    Lit un fichier CSV (colonnes : NOM_USUEL, LAT, LON, AAAAMMJJ[HH], RR (quotidien) ou RR1 (horaire))
    et produit un DataFrame avec colonnes [station_idx, year, date_dt, pr].
    """
    if echelle == "horaire":
        cols = ["NOM_USUEL", "LAT", "LON", "AAAAMMJJHH", "RR1"]
        date_col = "AAAAMMJJHH"
        param_col = "RR1"
        date_fmt = "%Y%m%d%H"
    elif echelle == "quotidienne":
        cols = ["NOM_USUEL", "LAT", "LON", "AAAAMMJJ", "RR"]
        date_col = "AAAAMMJJ"
        param_col = "RR"
        date_fmt = "%Y%m%d"
    else:
        raise ValueError(f"Échelle inconnue: {echelle}")

    df = pd.read_csv(csv_path, sep=";", dtype=str, low_memory=False)
    print(df)
    df = df[[c for c in cols if c in df.columns]].dropna(subset=[cols[-1]])  # enlever lignes sans la colonne RR / RR1

    # Convert lat/lon
    df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
    df.dropna(subset=["LAT", "LON"], inplace=True)

    # Station -> index
    def get_station_idx(row):
        key = (row["NOM_USUEL"], float(row["LAT"]), float(row["LON"]))
        return station_index[key]

    df["station_idx"] = df.apply(get_station_idx, axis=1)

    # parse précip
    df[param_col] = df[param_col].str.replace(",", ".", regex=False)
    df[param_col] = pd.to_numeric(df[param_col], errors="coerce")

    # parse date
    df["date_dt"] = pd.to_datetime(df[date_col], format=date_fmt, errors="coerce", utc=True)
    df.dropna(subset=["date_dt"], inplace=True)

    df["year"] = df["date_dt"].dt.year
    df.rename(columns={param_col: "pr"}, inplace=True)

    return df[["station_idx", "year", "date_dt", "pr"]]


def fill_nc(echelle: str,
             year: int,
             data_for_year: pd.DataFrame,
             station_dim_size: int,
             nc_path: Path):
    """
    Remplit le NetCDF de l'année correspondante, échelle = {horaire|quotidienne},
    à partir d'un DataFrame data_for_year ayant colonnes:
        station_idx, date_dt, pr
    """
    if data_for_year.empty:
        return  # Rien à écrire

    # On ouvre le NetCDF en mode append
    with Dataset(nc_path, "a") as ds:
        time_var = ds.variables["time"]
        pr_var = ds.variables["pr"]
        times = time_var[:]  # float64 array

        # On reconstruit l'axe de temps en DatetimeIndex
        base_time = np.datetime64("1949-12-01T00:00:00")
        dt_array = base_time + (times * np.timedelta64(1, "D"))
        dt_index = pd.to_datetime(dt_array)

        dt_map = {dt_index[i]: i for i in range(len(dt_index))}
        pr_data = pr_var[:, :]

        for row in data_for_year.itertuples(index=False):
            station_idx = row.station_idx
            date_dt = row.date_dt
            pr_val = row.pr

            if date_dt in dt_map:
                t_idx = dt_map[date_dt]
                pr_data[t_idx, station_idx] = pr_val

        pr_var[:, :] = pr_data


def pipeline_obs_from_zip_to_nc(config):
    """
    Pipeline complet :
      1) Téléchargement (si nécessaire) des ZIP (horaire, quotidien)
      2) Parcours des ZIP (pass 1) pour indexer toutes les stations
      3) Création des NetCDF annuels (1959->2022) (sous-arborescence par échelle)
      4) Relit tous les ZIP (pass 2), parse les CSV, répartit par année, remplit les NetCDF.
      5) Supprime les ZIP.
    """
    logger = get_logger(__name__)

    # Paramètres
    echelles_cfg = config.get("echelles", [])
    # Désormais, on s'attend toujours à une liste (ex: ["horaire", "quotidienne"])
    if not isinstance(echelles_cfg, list):
        raise ValueError("'echelles' doit être une liste dans la config, ex: ['horaire','quotidienne']")
    echelles = echelles_cfg

    # 1) Téléchargement
    logger.info("--- Début du téléchargement des ZIP")
    download_all_zips(echelles, logger, max_workers=16)  # max_workers à adapter
    logger.info("--- Téléchargement terminé.")


    # Chemins de sortie
    nc_dirs = {
        "horaire": Path(config["nc"].get("horaire", {}).get("path", {}).get("outputdir", "data/raw/observed/horaire")),
        "quotidienne": Path(config["nc"].get("quotidienne", {}).get("path", {}).get("outputdir", "data/raw/observed/quotidien"))
    }
    for echelle in echelles:
        nc_dirs[echelle].mkdir(parents=True, exist_ok=True)

    # Dossiers ZIP
    zip_dirs = {
        "horaire": Path("data/temp/horaire_zip"),
        "quotidienne": Path("data/temp/quotidien_zip")
    }

    # # ---------------------------------------------------------
    # # PASS 1 : Récupération des stations (on ne lit que noms/lat/lon)
    # # ---------------------------------------------------------

    # stations_dict = {e: set() for e in echelles}

    # for echelle in echelles:
    #     all_zips = list(zip_dirs[echelle].glob("*.zip"))
    #     for zip_file in all_zips:
    #         logger.info(f"[PASS1] Lecture des stations dans {zip_file}")
    #         with zipfile.ZipFile(zip_file, 'r') as z:
    #             z.extractall("temp_extract_pass1")

    #         for csv_path in Path("temp_extract_pass1").rglob("*.csv"):
    #             try:
    #                 if echelle == "horaire":
    #                     relevant_cols = ["NOM_USUEL", "LAT", "LON", "AAAAMMJJHH", "RR1"]
    #                 else:
    #                     relevant_cols = ["NOM_USUEL", "LAT", "LON", "AAAAMMJJ", "RR"]

    #                 df = pd.read_csv(csv_path, sep=";", dtype=str, usecols=lambda c: c in relevant_cols, low_memory=False)
    #                 df.dropna(subset=["LAT", "LON", relevant_cols[-1]], inplace=True)

    #                 df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
    #                 df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
    #                 df.dropna(subset=["LAT", "LON"], inplace=True)

    #                 for row in df.itertuples(index=False):
    #                     nom_usuel = getattr(row, "NOM_USUEL")
    #                     lat_val = getattr(row, "LAT")
    #                     lon_val = getattr(row, "LON")
    #                     stations_dict[echelle].add((nom_usuel, float(lat_val), float(lon_val)))
    #             except Exception as e:
    #                 logger.warning(f"Erreur lecture CSV {csv_path}: {e}")

    #         for f in Path("temp_extract_pass1").rglob("*"):
    #             f.unlink()
    #         try:
    #             Path("temp_extract_pass1").rmdir()
    #         except OSError:
    #             pass

    # station_index_map = {}
    # for echelle in echelles:
    #     sorted_stations = sorted(list(stations_dict[echelle]))
    #     station_index_map[echelle] = {
    #         st: i for i, st in enumerate(sorted_stations)
    #     }
    #     logger.info(f"{len(sorted_stations)} stations détectées pour l'échelle {echelle}")

    # # ---------------------------------------------------------
    # # Création des NetCDF : 1959->2022 pour chaque échelle
    # # ---------------------------------------------------------

    # for echelle in echelles:
    #     stations_for_ech = station_index_map[echelle]
    #     nb_stations = len(stations_for_ech)
    #     for annee in range(1959, 2023):
    #         nc_path = nc_dirs[echelle] / f"observed_{annee}01010000-{annee}12312359.nc"
    #         if nc_path.exists():
    #             logger.info(f"[INIT] Fichier déjà existant: {nc_path}, on saute")
    #             continue

    #         logger.info(f"[INIT] Création du NC pour {nc_path}")
    #         time_axis = get_time_axis(annee, echelle)
    #         ntime = len(time_axis)

    #         with Dataset(nc_path, "w", format="NETCDF4") as ds:
    #             ds.createDimension("time", ntime)
    #             ds.createDimension("station", nb_stations)

    #             time_var = ds.createVariable("time", "f8", ("time",))
    #             time_var.units = "days since 1949-12-01"
    #             time_var.calendar = "standard"
    #             base_time = np.datetime64("1949-12-01T00:00:00")
    #             time_naive = time_axis.tz_localize(None)
    #             offset_days = (time_naive.values - base_time) / np.timedelta64(1, "D")
    #             time_var[:] = offset_days

    #             pr_var = ds.createVariable("pr", "f4", ("time", "station"), fill_value=np.nan)
    #             pr_var.units = "mm/h ou mm/j"
    #             pr_var.long_name = "Précipitation"

    #             lat_var = ds.createVariable("lat", "f4", ("station",))
    #             lon_var = ds.createVariable("lon", "f4", ("station",))
    #             nom_var = ds.createVariable("station_name", str, ("station",))

    #             st_sorted = sorted(stations_for_ech.keys(), key=lambda s: stations_for_ech[s])
    #             lat_vals = [s[1] for s in st_sorted]
    #             lon_vals = [s[2] for s in st_sorted]
    #             nom_vals = [s[0] for s in st_sorted]

    #             lat_var[:] = lat_vals
    #             lon_var[:] = lon_vals
    #             for i, nom in enumerate(nom_vals):
    #                 nom_var[i] = nom

    # # ---------------------------------------------------------
    # # PASS 2 : Lecture / distribution des données dans chaque NetCDF
    # # ---------------------------------------------------------

    # for echelle in echelles:
    #     all_zips = list(zip_dirs[echelle].glob("*.zip"))
    #     for zip_file in all_zips:
    #         logger.info(f"[PASS2] Lecture des données dans {zip_file}")
    #         with zipfile.ZipFile(zip_file, 'r') as z:
    #             z.extractall("temp_extract_pass2")

    #         for csv_path in Path("temp_extract_pass2").rglob("*.csv"):
    #             try:
    #                 df_data = parse_csv(echelle, csv_path, station_index_map[echelle])
    #                 if df_data.empty:
    #                     continue
    #                 for year, df_year in df_data.groupby("year"):
    #                     if not (1959 <= year <= 2022):
    #                         continue
    #                     nc_path = nc_dirs[echelle] / f"observed_{year}01010000-{year}12312359.nc"
    #                     fill_nc(echelle,
    #                             year,
    #                             df_year,
    #                             len(station_index_map[echelle]),
    #                             nc_path)
    #             except Exception as e:
    #                 logger.warning(f"Erreur lecture CSV {csv_path}: {e}")

    #         for f in Path("temp_extract_pass2").rglob("*"):
    #             f.unlink()
    #         try:
    #             Path("temp_extract_pass2").rmdir()
    #         except OSError:
    #             pass

    #         # Suppression du .zip
    #         zip_file.unlink()

    # # ---------------------------------------------------------
    # # Nettoyage : suppression des fichiers CSV.gz téléchargés
    # # ---------------------------------------------------------
    # logger.info("Suppression des fichiers .csv.gz après traitement")
    # for echelle in echelles:
    #     dir_path = zip_dirs[echelle]
    #     for gz_file in dir_path.glob("*.csv.gz"):
    #         try:
    #             gz_file.unlink()
    #             logger.info(f"Supprimé : {gz_file}")
    #         except Exception as e:
    #             logger.warning(f"Impossible de supprimer {gz_file} : {e}")

    #     # Suppression des dossiers s'ils sont vides
    #     try:
    #         dir_path.rmdir()
    #         logger.info(f"Dossier supprimé : {dir_path}")
    #     except OSError:
    #         logger.info(f"Dossier non vide ou non supprimé : {dir_path}")

    # logger.info("Pipeline terminé")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline observations (ZIP) vers .nc.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/observed_settings.yaml",
        help="Chemin vers le fichier de configuration YAML (par défaut : config/observed_settings.yaml)"
    )
    args = parser.parse_args()
    config_path = args.config
    config = load_config(config_path)

    log_dir = config.get("log", {}).get("directory", "logs")
    logger = get_logger(__name__, log_to_file=True, log_dir=log_dir)

    logger.info(f"Démarrage du pipeline observations (ZIP) → .nc avec la config : {config_path}")
    pipeline_obs_from_zip_to_nc(config)
