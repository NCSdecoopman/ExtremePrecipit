import gzip
import argparse
from pathlib import Path
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr
from itertools import islice
import numcodecs

# Supposons qu'on ait déjà défini ces fonctions ou modules
from src.utils.config_tools import load_config
from src.utils.logger import get_logger


# ================================================================
# Fonctions de téléchargement
# ================================================================

def download_utils(dep: int, echelle: str, output_file: Path, full_url: str, logger) -> None:
    if output_file.exists():
        logger.info(f"Fichier échelle {echelle} déjà présent pour le département {dep:02d}")

        # Vérifie si le fichier est un gzip valide
        try:
            with gzip.open(output_file, 'rb') as f:
                f.read()
        except Exception:
            logger.warning(f"Fichier corrompu détecté : {output_file.name}, suppression et re-téléchargement.")
            output_file.unlink()  # suppression
        else:
            return  # fichier OK → on quitte

    # Téléchargement
    try:
        r = requests.get(full_url, stream=True)
        if r.status_code == 200:
            with open(output_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Téléchargement terminé pour le département {dep:02d} à l'échelle {echelle}")
        else:
            logger.warning(f"Impossible de DL {full_url} (status={r.status_code})")
            return
    except Exception as e:
        logger.warning(f"Erreur lors du téléchargement de {full_url} : {e}")
        return

    # Vérifie que le fichier téléchargé est bien valide (pas tronqué)
    try:
        with gzip.open(output_file, 'rb') as f:
            f.read()
    except Exception:
        logger.error(f"Fichier téléchargé corrompu : {output_file.name} — suppression.")
        output_file.unlink()
    else:
        logger.info(f"Fichier validé : {output_file.name}")


def download_quotidien_zip(dep: int, output_dir: Path, logger) -> None:
    """
    Télécharge les fichiers quotidiens pour un département donné selon :
    - https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/BASE/QUOT/Q_XX_previous-1950-2023_RR-T-Vent.csv.gz
    """
    base_url = "https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/BASE/QUOT"
    file_name = f"Q_{dep:02d}_previous-1950-2023_RR-T-Vent.csv.gz"
    full_url = f"{base_url}/{file_name}"
    output_file = output_dir / file_name

    download_utils(dep, 'quotidien', output_file, full_url, logger)


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
        if "quotidien" in echelles:
            tasks.append(("quotidien", dep, quotidien_dir))
        if "horaire" in echelles:
            tasks.append(("horaire", dep, horaire_dir))

    # Téléchargement en parallèle
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for echelle, dep, output_dir in tasks:
            if echelle == "horaire":
                futures.append(executor.submit(download_horaire_zip, dep, output_dir, logger))
            elif echelle == "quotidien":
                futures.append(executor.submit(download_quotidien_zip, dep, output_dir, logger))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Téléchargements"):
            try:
                future.result()
            except Exception as e:
                logger.warning(f"Erreur pendant un téléchargement : {e}")


# ================================================================
# Fonctions utilitaires
# ================================================================

def get_time_axis(annee: int, echelle: str) -> pd.DatetimeIndex:
    """
    Génère un axe de temps en fonction de l'échelle :
      - horaire : freq="H"
      - quotidien : freq="D"
    De {annee}-01-01 00:00:00 à {annee}-12-31 23:59:59
    """
    if echelle == "horaire":
        freq = "h"
        start = f"{annee}-01-01 00:00:00"
        end = f"{annee}-12-31 23:59:59"
    elif echelle == "quotidien":
        freq = "D"
        start = f"{annee}-01-01"
        end = f"{annee}-12-31"
    else:
        raise ValueError(f"Échelle inconnue: {echelle}")

    time_axis = pd.date_range(start=start, end=end, freq=freq, tz="UTC")
    return time_axis


def parse_datetime(row, echelle):
    """
    Fonction utilitaire qu'on n'utilise pas forcément dans fill_zarr_from_csv.
    Mais si on voulait l'utiliser, on ferait :
       row["datetime"] = parse_datetime(row, echelle)
    """
    if echelle == "horaire":
        return pd.to_datetime(str(row["AAAAMMJJHH"]), format="%Y%m%d%H", errors="coerce")
    else:
        return pd.to_datetime(str(row["AAAAMMJJ"]), format="%Y%m%d", errors="coerce")


# ================================================================
# Fonctions principales
# ================================================================

def get_station_metadata(file):
    """
    Récupère un DataFrame avec NOM_USUEL, LAT, LON
    pour chaque station trouvée dans le fichier CSV.
    """
    try:
        cols = ["NOM_USUEL", "LAT", "LON"]
        df = pd.read_csv(file, sep=";", usecols=cols, compression="gzip")
        df = df.drop_duplicates()
        return df
    except Exception as e:
        logger.warning(f"Erreur lecture stations depuis {file} : {e}")
        return pd.DataFrame(columns=["NOM_USUEL", "LAT", "LON"])


def generate_zarr_structure(echelle, stations_df, config):
    """
    Crée une structure Zarr vide pour chaque année de 1959 à 2022
    et y stocke un Dataset (time, points):
    - pr : précipitation
    """
    zarr_dir = Path(config["zarr"]["path"]["outputdir"])
    zarr_dir.mkdir(parents=True, exist_ok=True)

    stations_df = stations_df.sort_values(by=["NOM_USUEL", "LAT", "LON"]).reset_index(drop=True)

    lat = stations_df["LAT"].values.astype(np.float32)
    lon = stations_df["LON"].values.astype(np.float32)

    # Choix de la fréquence
    time_freq = "h" if echelle == "horaire" else "D"
    fill_value = config["zarr"]["variables"]["pr"]["fill_value"]

    # Création pour chaque année
    for year in range(1959, 2023):
        # Génère l'axe de temps (on peut aussi utiliser get_time_axis(year, echelle))time = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:59:59", freq=time_freq, tz=None)
        time = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:59:59", freq=time_freq, tz=None)

        ds = xr.Dataset(
            data_vars={
                "pr": (("time", "points"), np.full((len(time), len(lat)), fill_value, dtype=np.float32))
            },
            coords={
                "time": time,
                "points": np.arange(len(lat))  # index station
            }
        )
        ds = ds.assign_coords(lat=("points", lat), lon=("points", lon))

        # Configuration du compresseur
        codec = numcodecs.Blosc(**config["zarr"]["compressor"]["blosc"])
        encoding = {
            "pr": {"chunks": (len(time), 100), "compressor": codec},
            "lat": {"chunks": (len(lat),), "compressor": codec},
            "lon": {"chunks": (len(lon),), "compressor": codec},
        }

        output_path = zarr_dir / echelle / f"{year}.zarr"
        output_path.parent.mkdir(exist_ok=True, parents=True)
        ds.to_zarr(output_path, mode="w", encoding=encoding)
        logger.debug(f"Création tableau Zarr : {len(time)} temps x {len(lat)} stations")

def fill_wrapper(file_path, echelle, station_idx_map, config):
    """Wrapper pour rendre les arguments compatibles avec ProcessPoolExecutor"""
    try:
        fill_zarr_from_csv(file_path, echelle, station_idx_map, config)
        return file_path.name, "Traité"
    except Exception as e:
        return file_path.name, f"Erreur: {str(e)}"


def fill_zarr_from_csv(file, echelle, station_idx_map, config):
    """
    Lit un CSV gzip, parse la date et la precipitation RR ou RR1,
    localise la bonne année, ouvre le Zarr, met à jour la variable 'pr'.
    """
    try:
        # Choix des colonnes selon l'échelle
        val_col = "RR1" if echelle == "horaire" else "RR"
        date_col = "AAAAMMJJHH" if echelle == "horaire" else "AAAAMMJJ"

        # Lecture
        df = pd.read_csv(
            file, sep=";",
            usecols=["NOM_USUEL", "LAT", "LON", date_col, val_col],
            compression="gzip"
        )
        
        # Convertit en float ce qui peut l'être
        df[val_col] = pd.to_numeric(df[val_col], errors="coerce")

        # Parse la date
        if echelle == "horaire":
            df["datetime"] = pd.to_datetime(
                df[date_col].astype(str),
                format="%Y%m%d%H",
                errors="coerce"
            )
        else:
            df["datetime"] = pd.to_datetime(
                df[date_col].astype(str),
                format="%Y%m%d",
                errors="coerce"
            )

        # On localise en UTC
        df["datetime"] = df["datetime"].dt.tz_localize("UTC")

        # Filtre lignes invalides
        df = df.dropna(subset=["datetime", val_col])

        # Année de chaque mesure
        df["year"] = df["datetime"].dt.year

        zarr_base = Path(config["zarr"]["path"]["outputdir"]) / echelle

        # On boucle par année pour remplir chaque .zarr
        for year, df_y in df.groupby("year"):
            # cast year → int si besoin
            try:
                year = int(year)
            except Exception:
                continue

            # Bornes
            if year < 1959 or year > 2022:
                continue

            path = zarr_base / f"{year}.zarr"
            if not path.exists():
                # pas de zarr correspondant
                continue

            # Ouvre le zarr
            ds = xr.open_zarr(path, chunks=None)
            pr = ds["pr"].load()  # on charge en RAM (simple, mais potentiellement gourmand)

            # Mise à jour station par station
            for _, row_data in df_y.iterrows():
                # Identifie la station
                station = (
                    row_data["NOM_USUEL"],
                    round(row_data["LAT"], 4),
                    round(row_data["LON"], 4)
                )
                idx = station_idx_map.get(station, None)
                if idx is not None:

                    dt64 = np.datetime64(row_data["datetime"].tz_convert(None), "ns")
                    time_array = ds.time.values.astype("datetime64[ns]")
                    time_idx = np.searchsorted(time_array, dt64)

                    if 0 <= time_idx < pr.shape[0]:
                        pr[time_idx, idx] = row_data[val_col]

            # Reconstruit le Dataset et réécrit en mode 'append'
            pr_ds = xr.Dataset(
                {"pr": (("time", "points"), pr.data)},
                coords={"time": ds.time, "points": ds.points}
            )

            pr_ds = pr_ds.assign_coords(lat=("points", ds.lat.values),
                                        lon=("points", ds.lon.values))

            pr_ds.to_zarr(path, mode="a")

    except Exception as e:
        logger.warning(f"Erreur traitement fichier {file} : {e}")


def pipeline_csv_to_zarr(config):
    global logger
    logger = get_logger(__name__)
    echelles = config.get("echelles", ["horaire", "quotidien"])
    max_workers = config.get("workers", 8)

    # Étape 1 : Téléchargement
    if config["data"]["download"]:
        logger.info("--- Téléchargement des fichiers depuis Météo-France")
        download_all_zips(echelles, logger, max_workers=16)  # ajuster au besoin
        logger.info("--- Téléchargement terminé.")
    else:
        logger.info("--- Téléchargement desactivé.")

    # Étape 2 : Récupération des métadonnées
    meta_dir = Path(config["metadata"]["path"]["outputdir"])
    meta_dir.mkdir(parents=True, exist_ok=True)

    for echelle in ["quotidien"]:
        logger.info(f"--- Construction des métadonnées de {echelle}")
        zip_dir = Path(f"data/temp/{echelle}_zip")
        all_files = sorted(zip_dir.glob("*.csv.gz"))

        # Collecte parallèle des infos station
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            all_dfs = list(tqdm(executor.map(get_station_metadata, all_files), total=len(all_files)))
        stations_df = pd.concat(all_dfs).drop_duplicates().reset_index(drop=True)

        # Sauvegarde
        stations_df.to_csv(meta_dir / f"stations_metadata_{echelle}.csv", index=False)

        # Création du map station → index
        station_idx_map = {
            (row["NOM_USUEL"], round(row["LAT"], 4), round(row["LON"], 4)): i
            for i, row in stations_df.iterrows()
        }

        logger.info(f"{len(station_idx_map)} stations extraites pour {echelle}")
        # Affiche quelques exemples
        logger.info(dict(islice(station_idx_map.items(), 10)))

        # Étape 3 : Création des fichiers .zarr par année
        logger.info(f"--- Création des fichiers Zarr pour {echelle}")
        generate_zarr_structure(echelle, stations_df, config)

        # Étape 4 : Remplissage
        logger.info(f"--- Remplissage des fichiers Zarr pour {echelle}")
        # for file in tqdm(all_files, desc="Remplissage des fichiers"):
        #     tqdm.write(f"Traitement du fichier : {file.name}")
        #     fill_zarr_from_csv(file, echelle, station_idx_map, config)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(fill_wrapper, file, echelle, station_idx_map, config): file
                for file in all_files
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Remplissage des fichiers"):
                file_name, status = future.result()
                tqdm.write(f"{file_name} → {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline CSV.gz vers .zarr")
    parser.add_argument("--config", type=str, default="config/observed_settings.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    pipeline_csv_to_zarr(config)
