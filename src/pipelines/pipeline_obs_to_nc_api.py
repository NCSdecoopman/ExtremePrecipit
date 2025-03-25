import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import xarray as xr
import requests
import time
import io

from src.utils.config_tools import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- Fonction utilitaire
# Génère l'axe du temps
def get_time_axis(annee: int, echelle: str) -> pd.DatetimeIndex:
    if echelle == "horaire":
        freq = "h"
        start = f"{annee}-01-01 00:00:00"
        end = f"{annee}-12-31 23:59:00"
    elif echelle == "quotidienne":
        freq = "D"
        start = f"{annee}-01-01"
        end = f"{annee}-12-31"
    else:
        logger.error("Échelle inconnue")
        raise ValueError("Échelle inconnue")

    time_axis = pd.date_range(start=start, end=end, freq=freq, tz="UTC")
    return time_axis


# Génère les fichiers .nc vides
def init_netcdf_file(path, time_axis):
    ntime = len(time_axis)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with Dataset(path, "w", format="NETCDF4") as ds:
        # Définition des dimensions
        ds.createDimension("time", ntime)
        ds.createDimension("station", None)  # Dimension extensible

        # Variable time (fixe)
        time_var = ds.createVariable("time", "f8", ("time",))
        time_var.units = "days since 1949-12-01"
        time_var.calendar = "standard"

        # Convertir en tz-naive pour éviter l'erreur
        time_naive = time_axis.tz_localize(None)
        base_time = np.datetime64("1949-12-01T00:00:00")
        time_var[:] = (time_naive - base_time) / np.timedelta64(1, "D")

        # Variables stationnaires extensibles, chunkées (mais sans compression)
        pr_var = ds.createVariable("pr", "f4", ("time", "station"), chunksizes=(ntime, 1))
        lat_var = ds.createVariable("lat", "f4", ("station",), chunksizes=(1,))
        lon_var = ds.createVariable("lon", "f4", ("station",), chunksizes=(1,))
        stn_var = ds.createVariable("station", str, ("station",), chunksizes=(1,))

        pr_var.units = "mm/h ou mm/j"
        pr_var.long_name = "Precipitation"

        ds.description = "Fichier NetCDF vide initialisé pour écriture incrémentale"


# ---------------------------------------------------------------------------
# Fonction de génération des stations disponibles
def get_total_stations(
        id_departement: int,
        echelle: str,
        base_url: str, 
        headers: dict[str, str],
        parametre: str = 'precipitation'):
    
    url = f"{base_url}/liste-stations/{echelle}"
    params = {
        'id-departement': id_departement,
        'parametre': parametre
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        logger.info(f"Département {id_departement:02d} - {len(data)} stations trouvées à l'échelle {echelle}")
        return data
    else:
        logger.error(f"Erreur {response.status_code}: {response.text}")
        return []
    
# Pipeline de génération des stations
def pipeline_stations(meta_dir: str, base_url: str, headers: dict[str, str]) -> pd.DataFrame:
    logger.info("Recherche des stations dispobles dans Meteo-France...")
    # Récupération de tous les départements
    all_stations = []

    # Récupération de toutes les stations à enregistrement horaire ou journalière
    for echelle in ["horaire", "quotidienne"]:
        for dep in range(1, 96):
            stations = get_total_stations(dep, echelle, base_url, headers)
            for s in stations:
                all_stations.append({
                    "id": s["id"],
                    "lat": np.float32(s["lat"]),
                    "lon": np.float32(s["lon"]),
                    "nom": s["nom"],
                    "echelle": echelle
                })

    # Transformation en DataFrame
    df = pd.DataFrame(all_stations)

    # Pivot pour obtenir horaire / quotidienne en booléens
    df["horaire"] = df["echelle"] == "horaire"
    df["quotidienne"] = df["echelle"] == "quotidienne"

    df_final = df.groupby(["id", "lat", "lon", "nom"]).agg({
        "horaire": "max",
        "quotidienne": "max"
    }).reset_index()

    # Conversion des types
    df_final["id"] = df_final["id"].astype(str)
    df_final["horaire"] = df_final["horaire"].astype(bool)
    df_final["quotidienne"] = df_final["quotidienne"].astype(bool)

    # Sauvegarde
    logger.info("Metadonnées des stations disponibles finalisées :")
    logger.info(df_final.head())

    # Export CSV
    os.makedirs(meta_dir, exist_ok=True)
    meta_path = os.path.join(meta_dir, "metadonnees_observed.csv")
    df_final.to_csv(meta_path, index=False)
    logger.info(f"Metadonnées des stations disponibles enregistrées dans {meta_path}")

    return df_final


# ---------------------------------------------------------------------------
# Renvoi la clé du fichier associé à une station pour une date données
def commander_donnees_station(
    id_station: str,
    date_deb: str,
    date_fin: str,
    echelle: str,
    base_url: str,
    headers: dict
):
    url = f"{base_url}/commande-station/{echelle}"
    params = {
        "id-station": id_station,
        "date-deb-periode": date_deb,
        "date-fin-periode": date_fin
    }

    while True:
        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 202:
            data = response.json()
            return data["elaboreProduitAvecDemandeResponse"]["return"]
        
        elif response.status_code == 429:
            try:
                data = response.json()
                next_access_time = data.get("nextAccessTime")

                if next_access_time:
                    from dateutil import parser
                    from datetime import datetime, timezone

                    mois_fr_to_en = {
                        "janvier": "January", "février": "February", "mars": "March", "avril": "April",
                        "mai": "May", "juin": "June", "juillet": "July", "août": "August",
                        "septembre": "September", "octobre": "October", "novembre": "November", "décembre": "December"
                    }

                    for fr, en in mois_fr_to_en.items():
                        next_access_time = next_access_time.replace(fr, en)

                    next_access_time = next_access_time.replace(" UTC", "")
                    wait_until = parser.parse(next_access_time).astimezone(timezone.utc)
                    now = datetime.now(timezone.utc)
                    wait_seconds = (wait_until - now).total_seconds()

                    if wait_seconds > 0:
                        logger.info(f"Attente de {int(wait_seconds)} secondes avant de réessayer...")
                        time.sleep(wait_seconds + 1)
                        continue  # <-- on réessaie !
            except Exception as e:
                logger.error(f"Erreur de parsing du 429 : {e}")
                time.sleep(60)
                continue  # <-- on réessaie même si erreur de parsing

        else:
            logger.error(f"Erreur commande {id_station}: {response.status_code}")
            return None


    
# Récupérer un fichier CSV généré suite à la commande sur l'API
def recuperer_csv_depuis_commande(
    id_cmde: str,
    base_url: str,
    headers: dict,
    max_essais: int = 10,
    pause: int = 10
) -> str | None:
    url = f"{base_url}/commande/fichier?id-cmde={id_cmde}"

    for essai in range(1, max_essais + 1):
        response = requests.get(url, headers=headers)
        status = response.status_code

        if status == 201:  # Fichier prêt
            return response.content.decode('utf-8'), True

        elif status == 204:  # Fichier en cours de production
            time.sleep(pause)

        elif status == 429:  # Trop de requêtes
            try:
                data = response.json()
                next_access_time = data.get("nextAccessTime")

                if next_access_time:
                    from dateutil import parser
                    from datetime import datetime, timezone

                    mois_fr_to_en = {
                        "janvier": "January", "février": "February", "mars": "March", "avril": "April",
                        "mai": "May", "juin": "June", "juillet": "July", "août": "August",
                        "septembre": "September", "octobre": "October", "novembre": "November", "décembre": "December"
                    }

                    for fr, en in mois_fr_to_en.items():
                        next_access_time = next_access_time.replace(fr, en)

                    next_access_time = next_access_time.replace(" UTC", "")
                    wait_until = parser.parse(next_access_time).astimezone(timezone.utc)
                    now = datetime.now(timezone.utc)
                    wait_seconds = (wait_until - now).total_seconds()

                    if wait_seconds > 0:
                        logger.info(f"Attente de {int(wait_seconds)} secondes avant de réessayer...")
                        time.sleep(wait_seconds + 1)
                        continue
            except Exception as e:
                logger.error(f"Erreur de parsing du 429 : {e}")
                time.sleep(60)
                continue

        elif status == 500:
            msg = response.text.lower()
            if "absence de données" in msg:
                return None, "absence" # absence de données explicite
            else:
                logger.error(f"Erreur serveur 500 pour {id_cmde} : {response.text}")
            break

        elif status == 410:
            logger.warning(f"Commande expirée ou supprimée (410) pour {id_cmde}")
            break

        else:
            logger.error(f"Erreur inattendue : {status} - {response.text}")
            break

    logger.info(f"Abandon après {essai} essais pour la commande {id_cmde}")
    return None, False     # erreur


# Convertit une chaîne CSV (issue de l'API Météo-France) en DataFrame pandas
def csv_string_to_df_pd(csv_str: str, echelle: str) -> pd.DataFrame:
    # Vérification de l'échelle
    if echelle not in {"horaire", "quotidienne"}:
        logger.error(f"Échelle non reconnue : {echelle}")
        raise ValueError(f"Échelle non valide : {echelle}. Doit être 'horaire' ou 'quotidienne'.")

    try:
        df = pd.read_csv(io.StringIO(csv_str), sep=';', low_memory=False)
    except Exception as e:
        logger.error(f"Erreur lors de la lecture du CSV : {e}")
        raise

    # Choix du bon nom de colonne selon l'échelle
    col_precip = "RR1" if echelle == "horaire" else "RR"

    # Vérification de la présence des colonnes attendues
    expected_cols = {"POSTE", "DATE", col_precip}
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        logger.warning(f"Colonnes manquantes dans le CSV : {missing_cols}")
        raise ValueError(f"Colonnes manquantes : {missing_cols}")

    # Sélection et renommage
    df = df[["POSTE", "DATE", col_precip]].rename(columns={
        "POSTE": "id",
        "DATE": "date",
        col_precip: "pr"
    })

    # Conversion des dates
    date_format = "%Y%m%d%H" if echelle == "horaire" else "%Y%m%d"
    try:
        df["date"] = pd.to_datetime(df["date"], format=date_format, utc=True)
    except Exception as e:
        logger.warning(f"Erreur de conversion des dates : {e}")
        raise

    # Nettoyage des précipitations
    try:
        # Conversion des virgules décimales → points, conversion en float
        df["pr"] = df["pr"].astype(str).str.replace(",", ".", regex=False)
        df["pr"] = pd.to_numeric(df["pr"], errors="coerce")
    except Exception as e:
        logger.warning(f"Erreur de conversion des valeurs de précipitations : {e}")
        raise


    return df

# Génération des relevés des stations
def recuperer_serie_station(id_station, annee, echelle, base_url, headers, time_axis):
    date_deb = f"{annee}-01-01T00:00:00Z"
    date_fin = f"{annee}-12-31T23:59:59Z"

    cmde = commander_donnees_station(id_station, date_deb, date_fin, echelle, base_url, headers)
    if cmde:
        csv_str, status = recuperer_csv_depuis_commande(cmde, base_url, headers)

        if status == "absence":
            return np.full(len(time_axis), np.nan, dtype=np.float32)

        if status is not True:
            logger.error(f"Erreur ou échec récupération CSV pour {id_station} - {echelle} - {annee}")
            return None

        try:
            df = csv_string_to_df_pd(csv_str, echelle)
            serie = df.set_index("date").reindex(time_axis)["pr"].astype(float).to_numpy()

            return serie.astype(np.float32)

        except Exception as e:
            logger.error(f"Erreur parsing station {id_station} année {annee} : {e}")
            return None

    return None

# ---------------------------------------------------------------------------
# Pipeline principal

def pipeline_obs_to_nc(config_path: str):
    logger.info(f"Démarrage du pipeline avec la config : {config_path}")

    config = load_config(config_path)
    base_url = config["api_info"]["base_url"]
    api_key = config["api_info"]["api_key"]
    echelles = config.get("echelles", [])
    metadonnees_dir = config["metadonnees"]["path"]["outputdir"]
    nc_dir = config["nc"]["path"]["outputdir"]

    headers = {
        'apikey': api_key
    }

    # Lecture des stations
    #stations = pipeline_stations(metadonnees_dir, base_url, headers)
    stations = pd.read_csv(
        os.path.join(metadonnees_dir, "observed_stations.csv"),
        dtype={"id": str}
    )
    print(stations)

    # Génération des fichiers NetCDF vides si non existants
    os.makedirs(nc_dir, exist_ok=True)
    for echelle in echelles:
        for annee in range(1959, 2023):
            path = os.path.join(nc_dir, echelle, f"observed_{annee}01010000-{annee}12312359.nc")
            if os.path.exists(path):
                logger.info(f"Le fichier NetCDF {path} existe déjà : on ne le recrée pas")
            else:
                time_axis = get_time_axis(annee, echelle)
                init_netcdf_file(path, time_axis)


    fichiers_generes = set()

    # Initialiser ou charger le checkpoint
    log_dir = config.get("log", {}).get("directory", "logs")
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_path = os.path.join(log_dir, "checkpoint_pipeline_api_to_nc.csv")

    if os.path.exists(checkpoint_path):
        checkpoint = pd.read_csv(checkpoint_path, dtype={"id_station": str})
    else:
        checkpoint = pd.DataFrame(columns=["id_station", "annee", "echelle"])
        with open(checkpoint_path, "w") as f:
            f.write("id_station,annee,echelle\n")

    total_stations = len(stations)
    # Boucle principale
    for i, row in stations.iterrows():
        id_station = row["id"]
        lon_val = row["lon"]
        lat_val = row["lat"]
        logger.info(f"Traitement de la station id{id_station} : {i}/{total_stations}")

        for echelle in ["horaire", "quotidienne"]:
            logger.info(f"Traitement de l'échelle {echelle}")
            if not row[echelle]:
                continue

            for annee in range(1959, 2023):
                # Vérifie si déjà traité
                if ((checkpoint["id_station"] == id_station) &
                    (checkpoint["annee"] == annee) &
                    (checkpoint["echelle"] == echelle)).any():
                    logger.info(f"Déjà traité : {id_station} - {echelle} - {annee} → skip")
                    continue

                time_axis = get_time_axis(annee, echelle)
                val = recuperer_serie_station(id_station, annee, echelle, base_url, headers, time_axis)
                # Skip uniquement si parsing/API a échoué
                if val is None:
                    logger.warning(f"Erreur de parsing pour {id_station} - {echelle} - {annee}")
                    continue

                nc_path = os.path.join(nc_dir, echelle, f"observed_{annee}01010000-{annee}12312359.nc")

                try:
                    with Dataset(nc_path, mode="a") as ds:
                        nstations = ds.dimensions["station"].size
                        ds.variables["pr"][:, nstations] = val
                        ds.variables["lat"][nstations] = lat_val
                        ds.variables["lon"][nstations] = lon_val
                        ds.variables["station"][nstations] = id_station

                    # Ajout dans le checkpoint uniquement si tout a été écrit
                    with open(checkpoint_path, "a") as f:
                        f.write(f"{id_station},{annee},{echelle}\n")
                        logger.info(f"{annee} ajoutée avec succès au NetCDF et checkpointée")

                except Exception as e:
                    logger.error(f"Erreur écriture NetCDF {nc_path} pour station {id_station} : {e}")
                    continue  # ne rien ajouter au checkpoint


                fichiers_generes.add((echelle, annee))

    # Log résumé
    log_path = os.path.join(log_dir, "pipeline_obs_to_nc_resume.log")

    nb_total = stations.shape[0]
    nb_horaire = stations["horaire"].sum()
    nb_quot = stations["quotidienne"].sum()

    annees_par_echelle = {}
    for echelle, annee in fichiers_generes:
        annees_par_echelle.setdefault(echelle, []).append(annee)

    # Vérification des manquants : construction du "tout" puis comparaison avec checkpoint --
    years = range(1959, 2023)
    all_combos = []
    for _, row in stations.iterrows():
        station_id = row["id"]
        if row["horaire"]:
            for annee in years:
                all_combos.append((station_id, annee, "horaire"))
        if row["quotidienne"]:
            for annee in years:
                all_combos.append((station_id, annee, "quotidienne"))

    df_all = pd.DataFrame(all_combos, columns=["id_station", "annee", "echelle"])
    
    # Jointure pour repérer les absents
    df_merged = pd.merge(df_all, checkpoint,
                         on=["id_station", "annee", "echelle"],
                         how="left",
                         indicator=True)
    missing = df_merged[df_merged["_merge"] == "left_only"]

    with open(log_path, "w") as f:
        f.write("Résumé de la génération des NetCDF\n")
        f.write("-----------------------------------\n")
        f.write(f"Nombre total de stations uniques : {nb_total}\n")
        f.write(f"Nombre de stations horaires : {nb_horaire}\n")
        f.write(f"Nombre de stations quotidiennes : {nb_quot}\n\n")
        for echelle, annees in annees_par_echelle.items():
            annees_str = ', '.join(map(str, sorted(annees)))
            f.write(f"Années générées pour l'échelle {echelle} :\n{annees_str}\n\n")

        # Ajout de la liste des manquants
        if missing.empty:
            f.write("Toutes les données ont été téléchargées avec succès !\n")
        else:
            f.write("Les données suivantes sont manquantes :\n")
            f.write(missing[["id_station", "annee", "echelle"]].to_string(index=False))
            f.write("\n")




# ---------------------------------------------------------------------------
# Entrypoint

if __name__ == "__main__":
    pipeline_obs_to_nc("config/observed_settings.yaml")