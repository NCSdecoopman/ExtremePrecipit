import argparse
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import as_completed
import dask
from dask import delayed
from dask import compute
from dask.distributed import progress
from tqdm import tqdm

from sklearn.metrics import r2_score

from src.utils.config_tools import load_config
from src.utils.logger import get_logger

from dask.distributed import Client, LocalCluster
import time
from more_itertools import chunked

def start_dask_cluster(n_workers=28, threads_per_worker=2, memory_limit='10GB', retries=5):
    for attempt in range(retries):
        try:
            cluster = LocalCluster(
                n_workers=n_workers,
                threads_per_worker=threads_per_worker,
                memory_limit=memory_limit,
                processes=True
            )
            client = Client(cluster)
            return client, cluster
        except Exception as e:
            print(f"[Tentative {attempt+1}] Échec de démarrage du cluster : {e}")
            time.sleep(5)
    raise RuntimeError("Impossible de démarrer le cluster Dask.")


def find_matching_point(df, lat_obs, lon_obs, col_mod_lat: str="lat_mod", col_mod_lon: str="lon_mod"):
    # Calcul de la distance euclidienne
    distances = np.sqrt((df[col_mod_lat] - lat_obs)**2 + (df[col_mod_lon] - lon_obs)**2)
    idx_min = distances.idxmin()

    lat_mod = df.loc[idx_min, col_mod_lat]
    lon_mod = df.loc[idx_min, col_mod_lon]

    return lat_mod, lon_mod


def create_commun_point(df_obs: pd.DataFrame, df_mod: pd.DataFrame) -> pd.DataFrame:
    """
    Associe à chaque point observé (NUM_POSTE_obs) le point modélisé (NUM_POSTE_mod)
    spatialement le plus proche.
    """
    # Sélection et renommage clair des colonnes
    df_obs = df_obs[['NUM_POSTE', 'lat', 'lon']].rename(columns={
        'NUM_POSTE': 'NUM_POSTE_obs',
        'lat': 'lat_obs',
        'lon': 'lon_obs'
    })
    df_mod = df_mod[['NUM_POSTE', 'lat', 'lon']].rename(columns={
        'NUM_POSTE': 'NUM_POSTE_mod',
        'lat': 'lat_mod',
        'lon': 'lon_mod'
    })

    # Trouver les correspondances (lat/lon les plus proches)
    correspondances = []
    for _, row in tqdm(df_obs.iterrows(), total=len(df_obs), desc="Matching obs vs mod"):
        lat_obs, lon_obs = row["lat_obs"], row["lon_obs"]
        lat_mod, lon_mod = find_matching_point(df_mod, lat_obs, lon_obs)

        match_row = df_mod[
            (df_mod["lat_mod"] == lat_mod) & (df_mod["lon_mod"] == lon_mod)
        ]
        if match_row.empty:
            raise ValueError(f"Aucune correspondance trouvée pour ({lat_obs}, {lon_obs})")

        num_poste_mod = match_row["NUM_POSTE_mod"].values[0]

        correspondances.append({
            "NUM_POSTE_obs": int(row["NUM_POSTE_obs"]),
            "lat_obs": lat_obs,
            "lon_obs": lon_obs,
            "NUM_POSTE_mod": int(num_poste_mod),
            "lat_mod": lat_mod,
            "lon_mod": lon_mod
        })

    df_match = pd.DataFrame(correspondances)
    return df_match


# Retourne les années disponibles d'un répertoire
def get_available_years(path: Path) -> set:
    return {
        int(p.stem) for p in path.glob("*.zarr") if p.stem.isdigit()
    }

def clean_array(arr, fill_value, scale):
    arr = arr.astype("float32")
    arr[arr == fill_value] = np.nan
    return arr / scale

# Fonction de traitement
@delayed
def process_one_point(
    num_poste_obs,
    df_match,
    path_obs_zarr,
    path_mod_zarr,
    years,
    scale_factor_obs,
    scale_factor_mod,
    fill_value=-9999,
    output_path: Path = None
) -> str:
    try:
        row = df_match[df_match["NUM_POSTE_obs"] == num_poste_obs]
        if row.empty:
            raise ValueError(f"Aucune correspondance pour NUM_POSTE_obs = {num_poste_obs}")

        num_poste_mod = row["NUM_POSTE_mod"].item()
        dfs = []

        is_daily = "quotidien" in str(path_obs_zarr).lower()

        for year in years:
            # === Observations ===
            ds_obs = xr.open_zarr(path_obs_zarr / f"{year}.zarr")
            ds_obs = ds_obs.assign_coords(NUM_POSTE=("NUM_POSTE", ds_obs["NUM_POSTE"].values.astype(str)))
            pr_obs = ds_obs["pr"].sel(NUM_POSTE=str(num_poste_obs))
            time_obs = pd.to_datetime(ds_obs["time"].values)

            df_obs = pd.DataFrame({
                "time": time_obs,
                "pr_obs": clean_array(pr_obs.values, fill_value, scale_factor_obs)
            }).dropna()

            # === Modélisation ===
            ds_mod = xr.open_zarr(path_mod_zarr / f"{year}.zarr")
            ds_mod = ds_mod.assign_coords(NUM_POSTE=("NUM_POSTE", ds_mod["NUM_POSTE"].values.astype(str)))
            pr_mod = ds_mod["pr"].sel(NUM_POSTE=str(num_poste_mod))
            time_mod = pd.to_datetime(ds_mod["time"].values)

            df_mod_full = pd.DataFrame({
                "time": time_mod,
                "pr_mod": clean_array(pr_mod.values, fill_value, scale_factor_mod)
            }).dropna()

            if is_daily:
                # Agrégation de 06h J à 06h J+1 pour chaque date obs (timestamp à J+1 00:30)
                pr_mod_agg = []
                for t_obs in df_obs["time"]:
                    t0 = t_obs - pd.Timedelta(hours=18)  # 00:30 - 18h = 06h la veille
                    t1 = t_obs + pd.Timedelta(hours=5, minutes=30)  # 00:30 + 5h30 = 06h
                    mask = (df_mod_full["time"] >= t0) & (df_mod_full["time"] < t1)
                    total = df_mod_full.loc[mask, "pr_mod"].sum()
                    pr_mod_agg.append(total)

                df_obs["pr_mod"] = pr_mod_agg
                df = df_obs
            else:
                # Fusion simple sur les timestamps pour l’échelle horaire
                df_mod = df_mod_full
                df = pd.merge(df_obs, df_mod, on="time", how="left")

            dfs.append(df)

        dfs = [df for df in dfs if not df.empty and not df.isna().all().all()]
        if dfs:
            df_final = pd.concat(dfs, ignore_index=True)
        else:
            df_final = pd.DataFrame()  # ou gérer autrement le cas vide : renvoyer un empty

        if df_final.empty or len(df_final) < 2:
            filename = f"NUM_POSTE_{num_poste_obs}_R2_NaN.parquet"
        else:
            r2 = r2_score(df_final["pr_obs"], df_final["pr_mod"])
            if not -1 <= r2 <= 1 or np.isnan(r2):
                filename = f"NUM_POSTE_{num_poste_obs}_R2_NaN.parquet"
            else:
                filename = f"NUM_POSTE_{num_poste_obs}_R2_{r2:.3f}.parquet"

        # Sauvegarde
        output_file = output_path / filename
        df_final.to_parquet(output_file, index=False)

        return True

    except Exception as e:
        logger.error(f"Erreur pour NUM_POSTE_obs = {num_poste_obs}: {e}")
        return f"{num_poste_obs} FAILED"

    
    
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

    PATH_METADATA_OBS_VS_MOD = Path(config_obs["obs_vs_mod"]["metadata_path"]["outputdir"])
    OUTPUT_PATH = Path(config_obs["obs_vs_mod"]["path"]["outputdir"])

    for echelle in echelles:
        years_obs = get_available_years(PATH_ZARR_OBS/echelle)
        years_mod = get_available_years(PATH_ZARR_MOD/"horaire")
        ANNEES = sorted(years_obs & years_mod)

        logger.info(f"Traitement {echelle} de {min(ANNEES)} à {max(ANNEES)}")

        output_path_echelle = OUTPUT_PATH / echelle
        output_path_echelle.mkdir(parents=True, exist_ok=True)

        # === CHARGEMENT DES DONNÉES OBS & MOD (CENTRALISÉ) ===
        # métadonnées
        df_obs = pd.read_csv(f"{PATH_METADATA_OBS}/postes_{echelle}.csv")  # contient lat, lon observés
        df_mod = pd.read_csv(f"{PATH_METADATA_MOD}/postes_{echelle}.csv")  # contient lat, lon observés

        # Formation du fichiers de metadonnées de correspondance obs - mod
        df_match = create_commun_point(df_obs, df_mod)
        PATH_METADATA_OBS_VS_MOD.mkdir(parents=True, exist_ok=True)
        df_match.to_csv(PATH_METADATA_OBS_VS_MOD / f"obs_vs_mod_{echelle}.csv", index=False)
        logger.info(f"Fichier de correspondances coordonnées observées - modélisées enregistré sous {PATH_METADATA_OBS_VS_MOD}/obs_vs_mod_{echelle}.csv")

        # Liste des stations
        rows = list(df_match["NUM_POSTE_obs"].values)
        logger.info(f"{len(rows)} stations à traiter")

        tasks = [
            process_one_point(
                num_poste_obs=row,
                df_match=df_match,
                path_obs_zarr=PATH_ZARR_OBS / echelle,
                path_mod_zarr=PATH_ZARR_MOD / "horaire",
                years=ANNEES,
                scale_factor_obs=scale_factor_obs,
                scale_factor_mod=scale_factor_mod,
                fill_value=-9999,
                output_path=output_path_echelle
            ) for row in rows
        ]

        BATCH_SIZE = 100
        for i, chunk in enumerate(chunked(tasks, BATCH_SIZE)):
            logger.info(f"Traitement batch {i+1}/{len(tasks)//BATCH_SIZE+1}")
            futures = client.compute(chunk) # Calcul en parallèle
            #progress(futures) # Affiche une barre de progression en console
            results = client.gather(futures)

        # Compter les succès et les échecs de manière robuste
        n_success = sum(1 for r in results if r is True)
        n_fail = sum(1 for r in results if r is not True)

        logger.info("Résumé du traitement :")
        logger.info(f"  - {n_success} stations traitées avec succès")
        logger.info(f"  - {n_fail} stations en échec")


if __name__ == "__main__":
    client, cluster = start_dask_cluster()

    parser = argparse.ArgumentParser(description="Pipeline obs vs mod")
    parser.add_argument("--config_obs", type=str, default="config/observed_settings.yaml")
    parser.add_argument("--config_mod", type=str, default="config/modelised_settings.yaml")
    parser.add_argument("--echelle", type=str, choices=["horaire", "quotidien"], nargs="+", default=["horaire", "quotidien"])
    args = parser.parse_args()

    config_obs = load_config(args.config_obs)
    config_mod = load_config(args.config_mod)
    config_obs["echelles"] = args.echelle

    pipeline_obs_vs_mod(config_obs, config_mod)
