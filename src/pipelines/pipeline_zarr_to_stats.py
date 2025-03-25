import argparse
import os
import pandas as pd
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from datetime import timedelta

from src.utils.config_tools import load_config
from src.utils.logger import get_logger

# ---------------------------------------------------------------------------
def compute_daily_precip(ds_pr_mm: xr.DataArray) -> xr.DataArray:
    # Décale le temps de -6h pour que chaque "journée" commence à 6h
    pr_shifted = ds_pr_mm.copy()
    pr_shifted["time"] = pr_shifted["time"] - pd.Timedelta(hours=6)
    return pr_shifted.resample(time="1D").sum()

# ---------------------------------------------------------------------------
def compute_monthly_statistics(ds: xr.Dataset, var_name: str, year: int, month: int) -> pd.DataFrame:
    pr = ds[var_name]
    time = ds["time"].values

    # Filtrage temporel mensuel
    start = np.datetime64(f"{year}-{month:02d}-01")
    # Pour gérer les journées de 6h à 6h on va chercher jusqu'au lendemain
    end = np.datetime64(f"{year}-{month % 12 + 1:02d}-01") if month < 12 else np.datetime64(f"{year+1}-01-01")
    pr_month = pr.sel(time=slice(start, end))

    if pr_month.time.size == 0:
        logger.warning(f"Aucune donnée pour {year}-{month:02d}")
        return pd.DataFrame()

    # Conversion en mm
    pr_mm = pr_month

    # Statistiques horaires
    logger.info("Calcul du maximum horaire")
    max_mm_h = pr_mm.max(dim="time")
    max_time_h = pr_mm.argmax(dim="time")
    max_date_mm_h = pr_mm["time"].isel(time=max_time_h).dt.strftime("%Y-%m-%d")

    # Cumul mensuel
    logger.info("Calcul du cumul mensuel")
    pr_sum = pr_mm.sum(dim="time")

    # Journalier (6h → 6h)
    logger.info("Agrégation quotidienne (6h à 6h)")
    pr_daily = compute_daily_precip(pr_mm)

    logger.info("Calcul du maximum journalier")
    max_mm_j = pr_daily.max(dim="time")
    max_time_j = pr_daily.argmax(dim="time")
    max_date_mm_j = pr_daily["time"].isel(time=max_time_j).dt.strftime("%Y-%m-%d")

    # Jours avec > 1mm
    logger.info("Nombre de jours > 1 mm")
    n_days_gt1mm = (pr_daily > 1).sum(dim="time")

    df = pd.DataFrame({
        "lat": ds["lat"].values,
        "lon": ds["lon"].values,
        "sum_mm": pr_sum.values,
        "max_mm_h": max_mm_h.values,
        "max_date_mm_h": max_date_mm_h.values,
        "max_mm_j": max_mm_j.values,
        "max_date_mm_j": max_date_mm_j.values,
        "n_days_gt1mm": n_days_gt1mm.values
    })

    n_total = len(df)
    n_unique_coords = df[["lat", "lon"]].drop_duplicates().shape[0]

    logger.info(f"Nombre de points générés : {n_total} (dont {n_unique_coords} couples lat/lon uniques)")


    return df

# ---------------------------------------------------------------------------
def process_zarr_file(zarr_path: str, config: dict, output_root: str, overwrite: bool = False, log_status: dict = None):
    logger.info(f"Traitement de {zarr_path}")
    year = os.path.basename(zarr_path).split(".")[0]
    ds = xr.open_zarr(zarr_path)

    var_name = list(config["variables"].keys())[0]
    var_conf = config["variables"][var_name]

    # Gestion sentinelle
    fill_value = var_conf.get("fill_value", None)
    if fill_value is not None:
        ds[var_name] = ds[var_name].where(ds[var_name] != fill_value)

    # Application inverse du scale_factor
    scale_factor = var_conf.get("scale_factor", None)
    if scale_factor is not None and scale_factor != 0:
        ds[var_name] = ds[var_name] / scale_factor

    if log_status is not None:
        log_status[year] = {}

    # Boucle mois par mois
    for month in range(1, 13):
        out_dir = os.path.join(output_root, year)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{month:02d}.parquet")

        if os.path.exists(out_path) and not overwrite:
            logger.info(f"[SKIP] {out_path} existe déjà et overwrite=False")
            if log_status is not None:
                log_status[year][f"{month:02d}"] = "Généré"
            continue

        logger.info(f"Traitement du mois {month:02d}/{year}")
        df = compute_monthly_statistics(ds, var_name, int(year), month)
        if df.empty:
            logger.info(f"Aucune donnée pour {year}-{month:02d}")
            if log_status is not None:
                log_status[year][f"{month:02d}"] = "Absent"
            continue

        df.to_parquet(out_path, index=False)
        logger.info(f"Statistiques mensuelles sauvegardées dans {out_path}")
        if log_status is not None:
            log_status[year][f"{month:02d}"] = "Généré"

# ---------------------------------------------------------------------------
def pipeline_statistics_from_zarr(config_path: str):
    config = load_config(config_path)
    zarr_dir = config["zarr"]["path"]["outputdir"]
    stats_conf = config["statistics"]
    stats_dir = stats_conf["path"]["outputdir"]
    log_dir = config["log"]["directory"]
    overwrite = stats_conf.get("overwrite", False)

    os.makedirs(stats_dir, exist_ok=True)

    zarr_files = [
        os.path.join(zarr_dir, f) for f in os.listdir(zarr_dir)
        if f.endswith(".zarr")
    ]

    status_log = {}

    for zarr_path in zarr_files:
        process_zarr_file(zarr_path, config["zarr"], stats_dir, overwrite=overwrite, log_status=status_log)

    # Création du DataFrame final avec années en ligne et mois en colonnes
    log_df = pd.DataFrame.from_dict(status_log, orient="index")
    log_df = log_df.sort_index()
    log_df = log_df[[f"{m:02d}" for m in range(1, 13)]]  # Ordre des mois

    logger.info("Résumé final de la génération mensuelle :\n" + log_df.to_string())

    # Sauvegarde dans un fichier .log
    log_df.to_csv(os.path.join(stats_dir, "pipeline_zarr_to_stats_resume.log"))

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Parser la config
    parser = argparse.ArgumentParser(description="Pipeline statistiques mensuelles à partir de fichiers Zarr.")
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

    logger.info(f"Démarrage du pipeline Zarr → statistiques avec la config : {config_path}")

    with ProgressBar():
        pipeline_statistics_from_zarr(args.config)
