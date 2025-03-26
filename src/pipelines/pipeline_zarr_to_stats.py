import argparse
import os
import pandas as pd
import numpy as np
import xarray as xr
import dask.array as da
from dask.diagnostics import ProgressBar
from dask import compute
from concurrent.futures import ProcessPoolExecutor

from src.utils.config_tools import load_config
from src.utils.logger import get_logger

# ---------------------------------------------------------------------------
def compute_daily_precip(ds_pr_mm: xr.DataArray) -> xr.DataArray:
    pr_shifted = ds_pr_mm.copy()
    pr_shifted["time"] = pr_shifted["time"] - pd.Timedelta(hours=6)
    return pr_shifted.resample(time="1D").sum()

# ---------------------------------------------------------------------------
def compute_monthly_statistics(ds: xr.Dataset, var_name: str, year: int, month: int) -> pd.DataFrame:
    pr = ds[var_name]

    start = np.datetime64(f"{year}-{month:02d}-01")
    end = np.datetime64(f"{year + 1}-01-01") if month == 12 else np.datetime64(f"{year}-{month + 1:02d}-01")
    pr_month = pr.sel(time=slice(start, end))

    if pr_month.time.size == 0:
        print(f"Aucune donnée pour {month:02d}/{year}")
        return pd.DataFrame()

    pr_mm = pr_month

    # Agrégation horaire
    max_mm_h = pr_mm.max(dim="time")
    pr_sum = pr_mm.sum(dim="time")

    # Optimisation argmax (en numpy)
    argmax_h = da.argmax(pr_mm.data, axis=pr_mm.get_axis_num("time"))
    max_time_h = pr_mm["time"].values[argmax_h.compute()]
    max_date_mm_h = np.datetime_as_string(max_time_h, unit="D")

    # Cumul journalier (6h à 6h)
    pr_daily = compute_daily_precip(pr_mm)

    max_mm_j = pr_daily.max(dim="time")
    argmax_j = da.argmax(pr_daily.data, axis=pr_daily.get_axis_num("time"))
    max_time_j = pr_daily["time"].values[argmax_j.compute()]
    max_date_mm_j = np.datetime_as_string(max_time_j, unit="D")

    # Jours > 1 mm
    n_days_gt1mm = (pr_daily > 1).sum(dim="time")

    # Calcul final via Dask compute groupé
    max_mm_h, pr_sum, max_mm_j, n_days_gt1mm = compute(
        max_mm_h, pr_sum, max_mm_j, n_days_gt1mm
    )

    df = pd.DataFrame({
        "lat": ds["lat"].values,
        "lon": ds["lon"].values,
        "sum_mm": pr_sum,
        "max_mm_h": max_mm_h,
        "max_date_mm_h": max_date_mm_h,
        "max_mm_j": max_mm_j,
        "max_date_mm_j": max_date_mm_j,
        "n_days_gt1mm": n_days_gt1mm
    })

    n_total = len(df)
    n_unique_coords = df[["lat", "lon"]].drop_duplicates().shape[0]
    if n_total != n_unique_coords:
        print(f"Pour {month:02d}/{year} : {n_total} points générés pour {n_unique_coords} coordonnées uniques")

    return df

# ---------------------------------------------------------------------------
def process_zarr_file(zarr_path: str, config: dict, output_root: str, overwrite: bool = False, log_status: dict = None):
    year = os.path.basename(zarr_path).split(".")[0]

    ds = xr.open_zarr(zarr_path)
    ds = ds.chunk({"time": 24 * 32}) # 1 mois

    var_name = list(config["variables"].keys())[0]
    var_conf = config["variables"][var_name]

    fill_value = var_conf.get("fill_value", None)
    if fill_value is not None:
        ds[var_name] = ds[var_name].where(ds[var_name] != fill_value)

    scale_factor = var_conf.get("scale_factor", None)
    if scale_factor is not None and scale_factor != 0:
        ds[var_name] = ds[var_name] / scale_factor

    if log_status is not None:
        log_status[year] = {}

    for month in range(1, 13):
        out_dir = os.path.join(output_root, year)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{month:02d}.parquet")

        if os.path.exists(out_path) and not overwrite:
            print(f"[SKIP] {out_path} existe déjà")
            if log_status is not None:
                log_status[year][f"{month:02d}"] = "Généré"
            continue

        df = compute_monthly_statistics(ds, var_name, int(year), month)
        if df.empty:
            print(f"Aucune donnée pour {year}-{month:02d}")
            if log_status is not None:
                log_status[year][f"{month:02d}"] = "Absent"
            continue

        df.to_parquet(out_path, index=False, engine="pyarrow", compression="zstd")
        print(f"Statistiques {month:02d}/{year} sauvegardées dans {out_path}")
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

    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_zarr_file, z, config["zarr"], stats_dir, overwrite, status_log)
                for z in zarr_files]
        for f in futures:
            f.result()  # raise exception if any


    log_df = pd.DataFrame.from_dict(status_log, orient="index")
    log_df = log_df.sort_index()
    log_df = log_df[[f"{m:02d}" for m in range(1, 13)]]

    logger.info("Résumé final :\n" + log_df.to_string())
    log_df.to_csv(os.path.join(stats_dir, "pipeline_zarr_to_stats_resume.log"))

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline statistiques mensuelles à partir de fichiers Zarr.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/modelised_settings.yaml",
        help="Chemin vers le fichier de configuration YAML"
    )
    args = parser.parse_args()
    config_path = args.config
    config = load_config(config_path)

    log_dir = config.get("log", {}).get("directory", "logs")
    logger = get_logger(__name__, log_to_file=True, log_dir=log_dir)

    logger.info(f"Démarrage du pipeline Zarr → statistiques avec la config : {config_path}")

    with ProgressBar():
        pipeline_statistics_from_zarr(config_path)
