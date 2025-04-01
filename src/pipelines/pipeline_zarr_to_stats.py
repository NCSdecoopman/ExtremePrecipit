import argparse
import os
import pandas as pd
import numpy as np
import xarray as xr
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
from dask import compute
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.utils.config_tools import load_config
from src.utils.logger import get_logger

dask.config.set(scheduler="threads")  # ou "processes", selon la charge mémoire

SEASON_MONTHS = {
    "djf": [12, 1, 2],
    "mam": [3, 4, 5],
    "jja": [6, 7, 8],
    "son": [9, 10, 11],
    "hydro": [9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8],
}

def compute_daily_precip(ds_pr_mm: xr.DataArray) -> xr.DataArray:
    pr_shifted = ds_pr_mm.copy()
    pr_shifted["time"] = pr_shifted["time"] - pd.Timedelta(hours=6)
    return pr_shifted.resample(time="1D").sum()

def get_season_bounds(year: int, season: str):
    months = SEASON_MONTHS[season]
    if season == "djf":
        start = np.datetime64(f"{year - 1}-12-01")
        end = np.datetime64(f"{year}-03-01")
    elif season == "hydro":
        start = np.datetime64(f"{year - 1}-09-01")
        end = np.datetime64(f"{year}-09-01")
    else:
        start = np.datetime64(f"{year}-{months[0]:02d}-01")
        end_month = months[-1] + 1
        if end_month > 12:
            end = np.datetime64(f"{year + 1}-01-01")
        else:
            end = np.datetime64(f"{year}-{end_month:02d}-01")
    return start, end

def compute_statistics_for_period(pr: xr.DataArray, echelle: str) -> pd.DataFrame:
    n_time = pr.time.size
    n_hours = n_time if echelle == "horaire" else n_time * 24

    n_nan = da.isnan(pr).sum(dim="time")
    nan_ratio = (n_nan / n_time).compute()

    valid_mask = nan_ratio < 1.0
    valid_idx = np.where(valid_mask)[0]
    shape = pr.lat.size

    mean_mm_h = np.full(shape, np.nan)
    max_mm_h = np.full(shape, np.nan)
    max_date_mm_h = np.full(shape, np.nan, dtype=object)
    max_mm_j = np.full(shape, np.nan)
    max_date_mm_j = np.full(shape, np.nan, dtype=object)
    n_days_gt1mm = np.full(shape, np.nan)

    if len(valid_idx) > 0:
        pr_valid = pr.isel(points=valid_idx)

        if echelle == "horaire":
            pr_sum = pr_valid.sum(dim="time").compute().values
            mean_mm_h[valid_idx] = pr_sum / n_time

            max_mm_h_valid = pr_valid.max(dim="time")
            argmax_h = da.argmax(pr_valid.data, axis=pr_valid.get_axis_num("time"))
            max_time_h = pr_valid["time"].values[argmax_h.compute()]
            max_mm_h[valid_idx] = compute(max_mm_h_valid)[0].values
            max_date_mm_h[valid_idx] = np.datetime_as_string(max_time_h, unit="D")

            # Conversion en journalier
            pr_daily = compute_daily_precip(pr_valid)

        else:  # echelle == "quotidien"
            mean_mm_h_valid = (pr_valid.mean(dim="time") / 24).compute().values
            mean_mm_h[valid_idx] = mean_mm_h_valid

            # max_mm_h et max_date_mm_h doivent rester NaN
            # rien d'autre à faire ici
            pr_daily = pr_valid  # déjà journalier

        # Statistiques communes journalières
        max_mm_j_valid = pr_daily.max(dim="time")
        argmax_j = da.argmax(pr_daily.data, axis=pr_daily.get_axis_num("time"))
        max_time_j = pr_daily["time"].values[argmax_j.compute()]
        max_mm_j[valid_idx] = compute(max_mm_j_valid)[0].values
        max_date_mm_j[valid_idx] = np.datetime_as_string(max_time_j, unit="D")

        n_gt1mm = (pr_daily > 1).sum(dim="time").compute().values
        n_days_gt1mm[valid_idx] = n_gt1mm

    df = pd.DataFrame({
        "lat": pr["lat"].values,
        "lon": pr["lon"].values,
        "mean_mm_h": mean_mm_h,
        "max_mm_h": max_mm_h,
        "max_date_mm_h": max_date_mm_h,
        "max_mm_j": max_mm_j,
        "max_date_mm_j": max_date_mm_j,
        "n_days_gt1mm": n_days_gt1mm,
        "nan_ratio": nan_ratio.values
    })

    return df



def process_zarr_file_seasonal(
    zarr_path: str,
    config_zarr: dict,
    echelle: str,
    output_root: str,
    overwrite: bool,
    log_status: dict,
    logger
):
    year = int(os.path.basename(zarr_path).split(".")[0])
    var_name = list(config_zarr["variables"].keys())[0]
    var_conf = config_zarr["variables"][var_name]

    all_ds = []
    zarr_dir = os.path.dirname(zarr_path)
    for offset in [-1, 0]:
        y = year + offset
        p = os.path.join(zarr_dir, f"{y}.zarr")
        if os.path.exists(p):
            all_ds.append(xr.open_zarr(p).chunk({"time": 24 * 92}))

    if not all_ds:
        logger.warning(f"Aucun fichier Zarr trouvé pour {year}")
        return

    ds = xr.concat(all_ds, dim="time", combine_attrs="override")
    ds["time"] = pd.to_datetime(ds["time"].values)  # <- conversion explicite
    ds = ds.sortby("time")

    if not np.issubdtype(ds["time"].dtype, np.datetime64):
        ds["time"] = pd.to_datetime(ds["time"].values)

    fill_value = var_conf.get("fill_value", None)
    if fill_value is not None:
        ds[var_name] = ds[var_name].where(ds[var_name] != fill_value)

    scale_factor = var_conf.get("scale_factor", None)
    if scale_factor is not None and scale_factor != 0:
        ds[var_name] = ds[var_name] / scale_factor

    if log_status is not None:
        log_status[year] = {}

    for season in ["djf", "mam", "jja", "son"]:
        out_dir = os.path.join(output_root, str(year))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{season}.parquet")

        if os.path.exists(out_path) and not overwrite:
            logger.info(f"[SKIP] {out_path} existe déjà")
            log_status[year][season] = "Généré"
            continue

        start, end = get_season_bounds(year, season)
        ds_start = ds["time"].values[0]
        ds_end = ds["time"].values[-1]

        if start < ds_start or end > ds_end:
            logger.info(f"[SKIP] {season.upper()} {year} hors des bornes de {zarr_path}")
            log_status[year][season] = "Hors bornes"
            continue

        pr_season = ds[var_name].sel(time=slice(start, end))
        if pr_season.time.size == 0:
            logger.warning(f"Aucune donnée pour {season.upper()} {year}")
            log_status[year][season] = "Absent"
            continue

        df = compute_statistics_for_period(pr_season, echelle)
        if df.empty:
            logger.warning(f"Aucune statistique pour {season.upper()} {year}")
            log_status[year][season] = "Vide"
            continue

        df.to_parquet(out_path, index=False, engine="pyarrow", compression="zstd")
        logger.info(f"Statistiques {season.upper()} {year} sauvegardées dans {out_path}")
        log_status[year][season] = "Généré"


def compute_hydro_from_seasons(year: int, stats_dir: str, log_status: dict):
    try:
        dfs = []
        for season, y in {"son": year - 1, "djf": year, "mam": year, "jja": year}.items():
            path = os.path.join(stats_dir, str(y), f"{season}.parquet")
            if not os.path.exists(path):
                log_status.setdefault(year, {})["hydro"] = "Manque données"
                return
            df = pd.read_parquet(path)
            df["season"] = season
            dfs.append(df)

        full_df = pd.concat(dfs)
        df_grouped = full_df.groupby(["lat", "lon"], sort=False)

        df_hydro = df_grouped.agg({
            "mean_mm_h": "mean",
            "max_mm_h": "max",
            "max_mm_j": "max",
            "n_days_gt1mm": "sum"
        }).reset_index()

        # Initialisation avec NaN
        df_hydro["max_date_mm_h"] = np.nan
        df_hydro["max_date_mm_j"] = np.nan

        # Indices des dates maximales
        idx_h = df_grouped["max_mm_h"].transform("idxmax")
        idx_j = df_grouped["max_mm_j"].transform("idxmax")

        # Filtrer les idx valides uniquement (non-NaN)
        if idx_h.notna().any():
            idx_h_valid = idx_h.dropna().astype(int)
            max_date_mm_h = full_df.loc[idx_h_valid, ["lat", "lon", "max_date_mm_h"]].drop_duplicates(["lat", "lon"])
            df_hydro = df_hydro.merge(max_date_mm_h, on=["lat", "lon"], how="left", suffixes=("", "_h"))
            df_hydro["max_date_mm_h"] = df_hydro["max_date_mm_h_h"].combine_first(df_hydro["max_date_mm_h"])
            df_hydro.drop(columns=["max_date_mm_h_h"], inplace=True)

        if idx_j.notna().any():
            idx_j_valid = idx_j.dropna().astype(int)
            max_date_mm_j = full_df.loc[idx_j_valid, ["lat", "lon", "max_date_mm_j"]].drop_duplicates(["lat", "lon"])
            df_hydro = df_hydro.merge(max_date_mm_j, on=["lat", "lon"], how="left", suffixes=("", "_j"))
            df_hydro["max_date_mm_j"] = df_hydro["max_date_mm_j_j"].combine_first(df_hydro["max_date_mm_j"])
            df_hydro.drop(columns=["max_date_mm_j_j"], inplace=True)

        out_dir = os.path.join(stats_dir, str(year))
        os.makedirs(out_dir, exist_ok=True)
        df_hydro.to_parquet(os.path.join(out_dir, "hydro.parquet"), index=False, engine="pyarrow", compression="zstd")

        logger.info(f"[HYDRO] Statistiques HYDRO {year} sauvegardées dans {out_dir}")
        log_status.setdefault(year, {})["hydro"] = "Généré"

    except Exception as e:
        logger.error(f"[FAIL] Erreur durant l’agrégation HYDRO {year}: {e}")
        log_status.setdefault(year, {})["hydro"] = "Erreur"



def process_one_file(args):
    zarr_file, echelle, config_zarr, stats_dir, overwrite = args
    logger = get_logger(f"worker_{os.path.basename(zarr_file).split('.')[0]}", log_to_file=False)

    try:
        log_status = {}
        process_zarr_file_seasonal(
            zarr_file,
            config_zarr,
            echelle,
            stats_dir,
            overwrite,
            log_status,
            logger
        )
        logger.info(f"[OK] Traitement terminé pour {zarr_file}")
        return (zarr_file, log_status, None)
    except Exception as e:
        logger.error(f"[FAIL] Erreur pendant le traitement de {zarr_file}: {e}")
        return (zarr_file, None, str(e))


def pipeline_statistics_from_zarr_seasonal(config_path: str):
    config = load_config(config_path)
    stats_conf = config["statistics"]
    overwrite = stats_conf.get("overwrite", False)

    echelles = config.get("echelles")

    for echelle in echelles:
        logger.info(f"--- Traitement pour l’échelle : {echelle.upper()} ---")

        zarr_dir = os.path.join(config["zarr"]["path"]["outputdir"], echelle)
        stats_dir = os.path.join(stats_conf["path"]["outputdir"], echelle)
        log_dir = os.path.join(config["log"]["directory"], echelle)

        os.makedirs(stats_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        zarr_files = [
            os.path.join(zarr_dir, f)
            for f in os.listdir(zarr_dir)
            if f.endswith(".zarr")
        ]

        if not zarr_files:
            logger.warning(f"Aucun fichier Zarr trouvé pour l’échelle '{echelle}' dans {zarr_dir}")
            continue

        args_list = [
            (zf, echelle, config["zarr"], stats_dir, overwrite)
            for zf in zarr_files
        ]

        status_log = {}

        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(process_one_file, args) for args in args_list]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Traitement Zarr ({echelle})"):
                zarr_file, log_status, error = future.result()
                if error:
                    print(f"[ERROR] Error processing {zarr_file}: {error}")
                else:
                    year = int(os.path.basename(zarr_file).split(".")[0])
                    status_log[year] = log_status.get(year, {})

        # Post-traitement HYDRO pour toutes les échelles
        for year in sorted(status_log):
            compute_hydro_from_seasons(year, stats_dir, status_log)

        log_df = pd.DataFrame.from_dict(status_log, orient="index").sort_index()
        cols = [s for s in SEASON_MONTHS if s in log_df.columns]
        if "hydro" in log_df.columns:
            cols.append("hydro")
        if cols:
            log_df = log_df[cols]
        else:
            logger.warning("Aucune colonne saisonnière ou hydro disponible dans le log.")

        logger.info(f"[{echelle.upper()}] Résumé final :\n" + log_df.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline statistiques saisonnières à partir de fichiers Zarr.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/observed_settings.yaml",
        help="Chemin vers le fichier de configuration YAML"
    )
    args = parser.parse_args()
    config_path = args.config
    config = load_config(config_path)

    log_dir = config.get("log", {}).get("directory", "logs")
    logger = get_logger(__name__, log_to_file=True, log_dir=log_dir)

    logger.info(f"Démarrage du pipeline Zarr → statistiques saisonnières avec la config : {config_path}")

    with ProgressBar():
        pipeline_statistics_from_zarr_seasonal(config_path)
