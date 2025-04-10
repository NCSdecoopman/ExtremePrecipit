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

SEASON_MONTHS = {
    "djf": [12, 1, 2],
    "mam": [3, 4, 5],
    "jja": [6, 7, 8],
    "son": [9, 10, 11],
    "hydro": [9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8],
}

def compute_daily_precip(ds_pr_mm: xr.DataArray) -> xr.DataArray:
    return ds_pr_mm.resample(time="1d", offset="6h").sum()

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

def safe_combine_first(df, col_main, col_secondary):
    if df[col_secondary].dropna().empty:
        df[col_secondary] = pd.Series([np.nan] * len(df), index=df.index)
    df[col_main] = df[col_secondary].combine_first(df[col_main])
    df.drop(columns=[col_secondary], inplace=True)


def compute_statistics_for_period(
    pr: xr.DataArray,
    echelle: str,
    lat_ref: np.ndarray,
    lon_ref: np.ndarray
) -> pd.DataFrame:
    n_points = lat_ref.shape[0]
    n_time = pr.time.size

    # Initialisation des tableaux complets
    mean_mm_h = np.full(n_points, np.nan)
    max_mm_h = np.full(n_points, np.nan)
    max_date_mm_h = np.full(n_points, np.nan, dtype=object)
    max_mm_j = np.full(n_points, np.nan)
    max_date_mm_j = np.full(n_points, np.nan, dtype=object)
    n_days_gt1mm = np.full(n_points, np.nan)
    nan_ratio_arr = np.full(n_points, np.nan)

    if "points" in pr.dims:
        pr_points = pr.coords["points"].values
        pr = pr.chunk({"points": -1})

        # Ratio de NaNs
        n_nan = da.isnan(pr).sum(dim="time")
        nan_ratio = (n_nan / n_time).compute()
        nan_ratio_arr[pr_points] = nan_ratio

        valid_mask = nan_ratio < 1.0  # Points avec au moins une valeur valide
        if valid_mask.sum().item() == 0:
            return pd.DataFrame({
                "lat": lat_ref,
                "lon": lon_ref,
                "mean_mm_h": mean_mm_h,
                "max_mm_h": max_mm_h,
                "max_date_mm_h": max_date_mm_h,
                "max_mm_j": max_mm_j,
                "max_date_mm_j": max_date_mm_j,
                "n_days_gt1mm": n_days_gt1mm,
                "nan_ratio": nan_ratio_arr,
            })

        valid_idx = pr_points[valid_mask.values]
        pr_valid = pr.isel(points=valid_mask)

        if echelle == "horaire":
            mean_valid = pr_valid.mean(dim="time", skipna=True).compute().values
            mean_mm_h[valid_idx] = mean_valid

            max_mm_h_valid = pr_valid.max(dim="time", skipna=True).compute().values
            max_mm_h[valid_idx] = max_mm_h_valid

            argmax_h = da.argmax(pr_valid.data, axis=pr_valid.get_axis_num("time"))
            argmax_h_val = argmax_h.compute()
            pr_time = pr_valid["time"].values

            valid_h_mask = ~np.isnan(max_mm_h_valid)
            max_date_mm_h[valid_idx[valid_h_mask]] = np.datetime_as_string(
                pr_time[argmax_h_val[valid_h_mask]], unit="D"
            )

            pr_daily = compute_daily_precip(pr_valid)
        else:
            mean_valid = (pr_valid.mean(dim="time", skipna=True) / 24).compute().values
            mean_mm_h[valid_idx] = mean_valid
            pr_daily = pr_valid  # déjà au pas quotidien

        max_mm_j_valid = pr_daily.max(dim="time", skipna=True).compute().values
        max_mm_j[valid_idx] = max_mm_j_valid

        argmax_j = da.argmax(pr_daily.data, axis=pr_daily.get_axis_num("time"))
        argmax_j_val = argmax_j.compute()
        pr_time_j = pr_daily["time"].values

        valid_j_mask = ~np.isnan(max_mm_j_valid)
        max_date_mm_j[valid_idx[valid_j_mask]] = np.datetime_as_string(
            pr_time_j[argmax_j_val[valid_j_mask]], unit="D"
        )

        n_gt1mm = (pr_daily > 1).sum(dim="time", skipna=True).compute().values
        n_days_gt1mm[valid_idx] = n_gt1mm

    df = pd.DataFrame({
        "lat": lat_ref,
        "lon": lon_ref,
        "mean_mm_h": mean_mm_h,
        "max_mm_h": max_mm_h,
        "max_date_mm_h": max_date_mm_h,
        "max_mm_j": max_mm_j,
        "max_date_mm_j": max_date_mm_j,
        "n_days_gt1mm": n_days_gt1mm,
        "nan_ratio": nan_ratio_arr,
    })
    return df


def process_zarr_file_seasonal(
    zarr_path: str,
    config_zarr: dict,
    echelle: str,
    output_root: str,
    overwrite: bool,
    log_status: dict,
    logger,
    lat_ref: np.ndarray, 
    lon_ref: np.ndarray
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

    # Convertir explicitement en float avant division
    ds[var_name] = ds[var_name].astype("float32")

    fill_value = var_conf.get("fill_value", None)
    if fill_value is not None:
        ds[var_name] = ds[var_name].where(ds[var_name] != fill_value, np.nan)

    scale_factor = var_conf.get("scale_factor", None)
    if scale_factor is not None and scale_factor != 0:
        ds[var_name] = ds[var_name] / scale_factor

    if log_status is not None:
        log_status[year] = {}

    for season in ["djf", "mam", "jja", "son"]:
        # logger.info(f"[{season.upper()}] {year} : début du calcul")
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

        df = compute_statistics_for_period(pr_season, echelle, lat_ref, lon_ref)

        if df.shape[0] != lat_ref.shape[0]:
            logger.warning("Le tableau généré n'a pas la même taille que la grille")

        if df.empty:
            logger.warning(f"Aucune statistique pour {season.upper()} {year}")
            log_status[year][season] = "Vide"
            continue

        df.to_parquet(out_path, index=False, engine="pyarrow", compression="zstd")
        #logger.info(f"Statistiques {season.upper()} {year} sauvegardées dans {out_path}")
        log_status[year][season] = "Généré"



def compute_hydro_from_seasons(year: int, stats_dir: str, log_status: dict):
    try:
        dfs = []
        season_map = {"son": year - 1, "djf": year, "mam": year, "jja": year}
        
        # Chargement des fichiers saisonniers
        for season, y in season_map.items():
            path = os.path.join(stats_dir, str(y), f"{season}.parquet")
            if not os.path.exists(path):
                logger.info(f"[HYDRO] Fichier manquant : {path}")
                log_status.setdefault(year, {})["hydro"] = "Manque données"
                return
            df = pd.read_parquet(path)
            df["season"] = season
            dfs.append(df)

        full_df = pd.concat(dfs, ignore_index=True)
        df_grouped = full_df.groupby(["lat", "lon"], sort=False)

        # Agrégations numériques robustes
        df_hydro = df_grouped.agg({
            "mean_mm_h": lambda x: x.mean(skipna=True),
            "max_mm_h": lambda x: x.max(skipna=True),
            "max_mm_j": lambda x: x.max(skipna=True),
            "n_days_gt1mm": lambda x: x.sum(skipna=True),
            "nan_ratio": lambda x: x.mean(skipna=True),
        }).reset_index()

        # Initialisation colonnes de dates de max
        df_hydro["max_date_mm_h"] = np.nan
        df_hydro["max_date_mm_j"] = np.nan

        # Dates de maxima horaire
        if full_df["max_mm_h"].notna().any():
            idx_h = df_grouped["max_mm_h"].transform(lambda x: x.idxmax(skipna=True) if x.notna().any() else pd.NA)
            idx_h_valid = idx_h.dropna().astype(int)
            max_date_mm_h = full_df.loc[idx_h_valid, ["lat", "lon", "max_date_mm_h"]].drop_duplicates(["lat", "lon"])
            df_hydro = df_hydro.merge(max_date_mm_h, on=["lat", "lon"], how="left", suffixes=("", "_h"))
            safe_combine_first(df_hydro, "max_date_mm_h", "max_date_mm_h_h")

        # Dates de maxima journalier
        if full_df["max_mm_j"].notna().any():
            idx_j = df_grouped["max_mm_j"].transform(lambda x: x.idxmax(skipna=True) if x.notna().any() else pd.NA)
            idx_j_valid = idx_j.dropna().astype(int)
            max_date_mm_j = full_df.loc[idx_j_valid, ["lat", "lon", "max_date_mm_j"]].drop_duplicates(["lat", "lon"])
            df_hydro = df_hydro.merge(max_date_mm_j, on=["lat", "lon"], how="left", suffixes=("", "_j"))
            safe_combine_first(df_hydro, "max_date_mm_j", "max_date_mm_j_j")

        # Sauvegarde
        out_dir = os.path.join(stats_dir, str(year))
        os.makedirs(out_dir, exist_ok=True)
        df_hydro.to_parquet(os.path.join(out_dir, "hydro.parquet"), index=False, engine="pyarrow", compression="zstd")

        logger.info(f"[HYDRO] Statistiques HYDRO {year} sauvegardées dans {out_dir}")
        log_status.setdefault(year, {})["hydro"] = "Généré"

    except Exception as e:
        logger.error(f"[FAIL] Erreur durant l’agrégation HYDRO {year}: {e}")
        log_status.setdefault(year, {})["hydro"] = "Erreur"





def process_one_file(args):
    dask.config.set(scheduler="single-threaded")
    
    # ProgressBar().register()
    zarr_file, echelle, config_zarr, stats_dir, overwrite, lat_ref, lon_ref = args
    logger = get_logger(f"worker_{os.path.basename(zarr_file).split('.')[0]}", log_to_file=False)

    # Vérification de la grille
    ds_check = xr.open_zarr(zarr_file)
    if not np.allclose(ds_check["lat"].values, lat_ref) or not np.allclose(ds_check["lon"].values, lon_ref):
        logger.error(f"[GRILLE] Incohérence de la grille détectée dans {zarr_file}")
        return (zarr_file, None, "Grille différente")

    try:
        log_status = {}
        process_zarr_file_seasonal(
            zarr_file,
            config_zarr,
            echelle,
            stats_dir,
            overwrite,
            log_status,
            logger,
            lat_ref,
            lon_ref
        )
        logger.info(f"[OK] Traitement terminé pour {zarr_file}")
        return (zarr_file, log_status, None)
    except Exception as e:
        logger.error(f"[FAIL] Erreur pendant le traitement de {zarr_file}: {e}")
        return (zarr_file, None, str(e))


def pipeline_statistics_from_zarr_seasonal(config, max_workers: int = 48):
    global logger
    logger = get_logger(__name__)

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

        n_points_list = []
        valid_files = 0

        for zarr_file in sorted(zarr_files):
            try:
                ds = xr.open_zarr(zarr_file)
                if "points" in ds.dims:
                    n_points = ds.sizes["points"]
                    n_points_list.append(n_points)
                    valid_files += 1
                else:
                    logger.warning(f"{zarr_file}: Dimensions inconnues, fichier ignoré")
            except Exception as e:
                logger.error(f"{zarr_file}: Erreur de lecture ({e})")

        if valid_files > 0:
            logger.info(f"Nombre de fichiers valides : {valid_files}")

            # Vérifie si tous les fichiers ont le même nombre de points
            if all(p == n_points_list[0] for p in n_points_list):
                logger.info(f"Tous les fichiers ont le même nombre de points : {n_points_list[0]}")
            else:
                logger.warning("Les fichiers n'ont pas tous le même nombre de points.")
                logger.warning(f"Valeurs uniques : {sorted(set(n_points_list))}")
        else:
            logger.warning("Aucun fichier valide n'a été traité.")


        reference_file = zarr_files[0]
        ds_ref = xr.open_zarr(reference_file)
        lat_ref = ds_ref["lat"].values
        lon_ref = ds_ref["lon"].values

        logger.info(f"[{echelle.upper()}] Grille de référence : {lat_ref.shape[0]} latitudes et {lon_ref.shape[0]} longitudes")

        if not zarr_files:
            logger.warning(f"Aucun fichier Zarr trouvé pour l’échelle '{echelle}' dans {zarr_dir}")
            continue

        args_list = [
            (zf, echelle, config["zarr"], stats_dir, overwrite, lat_ref, lon_ref)
            for zf in zarr_files
        ]

        status_log = {}

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_one_file, args) for args in args_list]
            for future in as_completed(futures):
                zarr_file, log_status, error = future.result()
                if error:
                    print(f"[ERROR] Error processing {zarr_file}: {error}")
                else:
                    year = int(os.path.basename(zarr_file).split(".")[0])
                    status_log[year] = log_status.get(year, {})

        # Post-traitement HYDRO pour toutes les échelles
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(compute_hydro_from_seasons, year, stats_dir, status_log): year
                for year in sorted(status_log)
            }

            for future in as_completed(futures):
                year = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"[ERROR] Error processing lors du traitement de l'année {year}: {e}")

        log_df = pd.DataFrame.from_dict(status_log, orient="index").sort_index()
        logger.info(f"[{echelle.upper()}] Résumé final :\n" + log_df.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline statistiques saisonnières à partir de fichiers Zarr.")
    parser.add_argument("--config", type=str, default="config/observed_settings.yaml")
    parser.add_argument("--echelle", type=str, choices=["horaire", "quotidien"])
    args = parser.parse_args()

    config = load_config(args.config)
    config["echelles"] = [args.echelle]  # Ne traiter qu'une seule échelle

    pipeline_statistics_from_zarr_seasonal(config, max_workers=96)
