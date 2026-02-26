import argparse
from pathlib import Path
import numpy as np
import polars as pl

from src.utils.logger import get_logger
from src.utils.config_tools import load_config
from src.utils.data_utils import years_to_load, load_data, cleaning_data_observed


# -------------------------------
# Compute return level zTpa
# -------------------------------
def compute_zTpa(
    T: int,
    year_ref: int,
    min_year: int,
    max_year: int,
    df_params: pl.DataFrame,
    df_series: pl.DataFrame,
) -> pl.DataFrame:
    """
    Computes z_T(year_ref) for each station.
    Returns NUM_POSTE, zTpa.
    """

    rows = []
    for row in df_params.to_dicts():
        num = row["NUM_POSTE"]

        # Temporal data for this station (used for normalization only)
        ds = df_series.filter(pl.col("NUM_POSTE") == num)
        years_obs = ds["year"].to_numpy()

        # Time normalization identical to previous pipeline
        t_raw = (years_obs - min_year) / (max_year - min_year)
        tmin, tmax = t_raw.min(), t_raw.max()
        res0 = t_raw / (tmax - tmin)
        dx = res0.min() + 0.5

        # t_tilde value for year_ref
        t_ref_raw = (year_ref - min_year) / (max_year - min_year)
        t_ref = (t_ref_raw / (tmax - tmin)) - dx

        # Parameters retrieval
        mu0 = row["mu0"]
        sigma0 = row["sigma0"]
        xi = row["xi"]

        # Stationary model: mu1 = sigma1 = 0
        CT = (-np.log(1 - 1 / T)) ** (-xi) - 1

        zTpa = mu0 + (sigma0 / xi) * CT

        rows.append({"NUM_POSTE": num, "zTpa": zTpa})

    return pl.DataFrame(rows)


# -------------------------------
# MAIN
# -------------------------------
def main(config, args, T: int = 10):
    global logger
    logger = get_logger(__name__)

    echelles = config["echelles"]
    season = config["season"]
    gev_dir = config["gev"]["path"]["outputdir"]
    reduce_activate = config["reduce_activate"]

    if reduce_activate:
        suffix_save = "_reduce"
    else:
        suffix_save = ""

    for echelle in echelles:
        logger.info(f"--- Processing scale: {echelle.upper()} season: {season} ---")

        # GEV parameters reading
        path_dir = Path(gev_dir) / f"{echelle}{suffix_save}" / season
        best_model_path = path_dir / "gev_param_best_model.parquet"
        best_model = pl.read_parquet(best_model_path)

        # Force stationary
        best_model = best_model.with_columns([
            pl.lit("s_gev").alias("model"),
            pl.lit(0.0).alias("mu1"),
            pl.lit(0.0).alias("sigma1"),
        ])

        # Maxima loading
        input_dir = Path(config["statistics"]["path"]["outputdir"]) / echelle
        if "modelised" in str(input_dir):
            input_dir = Path(config["statistics"]["path"]["outputdir"]) / "horaire"
        # Temporal parameters
        if reduce_activate and echelle == "quotidien":
            mesure, min_year, max_year, len_serie = years_to_load("reduce", season, input_dir)
            suffix_save = "_reduce"
        elif reduce_activate and echelle == "horaire":
            mesure, min_year, max_year, len_serie = years_to_load("horaire_reduce", season, input_dir)
            suffix_save = "_reduce"
        else:
            mesure, min_year, max_year, len_serie = years_to_load(echelle, season, input_dir)
            suffix_save = ""
        cols = ["NUM_POSTE", mesure, "nan_ratio"]

        df = load_data(input_dir, season, echelle, cols, min_year, max_year)
        df = cleaning_data_observed(df, echelle, len_serie)
        df = df.drop_nulls(subset=[mesure])

        # Filtering
        df = df.filter(pl.col("NUM_POSTE").is_in(best_model["NUM_POSTE"].to_list()))

        # Compute zTpa for year_ref = 1992
        final_table = compute_zTpa(
            T=T,
            year_ref=1992,
            min_year=min_year,
            max_year=max_year,
            df_params=best_model,
            df_series=df,
        )

        # Save
        out_path = f"{path_dir}/niveau_retour.parquet"
        final_table.write_parquet(out_path)
        logger.info(f"Final table saved: {out_path}")


# -------------------------------
# Entrée CLI
# CLI Entry Point
# -------------------------------
def str2bool(v):
    return v == "True"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simplified pipeline → zTpa only")
    parser.add_argument("--config", type=str, default="config/observed_settings.yaml")
    parser.add_argument("--echelle", choices=["horaire", "quotidien"], nargs="+", default=["horaire"])
    parser.add_argument("--season", type=str, default="son")
    parser.add_argument("--reduce_activate", type=str2bool, default=False)

    args = parser.parse_args()

    config = load_config(args.config)
    config["config"] = args.config
    config["echelles"] = args.echelle
    config["season"] = args.season
    config["reduce_activate"] = args.reduce_activate

    main(config, args)
