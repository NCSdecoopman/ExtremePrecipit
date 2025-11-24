import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from typing import Tuple, Union

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from src.utils.config_tools import load_config
from src.utils.logger import get_logger
from src.utils.data_utils import load_season

import contextlib
import os
import sys

def load_data(intputdir: str, season: str, echelle: str, cols: tuple, min_year: int, max_year: int) -> pl.DataFrame:
    dataframes = []
    errors = []

    for year in range(min_year, max_year + 1):
        try:
            df = load_season(year, cols, season, intputdir)
            df = df.with_columns([
                pl.col(cols[1]).cast(pl.Utf8),
                pl.lit(year).alias("year")
            ])
            df = df.filter(pl.col("nan_ratio") <= 0.10)
            
            dataframes.append(df)

        except Exception as e:
            errors.append(f"{year} ({season}) : {e}")

    if errors:
        for err in errors:
            raise ValueError(f"Erreur : {err}")

    if not dataframes:
        raise ValueError("Aucune donnée chargée.")

    return pl.concat(dataframes, how="vertical")


def load_max_date(config):
    global logger
    logger = get_logger(__name__)
    
    echelles = config.get("echelles", "quotidien")
    season = config.get("season", "hydro")
    model_path = config.get("config", "config/observed_settings.yaml")

    for echelle in echelles:
        logger.info(f"--- Traitement de {model_path} \nEchelle : {echelle.upper()} ---")

        # Choix du répertoire de lecture
        model_path_name = Path(model_path).name

        if model_path_name == "observed_settings.yaml":
            input_dir = Path(config["statistics"]["path"]["outputdir"]) / echelle
            logger.info(f"Source STATION détectée → lecture dans : {input_dir}")

        elif model_path_name == "modelised_settings.yaml":
            input_dir = Path(config["statistics"]["path"]["outputdir"]) / "horaire"
            logger.info(f"Source AROME détectée → lecture dans : {input_dir}")

        else:
            logger.error(f"Nom de fichier de configuration non reconnu : {model_path_name}")
            sys.exit(1)

        # Fixation de l'échelle pour le choix des colonnes à lire
        mesure = "max_date_mm_h" if echelle == "horaire" else "max_date_mm_j"
        cols = ["NUM_POSTE", mesure, "nan_ratio"]

        # Liste des années disponibles
        years = [
            int(name) for name in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, name)) and name.isdigit() and len(name) == 4
        ]

        if years:
            min_year = min(years)
            max_year = max(years)
        else:
            logger.error("Aucune année valide trouvée.")

        if season in ["hydro", "djf"]:
            min_year+=1 # On commence en 1960

        logger.info(f"Chargement des données de {min_year} à {max_year} : {input_dir}")
        df = load_data(input_dir, season, echelle, cols, min_year, max_year)

        return df


def load_all_date(args):
    dataframe_max_date = {}
    for config_file in args.config:
        cfg = load_config(config_file)          
        cfg["config"]   = config_file          
        cfg["echelles"] = args.echelle
        cfg["season"]   = args.season

        df = load_max_date(cfg)

        if "observed" in config_file:
            dataframe_max_date["observed"] = df

            e = cfg["echelles"][0]
            mapping_path = Path(cfg["obs_vs_mod"]["metadata_path"]["outputdir"]) / f"obs_vs_mod_{e}.csv"

        else:
            dataframe_max_date["modelised"] = df

    if mapping_path is None:
        raise RuntimeError("No observed config found; cannot locate mapping file.")
    logger.info("--- Chargement du mapping ---")
    mapping = pd.read_csv(mapping_path)

    return dataframe_max_date, mapping


def comparaison_max_date(mapping, dataframe_max_date, echelle):
    # Choix de la colonne de mesure selon l'échelle
    mesure = "max_date_mm_h" if echelle == "horaire" else "max_date_mm_j"

    # 1. Load mapping into Polars
    mapping_pl = pl.DataFrame(
        mapping[["NUM_POSTE_obs", "NUM_POSTE_mod"]]
    ).with_columns([
        pl.col("NUM_POSTE_obs").cast(pl.Int64),
        pl.col("NUM_POSTE_mod").cast(pl.Int64)
    ])

    # 2. Rename & cast votre table observée
    obs_pl = (
        dataframe_max_date["observed"]
        .rename({
            "NUM_POSTE":      "NUM_POSTE_obs",
            mesure:          f"{mesure}_obs"
        })
        .with_columns([
            pl.col("NUM_POSTE_obs").cast(pl.Int64)
        ])
    )

    # 3. Rename & cast votre table modélisée
    mod_pl = (
        dataframe_max_date["modelised"]
        .rename({
            "NUM_POSTE":      "NUM_POSTE_mod",
            mesure:          f"{mesure}_mod"
        })
        .with_columns([
            pl.col("NUM_POSTE_mod").cast(pl.Int64)
        ])
    )

    # 4. Perform the chained joins
    comparison_df = (
        mapping_pl
        .join(obs_pl, on="NUM_POSTE_obs", how="inner")
        .join(mod_pl, on=["NUM_POSTE_mod", "year"], how="inner")
        .select([
            "NUM_POSTE_obs",
            f"{mesure}_obs",
            "NUM_POSTE_mod",
            f"{mesure}_mod",
            "year",
        ])
        # Parsing et calcul du delta
        .with_columns([
            pl.col(f"{mesure}_obs")
              .str.strptime(pl.Date, "%Y-%m-%d")
              .cast(pl.Int32)
              .alias("obs_days_since_epoch"),
            pl.col(f"{mesure}_mod")
              .str.strptime(pl.Date, "%Y-%m-%d")
              .cast(pl.Int32)
              .alias("mod_days_since_epoch"),
        ])
        .with_columns([
            (
                pl.col("mod_days_since_epoch")
                - pl.col("obs_days_since_epoch")
            ).alias("delta_days")
        ])
        .drop(["obs_days_since_epoch", "mod_days_since_epoch"])
        .drop_nulls(subset=["delta_days"])
    )

    return comparison_df




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline comparaison date maxima"
    )
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",
        default=[
            "config/modelised_settings.yaml",
            "config/observed_settings.yaml",
        ],
    )
    parser.add_argument(
        "--echelle",
        type=str,
        choices=["horaire", "quotidien"],
        nargs="+",
        default=["quotidien"],
    )
    parser.add_argument("--season", type=str, default="hydro")

    args = parser.parse_args()


    # Chargement des données
    dataframe_max_date, mapping = load_all_date(args)

    # Formation du df de comparaison
    comparison_df = comparaison_max_date(mapping, dataframe_max_date, args.echelle[0])

    print(comparison_df)

    # if args.season == "hydro":
    #     for signe in [-1,1]:
    #         for i in [360, 361, 362, 363, 364, 365, 366]:
    #             aff = comparison_df.filter(pl.col("delta_days") == i*signe)
    #             if aff is not None:
    #                 print(aff)


    # ─── Génération de l'histogramme de delta_days ───
    from pathlib import Path
    import matplotlib.pyplot as plt

    # 1) Création du dossier outputs s'il n'existe pas
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2) Extraction de delta_days en liste d'entiers
    delta = comparison_df["delta_days"].to_list()

    # 3) Calcul de la moyenne
    mean_delta = np.mean(delta)
    # 3b) Calcul de l’écart-type
    std_delta = np.std(delta)

    # création d’un tableau de poids pour que la somme des hauteurs = 100%  
    weights = np.ones_like(delta) * 100. / len(delta)

    # 4) Tracé de l'histogramme avec la moyenne en rouge
    plt.figure(figsize=[6,6])
    plt.hist(delta, bins=50, weights=weights)
    plt.axvline(
        mean_delta,
        color="red",         # ligne rouge
        linestyle="--",
        linewidth=1.5,
        label=f"Moy = {mean_delta:.1f} (± {std_delta:.0f})"
    )


    # Ajout du texte pour indiquer le nombre de valeurs comparées (n)
    plt.text(0.05, 0.9, f"n = {len(delta)}", transform=plt.gca().transAxes, fontsize=12)

    # Ajout du texte pour indiquer les pourcentages
    pos = 0.9
    for i in range(0, 5):
        pos = pos - 0.05
        val = (np.sum(np.abs(delta) == i) / len(delta)) * 100
        plt.text(0.05, pos, f"% maxima ± {i}j : {val:.2f}%", transform=plt.gca().transAxes, fontsize=12)

    # On force l'axe Y entre 0 et 10 %
    plt.ylim(0, 25) if args.echelle[0]=="horaire" else plt.ylim(0, 50)

    plt.xlabel("")
    if args.season == "hydro":
        plt.ylabel(f"% de stations", fontsize=15)
    plt.title(f"")
    plt.legend(loc='upper left', fontsize=13, frameon=False)
    plt.tight_layout()

    # 5) Sauvegarde dans outputs/
    plt.savefig(output_dir / f"histogram_delta_days_{args.echelle[0]}_{args.season}.png")
