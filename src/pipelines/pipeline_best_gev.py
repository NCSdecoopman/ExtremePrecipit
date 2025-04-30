import argparse
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.config_tools import load_config

import polars as pl

# Liste des modèles disponibles et leurs colonnes
MODEL_LIST = {
    "s_gev": ["mu0", "sigma0", "xi"],
    "ns_gev_m1": ["mu0", "mu1", "sigma0", "xi"],
    "ns_gev_m2": ["mu0", "sigma0", "sigma1", "xi"],
    "ns_gev_m3": ["mu0", "mu1", "sigma0", "sigma1", "xi"],
}

def get_all_model_outputs(path_dir: Path) -> dict[str, pl.DataFrame]:
    """
    Charge les fichiers GEV pour tous les modèles dans le dossier donné.
    """
    model_outputs = {}
    for model, columns in MODEL_LIST.items():
        f = path_dir / f"gev_param_{model}.parquet"
        if f.exists():
            df = pl.read_parquet(f).select(["NUM_POSTE"] + columns + ["log_likelihood"])
            model_outputs[model] = df.with_columns(pl.lit(model).alias("model"))
        else:
            logger.error(f"Modèle manquant : {model}")
    return model_outputs

def merge_and_select_best_model(model_outputs: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """
    Concatène tous les modèles, calcule l’AIC et sélectionne le meilleur par NUM_POSTE.
    """
    dfs = []
    for model, df in model_outputs.items():
        k = len(MODEL_LIST[model])  # nombre de paramètres
        df = df.with_columns((2 * k - 2 * pl.col("log_likelihood")).alias("AIC")) # Calcul de l'AIC
        dfs.append(df)

    # Concatène tous les dataframes en remplissant NaN si la colonne n'existe pas 4 modèles * tous les points
    df_all = pl.concat(dfs, how="diagonal")
    # Meilleur AIC par station
    best = (
        df_all.sort("AIC")
        .group_by("NUM_POSTE")
        .first()
    )

    # Ajoute colonnes manquantes pour harmonisation
    all_cols = ["NUM_POSTE", "mu0", "mu1", "sigma0", "sigma1", "xi", "model"]
    for col in all_cols:
        if col not in best.columns:
            best = best.with_columns(pl.lit(None).alias(col))

    return best.select(all_cols)

def pipeline_select_best_model(config) -> pl.DataFrame:
    """
    Lance le pipeline de sélection du meilleur modèle pour chaque station.
    """
    global logger
    logger = get_logger(__name__)
    
    echelles = config.get("echelles")
    model_path = config.get("model")
    gev_dir = config["gev"]["path"]["outputdir"]

    for echelle in echelles:
        logger.info(f"--- Traitement de {model_path} - échelle : {echelle.upper()} ---")
        path_dir = Path(gev_dir) / echelle

        # Charge les fichiers GEV pour tous les modèles dans le dossier donné
        model_outputs = get_all_model_outputs(path_dir)

        df_best = merge_and_select_best_model(model_outputs)    
        df_best.write_parquet(f"{path_dir}/gev_param_best_model.parquet")
        logger.info(f"Enregistré sous {path_dir}/gev_param_best_model.parquet")
        logger.info(df_best)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline cherche meilleur GEV.")
    parser.add_argument("--config", type=str, default="config/observed_settings.yaml")
    parser.add_argument("--echelle", type=str, choices=["horaire", "quotidien"], nargs="+", default=["horaire", "quotidien"])
    args = parser.parse_args()

    config = load_config(args.config)
    config["echelles"] = args.echelle
    config["model"] = args.config

    df_best = pipeline_select_best_model(config)
