import argparse
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.config_tools import load_config

import polars as pl
from scipy.stats import chi2

# Liste des modèles disponibles et leurs colonnes
MODEL_LIST = {
    "s_gev": ["mu0", "sigma0", "xi"],
    "ns_gev_m1": ["mu0", "mu1", "sigma0", "xi"],
    "ns_gev_m1_break_year": ["mu0", "mu1", "sigma0", "xi"],
    "ns_gev_m2": ["mu0", "sigma0", "sigma1", "xi"],
    "ns_gev_m2_break_year": ["mu0", "sigma0", "sigma1", "xi"],
    "ns_gev_m3": ["mu0", "mu1", "sigma0", "sigma1", "xi"],
    "ns_gev_m3_break_year": ["mu0", "mu1", "sigma0", "sigma1", "xi"],
}

NON_STATIONARY_MODELS = [k for k in MODEL_LIST if k != "s_gev"]


def get_model_outputs(path_dir: Path) -> dict[str, pl.DataFrame]:
    model_outputs = {}
    for model, columns in MODEL_LIST.items():
        f = path_dir / f"gev_param_{model}.parquet"
        if f.exists():
            df = pl.read_parquet(f).select(["NUM_POSTE"] + columns + ["log_likelihood"])
            model_outputs[model] = df.with_columns(pl.lit(model).alias("model"))
        else:
            logger.warning(f"Modèle manquant : {model}")
    return model_outputs


def select_best_model(outputs: dict[str, pl.DataFrame], seuil_choice: int=0.10) -> pl.DataFrame:
    if "s_gev" not in outputs:
        raise ValueError("Modèle stationnaire (s_gev) requis.")

    s_df = outputs["s_gev"]
    s_params = s_df.select(["NUM_POSTE", *MODEL_LIST["s_gev"]])
    s_loglik = s_df.select(["NUM_POSTE", "log_likelihood"]).rename({"log_likelihood": "ll_s_gev"})

    all_candidates = []

    for model in NON_STATIONARY_MODELS:
        if model not in outputs:
            continue

        ns_df = outputs[model]
        k_diff = len(MODEL_LIST[model]) - len(MODEL_LIST["s_gev"])

        joined = ns_df.join(s_loglik, on="NUM_POSTE", how="inner")
        lrt_stat = 2 * (joined["log_likelihood"] - joined["ll_s_gev"])
        pval = chi2.sf(lrt_stat.to_numpy(), df=k_diff)

        # Ajouter pval à la table du modèle NS
        candidate = joined.select(["NUM_POSTE", *MODEL_LIST[model]]).with_columns([
            pl.Series(name="pval", values=pval),
            pl.lit(model).alias("model")
        ])

        all_candidates.append(candidate)

    if not all_candidates:
        # Aucun modèle NS disponible → on retourne s_gev directement
        return s_params.with_columns([
            pl.lit("s_gev").alias("model"),
            pl.lit(1.0).alias("pval"),
            pl.lit(0.0).alias("mu1"),
            pl.lit(0.0).alias("sigma1")
        ]).select(["NUM_POSTE", "mu0", "mu1", "sigma0", "sigma1", "xi", "model"])

    all_ns = pl.concat(all_candidates, how="diagonal")

    # Pour chaque station : chercher le modèle avec la plus petite p-val
    min_pval = all_ns.sort(["NUM_POSTE", "pval"]).group_by("NUM_POSTE").agg([
        pl.first("pval").alias("pval"),
        pl.first("model").alias("model"),
        *[pl.first(col).alias(col) for col in ["mu0", "mu1", "sigma0", "sigma1", "xi"] if col in all_ns.columns]
    ])

    # Créer une version s_gev complète avec mu1 = sigma1 = 0
    s_full = s_params.with_columns([
        pl.lit("s_gev").alias("model"),
        pl.lit(1.0).alias("pval"),
        pl.lit(0.0).alias("mu1"),
        pl.lit(0.0).alias("sigma1")
    ])

    # Fusionner les deux
    merged = min_pval.join(s_full, on="NUM_POSTE", how="full", coalesce=True, suffix="_sgev")

    # Appliquer la logique finale
    final = merged.with_columns([
        pl.when(pl.col("pval") < seuil_choice).then(pl.col("model")).otherwise(pl.col("model_sgev")).alias("final_model"),
        pl.when(pl.col("pval") < seuil_choice).then(pl.col("mu0")).otherwise(pl.col("mu0_sgev")).alias("mu0"),
        pl.when(pl.col("pval") < seuil_choice).then(pl.col("mu1")).otherwise(pl.col("mu1_sgev")).alias("mu1"),
        pl.when(pl.col("pval") < seuil_choice).then(pl.col("sigma0")).otherwise(pl.col("sigma0_sgev")).alias("sigma0"),
        pl.when(pl.col("pval") < seuil_choice).then(pl.col("sigma1")).otherwise(pl.col("sigma1_sgev")).alias("sigma1"),
        pl.when(pl.col("pval") < seuil_choice).then(pl.col("xi")).otherwise(pl.col("xi_sgev")).alias("xi")
    ])

    return final.select(["NUM_POSTE", "mu0", "mu1", "sigma0", "sigma1", "xi", "final_model"]).rename({"final_model": "model"})




def pipeline_select_by_lrt(config):
    global logger
    logger = get_logger(__name__)

    echelles = config.get("echelles")
    season = config.get("season", "hydro")
    model_path = config.get("config")
    gev_dir = config["gev"]["path"]["outputdir"]

    for echelle in echelles:
        logger.info(f"--- Traitement {model_path} - échelle : {echelle.upper()} ---")
        path_dir = Path(gev_dir) / echelle / season
        outputs = get_model_outputs(path_dir)
        best = select_best_model(outputs)

        out_file = path_dir / "gev_param_best_model_lrt.parquet"
        best.write_parquet(out_file)
        logger.info(f"Résultat enregistré : {out_file}")
        logger.info(best)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline LRT pour choisir le meilleur modèle non stationnaire")
    parser.add_argument("--config", type=str, default="config/observed_settings.yaml")
    parser.add_argument("--echelle", type=str, choices=["horaire", "quotidien"], nargs="+", default=["horaire", "quotidien"])
    parser.add_argument("--season", type=str, default="hydro")
    args = parser.parse_args()

    config = load_config(args.config)
    config["echelles"] = args.echelle
    config["season"] = args.season
    config["config"] = args.config

    pipeline_select_by_lrt(config)
