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
NON_STATIONARY = [m for m in MODEL_LIST if m != "s_gev"]


def get_model_outputs(path_dir: Path) -> dict[str, pl.DataFrame]:
    """
    Charge les outputs parquet GEV pour chaque modèle.
    """
    outputs = {}
    for model, cols in MODEL_LIST.items():
        file = path_dir / f"gev_param_{model}.parquet"
        if file.exists():
            df = pl.read_parquet(file).select(["NUM_POSTE"] + cols + ["log_likelihood"]).with_columns(
                pl.lit(model).alias("model")
            )
            outputs[model] = df
        else:
            logger.warning(f"Modèle manquant: {model} ({file})")
    return outputs


def select_best_lrt(outputs: dict[str, pl.DataFrame], threshold: float = 0.10) -> pl.DataFrame:
    """
    Sélectionne le meilleur modèle par test du rapport de vraisemblance (LRT).
    Retourne DataFrame avec colonnes NUM_POSTE, mu0, mu1, sigma0, sigma1, xi, model.
    """
    if "s_gev" not in outputs:
        raise ValueError("Le modèle stationnaire 's_gev' est requis pour LRT.")

    # Stationnaire
    s_df = outputs["s_gev"]
    s_params = s_df.select(["NUM_POSTE", *MODEL_LIST["s_gev"]])
    s_log = s_df.select(["NUM_POSTE", "log_likelihood"]).rename({"log_likelihood": "ll_s"})

    candidates = []
    for m in NON_STATIONARY:
        if m not in outputs:
            continue
        df_ns = outputs[m]
        k_diff = len(MODEL_LIST[m]) - len(MODEL_LIST["s_gev"] )
        join = df_ns.join(s_log, on="NUM_POSTE", how="inner")
        lrt = 2 * (join["log_likelihood"] - join["ll_s"])
        pvals = chi2.sf(lrt.to_numpy(), df=k_diff)
        cand = join.select(["NUM_POSTE", *MODEL_LIST[m]]).with_columns([
            pl.Series("pval", pvals),
            pl.lit(m).alias("model")
        ])
        candidates.append(cand)

    if not candidates:
        # Retourne uniquement s_gev si pas de NS
        return s_params.with_columns([
            pl.lit("s_gev").alias("model"),
            pl.lit(1.0).alias("pval"),
            pl.lit(0.0).alias("mu1"),
            pl.lit(0.0).alias("sigma1"),
        ]).select(["NUM_POSTE", "mu0", "mu1", "sigma0", "sigma1", "xi", "model"])

    all_ns = pl.concat(candidates, how="diagonal")
    # Choisir par plus petite p-val
    best_ns = (
        all_ns.sort(["NUM_POSTE", "pval"]).
        group_by("NUM_POSTE").
        agg([
            pl.first("pval").alias("pval"),
            pl.first("model").alias("model"),
            *[pl.first(c).alias(c) for c in ["mu0", "mu1", "sigma0", "sigma1", "xi"] if c in all_ns.columns]
        ])
    )
    # Construire version s_gev complète
    s_full = s_params.with_columns([
        pl.lit("s_gev").alias("model"),
        pl.lit(1.0).alias("pval"),
        pl.lit(0.0).alias("mu1"),
        pl.lit(0.0).alias("sigma1"),
    ])
    merged = best_ns.join(s_full, on="NUM_POSTE", how="full", coalesce=True, suffix="_s")
    final = merged.with_columns([
        pl.when(pl.col("pval") < threshold).then(pl.col("model")).otherwise(pl.col("model_s")).alias("model"),
        pl.when(pl.col("pval") < threshold).then(pl.col("mu0")).otherwise(pl.col("mu0_s")).alias("mu0"),
        pl.when(pl.col("pval") < threshold).then(pl.col("mu1")).otherwise(pl.col("mu1_s")).alias("mu1"),
        pl.when(pl.col("pval") < threshold).then(pl.col("sigma0")).otherwise(pl.col("sigma0_s")).alias("sigma0"),
        pl.when(pl.col("pval") < threshold).then(pl.col("sigma1")).otherwise(pl.col("sigma1_s")).alias("sigma1"),
        pl.when(pl.col("pval") < threshold).then(pl.col("xi")).otherwise(pl.col("xi_s")).alias("xi"),
    ]).select(["NUM_POSTE", "mu0", "mu1", "sigma0", "sigma1", "xi", "model"])

    return final.filter(pl.col("xi").is_not_null())


def select_best_aic(outputs: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """
    Sélectionne le meilleur modèle par critère AIC pour chaque station.
    """
    dfs = []
    for m, df in outputs.items():
        k = len(MODEL_LIST[m])
        df = df.with_columns((2 * k - 2 * pl.col("log_likelihood")).alias("AIC"))
        dfs.append(df)
    all_df = pl.concat(dfs, how="diagonal").filter(pl.col("log_likelihood").is_not_null())
    best = all_df.sort("AIC").group_by("NUM_POSTE").first()
    # Ajouter colonnes manquantes
    cols = ["NUM_POSTE", "mu0", "mu1", "sigma0", "sigma1", "xi", "model"]
    for c in cols:
        if c not in best.columns:
            best = best.with_columns(pl.lit(None).alias(c))
    return best.select(cols)


def main(config, args):
    global logger
    logger = get_logger(__name__)

    gev_dir = config["gev"]["path"]["outputdir"]
    for e in args.echelle:
        logger.info(f"--- Traitement {args.method} - échelle: {e.upper()} ---")
        path_dir = Path(gev_dir) / e / args.season
        outputs = get_model_outputs(path_dir)
        if args.method == 'lrt':
            best = select_best_lrt(outputs, threshold=args.threshold)
            out_name = "gev_param_best_model_lrt.parquet"
        elif args.method == 'aic':
            best = select_best_aic(outputs)
            out_name = "gev_param_best_model_aic.parquet"
        else:
            raise KeyError("Methode non valide")
        out_path = path_dir / out_name
        best.write_parquet(out_path)
        logger.info(f"Résultat enregistré: {out_path}")
        logger.info(best)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de sélection du meilleur modèle GEV (LRT ou AIC)")
    parser.add_argument("--config", type=str, default="config/observed_settings.yaml")
    parser.add_argument("--echelle", choices=["horaire", "quotidien"], nargs='+', default=["horaire", "quotidien"])
    parser.add_argument("--season", type=str, default="hydro")
    parser.add_argument("--method", choices=["lrt", "aic"], default="lrt", help="Méthode de sélection: 'lrt' pour Test de Rapport de Vraisemblance, 'aic' pour critère AIC")
    parser.add_argument("--threshold", type=float, default=0.10, help="Seuil p-value pour LRT (uniquement si méthode 'lrt')")
    args = parser.parse_args()

    config = load_config(args.config)
    config["echelles"] = args.echelle
    config["season"] = args.season
    main(config, args)
