import argparse
from pathlib import Path
from tqdm.auto import tqdm

from src.utils.logger import get_logger
from src.utils.config_tools import load_config

import polars as pl
from scipy.stats import chi2
import numpy as np


# Liste des modèles disponibles et leurs paramètres
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
    Ne garde que les lignes ayant un log_likelihood non nul (fit réussi).
    """
    outputs = {}
    for model, cols in MODEL_LIST.items():
        file = path_dir / f"gev_param_{model}.parquet"
        if file.exists():
            df = pl.read_parquet(file).select(["NUM_POSTE"] + cols + ["log_likelihood"]).with_columns(
                pl.lit(model).alias("model")
            )
            # Filtrer ici : on ne garde que les lignes ayant un log_likelihood non null
            df = df.filter(~pl.col("log_likelihood").is_null())
            if df.height > 0:
                outputs[model] = df
            else:
                logger.warning(f"Aucun fit valide pour le modèle: {model} ({file})")
        else:
            logger.warning(f"Modèle manquant: {model} ({file})")
    return outputs


# ETAPE 1 : calculer les pval du likelihood ratio test des 6 modèles non stationnaires
def compute_lrt_pvals(outputs: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """
    Calcule la p-value du LRT pour chaque modèle non-stationnaire par rapport à s_gev.
    Retourne un DataFrame avec colonnes: NUM_POSTE, model, pval.
    """
    if "s_gev" not in outputs:
        raise ValueError("Le modèle stationnaire 's_gev' est requis.")

    # Log-likelihood du modèle stationnaire
    s_log = outputs["s_gev"].select(["NUM_POSTE", "log_likelihood"])\
                        .rename({"log_likelihood": "ll_s"})

    results = []
    for model in NON_STATIONARY:
        df = outputs.get(model)
        if df is None:
            continue
        # nombre de degrés de liberté supplémentaires
        k_diff = len(MODEL_LIST[model]) - len(MODEL_LIST["s_gev"])
        # jointure avec log-likelihood stationnaire
        joined = df.join(s_log, on="NUM_POSTE", how="inner")
        # statistique LRT
        lrt_stat = 2 * (joined["log_likelihood"] - joined["ll_s"])
        # p-value
        pvals = chi2.sf(lrt_stat.to_numpy(), df=k_diff)
        temp = pl.DataFrame({
            "NUM_POSTE": joined["NUM_POSTE"].to_list(),
            "model": [model] * joined.height,
            "pval": pvals
        })
        results.append(temp)

    return pl.concat(results, how="vertical")


# Pour chaque station, il regarde les p-values de tous les modèles non-stationnaires.
# Règle 1 : Si au moins un des deux modèles complexes (ns_gev_m3, ns_gev_m3_break_year) a une p-value ≤ 0.10, il retient celui des deux qui a la plus petite p-value.
# Règle 2 : Sinon, il retient le modèle (parmi tous) qui a la plus petite p-value.
# Il retourne pour chaque station le nom du « meilleur » modèle selon cette logique.

COMPLEX_MODELS   = ["ns_gev_m3", "ns_gev_m3_break_year"]
FALLBACK_MODELS  = [m for m in NON_STATIONARY if m not in COMPLEX_MODELS]

def select_best_non_stationary(outputs: dict[str, pl.DataFrame], threshold: float = 0.10) -> pl.DataFrame:
    """
    Pour chaque station (NUM_POSTE) choisit le « meilleur » modèle non-stationnaire :
    1) Si au moins un des deux modèles complexes a p-value ≤ `threshold`,
       on retient le modèle complexe ayant la plus petite p-value.
    2) Sinon, on retient le modèle – parmi les six – ayant la plus petite p-value.
    
    Retourne un DataFrame Polars (colonnes : NUM_POSTE, model).
    """
    # 1. Calcul/assemblage des p-values → un seul DataFrame
    pvals_df = compute_lrt_pvals(outputs)

    best_records = []

    # 2. Parcours des stations
    for poste in tqdm(
        pvals_df["NUM_POSTE"].unique().to_list(),
        desc="Sélection best model",
        unit="poste"
    ):
        sub = pvals_df.filter(pl.col("NUM_POSTE") == poste)
        if sub.is_empty():
            raise ValueError(f"Aucun résultat de LRT pour NUM_POSTE = {poste}")

        # 2a. Modèles complexes significatifs pour ce poste
        signif_complex = (
            sub
            .filter(
                pl.col("model").is_in(COMPLEX_MODELS) &
                (pl.col("pval") <= threshold)
            )
            .sort("pval") # ordre croissant
        )

        # 2b. Choix du meilleur modèle selon la règle 
        best_row = (
            signif_complex.head(1)                   # cas (1)
            if signif_complex.height > 0
            else sub.sort("pval").head(1)            # cas (2)
        ) # donne la plus petite p-value

        best_records.append(
            {
                "NUM_POSTE": poste,
                "model": best_row["model"][0]        # Polars Series → scalaire
            }
        )

    return pl.DataFrame(best_records)


def assemble_final_table(outputs: dict[str, pl.DataFrame], best: pl.DataFrame) -> pl.DataFrame:
    """
    Pour chaque station et modèle sélectionné, récupère les paramètres associés.
    Remplit les paramètres manquants par 0 pour standardiser les colonnes.
    """
    # Colonnes cibles (ordre commun à tous les modèles)
    target_cols = ["NUM_POSTE", "model", "mu0", "mu1", "sigma0", "sigma1", "xi", "log_likelihood"]

    records = []
    # On itère sur les choix du meilleur modèle pour chaque station
    for row in best.iter_rows(named=True):
        poste = row["NUM_POSTE"]
        model = row["model"]
        df_model = outputs[model]
        # Sélection de la ligne correspondante
        params = df_model.filter(pl.col("NUM_POSTE") == poste)
        if params.is_empty():
            # Aucun fit pour ce poste avec ce modèle : on saute
            continue
        else:
            # On récupère les valeurs et complète les colonnes manquantes à 0
            p = params.row(0, named=True)
            rec = {
                "NUM_POSTE": poste,
                "model": model,
                "mu0": p.get("mu0", 0.0),
                "mu1": p.get("mu1", 0.0),
                "sigma0": p.get("sigma0", 0.0),
                "sigma1": p.get("sigma1", 0.0),
                "xi": p.get("xi", 0.0),
                "log_likelihood": p.get("log_likelihood", None),
            }
        records.append(rec)
    return pl.DataFrame(records)[target_cols]


def norm_1delta_0centred_pandas(series):
    res0 = series.astype(float) / (series.max() - series.min())
    dx = res0.min() + 0.5
    return res0 - dx

def build_x_ttilde(
    df: pl.DataFrame,
    best_model: pl.DataFrame,
    break_year: int | None = None,
    mesure: str | None = None,
) -> pl.DataFrame:
    """
    Applique la normalisation de l'année comme dans pipeline_stats_to_gev.py et construit les colonnes demandées.
    """
    # Conversion en pandas pour la normalisation
    df_pd = df.to_pandas()
    min_year = df_pd["year"].min()
    max_year = df_pd["year"].max()

    if break_year is not None:
        if max_year == break_year:
            raise ValueError("`break_year` ne peut pas être égal à `max_year` (division par zéro).")
        df_pd["year_norm"] = np.where(
            df_pd["year"] < break_year,
            0,
            (df_pd["year"] - break_year) / (max_year - break_year)
        )
        has_break = True
        t_plus = np.where(df_pd["year"] < break_year, 0, df_pd["year"] - break_year)
    else:
        df_pd["year_norm"] = (df_pd["year"] - min_year) / (max_year - min_year)
        has_break = False
        t_plus = df_pd["year"] - min_year

    # Normalisation finale comme dans pipeline_stats_to_gev.py
    df_pd["year_norm"] = norm_1delta_0centred_pandas(df_pd["year_norm"])

    # x = year_norm
    df_pd["x"] = df_pd["year_norm"]
    # t_tilde = x (pour compatibilité)
    df_pd["t_tilde"] = df_pd["x"]
    # tmin, tmax
    tmin = df_pd["year"].min()
    tmax = df_pd["year"].max()
    df_pd["tmin"] = tmin
    df_pd["tmax"] = tmax
    df_pd["has_break"] = has_break
    df_pd["t_plus"] = t_plus

    # Conversion en polars et sélection des colonnes demandées
    df_pl = pl.from_pandas(df_pd)
    return df_pl.select([
        "NUM_POSTE", "x", "t_tilde",
        "tmin", "tmax",
        "has_break",
        "t_plus"
    ])


def main(config, args):
    global logger
    logger = get_logger(__name__)

    gev_dir = config["gev"]["path"]["outputdir"]
    for e in args.echelle:
        logger.info(f"--- Traitement échelle: {e.upper()} saison: {args.season}---")
        path_dir = Path(gev_dir) / e / args.season

        # 1. Charger les outputs bruts
        outputs = get_model_outputs(path_dir)

        # 2. Sélection du meilleur modèle non-stationnaire
        best = select_best_non_stationary(outputs, threshold=args.threshold)
        # `best` a 2 colonnes: NUM_POSTE, model

        # 3. Assembler le tableau final avec tous les paramètres
        final_table = assemble_final_table(outputs, best)

        # 4. Sauvegarde
        out_path = path_dir / "gev_param_best_model.parquet"
        final_table.write_parquet(out_path)
        logger.info(f"Tableau final enregistré: {out_path}")
        logger.info(final_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de sélection du meilleur modèle GEV (LRT ou AIC)")
    parser.add_argument("--config", type=str, default="config/observed_settings.yaml")
    parser.add_argument("--echelle", choices=["horaire", "quotidien"], nargs='+', default=["quotidien"])
    parser.add_argument("--season", type=str, default="son")
    parser.add_argument("--threshold", type=float, default=0.10, help="Seuil p-value pour LRT")
    args = parser.parse_args()

    config = load_config(args.config)
    config["echelles"] = args.echelle
    config["season"] = args.season
    main(config, args)
