import argparse
from pathlib import Path
from tqdm.auto import tqdm

from src.utils.logger import get_logger
from src.utils.config_tools import load_config

import polars as pl
from scipy.stats import chi2


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


# ETAPE 2 : si au moins l'un des 2 pval des modèles sur mu et sigma sont significatif (au seuil 10%),
# considérer que le meilleur modèle est celui qui a la plus petite pval. L'idée c'est de privilégier le modèle le plus complexe

# ETAPE 3 : sinon le meilleur modèle non stationnaire est celui parmi les 4 restants qui a la plus petite pval

def select_best_non_stationary(outputs: dict[str, pl.DataFrame], threshold: float = 0.10) -> pl.DataFrame:
    """
    Sélectionne pour chaque station le meilleur modèle non-stationnaire selon:
    1) Si la p-val de ns_gev_m3 ou ns_gev_m3_break_year ≤ threshold, on prend le modèle le plus complexe
       (parmi tous les non stationnaires) ayant la p-val la plus faible.
    2) Sinon, on prend le meilleur modèle parmi les 4 autres (excluant ns_gev_m3 et ns_gev_m3_break_year)
       avec la p-val la plus faible.

    Retourne un DataFrame avec colonnes: NUM_POSTE, model.
    """
    # Calcul des p-values
    pvals_df = compute_lrt_pvals(outputs)

    best_records = []
    # modèles fallback (excluant les deux modèles les plus complexes)
    fallback_models = [m for m in NON_STATIONARY if m not in ["ns_gev_m3", "ns_gev_m3_break_year"]]
    # modèles complexes 
    complex_models = [m for m in NON_STATIONARY if m in ["ns_gev_m3", "ns_gev_m3_break_year"]]

    # on récupère la liste des postes
    postes = pvals_df["NUM_POSTE"].unique().to_list()
    # boucle avec barre de progression
    for poste in tqdm(postes, desc="Sélection best model", unit="poste"):
        # selection du poste
        sub = pvals_df.filter(pl.col("NUM_POSTE") == poste)
        # p-values des modèles sur mu et sigma complexes
        pm3  = sub.filter(pl.col("model") == "ns_gev_m3")["pval"].to_list() or [1.0]
        pm3b = sub.filter(pl.col("model") == "ns_gev_m3_break_year")["pval"].to_list() or [1.0]
        # condition de significativité
        if min(pm3[0], pm3b[0]) <= threshold:
            sub_cp = sub.filter(pl.col("model").is_in(complex_models))
            candidats = dict(zip(sub_cp["model"], sub_cp["pval"]))
        else:
            sub_fb = sub.filter(pl.col("model").is_in(fallback_models))
            candidats = dict(zip(sub_fb["model"], sub_fb["pval"]))

        if not candidats:
            raise ValueError(f"Aucun modèle disponible pour NUM_POSTE = {poste}")

        # modèle avec p-val minimale
        best = min(candidats, key=candidats.get)
        best_records.append({"NUM_POSTE": poste, "model": best})

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
