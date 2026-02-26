import argparse
from pathlib import Path
from tqdm.auto import tqdm

from src.utils.logger import get_logger
from src.utils.config_tools import load_config

import polars as pl
from scipy.stats import chi2


# Available models and their parameters
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
    Loads GEV parquet outputs for each model.
    Only keeps rows with a non-zero log_likelihood (successful fit).
    """
    outputs = {}
    for model, cols in MODEL_LIST.items():
        file = path_dir / f"gev_param_{model}.parquet"
        if file.exists():
            df = pl.read_parquet(file).select(["NUM_POSTE"] + cols + ["log_likelihood"]).with_columns(
                pl.lit(model).alias("model")
            )
            # Filter: only keep rows with a valid log_likelihood
            df = df.filter(~pl.col("log_likelihood").is_null())
            if df.height > 0:
                outputs[model] = df
            else:
                logger.warning(f"No valid fit for model: {model} ({file})")
        else:
            logger.warning(f"Missing model: {model} ({file})")
    return outputs


# STEP 1: Compute LRT p-values for the 6 non-stationary models
def compute_lrt_pvals(outputs: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """
    Computes LRT p-value for each non-stationary model relative to s_gev.
    Returns a DataFrame with columns: NUM_POSTE, model, pval.
    """
    if "s_gev" not in outputs:
        raise ValueError("Stationary model 's_gev' is required.")

    # Stationary model log-likelihood
    s_log = outputs["s_gev"].select(["NUM_POSTE", "log_likelihood"])\
                        .rename({"log_likelihood": "ll_s"})

    results = []
    for model in NON_STATIONARY:
        df = outputs.get(model)
        if df is None:
            continue
        # extra degrees of freedom
        k_diff = len(MODEL_LIST[model]) - len(MODEL_LIST["s_gev"])
        # join with stationary log-likelihood
        joined = df.join(s_log, on="NUM_POSTE", how="inner")
        # LRT statistic
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


# Selection logic per station:
# Rule 1: If at least one complex model (ns_gev_m3, ns_gev_m3_break_year) has p-value ≤ 0.10, the one with the smallest p-value is kept.
# Rule 2: Otherwise, keep the model (among all 6) with the smallest p-value.

COMPLEX_MODELS   = ["ns_gev_m3", "ns_gev_m3_break_year"]
FALLBACK_MODELS  = [m for m in NON_STATIONARY if m not in COMPLEX_MODELS]

def select_best_non_stationary(outputs: dict[str, pl.DataFrame], threshold: float = 0.10) -> pl.DataFrame:
    """
    Chooses the "best" non-stationary model per station (NUM_POSTE):
    1) If a complex model has p-value ≤ `threshold`, use the complex one with smallest p-value.
    2) Otherwise, use the model (out of 6) with the smallest p-value.
    """
    # 1. Compute/assemble p-values
    pvals_df = compute_lrt_pvals(outputs)

    best_records = []

    # 2. Iterate through stations
    for poste in tqdm(
        pvals_df["NUM_POSTE"].unique().to_list(),
        desc="Selecting best model",
        unit="poste"
    ):
        sub = pvals_df.filter(pl.col("NUM_POSTE") == poste)
        if sub.is_empty():
            raise ValueError(f"No LRT results for NUM_POSTE = {poste}")

        # 2a. Significant complex models for this station
        signif_complex = (
            sub
            .filter(
                pl.col("model").is_in(COMPLEX_MODELS) &
                (pl.col("pval") <= threshold)
            )
            .sort("pval") # ascending
        )

        # 2b. Best model choice according to rules
        best_row = (
            signif_complex.head(1)                   # case (1)
            if signif_complex.height > 0
            else sub.sort("pval").head(1)            # case (2)
        ) # gives the smallest p-value

        best_records.append(
            {
                "NUM_POSTE": poste,
                "model": best_row["model"][0]        # Polars Series → scalaire
            }
        )

    return pl.DataFrame(best_records)


def assemble_final_table(outputs: dict[str, pl.DataFrame], best: pl.DataFrame) -> pl.DataFrame:
    """
    Retrieves parameters for each station and selected model.
    Fills missing parameters with 0 to standardize columns.
    """
    # Target columns (common order)
    target_cols = ["NUM_POSTE", "model", "mu0", "mu1", "sigma0", "sigma1", "xi", "log_likelihood"]

    records = []
    # Iterate on best choices per station
    for row in best.iter_rows(named=True):
        poste = row["NUM_POSTE"]
        model = row["model"]
        df_model = outputs[model]
        # Matching row selection
        params = df_model.filter(pl.col("NUM_POSTE") == poste)
        if params.is_empty():
            # No fit for this station with this model
            continue
        else:
            # Retrieve values and fill missing columns with 0
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
    reduce_activate = config.get("reduce_activate", False)

    if reduce_activate:
        suffix_save = "_reduce"
    else:
        suffix_save = ""

    for e in args.echelle:
        e = f"{e}{suffix_save}"
        logger.info(f"--- Processing scale: {e.upper()} season: {args.season} ---")
        path_dir = Path(gev_dir) / e / args.season

        # 1. Load raw outputs
        outputs = get_model_outputs(path_dir)

        # 2. Select best non-stationary model
        best = select_best_non_stationary(outputs, threshold=args.threshold)

        # 3. Assemble final table with all parameters
        final_table = assemble_final_table(outputs, best)

        # 4. Save
        out_path = path_dir / "gev_param_best_model.parquet"
        final_table.write_parquet(out_path)
        logger.info(f"Final table saved: {out_path}")
        logger.info(final_table)


def str2bool(v):
    if v == "True":
        return True
    else:
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GEV best model selection pipeline (LRT or AIC)")
    parser.add_argument("--config", type=str, default="config/observed_settings.yaml")
    parser.add_argument("--echelle", choices=["horaire", "quotidien", "quotidien_reduce"], nargs='+', default=["quotidien"])
    parser.add_argument("--season", type=str, default="son")
    parser.add_argument("--threshold", type=float, default=0.10, help="LRT p-value threshold")
    parser.add_argument("--reduce_activate", type=str2bool, default=False)
    args = parser.parse_args()

    config = load_config(args.config)
    config["echelles"] = args.echelle
    config["season"] = args.season
    config["reduce_activate"] = args.reduce_activate

    main(config, args)
