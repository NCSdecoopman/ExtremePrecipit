import argparse
import os
from pathlib import Path
from tqdm.auto import tqdm

from typing import Tuple

from src.utils.logger import get_logger
from src.utils.config_tools import load_config
from src.utils.data_utils import years_to_load, load_data, cleaning_data_observed

import numpy as np
import polars as pl

from numba import njit

from scipy.stats import chi2
from scipy.optimize import brentq


def build_x_ttilde(df: pl.DataFrame, min_year: int, max_year:int, best_model: pl.DataFrame, break_year: int | None = None, mesure: str = None) -> pl.DataFrame:
    """Converts dates into the time covariate ``t_tilde``.

    - **Main change:** the 0 to 1 scale is now determined by global ``min_year`` 
      and ``max_year`` calculated once on the entire DataFrame, rather than 
      per station's observation period.
    - ``tmin`` / ``tmax`` per station are kept for slope recalculation later.
    - ``break_year`` handling (eventual breakpoint) remains unchanged: 
      ``t_tilde = 0`` before the breakpoint, then normalization of time since.
    """

    # Harmonize column names
    df = df.rename({mesure: "x"})  # daily maxima are now "x"

    # Breakpoint information per station (from best GEV model)
    break_info = best_model.select([
        pl.col("NUM_POSTE"),
        pl.col("model").str.contains("_break_year").alias("has_break")
    ])

    # tmin / tmax per station (for slope per 10 years later)
    tminmax = (
        df.group_by("NUM_POSTE")
          .agg([
              pl.col("year").min().alias("tmin"),
              pl.col("year").max().alias("tmax")
          ])
    )

    # Joins: add has_break, tmin, tmax
    df = (
        df
        .join(break_info, on="NUM_POSTE", how="left")
        .join(tminmax,   on="NUM_POSTE", how="left")
        .with_columns([
            # "t_plus" column = breakpoint year (identical for all stations)
            pl.when(pl.col("has_break"))
              .then(pl.lit(break_year))
              .otherwise(None)
              .alias("t_plus")
        ])
    )

    # Compute normalized covariate t_tilde (global scale)
    df = df.with_columns([
        pl.when(~pl.col("has_break"))
          # No breakpoint: (t - min_year)/(max_year - min_year)
          .then((pl.col("year") - min_year) / (max_year - min_year))
          .otherwise(
              # With breakpoint: 0 before break_year, then (t - break_year)/(max_year - break_year)
              pl.when(pl.col("year") < break_year)
                .then(0.0)
                .otherwise((pl.col("year") - break_year) / (max_year - break_year))
          )
          .alias("t_tilde_raw")
    ])

    # Applies `norm_1delta_0centred` normalization per station
    def norm_1delta_0centred_polars(s):
        t = s.to_numpy()
        t_min = t.min()
        t_max = t.max()
        if t_max == t_min:
            # Degenerate case: all t are identical
            return pl.Series([-0.5] * len(t))
        res0 = t / (t_max - t_min)
        dx = res0.min() + 0.5
        return pl.Series(res0 - dx)

    df = (
        df.group_by("NUM_POSTE")
        .map_groups(
            lambda g: g.with_columns(
                norm_1delta_0centred_polars(g["t_tilde_raw"]).alias("t_tilde")
            )
        )
    )

    # --------------------------------------------------------------------------------
    return df.select([
        "NUM_POSTE", "x", "t_tilde",
        "tmin", "tmax",
        "has_break",
        "t_plus", "year"
    ])






def calculate_z_T1(T:int, mu1: float, sigma1: float, xi0: float) -> float:
    """
    Computes z_T,1 in z_T(t) = z_T,0 + z_T,1 * t
    """
    CT = (-np.log(1 - 1/T))**(-xi0) - 1
    return mu1 + (sigma1 / xi0) * CT

def compute_calculate_zT1(T, df):
    return df.select([
        pl.col("NUM_POSTE", "model", "mu0", "mu1", "sigma0", "sigma1", "xi"),
        pl.struct(["mu1", "sigma1", "xi"])
        .map_elements(lambda s: calculate_z_T1(T, s["mu1"], s["sigma1"], s["xi"]),
                        return_dtype=pl.Float64)
        .alias("z_T1")
    ])




def compute_zT_for_years(
    T: int,
    annees_retour: np.ndarray,  # requested years (must contain 2 values)
    min_year: int,
    max_year: int,
    df_params: pl.DataFrame,
    df_series: pl.DataFrame
) -> pl.DataFrame:
    """
    Computes z_T(x) for each station for two return years (a and b),
    and returns a DataFrame with NUM_POSTE, zTpa, zTpb, (zTpb-zTpa)/zTpa.
    """
    if len(annees_retour) != 2:
        raise ValueError("annees_retour must contain exactly two values (a and b)")
    a, b = annees_retour[0], annees_retour[1]
    rows = []
    for row in df_params.to_dicts():
        data_station = df_series.filter(pl.col("NUM_POSTE") == row["NUM_POSTE"])
        years_obs = data_station["year"].to_numpy()
        t_tilde_obs_raw = (years_obs - min_year) / (max_year - min_year)
        t_min_ret = t_tilde_obs_raw.min()
        t_max_ret = t_tilde_obs_raw.max()
        res0_obs = t_tilde_obs_raw / (t_max_ret - t_min_ret)
        dx = res0_obs.min() + 0.5

        t_tilde_retour = []
        for y in [a, b]:
            t_tilde_retour_raw = (y - min_year) / (max_year - min_year)
            res0_ret = t_tilde_retour_raw / (t_max_ret - t_min_ret)
            t_tilde = res0_ret - dx
            t_tilde_retour.append(t_tilde)
        t_tilde_retour = np.array(t_tilde_retour)
        # Compute z_T for a and b
        mu0, mu1, sigma0, sigma1, xi = row["mu0"], row["mu1"], row["sigma0"], row["sigma1"], row["xi"]
        CT = ((-np.log(1-1/T))**(-xi) - 1)
        zTpa = mu0 + mu1*t_tilde_retour[0] + (sigma0 + sigma1*t_tilde_retour[0])/xi * CT
        zTpb = mu0 + mu1*t_tilde_retour[1] + (sigma0 + sigma1*t_tilde_retour[1])/xi * CT
        ratio = (zTpb - zTpa) / zTpa * 100 if zTpa != 0 else np.nan
        rows.append({
            "NUM_POSTE": row["NUM_POSTE"],
            "zTpa": zTpa,
            "zTpb": zTpb,
            "z_T_p": ratio
        })

    return pl.DataFrame(rows)


# mu and sigma functions as a function of z
@njit
def mu1_zT1_func(T, z_T1, hat_sigma1, hat_xi0):
    CT = (-np.log(1 - 1/T)) ** (-hat_xi0) - 1
    return z_T1 - (hat_sigma1 / hat_xi0) * CT

@njit
def sigma1_zT1_func(T, z_T1, hat_mu1, hat_xi0):
    CT = (-np.log(1 - 1/T)) ** (-hat_xi0) - 1
    return hat_xi0 * (z_T1 - hat_mu1) / CT



# Profiled likelihood functions

# ────────────────────────── helper functions ──────────────────────────
@njit(inline='always')
def _invalid_term(term):
    """True if at least one value in `term` is <= 0."""
    return np.any(term <= 0.0)

@njit(inline='always')
def _invalid_sigma(sigma):
    """True if at least one value in `sigma` is <= 0."""
    return np.any(sigma <= 0.0)

# ───────────────────────────── M1 : variable mu ─────────────────────────────
@njit
def log_likelihood_M1(T, z_T1, x, t_tilde, mu0, sigma0, mu1, sigma1, xi0):
    """
    Temporal effect on mu
    """
    if sigma0 <= 0.0:
        return -np.inf

    mu1 = mu1_zT1_func(T, z_T1, sigma1, xi0)
    z = (x - (mu0 + mu1 * t_tilde)) / sigma0
    term = 1 + xi0 * z

    if _invalid_term(term):
        return -np.inf

    return -np.sum(
        np.log(sigma0)
        + (1 + 1/xi0) * np.log(term)
        + term ** (-1 / xi0)
    )

# ───────────────────────────── M2 : variable sigma ─────────────────────────
@njit
def log_likelihood_M2(T, z_T1, x, t_tilde, mu0, sigma0, mu1, sigma1, xi0):
    """
    Temporal effect on sigma
    """
    sigma1 = sigma1_zT1_func(T, z_T1, mu1, xi0)
    sigma = sigma0 + sigma1 * t_tilde

    if _invalid_sigma(sigma): 
        return -np.inf

    z = (x - mu0) / sigma
    term = 1 + xi0 * z

    if _invalid_term(term):
        return -np.inf

    return -np.sum(
        np.log(sigma)
        + (1 + 1/xi0) * np.log(term)
        + term ** (-1 / xi0)
    )

# ─────────────────────────── M3 : variable mu & sigma ─────────────────────
@njit
def log_likelihood_M3(T, z_T1, x, t_tilde, mu0, sigma0, mu1, sigma1, xi0):
    """
    Temporal effect on mu and sigma
    """
    mu1 = mu1_zT1_func(T, z_T1, sigma1, xi0)
    sigma = sigma0 + sigma1 * t_tilde

    if _invalid_sigma(sigma):
        return -np.inf

    mu = mu0 + mu1 * t_tilde
    z = (x - mu) / sigma
    term = 1 + xi0 * z

    if _invalid_term(term):
        return -np.inf

    return -np.sum(
        np.log(sigma)
        + (1 + 1/xi0) * np.log(term)
        + term ** (-1 / xi0)
    )

def select_log_likelihood(model: str):
    """Returns the likelihood function for the selected model"""
    model = model.lower()          # tolerate « M1 », « m1 », etc.
    if "m1" in model:              
        return log_likelihood_M1
    elif "m2" in model:
        return log_likelihood_M2
    elif "m3" in model:
        return log_likelihood_M3
    else:
        raise ValueError(f"Model not recognized: {model}")


# Function to find confidence interval
from scipy.optimize import root_scalar

# ------------------------------------------------------------------------------
# Profiled confidence interval for z_T1
# ------------------------------------------------------------------------------
def find_confidence_interval(
    z_hat: float,
    ll_func,                       # profiled log-likelihood function (M1/M2/M3)
    model_params: dict,            # T, x, t_tilde, mu0, sigma0, mu1, sigma1, xi0
    alpha: float = 0.10,           # 1 – alpha = confidence level
    tol: float = 1e-6,             # absolute & relative precision for Brent
    max_doublings: int = 1000,     # max number of R → 2R extensions
    R: float = 1.0,                # initial radius around ẑ
    min_sep: float = 1e-6          # min separation ẑ ↔ bound to avoid "stuck root"
) -> Tuple[float, float]:
    """
    Returns (z_left, z_right), bounds of the (1-alpha) profiled CI.

    Procedure:
      1. Compute deviance D(z) = 2·(ℓ̂ − ℓ(z)) − χ²_{1−α}.
      2. Find, left and right of ẑ, the first point where D changes sign, 
         doubling radius R progressively.
      3. Refine root using Brent once sign inversion is found.
      4. If no inversion after `max_doublings`, return −∞ or +∞.
    """
    # ---------------------------------------------------------------- χ² threshold
    chi2_thr = chi2.ppf(1.0 - alpha, df=1)

    # ---------------------------------------------------------------- max log-likelihood
    ll_hat = ll_func(
        model_params["T"], z_hat,
        model_params["x"], model_params["t_tilde"],
        model_params["mu0"], model_params["sigma0"],
        model_params["mu1"], model_params["sigma1"],
        model_params["xi0"]
    )

    # ---------------------------------------------------------------- deviance
    def D(z: float) -> float:
        ll_z = ll_func(
            model_params["T"], z,
            model_params["x"], model_params["t_tilde"],
            model_params["mu0"], model_params["sigma0"],
            model_params["mu1"], model_params["sigma1"],
            model_params["xi0"]
        )
        if np.isnan(ll_z):
            return np.inf                    # invalid zone → no root here
        return 2.0 * (ll_hat - ll_z) - chi2_thr

    # ---------------------------------------------------------------- bound search
    def _find_side(sign: int) -> float:
        """
        Search for bound on `sign` side (−1 = left, +1 = right).
        If the only root found is "too close" to ẑ, it's rejected 
        and the bound is considered infinite.
        """
        R_local = R
        f_ref   = D(z_hat)                  # D(z_hat) ≤ 0
        found_far_enough = False

        # relative proximity threshold (1e-4 = 0.01 %)
        eps_rel = 1e-4 * max(1.0, abs(z_hat))

        for _ in range(max_doublings):
            z_try = z_hat + sign * R_local
            f_try = D(z_try)

            # sign inversion?
            if np.sign(f_try) != np.sign(f_ref):
                lo, hi = (z_try, z_hat - sign * min_sep) if sign < 0 else \
                        (z_hat + sign * min_sep, z_try)
                try:
                    sol = root_scalar(D, bracket=[lo, hi],
                                    method="brentq", xtol=tol, rtol=tol)

                    dist = abs(sol.root - z_hat)
                    if dist < max(min_sep, eps_rel):
                        # stuck root: attempt one extra extension
                        if found_far_enough:           # 2nd time → abandon
                            return np.inf if sign > 0 else -np.inf
                        found_far_enough = True
                        R_local *= 4.0                 # widen further
                        f_ref   = f_try
                        continue

                    return sol.root                    # valid bound

                except ValueError:
                    pass  # widen further if Brentq fails

            R_local *= 2.0
        # no root found after all extensions
        return np.inf if sign > 0 else -np.inf


    # ---------------------------------------------------------------- execution
    z_left  = _find_side(-1)
    z_right = _find_side(+1)

    return z_left, z_right




# ------------------------------------------------------------------
# MAIN FUNCTION
# ------------------------------------------------------------------
def profile_loglikelihood_per_station(
    T: int,
    df_series: pl.DataFrame,
    df_zT1: pl.DataFrame,
    threshold: float = 0.10
) -> pl.DataFrame:
    """
    Computes profiled IC of z_T1 for each station via robust root search.
    """
    merged = df_series.join(df_zT1, on="NUM_POSTE", how="inner")
    chi2_thr = chi2.ppf(1 - threshold, df=1)
    rows = []

    for gkey, grp in tqdm(list(merged.group_by("NUM_POSTE")), desc="Profiling stations"):
        num_poste = str(gkey) if isinstance(gkey, str) else str(gkey[0]) 
        
        x, t_tilde = grp["x"].to_numpy(), grp["t_tilde"].to_numpy()
        model, mu0, mu1, sigma0, sigma1, xi0, z_hat = (
            grp[["model", "mu0", "mu1", "sigma0", "sigma1", "xi", "z_T1"]]
            .unique()
            .row(0)
        )

        model_params = {
            "T": T,
            "z_T1": z_hat, 
            "x": x,                   
            "t_tilde": t_tilde,       
            "mu0": mu0,              
            "sigma0": sigma0,          
            "mu1": mu1,                 
            "sigma1": sigma1,          
            "xi0": xi0                  
        }

        # Select log-likelihood function
        loglik_func = select_log_likelihood(model)
        # log-likelihood function at z_hat
        loglik_hat  = loglik_func(T, z_hat, x, t_tilde, mu0, sigma0, mu1, sigma1, xi0)

        # Compute confidence interval
        z_left, z_right = find_confidence_interval(z_hat, loglik_func, model_params, threshold)

        # ─── Checks & warnings ───────────────────────────────────────────
        if (
            np.isinf(z_left) or np.isinf(z_right) or np.isinf(loglik_hat)
            or np.isclose(z_left,  z_hat)  # bound stuck to point-estimate
            or np.isclose(z_right, z_hat)
        ):
            logger.warning(f"\nStation {num_poste} : bounds z_left={z_left}, z_hat={z_hat}, z_right={z_right}")


        # if num_poste=="20092001":
        #     logger.warning(f"\n\n Encadrement identique que z_hat pour {num_poste} [{z_left} , {z_right}] avec \n {model_params}")
        #     z_grid  = np.linspace(z_hat - 50, z_hat + 50, 3000)
        #     ll_vals = np.array([
        #         loglik_func(T, z, x, t_tilde, mu0, sigma0, mu1, sigma1, xi0)
        #         for z in z_grid
        #     ])

        #     mask = np.isfinite(ll_vals)      # True là où σ>0 ET 1+ξz>0
           


        #     plot_dir = "log-vrais"
        #     os.makedirs(plot_dir, exist_ok=True)
        #     import matplotlib.pyplot as plt
        #     plt.figure()
        #     plt.plot(z_grid[mask], ll_vals[mask], label="Log-vraisemblance")
        #     # Ligne verticale pour z_hat
        #     plt.axvline(z_hat, color='red', linestyle='--', label=f'ẑ={z_hat:.3f}')
        #     # Lignes pour bornes IC
        #     plt.axvline(z_left, color='green', linestyle=':', label=f'IC lower={z_left:.3f}')
        #     plt.axvline(z_right, color='green', linestyle=':', label=f'IC upper={z_right:.3f}')
        #     plt.xlabel('z_T1')
        #     plt.ylabel('Log-vraisemblance')
        #     plt.title(f'Station {num_poste} : profil de log-vraisemblance')
        #     plt.legend(loc='best')
        #     plt.tight_layout()
        #     plt.savefig(os.path.join(plot_dir, f"loglik_profile_{num_poste}.png"))
        #     plt.close()

        rows.append({
            "NUM_POSTE":    num_poste,
            "ic_lower":     z_left,
            "ic_upper":     z_right,
            "significant":  not (z_left <= 0 <= z_right)
        })

    return (
        pl.DataFrame(rows)
          .with_columns(pl.col("NUM_POSTE").cast(pl.Utf8))
    )









def main(config, args, T: int = 10):
    global logger
    logger = get_logger(__name__)

    echelles = config.get("echelles", "quotidien")
    season = config.get("season", "hydro")
    model_path = config.get("config", "config/observed_settings.yaml")
    gev_dir = config["gev"]["path"]["outputdir"]
    break_year = config.get("gev", {}).get("break_year", 1985)
    reduce_activate = config.get("reduce_activate", False)

    if reduce_activate:
        suffix_save = "_reduce"
    else:
        suffix_save = ""
    
    for echelle in echelles:
        logger.info(f"--- Processing scale: {echelle.upper()} season: {season} ---")

        # STEP 1 
        # Open NUM_POSTE ┆ model ┆ mu0 ┆ mu1 ┆ sigma0 ┆ sigma1 ┆ xi ┆ log_likelihood table
        path_dir = Path(gev_dir) / f"{echelle}{suffix_save}" / args.season
        best_model_path = path_dir / "gev_param_best_model.parquet"
        best_model = pl.read_parquet(best_model_path)

        # STEP 2
        # Open maxima series
        # Select reading directory
        model_path_name = Path(model_path).name

        if model_path_name == "observed_settings.yaml":
            input_dir = Path(config["statistics"]["path"]["outputdir"]) / echelle
            logger.info(f"STATION source detected → reading from: {input_dir}")

        elif model_path_name == "modelised_settings.yaml":
            input_dir = Path(config["statistics"]["path"]["outputdir"]) / "horaire"
            logger.info(f"AROME source detected → reading from: {input_dir}")

        else:
            logger.error(f"Config file name not recognized: {model_path_name}")

        # Data loading parameters
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

        logger.info(f"Loading data from {min_year} to {max_year}: {input_dir}")
        df = load_data(input_dir, season, echelle, cols, min_year, max_year)

        # Station selection based on max NaN
        df = cleaning_data_observed(df, echelle, len_serie)

        # NaN handling
        df = df.drop_nulls(subset=[mesure])

        # Filter stations with GEV results
        df = df.filter(pl.col("NUM_POSTE").is_in(best_model["NUM_POSTE"].to_list()))
        assert df["NUM_POSTE"].n_unique() == best_model["NUM_POSTE"].n_unique(), \
            "DataFrames do not have the same number of unique NUM_POSTE"

        # Normalize t into t_tilde or t_tilde* based on breakpoint presence
        df_series = build_x_ttilde(df, min_year, max_year, best_model, break_year, mesure)

        # Add "z_T1" column to best_model DataFrame
        df_zT1 = compute_calculate_zT1(T, best_model)

        # Compute z_T(1992) and z_T(2022) with normalization on [1992, 2022] grid
        z_levels = compute_zT_for_years(
            T,
            np.array([1992, max_year]),
            min_year,
            max_year,
            best_model,
            df_series
        )

        df_zT1 = df_zT1.join(z_levels, on="NUM_POSTE")

        # Profiled log-likelihood for each station around z_T1
        table_ic = profile_loglikelihood_per_station(
            T,
            df_series,
            df_zT1,
            threshold=args.threshold  # threshold for chi²
        )

        # Retrieve model and z_T1 from df_zT1
        model_info = df_zT1.select(["NUM_POSTE", "model", "z_T1", "z_T_p"])

        # Join with model parameters
        table_ic = table_ic.with_columns(pl.col("NUM_POSTE").cast(pl.Utf8))     # match on NUM_POSTE
        model_info = model_info.with_columns(pl.col("NUM_POSTE").cast(pl.Utf8)) # match on NUM_POSTE
        final_table = table_ic.join(model_info, on="NUM_POSTE", how="left")

        # Column reorganization
        final_table = final_table.select(["NUM_POSTE", "model", "z_T1", "ic_lower", "ic_upper", "significant", "z_T_p"])

        # RESULTS PER 10 YEARS SLOPE
        # 1) duration per station Δtᵢ = tmaxᵢ – tminᵢ (or t+ if breakpoint)
        delta_years = (
            df_series
            .group_by("NUM_POSTE")
            .agg(
                (
                    pl.when(pl.max("has_break"))               # Station-wide boolean
                    .then(pl.max("tmax") - pl.max("t_plus")) # tmax – t_plus if breakpoint
                    .otherwise(pl.max("tmax") - pl.min("tmin"))  # tmax – tmin otherwise
                ).alias("delta_year")
            )
            .with_columns(pl.col("NUM_POSTE").cast(pl.Utf8)) # match on NUM_POSTE
        )

        # 2) Join with final table
        final_table = (
            final_table
            .join(delta_years, on="NUM_POSTE")
            .with_columns([
                (10 / pl.col("delta_year")).alias("k10")
            ])
            .with_columns([
                (pl.col("z_T1")     * pl.col("k10")).alias("z_T1"),
                (pl.col("ic_lower") * pl.col("k10")).alias("ic_lower"),
                (pl.col("ic_upper") * pl.col("k10")).alias("ic_upper"),
            ])
            .drop(["delta_year", "k10"])
        )

        # Save
        out_path = f"{path_dir}/niveau_retour.parquet"
        final_table.write_parquet(out_path)
        logger.info(f"Final table saved: {out_path}")
        logger.info(final_table)


def str2bool(v):
    if v == "True":
        return True
    else:
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Return level and CI calculation pipeline")
    parser.add_argument("--config", type=str, default="config/observed_settings.yaml")
    parser.add_argument("--echelle", choices=["horaire", "quotidien"], nargs='+', default=["horaire"])
    parser.add_argument("--season", type=str, default="son")
    parser.add_argument("--threshold", type=float, default=0.10, help="CI threshold")
    parser.add_argument("--reduce_activate", type=str2bool, default=False)

    args = parser.parse_args()

    config = load_config(args.config)
    config["config"] = args.config
    config["echelles"] = args.echelle
    config["season"] = args.season
    config["reduce_activate"] = args.reduce_activate
    
    main(config, args)
