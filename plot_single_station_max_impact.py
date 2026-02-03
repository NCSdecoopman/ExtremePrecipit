import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

# Force non-interactive backend before pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.append(os.getcwd())

from src.utils.data_utils import load_data, cleaning_data_observed
from src.pipelines.pipeline_stats_to_gev import (
    gev_non_stationnaire,
    MODEL_REGISTRY,
    PARAM_DEFAULTS,
    init_gev_params_from_moments,
    NsDistribution,
    ObsWithCovar,
    FitNsDistribution,
    suppress_stdout,
)

import src.pipelines.pipeline_stats_to_gev as psg
from src.utils.logger import get_logger
psg.logger = get_logger("fit_single_station")

MONTH_LABELS_FR = {
    "jan": "Janvier",
    "fev": "Février",
    "mar": "Mars",
    "avr": "Avril",
    "mai": "Mai",
    "jui": "Juin",
    "juill": "Juillet",
    "aou": "Août",
    "sep": "Septembre",
    "oct": "Octobre",
    "nov": "Novembre",
    "dec": "Décembre",
}


def month_to_key(value: str) -> str:
    text = str(value).strip().lower()
    mapping = {
        "janvier": "jan", "janv": "jan", "jan": "jan",
        "fevrier": "fev", "fev": "fev", "feb": "fev",
        "mars": "mar", "mar": "mar",
        "avril": "avr", "avr": "avr", "apr": "avr",
        "mai": "mai", "may": "mai",
        "juin": "jui", "jun": "jui",
        "juillet": "juill", "juil": "juill", "jul": "juill",
        "aout": "aou", "aug": "aou",
        "septembre": "sep", "sept": "sep", "sep": "sep",
        "octobre": "oct", "oct": "oct",
        "novembre": "nov", "nov": "nov",
        "decembre": "dec", "dec": "dec",
    }
    if text not in mapping:
        raise ValueError(f"Mois invalide: {value}")
    return mapping[text]


def get_year_norm_fit(df: pd.DataFrame, min_year: int, max_year: int, model_name: str, break_year: int) -> np.ndarray:
    has_break = "_break_year" in model_name
    if has_break:
        return np.where(
            df["year"].to_numpy() < break_year,
            0.0,
            (df["year"].to_numpy() - break_year) / (max_year - break_year),
        )
    return (df["year"].to_numpy() - min_year) / (max_year - min_year)


def get_t_tilde_for_range(
    years: np.ndarray,
    obs_years: np.ndarray,
    min_year_global: int,
    max_year_global: int,
) -> np.ndarray:
    years = np.asarray(years, dtype=float)
    obs_years = np.asarray(obs_years, dtype=float)

    t_raw = (years - min_year_global) / (max_year_global - min_year_global)
    t_obs_raw = (obs_years - min_year_global) / (max_year_global - min_year_global)

    t_min_obs, t_max_obs = float(t_obs_raw.min()), float(t_obs_raw.max())
    if t_max_obs == t_min_obs:
        return np.full_like(t_raw, -0.5, dtype=float)

    res0_raw = t_raw / (t_max_obs - t_min_obs)
    res0_obs = t_obs_raw / (t_max_obs - t_min_obs)

    dx = float(res0_obs.min()) + 0.5
    return res0_raw - dx


def compute_rl(params: dict, t_norm: float, T: int = 10) -> float:
    mu0, mu1 = params["mu0"], params["mu1"]
    sigma0, sigma1 = params["sigma0"], params["sigma1"]
    xi = params["xi"]

    mu_t = mu0 + mu1 * t_norm
    sigma_t = sigma0 + sigma1 * t_norm

    if not np.isfinite(sigma_t) or sigma_t <= 0:
        return np.nan

    if xi != 0:
        CT = ((-np.log(1 - 1 / T)) ** (-xi) - 1)
        return float(mu_t + (sigma_t / xi) * CT)

    return float(mu_t - sigma_t * np.log(-np.log(1 - 1 / T)))


def params_to_dict(res: np.ndarray | None, model_name: str) -> dict | None:
    if res is None:
        return None

    if model_name in ["ns_gev_m1", "s_gev", "ns_gev_m1_break_year"]:
        return {"mu0": float(res[0]), "mu1": float(res[1]), "sigma0": float(res[2]), "sigma1": 0.0, "xi": float(res[3])}

    if model_name in ["ns_gev_m2", "ns_gev_m2_break_year"]:
        return {"mu0": float(res[0]), "mu1": 0.0, "sigma0": float(res[1]), "sigma1": float(res[2]), "xi": float(res[3])}

    if model_name in ["ns_gev_m3", "ns_gev_m3_break_year"]:
        return {"mu0": float(res[0]), "mu1": float(res[1]), "sigma0": float(res[2]), "sigma1": float(res[3]), "xi": float(res[4])}

    return None


def gev_non_stationnaire_first_valid(
    df: pd.DataFrame,
    col_val: str,
    model_name: str,
    init_params: dict[str, float] | None = None,
) -> tuple | None:
    """
    Version "avant correction" : retourne le premier fit fini, sans chercher
    la meilleure log-vraisemblance.
    """
    df = df.dropna(subset=[col_val])
    values = pd.Series(df[col_val].values, index=df.index)
    if values.empty:
        return None

    model_struct, param_names = MODEL_REGISTRY[model_name]

    mean_emp = values.mean()
    std_emp = values.std()
    xi_init = 0.1
    mu_init, sigma_init = init_gev_params_from_moments(mean_emp, std_emp, xi=xi_init)

    covar = pd.DataFrame({"x": df["year_norm"]}, index=df.index)
    obs = ObsWithCovar(values, covar)
    ns_dist = NsDistribution("gev", model_struct)
    bounds = [PARAM_DEFAULTS[param]["bounds"] for param in param_names]

    if init_params:
        mu_init = init_params.get("mu0", mu_init)
        sigma_init = init_params.get("sigma0", sigma_init)
        xi_init = init_params.get("xi", xi_init)

    custom_x0 = {
        "mu0": mu_init,
        "mu1": 0,
        "sigma0": sigma_init,
        "sigma1": 0,
        "xi": -xi_init,
    }

    if model_name in ["ns_gev_m3", "ns_gev_m3_break_year"]:
        if init_params:
            custom_x0["mu1"] = init_params.get("mu1", 0)
            custom_x0["sigma1"] = init_params.get("sigma1", 0)
        else:
            custom_x0["mu1"] = 0
            custom_x0["sigma1"] = 0

    x0 = [custom_x0[param] for param in param_names]

    optim_methods = []
    if model_name == "s_gev":
        bounds = [
            (0, 0) if param == "mu1" else PARAM_DEFAULTS[param]["bounds"]
            for param in param_names
        ]
    else:
        bounds = [PARAM_DEFAULTS[param]["bounds"] for param in param_names]
        optim_methods.append({"method": "BFGS", "x0": x0})

    bound_methods = ["L-BFGS-B", "TNC", "SLSQP", "Powell", "Nelder-Mead"]
    for method in bound_methods:
        optim_methods.append({"method": method, "x0": x0, "bounds": bounds})

    for optim_kwargs in optim_methods:
        try:
            fit = FitNsDistribution(obs, ns_distribution=ns_dist, fit_kwargs=[optim_kwargs])
            with suppress_stdout():
                fit.fit()
            log_likelihood = -fit.nllh()
            p = fit.ns_distribution.to_params_ts()
            param_values = list(p)
            param_values[-1] = -param_values[-1]  # xi
            if all(np.isfinite(param_values)):
                return tuple(param_values) + (log_likelihood,)
        except Exception:
            continue

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", required=True, help="NUM_POSTE")
    parser.add_argument("--month", required=True, help="Mois (ex: oct, octobre)")
    parser.add_argument("--echelle", default="horaire", choices=["horaire", "quotidien"])
    parser.add_argument("--t_ret", type=int, default=10)
    parser.add_argument("--out", default=None, help="Chemin de sortie PNG")
    args = parser.parse_args()

    month_key = month_to_key(args.month)
    month_label = MONTH_LABELS_FR.get(month_key, month_key)

    input_dir = Path(f"data/statisticals/observed/{args.echelle}")
    gev_root = Path(f"data/gev/observed/{args.echelle}")

    if args.echelle == "horaire":
        mesure = "max_mm_h"
        min_year_global = 1990
        max_year_global = 2022
        len_serie = 25
    else:
        mesure = "max_mm_j"
        min_year_global = 1959
        max_year_global = 2022
        len_serie = 50

    break_year = 1985

    best_path = gev_root / month_key / "gev_param_best_model.parquet"
    if not best_path.exists():
        raise FileNotFoundError(best_path)

    best_models_df = pl.read_parquet(best_path)

    cols = ["NUM_POSTE", mesure, "nan_ratio"]
    df = load_data(str(input_dir), month_key, args.echelle, cols, min_year_global, max_year_global)
    df = cleaning_data_observed(df, args.echelle, len_serie)
    df_pd = df.to_pandas()

    data_station = df_pd[df_pd["NUM_POSTE"] == str(args.station)].sort_values("year")
    if data_station.empty:
        raise ValueError("Aucune donnée pour cette station/mois.")

    row_best = best_models_df.filter(pl.col("NUM_POSTE") == str(args.station)).to_dicts()
    if not row_best:
        raise ValueError("Station absente des meilleurs modèles.")
    row_best = row_best[0]
    model_name = row_best["model"]
    init_p = {k: row_best[k] for k in ["mu0", "mu1", "sigma0", "sigma1", "xi"]}

    data_station = data_station.copy()
    data_station["year_norm"] = get_year_norm_fit(data_station, min_year_global, max_year_global, model_name, break_year)

    params_full_raw_after = gev_non_stationnaire(data_station, mesure, model_name, init_params=init_p)
    if params_full_raw_after is None:
        raise ValueError("Fit impossible (apres correction).")
    params_full_after = params_to_dict(np.asarray(params_full_raw_after, dtype=float), model_name)
    if params_full_after is None:
        raise ValueError("Parametres invalides (apres correction).")

    params_full_raw_before = gev_non_stationnaire_first_valid(data_station, mesure, model_name, init_params=init_p)
    if params_full_raw_before is None:
        raise ValueError("Fit impossible (avant correction).")
    params_full_before = params_to_dict(np.asarray(params_full_raw_before, dtype=float), model_name)
    if params_full_before is None:
        raise ValueError("Parametres invalides (avant correction).")

    idx_max = int(data_station[mesure].idxmax())
    data_no_max = data_station.drop(idx_max)
    params_no_max_raw_after = gev_non_stationnaire(data_no_max, mesure, model_name, init_params=init_p)
    params_no_max_after = params_to_dict(np.asarray(params_no_max_raw_after, dtype=float), model_name)
    if params_no_max_after is None:
        raise ValueError("Fit sans max impossible (apres correction).")

    params_no_max_raw_before = gev_non_stationnaire_first_valid(data_no_max, mesure, model_name, init_params=init_p)
    params_no_max_before = params_to_dict(np.asarray(params_no_max_raw_before, dtype=float), model_name)
    if params_no_max_before is None:
        raise ValueError("Fit sans max impossible (avant correction).")

    obs_years = data_station["year"].to_numpy()
    years_grid = np.linspace(min_year_global, max_year_global, 100)
    t_tilde_grid = get_t_tilde_for_range(years_grid, obs_years, min_year_global, max_year_global)

    rl_full_after = np.array([compute_rl(params_full_after, t, T=args.t_ret) for t in t_tilde_grid], dtype=float)
    rl_full_before = np.array([compute_rl(params_full_before, t, T=args.t_ret) for t in t_tilde_grid], dtype=float)
    rl_no_max_after = np.array([compute_rl(params_no_max_after, t, T=args.t_ret) for t in t_tilde_grid], dtype=float)
    rl_no_max_before = np.array([compute_rl(params_no_max_before, t, T=args.t_ret) for t in t_tilde_grid], dtype=float)

    def plot_base(ax):
        ax.plot(
            data_station["year"],
            data_station[mesure],
            color="black",
            marker="o",
            linestyle="-",
            alpha=0.3,
            label="Observations",
        )
        ax.scatter(
            [data_station.loc[idx_max, "year"]],
            [data_station.loc[idx_max, mesure]],
            color="red",
            s=100,
            edgecolors="black",
            zorder=5,
            label=f"Maximum ({int(data_station.loc[idx_max, 'year'])})",
        )
        ax.set_xlabel("Année")
        ax.set_ylabel("Précipitation (mm/h)" if args.echelle == "horaire" else "Précipitation (mm/j)")
        ax.grid(True, linestyle=":", alpha=0.7)

    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    plot_base(ax_before)
    ax_before.plot(years_grid, rl_full_before, color="#0055aa", linewidth=3, label=f"NL{args.t_ret} Complet")
    ax_before.plot(years_grid, rl_no_max_before, color="#00aa55", linewidth=3, linestyle="--", label=f"NL{args.t_ret} Sans max")
    ax_before.set_title(f"Avant correction")
    ax_before.legend(loc="lower right", framealpha=0.9)

    plot_base(ax_after)
    ax_after.plot(years_grid, rl_full_after, color="#0055aa", linewidth=3, label=f"NL{args.t_ret} Complet")
    ax_after.plot(years_grid, rl_no_max_after, color="#00aa55", linewidth=3, linestyle="--", label=f"NL{args.t_ret} Sans max")
    ax_after.set_title(f"Apres correction")
    ax_after.legend(loc="lower right", framealpha=0.9)

    fig.suptitle(f"Station {args.station} - {month_label} - Echelle {args.echelle}")

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path("outputs/temp") / f"station_{args.station}_{month_key}_{args.echelle}_impact_avant_apres.png"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(out_path)


if __name__ == "__main__":
    main()
