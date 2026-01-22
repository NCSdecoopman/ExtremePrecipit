
import sys
import os
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

# Fix: Force non-interactive backend
import matplotlib
matplotlib.use('Agg')

# Add project root to path
sys.path.append(os.getcwd())

from src.utils.data_utils import load_data, cleaning_data_observed
from src.pipelines.pipeline_stats_to_gev import gev_non_stationnaire

# Initialize logger
import src.pipelines.pipeline_stats_to_gev as psg
from src.utils.logger import get_logger
psg.logger = get_logger("fit_scrupulous")

# Configuration
INPUT_DIR = Path("data/statisticals/observed/horaire")
SEASON = "fev"
ECHELLE = "horaire"
MESURE = "max_mm_h"
MIN_YEAR_GLOBAL = 1990
MAX_YEAR_GLOBAL = 2022
LEN_SERIE = 25
T_RET = 10
BREAK_YEAR = 1985

# Styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.7
})

def get_year_norm_fit(df, min_year, max_year, model_name):
    has_break = "_break_year" in model_name
    if has_break:
        return np.where(df["year"] < BREAK_YEAR, 0, (df["year"] - BREAK_YEAR) / (max_year - BREAK_YEAR))
    else:
        return (df["year"] - min_year) / (max_year - min_year)

def get_t_tilde_for_range(years, obs_years, min_year_global, max_year_global):
    t_raw = (years - min_year_global) / (max_year_global - min_year_global)
    t_obs_raw = (obs_years - min_year_global) / (max_year_global - min_year_global)
    t_min_obs, t_max_obs = t_obs_raw.min(), t_obs_raw.max()
    if t_max_obs == t_min_obs: return np.full_like(t_raw, -0.5)
    res0_raw = t_raw / (t_max_obs - t_min_obs)
    res0_obs = t_obs_raw / (t_max_obs - t_min_obs)
    dx = res0_obs.min() + 0.5
    return res0_raw - dx

def compute_rl(params, t_norm, T=10):
    mu0, mu1, sigma0, sigma1, xi = params["mu0"], params["mu1"], params["sigma0"], params["sigma1"], params["xi"]
    mu_t = mu0 + mu1 * t_norm
    sigma_t = sigma0 + sigma1 * t_norm
    if xi != 0:
        CT = ((-np.log(1 - 1/T))**(-xi) - 1)
        return mu_t + (sigma_t / xi) * CT
    else:
        return mu_t - sigma_t * np.log(-np.log(1 - 1/T))

def fit_scrupulous(df, col_val, model_name, init_params=None):
    df = df.copy()
    df["year_norm"] = get_year_norm_fit(df, MIN_YEAR_GLOBAL, MAX_YEAR_GLOBAL, model_name)
    try:
        res = gev_non_stationnaire(df, col_val, model_name, init_params=init_params)
        if res and np.all(np.isfinite(res)): return res
    except:
        pass
    return None

def params_to_dict(res, model_name):
    if res is None: return None
    if model_name in ["ns_gev_m1", "s_gev", "ns_gev_m1_break_year"]:
        return {"mu0": res[0], "mu1": res[1], "sigma0": res[2], "sigma1": 0.0, "xi": res[3]}
    elif model_name in ["ns_gev_m2", "ns_gev_m2_break_year"]:
        return {"mu0": res[0], "mu1": 0.0, "sigma0": res[1], "sigma1": res[2], "xi": res[3]}
    elif model_name in ["ns_gev_m3", "ns_gev_m3_break_year"]:
        return {"mu0": res[0], "mu1": res[1], "sigma0": res[2], "sigma1": res[3], "xi": res[4]}
    return None

def run_analysis():
    best_models_df = pl.read_parquet(f"data/gev/observed/horaire/{SEASON}/gev_param_best_model.parquet")
    cols = ["NUM_POSTE", MESURE, "nan_ratio"]
    df = load_data(str(INPUT_DIR), SEASON, ECHELLE, cols, MIN_YEAR_GLOBAL, MAX_YEAR_GLOBAL)
    df = cleaning_data_observed(df, ECHELLE, LEN_SERIE)
    df_pd = df.to_pandas()
    
    # New stations with trends 200%-300%
    stations_to_plot = ["7068001", "26176001", "26211001", "26313001"]
    output_dir = Path("add_fig")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_dir = Path("add_fig")
    output_dir.mkdir(parents=True, exist_ok=True)
    multi_pdf_path = output_dir / "impact_max_all_stations.pdf"

    with PdfPages(multi_pdf_path) as pdf:
        for station in stations_to_plot:
            data_station = df_pd[df_pd["NUM_POSTE"] == station].sort_values("year")
            if data_station.empty:
                continue

            obs_years = data_station["year"].values
            row_best = best_models_df.filter(pl.col("NUM_POSTE") == station).to_dicts()[0]
            model_name = row_best["model"]
            init_p = {k: row_best[k] for k in ["mu0", "mu1", "sigma0", "sigma1", "xi"]}

            print(f"Traitement station {station} (Modèle: {model_name})...")

            params_full_raw = fit_scrupulous(data_station, MESURE, model_name, init_params=init_p)
            idx_max = data_station[MESURE].idxmax()
            data_no_max = data_station.drop(idx_max)
            params_no_max_raw = fit_scrupulous(data_no_max, MESURE, model_name, init_params=init_p)

            if params_full_raw is None or params_no_max_raw is None:
                continue

            params_full = params_to_dict(params_full_raw, model_name)
            params_no_max = params_to_dict(params_no_max_raw, model_name)

            fig, ax = plt.subplots(figsize=(10, 6))

            # Observations - like in plot_serie.py
            ax.plot(
                data_station["year"],
                data_station[MESURE],
                color="black",
                marker="o",
                linestyle="-",
                alpha=0.3,
                label="Observations",
            )

            # Highlight Maximum
            ax.scatter(
                [data_station.loc[idx_max, "year"]],
                [data_station.loc[idx_max, MESURE]],
                color="red",
                s=100,
                edgecolors="black",
                zorder=5,
                label=f"Maximum ({data_station.loc[idx_max, 'year']})",
            )

            years_grid = np.linspace(MIN_YEAR_GLOBAL, MAX_YEAR_GLOBAL, 100)
            t_tilde_grid = get_t_tilde_for_range(years_grid, obs_years, MIN_YEAR_GLOBAL, MAX_YEAR_GLOBAL)

            rl_full = [compute_rl(params_full, t, T=T_RET) for t in t_tilde_grid]
            rl_no_max = [compute_rl(params_no_max, t, T=T_RET) for t in t_tilde_grid]

            ax.plot(years_grid, rl_full, color="#0055aa", linewidth=3, label=f"Niveau de retour T={T_RET} (Complet)")
            ax.plot(
                years_grid,
                rl_no_max,
                color="#00aa55",
                linewidth=3,
                linestyle="--",
                label=f"Niveau de retour T={T_RET} (Sans max)",
            )

            # Trend calculation (1995-2022 as in plot_serie.py)
            t_tilde_1995 = get_t_tilde_for_range(np.array([1995]), obs_years, MIN_YEAR_GLOBAL, MAX_YEAR_GLOBAL)[0]
            t_tilde_2022 = get_t_tilde_for_range(np.array([2022]), obs_years, MIN_YEAR_GLOBAL, MAX_YEAR_GLOBAL)[0]

            rl1995_full = compute_rl(params_full, t_tilde_1995, T=T_RET)
            rl2022_full = compute_rl(params_full, t_tilde_2022, T=T_RET)
            trend_full = (rl2022_full - rl1995_full) / rl1995_full * 100

            rl1995_no_max = compute_rl(params_no_max, t_tilde_1995, T=T_RET)
            rl2022_no_max = compute_rl(params_no_max, t_tilde_2022, T=T_RET)
            trend_no_max = (rl2022_no_max - rl1995_no_max) / rl1995_no_max * 100

            # Adding trend text for both (as in earlier version)
            ax.text(
                0.02,
                0.95,
                f"Tendance (Full) : {trend_full:+.1f}%",
                transform=ax.transAxes,
                color="#0055aa",
                fontweight="bold",
                fontsize=12,
                va="top",
            )
            ax.text(
                0.02,
                0.90,
                f"Tendance (Sans Max) : {trend_no_max:+.1f}%",
                transform=ax.transAxes,
                color="#00aa55",
                fontweight="bold",
                fontsize=12,
                va="top",
            )

            ax.set_title(f"Station {station} - Mois de février - Echelle horaire")
            ax.set_xlabel("Année")
            ax.set_ylabel("Précipitation (mm/h)")

            ax.legend(loc="lower right", framealpha=0.9)
            plt.tight_layout()

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

if __name__ == "__main__":
    run_analysis()
