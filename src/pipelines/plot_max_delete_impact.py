import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

# Force non-interactive backend BEFORE importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Add project root to path
sys.path.append(os.getcwd())

from src.utils.data_utils import load_data, cleaning_data_observed
from src.pipelines.pipeline_stats_to_gev import gev_non_stationnaire

# Initialize logger used inside pipeline_stats_to_gev
import src.pipelines.pipeline_stats_to_gev as psg
from src.utils.logger import get_logger
psg.logger = get_logger("fit_scrupulous")

# -----------------------
# Configuration
# -----------------------
INPUT_DIR = Path("data/statisticals/observed/horaire")
GEV_ROOT = Path("data/gev/observed/horaire")

ECHELLE = "horaire"
MESURE = "max_mm_h"

MIN_YEAR_GLOBAL = 1990
MAX_YEAR_GLOBAL = 2022
LEN_SERIE = 25

T_RET = 10
BREAK_YEAR = 1985

TREND_MIN = 200
TREND_MAX = 300
TARGET_STATIONS = 50

MONTHS_FR = ["jan", "fev", "mar", "avr", "mai", "jui", "juill", "aou", "sep", "oct", "nov", "dec"]
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

# -----------------------
# Styling
# -----------------------
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


def get_year_norm_fit(df: pd.DataFrame, min_year: int, max_year: int, model_name: str) -> np.ndarray:
    """year_norm used by gev_non_stationnaire."""
    has_break = "_break_year" in model_name
    if has_break:
        return np.where(
            df["year"].to_numpy() < BREAK_YEAR,
            0.0,
            (df["year"].to_numpy() - BREAK_YEAR) / (max_year - BREAK_YEAR),
        )
    return (df["year"].to_numpy() - min_year) / (max_year - min_year)


def get_t_tilde_for_range(
    years: np.ndarray,
    obs_years: np.ndarray,
    min_year_global: int,
    max_year_global: int,
) -> np.ndarray:
    """
    Map arbitrary year grid to t-tilde consistent with the observed-year support,
    as used in your earlier plotting logic.
    """
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
    """Return level at return period T for a (possibly) time-varying GEV."""
    mu0, mu1 = params["mu0"], params["mu1"]
    sigma0, sigma1 = params["sigma0"], params["sigma1"]
    xi = params["xi"]

    mu_t = mu0 + mu1 * t_norm
    sigma_t = sigma0 + sigma1 * t_norm

    # Guard against invalid sigma
    if not np.isfinite(sigma_t) or sigma_t <= 0:
        return np.nan

    if xi != 0:
        CT = ((-np.log(1 - 1 / T)) ** (-xi) - 1)
        return float(mu_t + (sigma_t / xi) * CT)

    return float(mu_t - sigma_t * np.log(-np.log(1 - 1 / T)))


def fit_scrupulous(
    df: pd.DataFrame,
    col_val: str,
    model_name: str,
    init_params: dict | None = None,
) -> np.ndarray | None:
    """
    Fit model; return raw parameter vector from gev_non_stationnaire, or None on failure.
    """
    df = df.copy()
    df["year_norm"] = get_year_norm_fit(df, MIN_YEAR_GLOBAL, MAX_YEAR_GLOBAL, model_name)

    try:
        res = gev_non_stationnaire(df, col_val, model_name, init_params=init_params)
        if res is None:
            return None
        arr = np.asarray(res, dtype=float)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            return None
        return arr
    except Exception:
        return None


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


def pick_stations_by_month() -> list[tuple[str, str]]:
    """
    Select up to TARGET_STATIONS stations across months whose z_T_p is in [TREND_MIN, TREND_MAX]
    and significant=True. Round-robin across months, unique stations first.
    """
    candidates_by_month: dict[str, list[str]] = {}

    for month in MONTHS_FR:
        path_nr = GEV_ROOT / month / "niveau_retour.parquet"
        if not path_nr.exists():
            continue

        df = pl.read_parquet(path_nr, columns=["NUM_POSTE", "z_T_p", "significant"])
        df = df.filter(
            (pl.col("z_T_p") >= TREND_MIN) &
            (pl.col("z_T_p") <= TREND_MAX) &
            (pl.col("significant") == True)
        )

        stations = (
            df.select(pl.col("NUM_POSTE").cast(pl.Utf8))
              .get_column("NUM_POSTE")
              .to_list()
        )

        # Copy list so we can pop safely
        candidates_by_month[month] = list(stations)

    selected: list[tuple[str, str]] = []
    used_stations: set[str] = set()

    # Round-robin across months, unique stations first
    while len(selected) < TARGET_STATIONS:
        progressed = False

        for month in MONTHS_FR:
            stations = candidates_by_month.get(month, [])
            while stations and stations[0] in used_stations:
                stations.pop(0)

            if stations:
                station = stations.pop(0)
                selected.append((month, station))
                used_stations.add(station)
                progressed = True

                if len(selected) >= TARGET_STATIONS:
                    break

        if not progressed:
            break

    # If still short, allow duplicates
    if len(selected) < TARGET_STATIONS:
        for month in MONTHS_FR:
            stations = candidates_by_month.get(month, [])
            for station in stations:
                selected.append((month, station))
                if len(selected) >= TARGET_STATIONS:
                    break
            if len(selected) >= TARGET_STATIONS:
                break

    return selected


def run_analysis():
    output_dir = Path("add_fig")
    output_dir.mkdir(parents=True, exist_ok=True)
    multi_pdf_path = output_dir / "impact_max_all_stations.pdf"

    selections = pick_stations_by_month()
    if not selections:
        print("Aucune station trouvée dans l'intervalle de tendance demandé.")
        return

    # Cache per-month loaded datasets (best model params + cleaned obs dataframe)
    month_cache: dict[str, tuple[pl.DataFrame, pd.DataFrame]] = {}

    with PdfPages(multi_pdf_path) as pdf:
        for month, station in selections:
            if month not in month_cache:
                best_path = GEV_ROOT / month / "gev_param_best_model.parquet"
                if not best_path.exists():
                    continue

                best_models_df = pl.read_parquet(best_path)

                cols = ["NUM_POSTE", MESURE, "nan_ratio"]
                df = load_data(str(INPUT_DIR), month, ECHELLE, cols, MIN_YEAR_GLOBAL, MAX_YEAR_GLOBAL)
                df = cleaning_data_observed(df, ECHELLE, LEN_SERIE)
                df_pd = df.to_pandas()

                month_cache[month] = (best_models_df, df_pd)
            else:
                best_models_df, df_pd = month_cache[month]

            data_station = df_pd[df_pd["NUM_POSTE"] == station].sort_values("year")
            if data_station.empty:
                continue

            obs_years = data_station["year"].to_numpy()

            row_best = best_models_df.filter(pl.col("NUM_POSTE") == station).to_dicts()
            if not row_best:
                continue
            row_best = row_best[0]

            model_name = row_best["model"]
            init_p = {k: row_best[k] for k in ["mu0", "mu1", "sigma0", "sigma1", "xi"]}

            print(f"Traitement station {station} ({month}, Modèle: {model_name})...")

            params_full_raw = fit_scrupulous(data_station, MESURE, model_name, init_params=init_p)
            idx_max = int(data_station[MESURE].idxmax())
            data_no_max = data_station.drop(idx_max)
            params_no_max_raw = fit_scrupulous(data_no_max, MESURE, model_name, init_params=init_p)

            if params_full_raw is None or params_no_max_raw is None:
                continue

            params_full = params_to_dict(params_full_raw, model_name)
            params_no_max = params_to_dict(params_no_max_raw, model_name)
            if params_full is None or params_no_max is None:
                continue

            fig, ax = plt.subplots(figsize=(10, 6))

            # Observations
            ax.plot(
                data_station["year"],
                data_station[MESURE],
                color="black",
                marker="o",
                linestyle="-",
                alpha=0.3,
                label="Observations",
            )

            # Highlight maximum
            ax.scatter(
                [data_station.loc[idx_max, "year"]],
                [data_station.loc[idx_max, MESURE]],
                color="red",
                s=100,
                edgecolors="black",
                zorder=5,
                label=f"Maximum ({int(data_station.loc[idx_max, 'year'])})",
            )

            years_grid = np.linspace(MIN_YEAR_GLOBAL, MAX_YEAR_GLOBAL, 100)
            t_tilde_grid = get_t_tilde_for_range(years_grid, obs_years, MIN_YEAR_GLOBAL, MAX_YEAR_GLOBAL)

            rl_full = np.array([compute_rl(params_full, t, T=T_RET) for t in t_tilde_grid], dtype=float)
            rl_no_max = np.array([compute_rl(params_no_max, t, T=T_RET) for t in t_tilde_grid], dtype=float)

            if not np.any(np.isfinite(rl_full)) or not np.any(np.isfinite(rl_no_max)):
                plt.close(fig)
                continue

            ax.plot(years_grid, rl_full, color="#0055aa", linewidth=3, label=f"Niveau de retour T={T_RET} (Complet)")
            ax.plot(
                years_grid,
                rl_no_max,
                color="#00aa55",
                linewidth=3,
                linestyle="--",
                label=f"Niveau de retour T={T_RET} (Sans max)",
            )

            # Trend calculation (1995 -> 2022)
            t_tilde_1995 = get_t_tilde_for_range(np.array([1995.0]), obs_years, MIN_YEAR_GLOBAL, MAX_YEAR_GLOBAL)[0]
            t_tilde_2022 = get_t_tilde_for_range(np.array([2022.0]), obs_years, MIN_YEAR_GLOBAL, MAX_YEAR_GLOBAL)[0]

            rl1995_full = compute_rl(params_full, t_tilde_1995, T=T_RET)
            rl2022_full = compute_rl(params_full, t_tilde_2022, T=T_RET)
            rl1995_no_max = compute_rl(params_no_max, t_tilde_1995, T=T_RET)
            rl2022_no_max = compute_rl(params_no_max, t_tilde_2022, T=T_RET)

            # Guard divisions
            if not np.isfinite(rl1995_full) or rl1995_full == 0 or not np.isfinite(rl2022_full):
                plt.close(fig)
                continue
            if not np.isfinite(rl1995_no_max) or rl1995_no_max == 0 or not np.isfinite(rl2022_no_max):
                plt.close(fig)
                continue

            trend_full = (rl2022_full - rl1995_full) / rl1995_full * 100.0
            trend_no_max = (rl2022_no_max - rl1995_no_max) / rl1995_no_max * 100.0

            # Keep only cases where removing max decreases the trend
            if trend_full <= trend_no_max:
                plt.close(fig)
                continue

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

            month_label = MONTH_LABELS_FR.get(month, month)
            ax.set_title(f"Station {station} - {month_label} - Echelle horaire")
            ax.set_xlabel("Année")
            ax.set_ylabel("Précipitation (mm/h)")

            ax.legend(loc="lower right", framealpha=0.9)
            plt.tight_layout()

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


if __name__ == "__main__":
    run_analysis()
