import sys
import os
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Force non-interactive backend
import matplotlib
matplotlib.use("Agg")

# Add project root to path
sys.path.append(os.getcwd())

from src.utils.data_utils import load_data, cleaning_data_observed
from src.pipelines.pipeline_stats_to_gev import gev_non_stationnaire

# Configuration
INPUT_ROOT = Path("data/statisticals/observed/horaire")
GEV_ROOT = Path("data/gev/observed/horaire")
ECHELLE = "horaire"
MESURE = "max_mm_h"
MIN_YEAR_GLOBAL = 1990
MAX_YEAR_GLOBAL = 2022
LEN_SERIE = 25
T_RET = 10
BREAK_YEAR = 1985

MONTHS_FR = ["jan", "fev", "mar", "avr", "mai", "jui", "juill", "aou", "sep", "oct", "nov", "dec"]
MONTHS_EN = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

CACHE_DIR = Path("add_fig")
CACHE_RATIO_PATH = CACHE_DIR / "boxplot_robustness_all_months_ratio_signif.parquet"


def get_year_norm_fit(df, min_year, max_year, model_name):
    has_break = "_break_year" in model_name
    if has_break:
        return np.where(df["year"] < BREAK_YEAR, 0, (df["year"] - BREAK_YEAR) / (max_year - BREAK_YEAR))
    return (df["year"] - min_year) / (max_year - min_year)


def get_t_tilde_for_range(years, obs_years, min_year_global, max_year_global):
    t_raw = (years - min_year_global) / (max_year_global - min_year_global)
    t_obs_raw = (obs_years - min_year_global) / (max_year_global - min_year_global)
    t_min_obs, t_max_obs = t_obs_raw.min(), t_obs_raw.max()
    if t_max_obs == t_min_obs:
        return np.full_like(t_raw, -0.5)
    res0_raw = t_raw / (t_max_obs - t_min_obs)
    res0_obs = t_obs_raw / (t_max_obs - t_min_obs)
    dx = res0_obs.min() + 0.5
    return res0_raw - dx


def compute_rl(params, t_norm, T=10):
    mu0, mu1, sigma0, sigma1, xi = params["mu0"], params["mu1"], params["sigma0"], params["sigma1"], params["xi"]
    mu_t = mu0 + mu1 * t_norm
    sigma_t = sigma0 + sigma1 * t_norm
    if xi != 0:
        CT = ((-np.log(1 - 1 / T)) ** (-xi) - 1)
        return mu_t + (sigma_t / xi) * CT
    return mu_t - sigma_t * np.log(-np.log(1 - 1 / T))


def fit_scrupulous(df, col_val, model_name, init_params=None):
    df = df.copy()
    df["year_norm"] = get_year_norm_fit(df, MIN_YEAR_GLOBAL, MAX_YEAR_GLOBAL, model_name)
    try:
        res = gev_non_stationnaire(df, col_val, model_name, init_params=init_params)
        if res and np.all(np.isfinite(res)):
            return res
    except Exception:
        pass
    return None


def params_to_dict(res, model_name):
    if res is None:
        return None
    if model_name in ["ns_gev_m1", "s_gev", "ns_gev_m1_break_year"]:
        return {"mu0": res[0], "mu1": res[1], "sigma0": res[2], "sigma1": 0.0, "xi": res[3]}
    if model_name in ["ns_gev_m2", "ns_gev_m2_break_year"]:
        return {"mu0": res[0], "mu1": 0.0, "sigma0": res[1], "sigma1": res[2], "xi": res[3]}
    if model_name in ["ns_gev_m3", "ns_gev_m3_break_year"]:
        return {"mu0": res[0], "mu1": res[1], "sigma0": res[2], "sigma1": res[3], "xi": res[4]}
    return None


def _process_station(task):
    station, data_station, row_best = task
    if data_station is None or data_station.empty or row_best is None:
        return None

    model_name = row_best["model"]
    init_p = {k: row_best[k] for k in ["mu0", "mu1", "sigma0", "sigma1", "xi"]}
    obs_years = data_station["year"].values

    t_95 = get_t_tilde_for_range(np.array([1995]), obs_years, MIN_YEAR_GLOBAL, MAX_YEAR_GLOBAL)[0]
    t_22 = get_t_tilde_for_range(np.array([2022]), obs_years, MIN_YEAR_GLOBAL, MAX_YEAR_GLOBAL)[0]

    res_full = fit_scrupulous(data_station, MESURE, model_name, init_params=init_p)
    if res_full is None:
        return None
    p_f = params_to_dict(res_full, model_name)
    if p_f is None:
        return None
    rl22_f, rl95_f = compute_rl(p_f, t_22, T_RET), compute_rl(p_f, t_95, T_RET)
    if rl95_f == 0 or not np.isfinite(rl22_f) or not np.isfinite(rl95_f):
        return None
    tf = (rl22_f - rl95_f) / rl95_f * 100

    data_nm = data_station.drop(data_station[MESURE].idxmax())
    res_nm = fit_scrupulous(data_nm, MESURE, model_name, init_params=init_p)
    if res_nm is None:
        return None
    p_nm = params_to_dict(res_nm, model_name)
    if p_nm is None:
        return None
    rl22_nm, rl95_nm = compute_rl(p_nm, t_22, T_RET), compute_rl(p_nm, t_95, T_RET)
    if rl95_nm == 0 or not np.isfinite(rl22_nm) or not np.isfinite(rl95_nm):
        return None
    tnm = (rl22_nm - rl95_nm) / rl95_nm * 100

    if tnm == 0 or not np.isfinite(tnm):
        return None
    return station, tf, tnm, (tf / tnm)


def run_diff_plot():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Total significant stations per month (from niveau_retour.parquet)
    signif_counts: dict[str, int] = {}
    for m_fr, m_en in zip(MONTHS_FR, MONTHS_EN):
        signif_path = GEV_ROOT / m_fr / "niveau_retour.parquet"
        if not signif_path.exists():
            signif_counts[m_en] = 0
            continue
        df_signif = pl.read_parquet(signif_path, columns=["NUM_POSTE", "significant"])
        signif_counts[m_en] = (
            df_signif.filter(pl.col("significant") == True)
                     .select(pl.col("NUM_POSTE").cast(pl.Utf8))
                     .unique()
                     .height
        )

    if CACHE_RATIO_PATH.exists():
        res_df = pd.read_parquet(CACHE_RATIO_PATH)
    else:
        all_rows = []
        for m_fr, m_en in zip(MONTHS_FR, MONTHS_EN):
            print(f"\nProcessing {m_en}...")
            best_model_path = GEV_ROOT / m_fr / "gev_param_best_model.parquet"
            signif_path = GEV_ROOT / m_fr / "niveau_retour.parquet"
            if not best_model_path.exists() or not signif_path.exists():
                continue

            best_models_df = pl.read_parquet(best_model_path)
            df_signif = pl.read_parquet(signif_path, columns=["NUM_POSTE", "significant"])
            signif_stations = set(
                df_signif.filter(pl.col("significant") == True)
                         .select(pl.col("NUM_POSTE").cast(pl.Utf8))
                         .get_column("NUM_POSTE")
                         .to_list()
            )

            cols = ["NUM_POSTE", MESURE, "nan_ratio"]
            df = load_data(str(INPUT_ROOT), m_fr, ECHELLE, cols, MIN_YEAR_GLOBAL, MAX_YEAR_GLOBAL)
            df = cleaning_data_observed(df, ECHELLE, LEN_SERIE)
            df_pd = df.to_pandas()
            df_pd = df_pd[df_pd["NUM_POSTE"].astype(str).isin(signif_stations)]

            stations = df_pd["NUM_POSTE"].unique()
            best_by_station = {row["NUM_POSTE"]: row for row in best_models_df.to_dicts()}

            tasks = []
            for station in stations:
                row_best = best_by_station.get(station)
                if row_best is None:
                    continue
                data_station = df_pd[df_pd["NUM_POSTE"] == station].sort_values("year")
                if data_station.empty:
                    continue
                tasks.append((station, data_station, row_best))

            with ProcessPoolExecutor(max_workers=96) as executor:
                futures = {executor.submit(_process_station, task): task[0] for task in tasks}
                progress_bar = tqdm(total=len(futures), desc=m_en)

                for future in as_completed(futures):
                    res = future.result()
                    if res is None:
                        progress_bar.update(1)
                        continue
                    station, tf, tnm, ratio = res
                    all_rows.append(
                        {
                            "Month": m_en,
                            "NUM_POSTE": str(station),
                            "trend_full": tf,
                            "trend_no_max": tnm,
                            "ratio": ratio,
                        }
                    )
                    progress_bar.update(1)

                progress_bar.close()

        res_df = pd.DataFrame(all_rows)
        res_df.to_parquet(CACHE_RATIO_PATH, index=False)

    fig, ax = plt.subplots(figsize=(12, 5))

    data_ratio = [res_df[res_df["Month"] == m]["ratio"] for m in MONTHS_EN]
    pos = np.arange(len(MONTHS_EN))

    bp = ax.boxplot(
        data_ratio,
        positions=pos,
        widths=0.55,
        patch_artist=True,
        boxprops=dict(facecolor="lightgray", color="black"),
        medianprops=dict(color="black", linewidth=1.5),
        showfliers=False,
    )

    # Add n per month above each boxplot (use whisker top)
    def whisker_top(boxplot_dict, idx):
        whiskers = boxplot_dict["whiskers"][2 * idx : 2 * idx + 2]
        return max(np.max(w.get_ydata()) for w in whiskers)

    y0, y1 = ax.get_ylim()
    y_range = y1 - y0
    y_pad = max(0.02, 0.04 * y_range) if np.isfinite(y_range) else 0.02

    for i, m in enumerate(MONTHS_EN):
        n = signif_counts.get(m, 0)
        if n == 0:
            continue
        ymax = whisker_top(bp, i)
        ax.annotate(
            f"n={n}",
            xy=(i, ymax),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            annotation_clip=False,
        )
    ax.set_ylim(top=y1 + y_pad)

    ax.set_xticks(pos)
    ax.set_xticklabels(MONTHS_EN, rotation=45)
    ax.set_xlabel("")
    ax.set_ylabel("Ratio of trends = complete series / series without maximum value")
    ax.set_title("")

    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
    ax.grid(axis="y", which="major", linestyle="--", alpha=0.8)
    ax.grid(axis="y", which="minor", linestyle=":", alpha=0.5)
    ax.axhline(1, color="black", linestyle="--", linewidth=1)

    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.18, top=0.90)
    output_path = CACHE_DIR / "boxplot_robustness_all_months_ratio_signif.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    run_diff_plot()
