import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from scipy.stats import genextreme
import matplotlib.cm as cm

from src.utils.data_utils import load_data, years_to_load, cleaning_data_observed
from src.pipelines.pipeline_best_to_niveau_retour import build_x_ttilde

# Configuration
STATION_ID = "45330001"
SEASON = "hydro"
GEV_PARAMS_PATH = Path("data/gev/observed/quotidien/hydro/gev_param_best_model.parquet")
STATS_DIR = Path("data/statisticals/observed/quotidien")
OUTPUT_DIR = Path("add_fig")
BREAK_YEAR = 1985

def load_station_data(station_id):
    mesure, min_y, max_y, len_serie = years_to_load("quotidien", SEASON, str(STATS_DIR))
    df = load_data(str(STATS_DIR), SEASON, "quotidien", ["NUM_POSTE", "max_mm_j", "nan_ratio"], min_y, max_y)
    
    # Apply standard cleaning
    df = cleaning_data_observed(df, "quotidien", len_serie=len_serie)
    
    station_df = df.filter(pl.col("NUM_POSTE") == station_id).sort("year")
    return station_df, min_y, max_y, mesure

def get_params(station_id):
    df = pl.read_parquet(GEV_PARAMS_PATH)
    params = df.filter(pl.col("NUM_POSTE") == station_id).to_dicts()[0]
    return params

def gev_pdf(x, mu, sigma, xi):
    # genextreme in scipy uses c = -xi
    return genextreme.pdf(x, -xi, loc=mu, scale=sigma)

def gev_return_level(T, mu, sigma, xi):
    if T <= 1: return np.nan
    # Formula used in src: zT = mu + (sigma/xi) * ((-log(1-1/T))**(-xi) - 1)
    return mu + (sigma / xi) * ((-np.log(1 - 1/T))**(-xi) - 1)

def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load data
    df_station, min_y, max_y, mesure = load_station_data(STATION_ID)
    params = get_params(STATION_ID)
    
    # 2. Get best model info for build_x_ttilde
    best_model = pl.read_parquet(GEV_PARAMS_PATH).filter(pl.col("NUM_POSTE") == STATION_ID)
    
    # 3. Apply standard normalization build_x_ttilde
    df_normalized = build_x_ttilde(df_station, min_y, max_y, best_model, BREAK_YEAR, mesure)
    
    mu0, mu1 = params['mu0'], params['mu1']
    sigma0, sigma1 = params['sigma0'], params['sigma1']
    xi = params['xi']
    
    years = df_normalized['year'].to_numpy()
    maxima = df_normalized['x'].to_numpy()
    t_tilde = df_normalized['t_tilde'].to_numpy()
    
    # 4. Compute trends
    mu_t = mu0 + mu1 * t_tilde
    sigma_t = sigma0 + sigma1 * t_tilde
    z10_t = [gev_return_level(10, m, s, xi) for m, s in zip(mu_t, sigma_t)]
    
    # 5. Plotting
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(2, 2)
    
    # Panel A: Time Series
    ax_a = fig.add_subplot(gs[0, :])
    ax_a.scatter(years, maxima, color='black', s=15, label='Annual maximum')
    ax_a.plot(years, z10_t, color='red', linestyle='-', label='10-year return level')
    ax_a.set_xlabel('Year')
    ax_a.set_ylabel('Annual maximum (mm)')
    ax_a.set_title(f'A: Time series')
    ax_a.legend()
    ax_a.grid(True, linestyle=':', alpha=0.6)
    
    # Panel B: GEV Densities
    ax_b = fig.add_subplot(gs[1, 0])
    x_range = np.linspace(0, np.max(maxima) * 1.5, 500)
    
    # Years to plot
    years_to_plot = [1970, 1990, 2000, 2010, 2020]
    colors = cm.viridis(np.linspace(0, 1, len(years_to_plot)))
    
    # Helper to get t_tilde for arbitrary years matching build_x_ttilde logic
    # Following src/utils/plot_comparaison_gev.py compute_zT_for_years_VERSION
    years_obs = df_station["year"].to_numpy()
    t_tilde_obs_raw = (years_obs - min_y) / (max_y - min_y)
    t_min_ret = t_tilde_obs_raw.min()
    t_max_ret = t_tilde_obs_raw.max()
    res0_obs = t_tilde_obs_raw / (t_max_ret - t_min_ret)
    dx = res0_obs.min() + 0.5
    
    has_break = "_break_year" in params['model']

    for yr, color in zip(years_to_plot, colors):
        if has_break and yr < BREAK_YEAR:
            tt = -0.5 # t_tilde = 0 - 0.5 if centered. Wait.
            # Let's re-calculate tt exactly as in src
            t_raw = 0.0
        else:
            if has_break:
                t_raw = (yr - BREAK_YEAR) / (max_y - BREAK_YEAR)
            else:
                t_raw = (yr - min_y) / (max_y - min_y)
        
        tt = t_raw / (t_max_ret - t_min_ret) - dx
        
        m = mu0 + mu1 * tt
        s = sigma0 + sigma1 * tt
        pdf = gev_pdf(x_range, m, s, xi)
        ax_b.plot(x_range, pdf, color=color, label=f'{yr}')
        
    ax_b.set_xlabel('Annual maximum (mm)')
    ax_b.set_ylabel('Density (-)')
    ax_b.set_title('B: Associated GEV density plot')
    ax_b.legend(title='Year')
    ax_b.grid(True, linestyle=':', alpha=0.6)
    
    # Panel C: Return Level Plot
    ax_c = fig.add_subplot(gs[1, 1])
    return_periods = np.logspace(0.1, 2.5, 100) # T from ~1.2 to ~300
    
    for yr, color in zip(years_to_plot, colors):
        if has_break and yr < BREAK_YEAR:
            t_raw = 0.0
        else:
            if has_break:
                t_raw = (yr - BREAK_YEAR) / (max_y - BREAK_YEAR)
            else:
                t_raw = (yr - min_y) / (max_y - min_y)
        
        tt = t_raw / (t_max_ret - t_min_ret) - dx
        
        m = mu0 + mu1 * tt
        s = sigma0 + sigma1 * tt
        rls = [gev_return_level(T, m, s, xi) for T in return_periods]
        ax_c.plot(return_periods, rls, color=color, label=f'{yr}')
        
    ax_c.set_xscale('log')
    ax_c.set_xlabel('Return period (years)')
    ax_c.set_ylabel('Return level (mm)')
    ax_c.set_title('C: Associated return level plot')
    ax_c.legend(title='Year')
    ax_c.grid(True, linestyle=':', alpha=0.6)
    ax_c.xaxis.set_major_formatter(plt.ScalarFormatter())

    plt.tight_layout()
    output_path = OUTPUT_DIR / "figure_3_replication.pdf"
    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"Replication plot saved to {output_path}")

if __name__ == "__main__":
    run()
