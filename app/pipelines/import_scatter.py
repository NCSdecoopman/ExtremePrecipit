import streamlit as st
import polars as pl

from app.utils.hist_utils import plot_histogramme, plot_histogramme_comparatif
from app.utils.scatter_plot_utils import generate_scatter_plot_interactive
from app.utils.data_utils import match_and_compare
from app.utils.stats_utils import generate_metrics

def pipeline_scatter(params_load):
    result, stat_choice_key, scale_choice_key, stat_choice, unit_label, height = params_load

    df_modelised_load = result["modelised_load"]
    df_observed_load = result["observed_load"]
    n_tot_mod = df_modelised_load.select(pl.col("NUM_POSTE").n_unique()).item()
    n_tot_obs = df_observed_load.select(pl.col("NUM_POSTE").n_unique()).item()
    
    if stat_choice_key not in ["date", "month"]:
        echelle = "horaire" if scale_choice_key == "mm_h" else "quotidien"
        df_obs_vs_mod = pl.read_csv(f"data/metadonnees/obs_vs_mod/obs_vs_mod_{echelle}.csv")
        obs_vs_mod = match_and_compare(result["observed"], result["modelised"], result["column"], df_obs_vs_mod)
        if obs_vs_mod is not None and obs_vs_mod.height > 0:            
            fig = generate_scatter_plot_interactive(obs_vs_mod, stat_choice, unit_label, height)
            me, mae, rmse, r2 = generate_metrics(obs_vs_mod)

            return n_tot_mod, n_tot_obs, me, mae, rmse, r2, fig

        else:
            fig = plot_histogramme(result["modelised"], result["column"], stat_choice, stat_choice_key, unit_label, height)
            
            return n_tot_mod, n_tot_obs, None, None, None, None, fig
    
    else:
        fig = plot_histogramme_comparatif(result["observed"], result["modelised"], result["column"], stat_choice, stat_choice_key, unit_label, height)
        
        return n_tot_mod, n_tot_obs, None, None, None, None, fig
    