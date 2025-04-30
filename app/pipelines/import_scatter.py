import streamlit as st
import polars as pl

from app.utils.hist_utils import plot_histogramme, plot_histogramme_comparatif
from app.utils.scatter_plot_utils import generate_scatter_plot_interactive
from app.utils.show_info import show_info_data, show_info_metric
from app.utils.data_utils import match_and_compare
from app.utils.stats_utils import generate_metrics

def pipeline_scatter(params_load):
    df_modelised_load, df_observed_load, column_to_show, result_df_modelised, result_df_modelised_show, result_df_observed, stat_choice_key, scale_choice_key, stat_choice,unit_label, height = params_load

    col0bis, col1bis, col2bis, col3bis, col4bis, col5bis, col6bis = st.columns(7)
    n_tot_mod = df_modelised_load.select(pl.col("NUM_POSTE").n_unique()).item()
    n_tot_obs = df_observed_load.select(pl.col("NUM_POSTE").n_unique()).item()
    show_info_data(col0bis, "CP-AROME map", result_df_modelised_show.shape[0], n_tot_mod)
    show_info_data(col1bis, "Stations", result_df_observed.shape[0], n_tot_obs)
    
    if stat_choice_key not in ["date", "month"]:
        echelle = "horaire" if scale_choice_key == "mm_h" else "quotidien"
        df_obs_vs_mod = pl.read_csv(f"data/metadonnees/obs_vs_mod/obs_vs_mod_{echelle}.csv")
        obs_vs_mod = match_and_compare(result_df_observed, result_df_modelised, column_to_show, df_obs_vs_mod)
        if obs_vs_mod is not None and obs_vs_mod.height > 0:            
            fig = generate_scatter_plot_interactive(obs_vs_mod, stat_choice, unit_label, height)
            st.plotly_chart(fig, use_container_width=True)
            me, mae, rmse, r2 = generate_metrics(obs_vs_mod)
            show_info_data(col2bis, "CP-AROME plot", result_df_modelised.shape[0], n_tot_mod)
            show_info_metric(col3bis, "ME", me)
            show_info_metric(col4bis, "MAE", mae)
            show_info_metric(col5bis, "RMSE", rmse)
            show_info_metric(col6bis, "r²", r2)

        else:
            st.write("Changer les paramètres afin de générer des stations pour visualiser les scatter plot")
            plot_histogramme(result_df_modelised, column_to_show, stat_choice, stat_choice_key, unit_label, height)
    
    else:                
        plot_histogramme_comparatif(result_df_observed, result_df_modelised, column_to_show, stat_choice, stat_choice_key, unit_label, height)

    return me, r2