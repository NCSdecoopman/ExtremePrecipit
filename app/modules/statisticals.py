import streamlit as st
import streamlit.components.v1 as components

from app.utils.config_utils import *
from app.utils.menus_utils import *
from app.utils.data_utils import *
from app.utils.stats_utils import *
from app.utils.map_utils import *
from app.utils.legends_utils import *
from app.utils.hist_utils import *
from app.utils.scatter_plot_utils import *

import pydeck as pdk

import polars as pl
import pandas as pd


@st.cache_data
def load_data_cached(type_data: str, echelle: str, min_year: int, max_year: int, season_key: str, config) -> pl.DataFrame:
    """
    Version cachée qui retourne un DataFrame Pandas pour la sérialisation.
    """
    df_polars = load_data(type_data, echelle, min_year, max_year, season_key, config)
    return df_polars.to_pandas()


def show_info_data(col, label, n_points_valides, n_points_total):
    return col.markdown(f"""
            **{label}**  
            {n_points_valides} / {n_points_total}  
            Tx couverture : {(n_points_valides / n_points_total * 100):.1f}%
            """)

def show_info_metric(col, label, metric):
    return col.markdown(f"""
            **{label}**  
            {metric:.3f}
            """)

def show(config_path):
    st.markdown("<h3>Visualisation des précipitations</h3>", unsafe_allow_html=True)
    config = load_config(config_path)

    STATS, SEASON, SCALE = menu_config()

    min_years = config["years"]["min"]
    max_years = config["years"]["max"]

    params = menu_statisticals(
        min_years,
        max_years,
        STATS,
        SEASON
    )

    if params is not None:
        stat_choice, quantile_choice, min_year_choice, max_year_choice, season_choice, scale_choice, missing_rate = params
    else:
        st.warning("Merci de configurer vos paramètres puis de cliquer sur **Lancer l’analyse** pour afficher les résultats.")
        st.stop()  # Stoppe l'exécution ici si pas validé

    stat_choice_key = STATS[stat_choice]
    season_choice_key = SEASON[season_choice]
    scale_choice_key = SCALE[scale_choice]

    try:    
        df_modelised_load = load_data_cached(
            'modelised', 'horaire',
            min_year_choice,
            max_year_choice,
            season_choice_key,
            config
        )
    except Exception as e:
        st.error(f"Erreur lors du chargement des données modélisées : {e}")
        return


    try:
        df_observed_load = load_data_cached(
            'observed', 'horaire' if scale_choice_key == 'mm_h' else 'quotidien',
            min_year_choice,
            max_year_choice,
            season_choice_key,
            config
        )
    except Exception as e:
        st.error(f"Erreur lors du chargement des données observées : {e}")
        return

    df_observed_load = pl.from_pandas(df_observed_load) if isinstance(df_observed_load, pd.DataFrame) else df_observed_load
    df_modelised_load = pl.from_pandas(df_modelised_load) if isinstance(df_modelised_load, pd.DataFrame) else df_modelised_load

    # Selection des données observées
    df_observed = cleaning_data_observed(df_observed_load, missing_rate)
    
    # Calcul des statistiques
    result_df_modelised = compute_statistic_per_point(df_modelised_load, stat_choice_key)
    result_df_observed = compute_statistic_per_point(df_observed, stat_choice_key)

    # Obtention de la colonne étudiée
    column_to_show = get_stat_column_name(stat_choice_key, scale_choice_key)
    
    # Retrait des extrêmes pour l'affichage uniquement
    if stat_choice_key not in ["month", "date"]:
        percentile_95 = result_df_modelised.select(
            pl.col(column_to_show).quantile(quantile_choice, "nearest")
        ).item()

        result_df_modelised_show = result_df_modelised.filter(
            pl.col(column_to_show) <= percentile_95
        )

    # Ajout de l'altitude
    result_df_modelised_show = add_alti(result_df_modelised_show, type='model')    
    result_df_observed = add_alti(result_df_observed, type='horaire' if scale_choice_key == 'mm_h' else 'quotidien')    

    # Définir l'échelle personnalisée continue
    colormap = echelle_config("continu" if stat_choice_key != "month" else "discret")

    # Normalisation de la légende
    result_df_modelised_show, vmin, vmax = formalised_legend(result_df_modelised_show, column_to_show, colormap)

    # Créer le layer Pydeck
    layer = create_layer(result_df_modelised_show)
    
    # Ajouter les points observés avec la même échelle
    result_df_observed, _, _ = formalised_legend(result_df_observed, column_to_show, colormap, vmin, vmax)
    scatter_layer = create_scatter_layer(result_df_observed, radius=1500)

    # Tooltip
    unit_label = get_stat_unit(stat_choice_key, scale_choice_key)
    tooltip = create_tooltip(unit_label)

    # View de la carte
    view_state = pdk.ViewState(latitude=46.9, longitude=1.7, zoom=5)
    
    col1, col2, col3 = st.columns([1, 0.15, 1])
    height = 600

    with col1:
        deck = plot_map([layer, scatter_layer], view_state, tooltip)
        html = deck.to_html(as_string=True, notebook_display=False)

        st.markdown(
            f"""
            <div style='text-align: left; margin-bottom: 10px;'>
                <b>{stat_choice} des précipitations de {min_year_choice} à {max_year_choice} ({season_choice.lower()})</b>
            </div>
            """,
            unsafe_allow_html=True
        )
        components.html(html, height=height, scrolling=False)
        st.markdown(
            """
            <div style='text-align: left; font-size: 0.8em; color: grey; margin-top: -15px;'>
                Données CP-RCM, 2.5 km, forçage ERA5, réanalyse ECMWF
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        display_vertical_color_legend(height, colormap, vmin, vmax, n_ticks=8, label=unit_label)

    with col3:
        col0bis, col1bis, col2bis, col3bis, col4bis, col5bis, col6bis = st.columns(7)
        show_info_data(col0bis, "CP-AROME map", result_df_modelised_show.shape[0], df_modelised_load.select(['lat', 'lon']).unique().shape[0])
        show_info_data(col1bis, "Stations Météo-France", result_df_observed.shape[0], df_observed_load.select(['lat', 'lon']).unique().shape[0])
       
        if stat_choice_key not in ["date", "month"]:
            obs_vs_mod = match_and_compare(result_df_observed, result_df_modelised, column_to_show)
            
            if obs_vs_mod is not None and obs_vs_mod.height > 0:            
                fig = generate_scatter_plot_interactive(obs_vs_mod, stat_choice, unit_label, height-100)
                st.plotly_chart(fig, use_container_width=True)
                me, mae, rmse, r2 = generate_metrics(obs_vs_mod)
                show_info_data(col2bis, "CP-AROME plot", result_df_modelised.shape[0], df_modelised_load.select(['lat', 'lon']).unique().shape[0])
                show_info_metric(col3bis, "ME", me)
                show_info_metric(col4bis, "MAE", mae)
                show_info_metric(col5bis, "RMSE", rmse)
                show_info_metric(col6bis, "R²", r2)

            
            else:
                st.write("Changer les paramètres afin de générer des stations pour visualiser les scatter plot")
                plot_histogramme(result_df_modelised, column_to_show, stat_choice, stat_choice_key, unit_label, height)
        
        else:                
            plot_histogramme_comparatif(result_df_observed, result_df_modelised, column_to_show, stat_choice, stat_choice_key, unit_label, height)