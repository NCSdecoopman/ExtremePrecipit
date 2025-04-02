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


@st.cache_data
def load_data_cached(type_data, echelle, min_year, max_year, season_key, config):
    return load_data(type_data, echelle, min_year, max_year, season_key, config)

def cleaning_data_observed(df, nan_limit: int = 0.1):
    df_observed = df.copy()

    # Étape 1 : suppression directe des lignes avec un nan_ratio > 0.1
    df_cleaned = df_observed[df_observed["nan_ratio"] <= nan_limit].copy()

    # Étape 2 : identifier les lat/lon problématiques dans le df original
    bad_coords = df_observed[df_observed["nan_ratio"] > nan_limit][["lat", "lon"]].drop_duplicates()

    # Supprimer toutes les lignes ayant ces lat/lon
    df_cleaned = df_cleaned.merge(bad_coords, on=["lat", "lon"], how="left", indicator=True)
    df_cleaned = df_cleaned[df_cleaned["_merge"] == "left_only"].drop(columns=["_merge"])

    return df_cleaned


def show(config_path):
    st.markdown("<h3>Visualisation des précipitations</h3>", unsafe_allow_html=True)
    config = load_config(config_path)

    STATS, SEASON, SCALE = menu_config()

    min_years = config["years"]["min"]
    max_years = config["years"]["max"]

    stat_choice, min_year_choice, max_year_choice, season_choice, scale_choice, missing_rate = menu_statisticals(
        min_years,
        max_years,
        STATS,
        SEASON
    )

    stat_choice_key = STATS[stat_choice]
    season_choice_key = SEASON[season_choice]
    scale_choice_key = SCALE[scale_choice]

    try:    
        df_modelised = load_data_cached(
            'modelised', 'horaire',
            min_year_choice,
            max_year_choice,
            season_choice_key,
            config
        )
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        return
    
    try:
        df_observed = load_data_cached(
            'observed', 'horaire' if scale_choice_key == 'mm_h' else 'quotidien',
            min_year_choice,
            max_year_choice,
            season_choice_key,
            config
        )
        df_observed = cleaning_data_observed(df_observed, missing_rate)
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        return
    
    # Calcul des statistiques
    result_df_modelised = compute_statistic_per_point(df_modelised, stat_choice_key)
    result_df_observed = compute_statistic_per_point(df_observed, stat_choice_key)

    column_to_show = get_stat_column_name(stat_choice_key, scale_choice_key)

    # Définir l'échelle personnalisée continue
    colormap = echelle_config("continu" if stat_choice_key != "month" else "discret")

    # Normalisation de la légende
    result_df_modelised, vmin, vmax = formalised_legend(result_df_modelised, column_to_show, colormap)

    # Créer le layer Pydeck
    layer = create_layer(result_df_modelised)
    
    # Ajouter les points observés avec la même échelle
    result_df_observed, _, _ = formalised_legend(result_df_observed, column_to_show, colormap, vmin, vmax)
    scatter_layer = create_scatter_layer(result_df_observed, radius=1500)

    # Tooltip
    unit_label = get_stat_unit(stat_choice_key, scale_choice_key)
    tooltip = create_tooltip(stat_choice, unit_label)

    # View de la carte
    view_state = pdk.ViewState(latitude=46.9, longitude=1.7, zoom=5)

    st.write(f"Nombre de stations chargées : {result_df_observed.shape[0]}")
    
    col1, col2, col3 = st.columns([2.8, 0.3, 2.5])
    height = 600

    with col1:
        deck = plot_map([layer, scatter_layer], view_state, tooltip)
        html = deck.to_html(as_string=True, notebook_display=False)

        # Injecte un style CSS pour contrôler la largeur de la carte
        html = html.replace(
            "<head>",
            """
            <head>
            <style>
                body { background-color: white !important; }
                .deckgl-wrapper {
                    width: 600px !important;
                    margin: auto;
                }
                canvas {
                    width: 600px !important;
                }
            </style>
            """
        )

        components.html(html, height=height, scrolling=False)

    with col2:
        display_vertical_color_legend(height, colormap, vmin, vmax, n_ticks=8, label=unit_label)

    with col3:
        plot_histogramme(result_df_modelised, column_to_show, stat_choice, stat_choice_key, unit_label, height)

    #show_scatter_plot(result_df_observed, result_df_modelised)