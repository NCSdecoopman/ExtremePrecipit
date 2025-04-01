import streamlit as st
import streamlit.components.v1 as components

from app.utils.config_utils import *
from app.utils.menus_utils import *
from app.utils.data_utils import *
from app.utils.stats_utils import *
from app.utils.map_utils import *
from app.utils.legends_utils import *

import pydeck as pdk

@st.cache_data
def load_data_cached(min_year, max_year, season_key, config):
    return load_arome_data(min_year, max_year, season_key, config)

def show(config_path):
    st.markdown("<h3>Visualisation des précipitations</h3>", unsafe_allow_html=True)
    config = load_config(config_path)

    STATS, SEASON, SCALE = menu_config()

    min_years = config["years"]["min"]
    max_years = config["years"]["max"]

    stat_choice, min_year_choice, max_year_choice, season_choice, scale_choice = menu_statisticals(
        min_years,
        max_years,
        STATS,
        SEASON
    )

    stat_choice_key = STATS[stat_choice]
    season_choice_key = SEASON[season_choice]
    scale_choice_key = SCALE[scale_choice]

    try:
        df_all = load_data_cached(
            min_year_choice,
            max_year_choice,
            season_choice_key,
            config
        )
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        return

    # Calcul des statistiques
    result_df = compute_statistic_per_point(df_all, stat_choice_key)
    column_to_show = get_stat_column_name(stat_choice_key, scale_choice_key)

    # Définir l'échelle personnalisée continue
    colormap = echelle_config("continu" if stat_choice_key != "month" else "discret")

    # Normalisation de la légende
    result_df, vmin, vmax = formalised_legend(result_df, column_to_show, colormap)

    # Créer le layer Pydeck
    layer = create_layer(result_df)

    # Tooltip
    unit_label = get_stat_unit(stat_choice_key, scale_choice_key)
    tooltip = create_tooltip(stat_choice, unit_label)

    # View de la carte
    view_state = pdk.ViewState(latitude=46.9, longitude=1.7, zoom=5)

    col1, col2, col3 = st.columns([3, 1, 2.5])  # Carte large, légende étroite
    height = 600

    with col1:
        deck = plot_map(layer, view_state, tooltip)
        html = deck.to_html(as_string=True, notebook_display=False)

        # Ajoute une règle CSS directement dans le <head>
        html = html.replace(
            "<head>",
            "<head><style>body { background-color: white; !important; }</style>"
        )

        components.html(html, height=height, scrolling=False)

    with col2:
        display_vertical_color_legend(height, colormap, vmin, vmax, n_ticks=8, label=unit_label)