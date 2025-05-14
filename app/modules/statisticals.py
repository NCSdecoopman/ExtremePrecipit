import streamlit as st

from app.utils.map_utils import plot_map
from app.utils.legends_utils import get_stat_unit

from app.pipelines.import_data import pipeline_data
from app.pipelines.import_config import pipeline_config
from app.pipelines.import_map import pipeline_map
from app.pipelines.import_scatter import pipeline_scatter
from app.utils.show_info import show_info_data, show_info_metric

def show(
    config_path: dict, 
    height: int=600
):
    st.markdown("<h3>Visualisation des précipitations</h3>", unsafe_allow_html=True)
    
    # Chargement des config
    params_config = pipeline_config(config_path)
    config = params_config["config"]
    stat_choice = params_config["stat_choice"]
    season_choice = params_config["season_choice"]
    stat_choice_key = params_config["stat_choice_key"]
    scale_choice_key = params_config["scale_choice_key"]
    min_year_choice = params_config["min_year_choice"]
    max_year_choice = params_config["max_year_choice"]
    season_choice_key = params_config["season_choice_key"]
    missing_rate = params_config["missing_rate"]
    quantile_choice = params_config["quantile_choice"]
    
    # Préparation des paramètres pour pipeline_data
    params_load = (
        stat_choice_key,
        scale_choice_key,
        min_year_choice,
        max_year_choice,
        season_choice_key,
        missing_rate,
        quantile_choice
    )

    # Obtention des données
    result = pipeline_data(params_load, config, use_cache=True)

    # Chargement des affichages graphiques
    unit_label = get_stat_unit(stat_choice_key, scale_choice_key)
    params_map = (
        stat_choice_key,
        result,
        unit_label,
        height
    )
    layer, scatter_layer, tooltip, view_state, html_legend = pipeline_map(params_map)
    
    col1, col2, col3 = st.columns([1, 0.15, 1])

    with col1:
        deck = plot_map([layer, scatter_layer], view_state, tooltip)
        st.markdown(
            f"""
            <div style='text-align: left; margin-bottom: 10px;'>
                <b>{stat_choice} des précipitations de {min_year_choice} à {max_year_choice} ({season_choice.lower()})</b>
            </div>
            """,
            unsafe_allow_html=True
        )
        if deck:
            st.pydeck_chart(deck, use_container_width=True, height=height)
        st.markdown(
            """
            <div style='text-align: left; font-size: 0.8em; color: grey; margin-top: 0px;'>
                Données CP-RCM, 2.5 km, forçage ERA5, réanalyse ECMWF
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(html_legend, unsafe_allow_html=True)        

    with col3:
        params_scatter = (
            result,
            stat_choice_key, 
            scale_choice_key, 
            stat_choice,unit_label, 
            height
        )
        n_tot_mod, n_tot_obs, me, mae, rmse, r2, scatter = pipeline_scatter(params_scatter)

        col0bis, col1bis, col2bis, col3bis, col4bis, col5bis, col6bis = st.columns(7)

        show_info_data(col0bis, "CP-AROME map", result["modelised_show"].shape[0], n_tot_mod)
        show_info_data(col1bis, "Stations", result["observed_show"].shape[0], n_tot_obs)
        show_info_data(col2bis, "CP-AROME plot", result["modelised"].shape[0], n_tot_mod)
        show_info_metric(col3bis, "ME", me)
        show_info_metric(col4bis, "MAE", mae)
        show_info_metric(col5bis, "RMSE", rmse)
        show_info_metric(col6bis, "r²", r2)

        st.plotly_chart(scatter, use_container_width=True)