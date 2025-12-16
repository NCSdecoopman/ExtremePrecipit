import streamlit as st

from app.utils.map_utils import plot_map
from app.pipelines.import_data import pipeline_data_gev
from app.pipelines.import_config import pipeline_config
from app.pipelines.import_map import pipeline_map
from app.pipelines.import_scatter import pipeline_scatter
from app.utils.show_info import show_info_data, show_info_metric


def show(
    config_path: dict,
    show_param: bool=False,
    height: int=600
):
    # Chargement des données
    params_config = pipeline_config(config_path, type="gev", show_param=show_param)
    params_config["stat_choice"] = f"{params_config['param_choice_pres']}"
    
    if params_config["stat_choice"] == "Δqᵀ":
        params_config["unit"] = f"{params_config['unit']}/{params_config['par_X_annees']} ans"
        title = f"Changements du niveau de retour {params_config['T_choice']} ans par {params_config['par_X_annees']} ans du modèle {params_config['model_name_pres']} en {params_config['season_choice'].lower()}"

    elif params_config["stat_choice"] in ["ΔE", "ΔVar", "ΔCV"]:
        base_unit = params_config.get("unit", "")  # ex: "mm/j"
        
        if params_config["stat_choice"] == "ΔE":
            params_config["unit"] = base_unit  # ΔE a la même unité que les données (mm, mm/h…)
        
        elif params_config["stat_choice"] == "ΔVar":
            if base_unit:
                params_config["unit"] = f"{base_unit}²"  # mm²/j²
            else:
                params_config["unit"] = ""  # au cas où unit était vide
        
        elif params_config["stat_choice"] == "ΔCV":
            params_config["unit"] = ""  # CV est adimensionnel

        STAT_TITLES = {
            None: lambda p: f"Changements du niveau de retour {p['T_choice']} ans par {p['par_X_annees']} ans",
            "Δqᵀ": lambda p: f"Changements du niveau de retour {p['T_choice']} ans par {p['par_X_annees']} ans",
            "ΔE":   lambda p: f"Changements de la moyenne des extrêmes par {p['par_X_annees']} ans",
            "ΔVar": lambda p: f"Changements de la variance des extrêmes par {p['par_X_annees']} ans",
            "ΔCV":  lambda p: f"Changements du coefficient de variation des extrêmes par {p['par_X_annees']} ans",
        }

        stat_desc = STAT_TITLES.get(params_config["stat_choice"], lambda p: "Statistique inconnue")(params_config)

        title = (
            f"{stat_desc} du modèle {params_config['model_name_pres']} "
            f"en {params_config['season_choice'].lower()}"
        )

    else:
        params_config["unit"] = ""
        title = f"Changements du niveau de retour 10 ans par 10 ans"
    
    result = pipeline_data_gev(params_config)
    result["stat_choice_key"] = None

    # Chargement des affichages graphiques
    params_map = (
        result["stat_choice_key"],
        result,
        params_config["unit"],
        height
    )
    layer, scatter_layer, tooltip, view_state, html_legend = pipeline_map(params_map)
    
    col1, col2, col3 = st.columns([1, 0.15, 1])

    with col1:
        deck = plot_map([layer, scatter_layer], view_state, tooltip)
        st.markdown(
            f"""
            <div style='text-align: left; margin-bottom: 10px;'>
                <b>{title}</b>
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
            result["stat_choice_key"], 
            params_config["scale_choice_key"], 
            params_config["stat_choice"],
            params_config["unit"], 
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