import streamlit as st

from app.utils.config_utils import load_config, menu_config_statisticals, menu_config_gev
from app.utils.menus_utils import menu_statisticals, menu_gev

def pipeline_config(config_path: str, type: str): 
    # Chargement de la configuration
    config = load_config(config_path)

    min_years = config["years"]["min"]
    max_years = config["years"]["max"]

    if type == "stat":
        STATS, SEASON, SCALE = menu_config_statisticals()

        params = menu_statisticals(
            min_years,
            max_years,
            STATS,
            SEASON
        )

        if params is None:
            st.info("Les paramètres d’analyse ne sont pas encore définis. Merci de les configurer pour lancer l’analyse.")
            st.stop()

        stat_choice, quantile_choice, min_year_choice, max_year_choice, season_choice, scale_choice, missing_rate = params

        return {
            "config": config,
            "stat_choice": stat_choice,
            "season_choice": season_choice,
            "stat_choice_key": STATS[stat_choice],
            "scale_choice_key": SCALE[scale_choice],
            "min_year_choice": min_year_choice,
            "max_year_choice": max_year_choice,
            "season_choice_key": SEASON[season_choice],
            "missing_rate": missing_rate,
            "quantile_choice": quantile_choice
        }

    elif type == "gev":
        MODEL_PARAM, MODEL_NAME = menu_config_gev()

        params = menu_gev(
            config,
            MODEL_NAME,
            MODEL_PARAM,
            show_param=False
        )

        if params is None:
            st.info("Les paramètres d’analyse ne sont pas encore définis. Merci de les configurer pour lancer l’analyse.")
            st.stop()        

        return {
            "config": config,
            **params
        }
