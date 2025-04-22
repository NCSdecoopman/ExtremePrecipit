import streamlit as st

from app.utils.config_utils import load_config, menu_config
from app.utils.menus_utils import menu_statisticals

def pipeline_config(config_path): 
    # Chargement de la configuration
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
        st.info("Les paramètres d’analyse ne sont pas encore définis. Merci de les configurer pour lancer l’analyse.")
        st.stop()

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