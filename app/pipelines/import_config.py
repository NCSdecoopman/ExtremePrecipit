import streamlit as st

from app.utils.config_utils import load_config, menu_config_statisticals, menu_config_gev
from app.utils.menus_utils import menu_statisticals, menu_gev




import streamlit as st

from pathlib import Path
from functools import reduce

from app.utils.config_utils import *
from app.utils.menus_utils import *
from app.utils.data_utils import *
from app.utils.stats_utils import *
from app.utils.map_utils import *
from app.utils.legends_utils import *
from app.utils.hist_utils import *
from app.utils.scatter_plot_utils import *
from app.utils.show_info import show_info_metric
from app.utils.gev_utils import compute_return_levels_ns

import pydeck as pdk
import polars as pl
import numpy as np

from app.pipelines.import_data import pipeline_data
from app.pipelines.import_map import pipeline_map

from app.utils.data_utils import standardize_year, filter_nan



def pipeline_config(config_path: str, type: str, show_param: bool=False): 
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
        _, SEASON, _ = menu_config_statisticals()

        params = menu_gev(
            config,
            MODEL_NAME,
            MODEL_PARAM,
            SEASON,
            show_param=show_param
        )

        if params is None:
            st.info("Les paramètres d’analyse ne sont pas encore définis. Merci de les configurer pour lancer l’analyse.")
            st.stop()        

        return {
            "config": config,
            **params
        }
