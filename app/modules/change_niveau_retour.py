import streamlit as st
import pydeck as pdk
from pathlib import Path

import polars as pl
import numpy as np
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

from app.utils.config_utils import load_config
from app.utils.map_utils import create_layer, create_scatter_layer, plot_map
from app.utils.legends_utils import formalised_legend, display_vertical_color_legend
from app.utils.data_utils import add_metadata

from app.pipelines.import_map import pipeline_map
from app.pipelines.import_scatter import pipeline_scatter

def filter_nan(df: pl.DataFrame, columns: list[str]):
    return df.drop_nulls(subset=columns)

def show(config_path):
    config = load_config(config_path)

    st.title("Carte des changements du niveau de retour (mm/10 ans)")

    # Sélection des paramètres
    col0, col1, col2 = st.columns(3)
    with col0:
        echelle_humaine = st.selectbox("Échelle temporelle", ["Journalière", "Horaire"])
        echelle = "quotidien" if echelle_humaine.lower() == "journalière" else "horaire"
        unit = "mm/j" if echelle == "quotidien" else "mm/h"

    with col1:
        T_choice = st.slider("Choix de la période de retour T (en années)", 10, 100, 20, step=10)

    with col2:
        quantile_choice = st.slider(
            "Percentile de retrait",
            min_value=0.950,
            max_value=1.00,
            value=0.999,
            step=0.001,
            format="%.3f",
            key="quantile_choice"
        )

    # Bornes des années utilisées pour la standardisation
    min_year, max_year = 1960, 2015
    std_year = (max_year - min_year) / 2  # σₜ = (max - min) / 2

    # Lecture des données GEV
    mod_dir = Path(config["gev"]["modelised"]) / echelle
    obs_dir = Path(config["gev"]["observed"]) / echelle

    df_modelised = pl.read_parquet(mod_dir / "gev_param_best_model.parquet")
    df_observed = pl.read_parquet(obs_dir / "gev_param_best_model.parquet")

    df_modelised = filter_nan(df_modelised, "xi") # xi est toujours valable   
    df_observed = filter_nan(df_observed, "xi") # xi est toujours valable

    df_modelised = add_metadata(df_modelised, "mm_h" if echelle == "horaire" else "mm_j", type="modelised")
    df_observed = add_metadata(df_observed, "mm_h" if echelle == "horaire" else "mm_j", type="observed")

    # Remplace les valeurs manquantes (None) par 0 pour les paramètres manquants
    def ensure_mu1_sigma1(df: pl.DataFrame) -> pl.DataFrame:
        cols = df.columns
        df_new = df
        if "mu1" not in cols:
            df_new = df_new.with_columns(pl.lit(0.0).alias("mu1"))
        else:
            df_new = df_new.with_columns(pl.col("mu1").fill_null(0.0))
        if "sigma1" not in cols:
            df_new = df_new.with_columns(pl.lit(0.0).alias("sigma1"))
        else:
            df_new = df_new.with_columns(pl.col("sigma1").fill_null(0.0))
        return df_new

    df_modelised = ensure_mu1_sigma1(df_modelised)
    df_observed = ensure_mu1_sigma1(df_observed)

    column_to_show = "delta_qT"

    # Conversion des dérivées ∂qT/∂t du temps standardisé → temps réel
    # ∂μ/∂tₙᵒʳᵐ = μ₁  donc ∂μ/∂année = μ₁ / σₜ
    # ∂σ/∂année = σ₁ / σₜ
    mu1_natural = pl.col("mu1") / std_year
    sigma1_natural = pl.col("sigma1") / std_year

    # Calcul du terme C_T :  Cₜ = (−log(1−1/T))^(−ξ) − 1
    log_term = -np.log(1 - 1 / T_choice)
    C_T_expr = (log_term ** (-pl.col("xi"))) - 1

    # Variation décennale de qT :
    # ΔqT = (μ₁ + (σ₁ / ξ) × Cₜ) × 10   (dans l’espace temporel réel)
    df_modelised_delta = df_modelised.with_columns([
        ((mu1_natural + (sigma1_natural / pl.col("xi")) * C_T_expr) * 10).alias("delta_qT")
    ])
    df_observed_delta = df_observed.with_columns([
        ((mu1_natural + (sigma1_natural / pl.col("xi")) * C_T_expr) * 10).alias("delta_qT")
    ])


    ### MAP

    df_modelised_show = dont_show_extreme(df_modelised_delta, column_to_show, quantile_choice, stat_choice_key=None)

    # Chargement des affichages graphiques
    height=650
    params_map = (
        "TEST",
        column_to_show,
        df_modelised_show,
        df_observed_delta,
        unit,
        height
    )
    layer, scatter_layer, tooltip, view_state, html_legend = pipeline_map(params_map)
    
    col1, col2, col3 = st.columns([1, 0.15, 1])

    with col1:
        deck = plot_map([layer, scatter_layer], view_state, tooltip)
        # st.markdown(
        #     f"""
        #     <div style='text-align: left; margin-bottom: 10px;'>
        #         <b>{stat_choice} des précipitations de {min_year_choice} à {max_year_choice} ({season_choice.lower()})</b>
        #     </div>
        #     """,
        #     unsafe_allow_html=True
        # )
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
            df_modelised, 
            df_observed, 
            column_to_show, 
            df_modelised_delta, 
            df_modelised_show,  # show
            df_observed_delta, 
            "TEST BIS", 
            "TEST BIS BIS", 
            "Niveau de retour/10 ans",
            unit, 
            height
        )
        pipeline_scatter(params_scatter)
