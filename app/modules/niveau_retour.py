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

import pydeck as pdk
import polars as pl
import numpy as np

from scipy.stats import genextreme

from app.pipelines.import_data import pipeline_data
from app.pipelines.import_map import pipeline_map




# --- Quantile GEV ---
# Soit :
#   μ(t)     = μ₀ + μ₁ × t                  # localisation dépendante du temps
#   σ(t)     = σ₀ + σ₁ × t                  # échelle dépendante du temps
#   ξ        = constante                    # forme
#   T        = période de retour (années)
#   p        = 1 − 1 / T                    # probabilité non-excédée associée

# La quantile notée qᵀ(t) (précipitation pour une période de retour T à l’année t) s’écrit :
#   qᵀ(t) = μ(t) + [σ(t) / ξ] × [ (−log(1 − p))^(−ξ) − 1 ]
#   qᵀ(t) = (μ₀ + μ₁ × t) + [(σ₀ + σ₁ × t) / ξ] × [ (−log(1 − (1/T)))^(−ξ) − 1 ]

def standardize_year(year: float, min_year: int, max_year: int) -> float:
    """
    Centre et réduit une année `year` en utilisant min_year et max_year.
    """
    mean = (min_year + max_year) / 2
    std = (max_year - min_year) / 2
    return (year - mean) / std

import math
def safe_compute_return(row, T_array, t_norm):
    """Force les None à 0 et retourne 0 si résultat invalide."""
    params = {
        "mu0": row["mu0"] or 0,
        "mu1": row["mu1"] or 0,
        "sigma0": row["sigma0"] or 0,
        "sigma1": row["sigma1"] or 0,
        "xi": row["xi"] or 0,
    }

    return compute_return_levels_ns(params, T_array, t_norm)

# Calcul du ratio de stations valides
def compute_valid_ratio(df: pl.DataFrame, param_list: list[str]) -> float:
    n_total = df.height
    n_valid = df.drop_nulls(subset=param_list).height
    return round(n_valid / n_total, 3) if n_total > 0 else 0.0

def filter_nan(df: pl.DataFrame, columns: list[str]):
    return df.drop_nulls(subset=columns)

def filter_percentile_all(df: pl.DataFrame, quantile_choice: float, columns: list[str]):
    quantiles = {
        col: df.select(pl.col(col).quantile(quantile_choice, "nearest")).item()
        for col in columns
    }
    filter_expr = reduce(
        lambda acc, col: acc & (pl.col(col) <= quantiles[col]),
        columns[1:],
        pl.col(columns[0]) <= quantiles[columns[0]]
    )
    return df.filter(filter_expr)

def show(config_path):
    config = load_config(config_path)

    col0, col1, col2, col3 = st.columns(4)
    with col0:
        Echelle = st.selectbox("Choix de l'échelle temporelle", ["Journalière", "Horaire"], key="scale_choice")
        echelle = "quotidien" if Echelle.lower() == "journalière" else "horaire"
        unit = "mm/j" if echelle == "quotidien" else "mm/h"      

    mod_dir = Path(config["gev"]["modelised"]) / echelle
    obs_dir = Path(config["gev"]["observed"]) / echelle

    with col1:
        quantile_choice = st.slider(
            "Percentile de retrait",
            min_value=0.950,
            max_value=1.00,
            value=0.999,
            step=0.001,
            format="%.3f",
            key="quantile_choice"
        )


    min_year_choice = 1960
    max_year_choice = 2015

    with col2:
        year_choice = st.slider(
            "Choix de t",
            min_value=min_year_choice,
            max_value=max_year_choice,
            value=max_year_choice,
            step=1,
            key="year_choice"
        )

    with col3:
        T_choice = st.slider(
            "Choix de T",
            min_value=10,
            max_value=100,
            value=20,
            step=10,
            key="T_choice"
        )

    column_to_show = "qT"

    df_modelised = pl.read_parquet(mod_dir / "gev_param_best_model.parquet")
    df_observed = pl.read_parquet(obs_dir / "gev_param_best_model.parquet")

    df_modelised = filter_nan(df_modelised, "xi") # xi est toujours valable   
    df_observed = filter_nan(df_observed, "xi") # xi est toujours valable

    df_modelised = add_metadata(df_modelised, "mm_h" if echelle == "horaire" else "mm_j", type="modelised")
    df_observed = add_metadata(df_observed, "mm_h" if echelle == "horaire" else "mm_j", type="observed")       

    t_norm = standardize_year(year_choice, min_year_choice, max_year_choice)

    # Période de retour choisie sous forme de tableau numpy (même un seul T)
    T_array = np.array([T_choice])

    # Crée la colonne avec compute_return_levels_ns
    df_modelised = df_modelised.with_columns(
        pl.struct(["mu0", "mu1", "sigma0", "sigma1", "xi"]).map_elements(
            lambda row: safe_compute_return(row, T_array, t_norm),
            return_dtype=pl.Float64
        ).alias(column_to_show)
    )

    df_observed = df_observed.with_columns(
        pl.struct(["mu0", "mu1", "sigma0", "sigma1", "xi"]).map_elements(
            lambda row: safe_compute_return(row, T_array, t_norm),
            return_dtype=pl.Float64
        ).alias(column_to_show)
    )

    df_modelised_show = dont_show_extreme(df_modelised, column_to_show, quantile_choice, stat_choice_key=None)

    colormap = echelle_config("continu", n_colors=15)
    view_state = pdk.ViewState(latitude=46.9, longitude=1.7, zoom=4.6)
    tooltip = create_tooltip("")

    vmin_modelised = df_modelised_show[column_to_show].min()
    df_modelised_leg, vmin, vmax = formalised_legend(df_modelised_show, column_to_show, colormap, vmin=vmin_modelised)
    df_observed_leg, _, _ = formalised_legend(df_observed, column_to_show, colormap, vmin, vmax)

    layer = create_layer(df_modelised_leg)
    scatter_layer = create_scatter_layer(df_observed_leg, radius=1500)
    deck = plot_map([layer, scatter_layer], view_state, tooltip)

    df_obs_vs_mod = pl.read_csv(f"data/metadonnees/obs_vs_mod/obs_vs_mod_{echelle}.csv")
    obs_vs_mod = match_and_compare(df_observed, df_modelised, column_to_show, df_obs_vs_mod)
    
    colmap, colplot = st.columns([0.45, 0.55])

    height=450

    with colmap:
        colmapping, collegend = st.columns([0.85, 0.15])
        with colmapping:
            st.markdown(f"AROME : {df_modelised_leg.height} et stations : {df_observed_leg.height}", unsafe_allow_html=True)
            st.pydeck_chart(deck, use_container_width=True, height=height)


        with collegend:
            html_legend = display_vertical_color_legend(height, colormap, vmin, vmax, n_ticks=15, label=f"Niveau de retour {T_choice} ans ({unit})")
            st.markdown(html_legend, unsafe_allow_html=True)


    with colplot:
        if obs_vs_mod is not None and obs_vs_mod.height > 0:
            fig = generate_scatter_plot_interactive(obs_vs_mod, f"Niveau de retour {T_choice} ans", unit, height)
            me, mae, rmse, r2 = generate_metrics(obs_vs_mod)
            col000, col0m, colm1, colm2, colm3, colm4 = st.columns(6)
            with col0m:
                st.markdown(f"<b>Points comparés</b><br>{obs_vs_mod.shape[0]}", unsafe_allow_html=True)
            show_info_metric(colm1, "ME", me)
            show_info_metric(colm2, "MAE", mae)
            show_info_metric(colm3, "RMSE", rmse)
            show_info_metric(colm4, "r²", r2)
            
            # Affiche le graphique avec mode sélection activé
            st.plotly_chart(fig)
