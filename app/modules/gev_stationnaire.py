import streamlit as st
from streamlit_folium import st_folium
import folium

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

import pydeck as pdk
import polars as pl

def filter_nan(df: pl.DataFrame):
    return df.drop_nulls(subset=["xi", "mu", "sigma"])

def filter_percentile_all(df: pl.DataFrame, quantile_choice: float, columns: list[str]):
    from functools import reduce
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
    st.markdown("<h3>Paramètres GEV</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        Echelle = st.selectbox("Choix de l'échelle temporelle", ["Journalière", "Horaire"], key="scale_choice")
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

    echelle = Echelle.lower()
    echelle = "quotidien" if echelle == "journalière" else echelle

    config = load_config(config_path)
    mod_dir = Path(config["gev"]["modelised"]) / echelle
    obs_dir = Path(config["gev"]["observed"]) / echelle

    try:
        df_modelised = pl.read_parquet(mod_dir / "gev_param.parquet")
    except Exception as e:
        st.error(f"Erreur lors du chargement des paramètres modélisés : {e}")
        return

    try:
        df_observed = pl.read_parquet(obs_dir / "gev_param.parquet")
    except Exception as e:
        st.error(f"Erreur lors du chargement des paramètres observés : {e}")
        return
    
    # Suppression des NaN
    df_observed = filter_nan(df_observed)

    df_modelised = add_alti(df_modelised, type='model')
    df_observed = add_alti(df_observed, type=echelle)

    params_gev = {
        "xi": "ξ",
        "mu": "μ",
        "sigma": "σ"
    }

    df_modelised_show = filter_percentile_all(df_modelised, quantile_choice, ["xi", "mu", "sigma"])

    colormap = echelle_config("continu", n_colors=15)
    view_state = pdk.ViewState(latitude=46.9, longitude=1.7, zoom=4.6)
    tooltip = create_tooltip("")

    col1, col2, col3 = st.columns(3)
    for param, label in params_gev.items():
        vmin_modelised = df_modelised_show[param].min()
        df_modelised_leg, vmin, vmax = formalised_legend(df_modelised_show, param, colormap, vmin=vmin_modelised)
        df_observed_leg, _, _ = formalised_legend(df_observed, param, colormap, vmin, vmax)

        layer = create_layer(df_modelised_leg)
        scatter_layer = create_scatter_layer(df_observed_leg, radius=1500)

        deck = plot_map([layer, scatter_layer], view_state, tooltip)
        with eval(f"col{list(params_gev).index(param)+1}"):
            cola, colb = st.columns([0.9, 0.1])
            with cola:
                st.markdown(f"<b>Visualisation de {label}</b>", unsafe_allow_html=True)
                st.pydeck_chart(deck, use_container_width=True, height=450)
            with colb:
                display_vertical_color_legend(500, colormap, vmin, vmax, n_ticks=15, label=label)
    

            
