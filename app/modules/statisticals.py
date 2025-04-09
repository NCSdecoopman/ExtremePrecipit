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

import polars as pl
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


@st.cache_data
def load_data_cached(type_data: str, echelle: str, min_year: int, max_year: int, season_key: str, config) -> pl.DataFrame:
    """
    Version cachée qui retourne un DataFrame Pandas pour la sérialisation.
    """
    df_polars = load_data(type_data, echelle, min_year, max_year, season_key, config)
    return df_polars.to_pandas()


def find_matching_point(df_model: pl.DataFrame, lat_obs: float, lon_obs: float):
    df_model = df_model.with_columns([
        ((pl.col("lat") - lat_obs) ** 2 + (pl.col("lon") - lon_obs) ** 2).sqrt().alias("dist")
    ])
    closest_row = df_model.sort("dist").select(["lat", "lon"]).row(0)
    return closest_row  # (lat, lon)


def match_and_compare(obs_df: pl.DataFrame, mod_df: pl.DataFrame, column_to_show: str) -> pl.DataFrame:
    # Convert to numpy arrays
    obs_coords = np.vstack((obs_df["lat"], obs_df["lon"])).T
    mod_coords = np.vstack((mod_df["lat"], mod_df["lon"])).T
    mod_values = mod_df[column_to_show].to_numpy()

    # Build KDTree
    tree = cKDTree(mod_coords)
    dist, idx = tree.query(obs_coords, k=1)

    matched_data = {
        "lat": obs_df["lat"],
        "lon": obs_df["lon"],
        "pr_obs": obs_df[column_to_show],
        "pr_mod": mod_values[idx]
    }

    return pl.DataFrame(matched_data)

def generate_metrics(df: pl.DataFrame, x_label: str = "pr_mod", y_label: str = "pr_obs"):
    x = df[x_label].to_numpy()
    y = df[y_label].to_numpy()

    rmse = np.sqrt(mean_squared_error(y, x))
    mae = mean_absolute_error(y, x)
    bias = np.mean(x - y)
    r2 = r2_score(y, x)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<b>Biais</b> : {bias:.3f}", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<b>MAE</b> : {mae:.3f}", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<b>RMSE</b> : {rmse:.3f}", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<b>R²</b> : {r2:.3f}", unsafe_allow_html=True)

def generate_scatter_plot_interactive(df: pl.DataFrame, stat_choice: str, unit_label: str, height: int,
                                      x_label: str = "pr_mod", y_label: str = "pr_obs"):
    df_pd = df.select([x_label, y_label]).to_pandas()

    fig = px.scatter(
        df_pd,
        x=x_label,
        y=y_label,
        title="",
        opacity=0.5,
        width=height,
        height=height,
        labels={
            x_label: f"{stat_choice} du modèle AROME ({unit_label})",
            y_label: f"{stat_choice} des stations ({unit_label})"
        },
        hover_data=None
    )

    fig.update_traces(
        hovertemplate=f"{x_label} : %{{x:.1f}}<br>{y_label} : %{{y:.1f}}<extra></extra>"
    )

    min_val = min(df_pd[x_label].min(), df_pd[y_label].min())
    max_val = max(df_pd[x_label].max(), df_pd[y_label].max())

    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='y = x',
            hoverinfo='skip'
        )
    )

    return fig



def show(config_path):
    st.markdown("<h3>Visualisation des précipitations</h3>", unsafe_allow_html=True)
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
        st.warning("Merci de configurer vos paramètres puis de cliquer sur **Lancer l’analyse** pour afficher les résultats.")
        st.stop()  # Stoppe l'exécution ici si pas validé

    stat_choice_key = STATS[stat_choice]
    season_choice_key = SEASON[season_choice]
    scale_choice_key = SCALE[scale_choice]

    try:    
        df_modelised_load = load_data_cached(
            'modelised', 'horaire',
            min_year_choice,
            max_year_choice,
            season_choice_key,
            config
        )
    except Exception as e:
        st.error(f"Erreur lors du chargement des données modélisées : {e}")
        return


    try:
        df_observed_load = load_data_cached(
            'observed', 'horaire' if scale_choice_key == 'mm_h' else 'quotidien',
            min_year_choice,
            max_year_choice,
            season_choice_key,
            config
        )
    except Exception as e:
        st.error(f"Erreur lors du chargement des données observées : {e}")
        return
    

    df_observed_load = pl.from_pandas(df_observed_load) if isinstance(df_observed_load, pd.DataFrame) else df_observed_load
    df_modelised_load = pl.from_pandas(df_modelised_load) if isinstance(df_modelised_load, pd.DataFrame) else df_modelised_load


    # Selection des données observées
    df_observed = cleaning_data_observed(df_observed_load, missing_rate)
    
    # Calcul des statistiques
    result_df_modelised = compute_statistic_per_point(df_modelised_load, stat_choice_key)
    result_df_observed = compute_statistic_per_point(df_observed, stat_choice_key)

    # Obtention de la colonne étudiée
    column_to_show = get_stat_column_name(stat_choice_key, scale_choice_key)
    
    # Retrait des extrêmes
    if stat_choice_key not in ["month", "date"]:
        percentile_95 = result_df_modelised.select(
            pl.col(column_to_show).quantile(quantile_choice, "nearest")
        ).item()

        result_df_modelised = result_df_modelised.filter(
            pl.col(column_to_show) <= percentile_95
        )

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
    
    col1, col2, col3 = st.columns([1.9, 0.5, 2.2])
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
        st.info(f"CP-AROM : {result_df_modelised.shape[0]}/{df_modelised_load.select(['lat', 'lon']).unique().shape[0]} | Stations Météo-France : {result_df_observed.shape[0]}/{df_observed_load.select(['lat', 'lon']).unique().shape[0]}")
        
        if stat_choice_key not in ["date", "month"]:
            obs_vs_mod = match_and_compare(result_df_observed, result_df_modelised, column_to_show)
            
            if obs_vs_mod is not None and obs_vs_mod.height > 0:            
                fig = generate_scatter_plot_interactive(obs_vs_mod, stat_choice, unit_label, height-100)
                st.plotly_chart(fig, use_container_width=True)
                generate_metrics(obs_vs_mod)
            
            else:
                plot_histogramme(result_df_modelised, column_to_show, stat_choice, stat_choice_key, unit_label, height)
        
        else:
            plot_histogramme_comparatif(result_df_observed, result_df_modelised, column_to_show, stat_choice, stat_choice_key, unit_label, height)