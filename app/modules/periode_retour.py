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

def find_matching_point_fast(df: pl.DataFrame, lat_obs: float, lon_obs: float) -> tuple[float, float]:
    df_with_dist = df.with_columns(
        (
            (pl.col("lat") - lat_obs).pow(2) + (pl.col("lon") - lon_obs).pow(2)
        ).sqrt().alias("distance")
    )
    idx = df_with_dist["distance"].arg_min()
    row = df_with_dist[idx]

    return row["lat"], row["lon"]

def show(config_path):
    st.markdown("<h3>Paramètres GEV</h3>", unsafe_allow_html=True)

    Echelle = st.selectbox("Choix de l'échelle temporelle", ["Journalière", "Horaire"], key="scale_choice")
    echelle = Echelle.lower()
    echelle = "quotidien" if echelle == "journalière" else echelle

    config = load_config(config_path)
    mod_dir = Path(config["gev"]["modelised"]) / echelle
    obs_dir = Path(config["gev"]["observed"]) / echelle

    try:
        df_modelised = pl.read_parquet(mod_dir / "gev_retour.parquet")
    except Exception as e:
        st.error(f"Erreur lors du chargement des paramètres modélisés : {e}")
        return

    try:
        df_observed = pl.read_parquet(obs_dir / "gev_retour.parquet")
    except Exception as e:
        st.error(f"Erreur lors du chargement des paramètres observés : {e}")
        return
    
    # Suppression des NaN
    df_observed = filter_nan(df_observed) 

    # Carte Folium
    m = folium.Map(location=[46.9, 1.7], zoom_start=5.5, tiles="OpenStreetMap")

    # Ajout des points
    for row in df_observed.iter_rows(named=True):
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4,
            color="white",           # bordure
            fill=True,               # <== indispensable pour un cercle plein
            fill_color="white",      # couleur de remplissage
            fill_opacity=1.0,        # opacité du remplissage (1 = opaque)
            tooltip=f"lat: {row['lat']:.2f}, lon: {row['lon']:.2f}"
        ).add_to(m)

    col1, col2 = st.columns([2, 3.5])
    with col1:
        # Capture du clic utilisateur
        m.add_child(folium.LatLngPopup())
        map_output = st_folium(m, width=600, height=600)
        # Traitement du clic
        if map_output and map_output.get("last_clicked"):
            lat_clicked = map_output["last_clicked"]["lat"]
            lon_clicked = map_output["last_clicked"]["lng"]

            st.success(f"Point sélectionné : lat = {lat_clicked:.4f}, lon = {lon_clicked:.4f}")
            # Trouver le fichier le plus proche (tolérance arrondi 3 décimales)
            lat, lon = find_matching_point(df_observed, lat_clicked, lon_clicked)
        else:
            row = None

    with col2: 
        if lat and lon is not None:


        else:
            st.write("Cliquer sur un point pour l'affichage")

            
