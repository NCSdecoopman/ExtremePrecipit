import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import os
import datetime

from app.utils.plot import plot_map

import os

SEASON_MAP = {
    "Année hydrologique": "hy",
    "Hiver": "djf",
    "Printemps": "mam",
    "Été": "jja",
    "Automne": "son"
}

SCALE_MAP = {
    "mm/h": "mm_h",
    "mm/j": "mm_j"
}

STATS = {
    "μ": "loc",
    "σ": "scale",
    "ξ": "c",
    "Test ADF": "ADF",
    "Test KPSS" : "KPSS",
    "Test MK": "MK",
    "Test LB": "LB",
    "Test KS": "KS"
}

# Charger les données sur une période sélectionnée
@st.cache_data
def load_data(file_name, outputdir):
    file_path = os.path.join(outputdir, "gev", f"{file_name}_max_gev.parquet")
    df = pd.read_parquet(file_path)
    return df

# Fonction de conversion
def interpret_tests(row):
    return pd.Series({
        'ADF': 'Stationnaire' if row['adf_pval'] < 0.05 else 'Non stationnaire',
        'KPSS': 'Stationnaire' if row['kpss_pval'] > 0.05 else 'Non stationnaire',
        'MK': 'Tendance' if row['mk_pval'] < 0.05 else 'Sans tendance',
        'LB': 'Autocorrélation' if row['lb_pval'] < 0.05 else 'Sans autocorrélation',
        'KS': 'Bon ajustement' if row['ks_pval'] >= 0.05 else 'Mauvais ajustement'
    })
    
def show(OUTPUT_DIR, years):  
    start_year = min(years)
    end_year = max(years)

    # Sélection de la saison et de la statistique à afficher
    col1, col2 = st.columns([4, 1])
    with col1:
        saison_selection = st.radio(
            "Saison", 
            ["Année hydrologique", "Hiver", "Printemps", "Été", "Automne"],
            horizontal=True, label_visibility="collapsed"
        )

    with col2:
        echelle_selection = st.selectbox("Echelle", ["mm/h", "mm/j"], label_visibility="collapsed")

    if saison_selection in ["Année hydrologique", "Hiver"]:
        start_year += 1

    # On définit ici le mapping
    scale_key = SCALE_MAP[echelle_selection]
    season_key = SEASON_MAP[saison_selection]

    file_name = f"{scale_key}_{season_key}"   
    df = load_data(file_name, OUTPUT_DIR)
    # Application
    df_bool = df.copy()
    df_bool[['ADF', 'KPSS', 'MK', 'LB', 'KS']] = df.apply(interpret_tests, axis=1)

    if season_key in ["hy", "djf"]:
        title_year = f"de {start_year-1} à {end_year}"
    else:
        if start_year == end_year:
            title_year = f"en {start_year}"
        else:
            title_year = f"de {start_year} à {end_year}"

    if season_key == "hy":
        date_title_map = f"01/09 au 31/08 {title_year}"
    elif season_key == "djf":
        date_title_map = f"01/12 au 28/02 {title_year}"
    elif season_key == "mam":
        date_title_map = f"01/03 au 31/05 {title_year}"
    elif season_key == "jja":
        date_title_map = f"01/06 au 31/08 {title_year}"
    elif season_key == "son":
        date_title_map = f"01/09 au 31/11 {title_year}"

    title_legend = echelle_selection

    custom_colorscale = [
        [0.0, "white"],  
        [0.01, "lightblue"],
        [0.10, "blue"],
        [0.30, "darkblue"],  
        [0.50, "green"], 
        [0.60, "yellow"],
        [0.70, "red"],  
        [0.80, "darkred"],  
        [1.0, "#654321"]
    ]

    tabs = st.tabs([s for s in STATS.keys()])
    
    for i, tab in enumerate(tabs):
        with tab:
            stat_label = list(STATS.keys())[i]
            stat_suffix = STATS[stat_label]

            fig_map = px.scatter_mapbox(
                df_bool,
                lat="lat",
                lon="lon",
                color=stat_suffix,
                color_continuous_scale="cividis",
                title=f"{stat_label} du {date_title_map}",
                width=1000,
                height=700,
            )
                            
            plot_map(fig_map, title_legend="")