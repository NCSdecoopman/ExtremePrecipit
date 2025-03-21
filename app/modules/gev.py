import os
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1" # Supprime l'avertissement console Hugging Face
os.environ["TQDM_DISABLE"] = "1" # Supprime la barre de téléchargement Hugging Face

import lzma
import json
import datetime

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

import streamlit as st
from streamlit_plotly_events import plotly_events

from huggingface_hub import hf_hub_download

import pyarrow.parquet as pq

# -----------------------------------------------------------
# 1) Mappings and constants
# -----------------------------------------------------------
SEASON_MAP = {
    "Année hydrologique": "hy",
    "Hiver": "djf",
    "Printemps": "mam",
    "Été": "jja",
    "Automne": "son",
}

SCALE_MAP = {
    "Horaire": "mm_h",
    "Journalière": "mm_j",
}

LEGEND_MAP = {
    "Horaire": "mm/h",
    "Journalière": "mm/j",
}

STATS = {
    "Stationnaire": "stationnaire",
    "Non stationnaire": "non_stationnaire"
}

# -----------------------------------------------------------
# 2) Cached loading functions
# -----------------------------------------------------------
@st.cache_data
def load_gev_parquet(file_name: str, base_dir: str = "data", repo_id: str = None) -> pd.DataFrame:
    """
    Reads a precomputed stats Parquet file.
    
    - If base_dir is provided, loads from the local directory.
    - If repo_id is provided, fetches the file directly from Hugging Face Hub.
    
    Example usage:
        Local: load_gev_parquet("mm_j_mam_mean", base_dir="data")
        HF:    load_gev_parquet("mm_j_mam_mean", repo_id="ncsdecoopman/extreme-precip-binaires")
    """

    file_parts = ["result", "gev", f"{file_name}.parquet"]

    if base_dir:
        # Local loading
        base_name = os.path.basename(os.path.normpath(base_dir))
        if base_name == "data":
            full_path = os.path.join(base_dir, *file_parts)
        elif base_name == "result":
            full_path = os.path.join(base_dir, "gev", f"{file_name}.parquet")
        else:
            raise ValueError("base_dir should point to 'data' or 'result' directory")
        return pd.read_parquet(full_path)
    
    elif repo_id:
        # Load from Hugging Face Hub with POSIX path
        file_relative_path = Path(*file_parts).as_posix()
        hf_path = hf_hub_download(
            repo_id=repo_id,
            filename=file_relative_path,
            repo_type="dataset"
        )
        return pd.read_parquet(hf_path)
    
    else:
        raise ValueError("Provide either `base_dir` or `repo_id`")
    

@st.cache_data
def load_max_parquet(file_name: str, lat: float, lon: float, base_dir: str = "data", repo_id: str = None) -> pd.DataFrame:
    file_parts = ["result", "preanalysis", "stats", f"{file_name}.parquet"]

    if base_dir:
        base_name = os.path.basename(os.path.normpath(base_dir))
        if base_name == "data":
            full_path = os.path.join(base_dir, *file_parts)
        elif base_name == "result":
            full_path = os.path.join(base_dir, "preanalysis", "stats", f"{file_name}.parquet")
        else:
            raise ValueError("base_dir should point to 'data' or 'result' directory")
    
    elif repo_id:
        file_relative_path = Path(*file_parts).as_posix()
        full_path = hf_hub_download(
            repo_id=repo_id,
            filename=file_relative_path,
            repo_type="dataset"
        )
    else:
        raise ValueError("Provide either `base_dir` or `repo_id`")

    # Lecture partielle avec filtre lat/lon
    table = pq.read_table(
        full_path,
        filters=[
            ('lat', '=', lat),
            ('lon', '=', lon)
        ]
    )
    df_filtered = table.to_pandas()
    return df_filtered
    

@st.cache_data
def compute_z_T(mu, sigma, xi, start_T=1, end_T=1000, num=1000):
    """
    Calcule le z_T de la GEV pour un vecteur de temps de retour T.
    """
    # On génère T directement sur l’échelle logarithmique entre start_T et end_T (ex: 1 à 1000 ans)
    T = np.logspace(np.log10(start_T), np.log10(end_T), num=num)
    
    if xi != 0:
        z_T = mu + (sigma / xi) * ((-np.log(1 - 1/T))**(-xi) - 1)
    else:
        z_T = mu - sigma * np.log(-np.log(1 - 1/T))
    
    return T, z_T



# -----------------------------------------------------------
# The Streamlit app
# -----------------------------------------------------------
def show(OUTPUT_DIR, years):

    st.markdown(f"<h3>Visualisation des GEV de {min(years)} à {max(years)}</h3>", unsafe_allow_html=True)

    # D'abord on définit stat_label pour l'utiliser plus tard :
    col1, col3 = st.columns([1,2])
    with col1:
        stat_label = st.selectbox("Choix de la GEV étudiée", list(STATS.keys()))

    with col3:

        st.markdown("""
            <style>
            /* Cacher les ticks-bar min et max sous la barre du slider */
            div[data-testid="stSliderTickBarMin"],
            div[data-testid="stSliderTickBarMax"] {
                display: none !important;
            }
            /* Réduire l'espace vertical du slider */
            .stSlider > div[data-baseweb="slider"] {
                margin-bottom: -10px;
            }
            /* Remonter la barre + les poignées */
            .stSlider {
                transform: translateY(-17px);
            }
            </style>
        """, unsafe_allow_html=True)

        start_T, end_T = st.slider(
            f"Sélection du temps de retour entre 1 et 1000",
            min_value=1,
            max_value=1000,
            value=(1, 1000)
        )

    # Ensuite on peut utiliser stat_label dans ce bloc
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        saison_selection = st.selectbox(
            "Choix de la saison",
            list(SEASON_MAP.keys())
        )

    with col2:
        echelle_selection = st.selectbox("Choix de l'échelle temporelle", ["Horaire","Journalière"])

    with col3:
        onglet_label = st.selectbox("Choix du paramètre", ["mu","sigma", "xi"])

    # -------------------------------------------------------
    # Build the file name, load stats
    # -------------------------------------------------------
    scale_key = SCALE_MAP[echelle_selection]  # "mm_h" or "mm_j"
    stat_key  = STATS[stat_label]             # "stationnaire" ou "non_stationnaire"
    season_key = SEASON_MAP[saison_selection] # "djf","mam","hy", etc.

    file_name_gev = f"{scale_key}_{season_key}_max_gev_{stat_key}"

    with st.spinner(f"Génération de la carte..."):
        df = load_gev_parquet(file_name_gev)

    date_title = f"de {min(years)} à {max(years)}"

    # Some textual info
    if season_key == "hy":
        date_title_map = f"01/09 au 31/08 {date_title}"
    elif season_key == "djf":
        date_title_map = f"01/12 au 28/02 {date_title}"
    elif season_key == "mam":
        date_title_map = f"01/03 au 31/05 {date_title}"
    elif season_key == "jja":
        date_title_map = f"01/06 au 31/08 {date_title}"
    elif season_key == "son":
        date_title_map = f"01/09 au 30/11 {date_title}"
    else:
        date_title_map = date_title  # fallback

    title_map = f"GEV {stat_label.lower()} sur les précipitations maximales du {date_title_map}"

    # Colorbar legend text
    cbar_legend = LEGEND_MAP[echelle_selection]

    # -------------------------------------------------------
    # Make the interactive map with px.scatter_mapbox
    # -------------------------------------------------------
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

    st.markdown(
        f"""
        <div style='text-align: center; font-weight: bold; margin-top: 15px'>
            {title_map}
        </div>
        """,
        unsafe_allow_html=True
    )

    fig_map = px.scatter_mapbox(
        df,
        lat="lat",
        lon="lon",
        color=onglet_label,
        color_continuous_scale=custom_colorscale,
        title=title_map,
        height=500,
        zoom=4.5,
        center=dict(lat=46.6, lon=2.2),
    )
    fig_map.update_layout(
        mapbox_style="carto-darkmatter",
        margin=dict(l=0,r=0,t=0,b=5),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    
    # colorbar label
    fig_map.update_coloraxes(
        colorbar=dict(
            title=dict(
                text="", # cbar_legend
                font=dict(color='white')
            ),
            tickfont=dict(color='white')
        )
    )

    # Use plotly_events to capture click
    selected_points = plotly_events(
        fig_map,
        click_event=True,     # single click
        select_event=False,   # no lasso
        hover_event=False,
        override_height=500
    )
    # Source alignée à droite sous la carte
    st.markdown(
        """
        <div style='text-align: left; font-size: 0.8em; color: lightgray; margin-top: -15px;'>
            Données CP-RCM, 2.5 km, forçage ERA5, réanalyse ECMWF
        </div>
        """,
        unsafe_allow_html=True
    )


    # -------------------------------------------------------
    # If a point is clicked, display its time series
    # -------------------------------------------------------
    # Après avoir chargé le DataFrame df contenant les colonnes ci-dessous:
    # ["mu", "mu_ci_lower", "mu_ci_upper", "sigma", "sigma_ci_lower", "sigma_ci_upper", "xi", "xi_ci_lower", "xi_ci_upper"]

    # Sélection du point cliqué
    if st.session_state["selected_point"] is not None:
        lat_clicked = st.session_state["selected_point"]["lat"]
        lon_clicked = st.session_state["selected_point"]["lon"]

        lat_str = f"{lat_clicked:.4f}"
        lon_str = f"{lon_clicked:.4f}"
        lat = float(lat_str)
        lon = float(lon_str)

        row = df[(df['lat'] == lat) & (df['lon'] == lon)].iloc[0]

        mu = row['mu']
        mu_ci_lower = row['mu_ci_lower']
        mu_ci_upper = row['mu_ci_upper']
        sigma = row['sigma']
        sigma_ci_lower = row['sigma_ci_lower']
        sigma_ci_upper = row['sigma_ci_upper']
        xi = row['xi']
        xi_ci_lower = row['xi_ci_lower']
        xi_ci_upper = row['xi_ci_upper']

        # Courbe centrale
        T, z_T = compute_z_T(mu, sigma, xi, start_T, end_T, num=1000)
        # Courbes pour bornes inf et sup (brute force mais robuste)
        _, z_T_lower = compute_z_T(mu_ci_lower, sigma_ci_lower, xi_ci_lower, start_T, end_T, num=1000)
        _, z_T_upper = compute_z_T(mu_ci_upper, sigma_ci_upper, xi_ci_upper, start_T, end_T, num=1000)

        df_max = load_max_parquet(f"{scale_key}_{season_key}_max", lat, lon)

        df_max_sorted = df_max.sort_values(by="pr", ascending=False).reset_index(drop=True)
        n = len(df_max_sorted)
        df_max_sorted['rank'] = np.arange(1, n + 1)
        df_max_sorted['T_emp'] = (n + 1) / df_max_sorted['rank']

        fig_gev = go.Figure()

        # Bande d'incertitude (entre lower et upper)
        fig_gev.add_trace(go.Scatter(
            x=np.concatenate([T, T[::-1]]),
            y=np.concatenate([z_T_lower, z_T_upper[::-1]]),
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name="Intervalle de confiance 95%"
        ))

        # Courbe GEV centrale
        fig_gev.add_trace(go.Scatter(
            x=T,
            y=z_T,
            mode='lines',
            name='GEV théorique',
            line=dict(width=2)
        ))

        # Points empiriques
        fig_gev.add_trace(go.Scatter(
            x=df_max_sorted['T_emp'],
            y=df_max_sorted['pr'],
            mode='markers',
            name='Observations',
            marker=dict(size=5, symbol='circle', line=dict(width=1)),
            text=df_max_sorted['year'],
            hovertemplate="Année: %{text}<br>T emp: %{x:.1f} ans<br>Max: %{y:.1f} mm"
        ))

        fig_gev.update_layout(
            title="",
            xaxis_title="Temps de retour T (années)",
            yaxis_title="Valeur de retour z(T) (mm)",
            template="plotly_white",
            height=500,
            xaxis_type="log"
        )

        # Titre
        st.markdown(
            f"""
            <div style='text-align: center; font-weight: bold; margin-top: 15px'>
                Courbe de niveau de retour (GEV) du point sélectionné : ({lat_clicked:.3f}, {lon_clicked:.3f})
            </div>
            """,
            unsafe_allow_html=True
        )

        st.plotly_chart(fig_gev, use_container_width=True)

        # Résumé des paramètres + IC
        st.markdown(
            f"""
            <div style='text-align: left; font-size: 0.8em; color: lightgray; margin-top: -15px;'>
                Intervalle de confiance généré par bootstrap (n = 100)<br>
                μ = {mu:.2f} [{mu_ci_lower:.2f}, {mu_ci_upper:.2f}], 
                σ = {sigma:.2f} [{sigma_ci_lower:.2f}, {sigma_ci_upper:.2f}], 
                ξ = {xi:.2f} [{xi_ci_lower:.2f}, {xi_ci_upper:.2f}]<br>
                Données CP-RCM, 2.5 km, forçage ERA5, réanalyse ECMWF
            </div>
            """,
            unsafe_allow_html=True
        )


            


    #     data_arr = load_time_series(lat_str, lon_str, sentinel, sf)

    #     if data_arr is None:
    #         st.warning(f"Pas de série temporelle disponible pour le point ({lat_clicked:.3f}, {lon_clicked:.3f}).")

    #     else:
    #         # Figure out the correct "season key"
    #         s_key = SEASON_MAP[saison_selection]
    #         # For the timeseries, we do not shift the start year for "Hiver" or "hy" because
    #         # the date slicing is done in get_season_dates(...) anyway. We'll just pass the user-chosen range
    #         # i.e. "1980 to 1985", that function knows "djf" means Dec(1979)->Feb(1980).

    #         with st.spinner(f"Génération de la série temporelle du point ({lat_str}, {lon_str})..."):
    #             indices, dates = get_line_indices(s_key, start_year, end_year, min_time)

    #             # Build a Pandas time series at hourly resolution
    #             series_hr = pd.Series(data_arr[indices], index=dates, name="Prec (hr)")

    #             # If "Journalière", sum from 06h to 06h
    #             if echelle_selection == "Journalière":
    #                 # sum every 1D with offset of 6H
    #                 daily = series_hr.resample("1D", offset="6h").sum()
    #                 # shift index by 6 hours
    #                 daily.index = daily.index + pd.Timedelta(hours=6)
    #                 final_series = daily
    #             else:
    #                 final_series = series_hr

    #             # Plot with Plotly
    #             fig_ts = go.Figure()
    #             fig_ts.add_trace(go.Scatter(
    #                 x=final_series.index,
    #                 y=final_series.values,
    #                 mode='lines',
    #                 name="Précip."
    #             ))
    #             fig_ts.update_layout(
    #                 title="",
    #                 xaxis_title="Date",
    #                 yaxis_title=LEGEND_MAP[echelle_selection],
    #                 template="plotly_white",
    #                 height=500
    #             )
                
    #             st.markdown(
    #                 f"""
    #                 <div style='text-align: center; font-weight: bold; margin-top: 15px'>
    #                     Série temporelle au point ({lat_clicked:.3f}, {lon_clicked:.3f}) du {date_title_map}
    #                 </div>
    #                 """,
    #                 unsafe_allow_html=True
    #             )
                
    #             st.plotly_chart(fig_ts, use_container_width=True)
    #             # Source
    #             st.markdown(
    #                 """
    #                 <div style='text-align: left; font-size: 0.8em; color: lightgray; margin-top: -15px;'>
    #                     Données CP-RCM, 2.5 km, forçage ERA5, réanalyse ECMWF
    #                 </div>
    #                 """,
    #                 unsafe_allow_html=True
    #             )
    # else:
    #     st.info("Cliquer sur la carte ci-dessus pour afficher une série temporelle.")