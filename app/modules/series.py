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
    "Moyenne": "mean",
    "Maximum": "max",
    "Cumul": "sum",      # valable en échelle Journalière
    "Date du maximum": "date",
    "Jour de pluie": "numday",  # valable en échelle Journalière
}

STATS_NATIONALE = {
    "mean": "moyenne",
    "max": "maximale",
    "sum": "cumulée",      # valable en échelle Journalière
    "date": "",
    "numday": "en jour de pluie",  # valable en échelle Journalière
}

# -----------------------------------------------------------
# 2) Cached loading functions
# -----------------------------------------------------------
@st.cache_data
def load_stats_parquet(file_name: str, base_dir: str = None, repo_id: str = "ncsdecoopman/extreme-precip-binaires") -> pd.DataFrame:
    """
    Reads a precomputed stats Parquet file.
    
    - If base_dir is provided, loads from the local directory.
    - If repo_id is provided, fetches the file directly from Hugging Face Hub.
    
    Example usage:
        Local: load_stats_parquet("mm_j_mam_mean", base_dir="data")
        HF:    load_stats_parquet("mm_j_mam_mean", repo_id="ncsdecoopman/extreme-precip-binaires")
    """

    file_parts = ["result", "preanalysis", "stats", f"{file_name}.parquet"]

    if base_dir:
        # Local loading
        base_name = os.path.basename(os.path.normpath(base_dir))
        if base_name == "data":
            full_path = os.path.join(base_dir, *file_parts)
        elif base_name == "result":
            full_path = os.path.join(base_dir, "preanalysis", "stats", f"{file_name}.parquet")
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

# def load_stats_parquet(file_name: str, output_dir: str) -> pd.DataFrame:
#     """
#     Reads one of the precomputed stats Parquet files.
#     E.g. file_name = "mm_j_mam_mean" => data/result/preanalysis/stats/mm_j_mam_mean.parquet
#     """
#     full_path = os.path.join(output_dir, "preanalysis", "stats", f"{file_name}.parquet")
#     return pd.read_parquet(full_path)

@st.cache_data
def load_metadata(metadata_path="data/binaires/metadata.json"):
    """
    Reads metadata.json to get scale_factor, sentinel_value, start time, etc.
    """
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    scale_factor = metadata["scale_factor"]
    sentinel_value = metadata["sentinel_value"]
    min_time = pd.to_datetime(metadata["min_time"])
    time_step_hours = metadata["time_step_hours"]
    return scale_factor, sentinel_value, min_time, time_step_hours

## Fonction de chargement des données Hungging Face
@st.cache_data
def load_time_series(lat_str: str, lon_str: str, sentinel: float, sf: float) -> np.ndarray:
    """
    Loads the hourly precip for a single point from .bin.xz file stored on Hugging Face.
    """
    # On découpe comme dans le script de génération
    lat_prefix = lat_str.split(".")[0]  # e.g. "41.6686" -> "41"
    lon_prefix = lon_str.split(".")[0]  # e.g. "9.0327"  -> "9"

    hf_path = f"lat_{lat_prefix}_lon_{lon_prefix}/ts_{lat_str}_{lon_str}.bin.xz"

    try:
        # Download file on-demand from HF
        file_path = hf_hub_download(
            repo_id="ncsdecoopman/extreme-precip-binaires",
            filename=hf_path,
            repo_type="dataset"
        )
    except Exception:
        return None  # File not found or other HF issue
    
    # Open and decode .bin.xz as before
    with lzma.open(file_path, "rb") as f:
        raw = f.read()
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        data[data == sentinel] = np.nan
        data /= sf
    return data

## Fonction de chargement des données en locale
# def load_time_series(lat_str: str, lon_str: str, sentinel: float, sf: float) -> np.ndarray:
#     """
#     Loads the hourly precip for a single point from compressed .bin.xz file.
#     Replaces sentinel values by NaN, then divides by scale factor.
#     """
#     path = f"data/binaires/individuels/ts_{lat_str}_{lon_str}.bin.xz"
    
#     if not os.path.exists(path):
#         return None
    
#     with lzma.open(path, "rb") as f:
#         raw = f.read()
#         data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
#         data[data == sentinel] = np.nan
#         data /= sf
#     return data

# -----------------------------------------------------------
# 3) Helper functions
# -----------------------------------------------------------
def filter_years(df: pd.DataFrame, start_yr: int, end_yr: int) -> pd.DataFrame:
    return df[(df["year"] >= start_yr) & (df["year"] <= end_yr)]

def group_by_coord(df: pd.DataFrame, stat: str) -> pd.DataFrame:
    """
    Aggregates the data over (lat, lon).
    - If `stat == "date"`, picks the date associated with the max precip per (lat,lon).
    - If `stat == "numday"`, uses mean (since it's the daily # of rainy days).
    - Else uses the named aggregation (mean, max, sum).
    """
    if stat == "date":
        # we need the row of the max from "max" column
        # i.e. the row where 'max' is largest, so we can store the corresponding 'date'
        df_max = df.loc[df.groupby(['lat','lon'])['max'].idxmax(), ['lat','lon','date','max']]
        return df_max.rename(columns={'date':'pr'})  # store the date into a 'pr' col
    else:
        agg_col = "mean" if stat == "numday" else stat
        grouped = df.groupby(["lat", "lon"])["pr"].agg(agg_col).reset_index()
        return grouped
    
def groupe_by_year(df):
    return df.groupby(["year"])["pr"].agg([f'mean'] + ['std']).reset_index()

def get_season_dates(season_key: str, year: int):
    """
    Returns (start, end) datetime for the given season key & year.
    """
    if season_key == "hy":
        return datetime.datetime(year-1, 9, 1), datetime.datetime(year, 8, 31, 23, 59)
    elif season_key == "djf":
        return datetime.datetime(year-1, 12, 1), datetime.datetime(year, 2, 28, 23, 59)
    elif season_key == "mam":
        return datetime.datetime(year, 3, 1), datetime.datetime(year, 5, 31, 23, 59)
    elif season_key == "jja":
        return datetime.datetime(year, 6, 1), datetime.datetime(year, 8, 31, 23, 59)
    elif season_key == "son":
        return datetime.datetime(year, 9, 1), datetime.datetime(year, 11, 30, 23, 59)
    else:
        raise ValueError("Season key unknown: " + season_key)

@st.cache_data
def get_line_indices(season_key: str, start_year: int, end_year: int, min_time: pd.Timestamp):
    """
    Builds a list of all hourly indices (since min_time) for the selected season(s) from start_year to end_year.
    """
    indices = []
    dates = []
    for y in range(start_year, end_year+1):
        st_dt, en_dt = get_season_dates(season_key, y)
        idx_start = int((st_dt - min_time).total_seconds() // 3600)
        idx_end   = int((en_dt - min_time).total_seconds() // 3600)
        for i in range(idx_start, idx_end+1):
            indices.append(i)
            dt = min_time + pd.Timedelta(hours=i)
            dates.append(dt)
    return indices, pd.to_datetime(dates)

# -----------------------------------------------------------
# 4) The Streamlit app
# -----------------------------------------------------------
def show(OUTPUT_DIR, years):

    if "selected_point" not in st.session_state:
        st.session_state["selected_point"] = None

    # D'abord on définit stat_label pour l'utiliser plus tard :
    col1, col2, col3 = st.columns([1.3,1.3,2])
    with col1:
        stat_label = st.selectbox("Statistique", list(STATS.keys()))
    with col2:
        onglet_label = st.selectbox("Échelle spatiale", ["Locale","Nationale"])
    with col3:
        start_year, end_year = st.slider(
            "Années",
            min_value=min(years),
            max_value=max(years),
            value=(min(years), max(years))
        )

    # Ensuite on peut utiliser stat_label dans ce bloc
    col1, col2 = st.columns([4,1])
    with col1:
        saison_selection = st.radio(
            "Saison",
            list(SEASON_MAP.keys()),
            horizontal=True
        )

    with col2:
        if stat_label in ["Cumul","Jour de pluie"]:
            echelle_selection = st.selectbox("Échelle", ["Journalière"])
        else:
            echelle_selection = st.selectbox("Échelle", ["Horaire","Journalière"])

    # Logique de validation
    if saison_selection in ["Année hydrologique", "Hiver"] and start_year == end_year:
        st.error("Pour cette saison, choisissez au moins un an d'écart.")
        st.stop()


    # Adjust year for filter if necessary
    # "Hiver" or "Année hydrologique" straddle two calendar years,
    # so effectively the "start" is the next year.
    if saison_selection in ["Année hydrologique", "Hiver"]:
        # Example: For "Hiver" 1980 => actual range is Dec 1979 to Feb 1980
        # so we want to filter by year >= (start_year+1)
        start_year_for_filter = start_year + 1
    else:
        start_year_for_filter = start_year

    # -------------------------------------------------------
    # Build the file name, load stats
    # -------------------------------------------------------
    scale_key = SCALE_MAP[echelle_selection]  # "mm_h" or "mm_j"
    stat_key  = STATS[stat_label]             # "mean","max","sum","date","numday"
    season_key = SEASON_MAP[saison_selection] # "djf","mam","hy", etc.

    file_name = f"{scale_key}_{season_key}_{stat_key}"

    with st.spinner(f"Génération de la carte..."):
        df = load_stats_parquet(file_name)

        # If stat is "date", we also need the "max" to find the row that gave that date
        if stat_key == "date":
            df_max = load_stats_parquet(f"{scale_key}_{season_key}_max")
            df = df.rename(columns={"pr":"date"})
            df_max = df_max.rename(columns={"pr":"max"})
            # Merge them on (lat,lon,year)
            df = df.merge(df_max, on=["lat","lon","year"], how="left")

        # Filter by year
        df_filt = filter_years(df, start_year_for_filter, end_year)

        # Group by lat/lon
        df_agg = group_by_coord(df_filt, stat_key)

    # Possibly scale the 'sum' from mm to m, etc. (optional)
    if stat_key == "sum":
        df_agg["pr"] /= 1000.0  # just an example, so that it's in 'm'

    # Build a title
    if season_key in ["hy","djf"]:
        # e.g. "de 1979 à 1985"
        date_title = f"de {start_year} à {end_year}"
    else:
        if start_year == end_year:
            date_title = f"en {start_year}"
        else:
            date_title = f"de {start_year} à {end_year}"

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

    if stat_key == "date":
        title_map = f"Survenue de la précipitation maximale du {date_title_map}"
    elif stat_key == "numday":
        title_map = f"Nombre de jours (>1 mm) du {date_title_map}"
    else:
        title_map = f"{stat_label} de précipitation du {date_title_map}"

    # Colorbar legend text
    if stat_key == "sum":
        cbar_legend = "m"  # after dividing by 1000
    elif stat_key == "numday":
        cbar_legend = "Jours"
    elif stat_key == "date":
        cbar_legend = "Date"
    else:
        cbar_legend = LEGEND_MAP[echelle_selection]

    # If "date", we convert the date to numeric for color scale
    tick_positions, tick_labels = None, None
    if stat_key == "date":
        df_agg["pr"] = pd.to_datetime(df_agg["pr"])
        # Reference start date for color offset
        if saison_selection in ["Année hydrologique","Hiver"]:
            reference_start = datetime.datetime(start_year-1, 1, 1)
        else:
            reference_start = datetime.datetime(start_year, 1, 1)

        df_agg["days_since_start"] = (df_agg["pr"] - reference_start).dt.days
        days_min = df_agg["days_since_start"].min()
        days_max = df_agg["days_since_start"].max()

        # We pick ~8 ticks
        nb_ticks = 8
        tick_positions = np.linspace(days_min, days_max, nb_ticks)
        # If only a short range, maybe show year-month, else year
        if (end_year - start_year) < nb_ticks - 1:
            tick_labels = pd.to_datetime(
                tick_positions, unit='D', origin=reference_start
            ).strftime('%Y-%m')
        else:
            tick_labels = pd.to_datetime(
                tick_positions, unit='D', origin=reference_start
            ).strftime('%Y')

        df_agg = df_agg.drop(columns=["pr"])
        df_agg = df_agg.rename(columns={"days_since_start":"pr"})

    if onglet_label == "Locale":

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

        st.write(f"**{title_map}**")

        fig_map = go.Figure(go.Scattermapbox(
            lat=df_agg["lat"],
            lon=df_agg["lon"],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=9,
                color=df_agg["pr"],
                colorscale=custom_colorscale,
                colorbar=dict(
                    title=dict(
                        text=cbar_legend,
                        font=dict(color="white")
                    ),
                    tickfont=dict(color="white"),
                    bgcolor="rgba(0,0,0,0)"
                ),
                showscale=True
            ),
            hoverinfo='lat+lon+text',
            text=[f"{pr:.2f} {cbar_legend}" for pr in df_agg["pr"]]
        ))

        fig_map.update_layout(
            mapbox=dict(
                style="carto-darkmatter",
                center=dict(lat=46.6, lon=2.2),
                zoom=4.5
            ),
            margin=dict(l=0, r=0, t=0, b=5),
            paper_bgcolor="rgba(0,0,0,0)"
        )

        # If date, set custom tick vals
        if stat_key == "date" and tick_positions is not None and tick_labels is not None:
            fig_map.update_layout(
                coloraxis_colorbar_tickvals=tick_positions.tolist(),
                coloraxis_colorbar_ticktext=tick_labels.tolist(),
            )
        # colorbar label
        fig_map.update_coloraxes(
            colorbar=dict(
                title=dict(
                    text=cbar_legend,
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

        # -------------------------------------------------------
        # If a point is clicked, display its time series
        # -------------------------------------------------------
        if selected_points:
            # figure out which row was clicked
            idx = selected_points[0]["pointIndex"]
            lat_clicked = df_agg.iloc[idx]["lat"]
            lon_clicked = df_agg.iloc[idx]["lon"]

            # Mise à jour de la session
            st.session_state["selected_point"] = {"lat": lat_clicked, "lon": lon_clicked}

        if st.session_state["selected_point"] is not None:
            lat_clicked = st.session_state["selected_point"]["lat"]
            lon_clicked = st.session_state["selected_point"]["lon"]

            # Load the timeseries
            sf, sentinel, min_time, time_step_hours = load_metadata()
            # Because in your .bin.xz naming, lat/lon are stored as strings,
            # we need the same exact format used in the file. So let's find them
            # in the original data or just create string formats:
            # We'll assume 3 decimals for lat/lon or you can do a more direct approach:
            lat_str = f"{lat_clicked:.4f}" # forcer 4 décimales
            lon_str = f"{lon_clicked:.4f}"

            # Sometimes your data may not store trailing zeros, e.g. "5.0" => "5"
            # If that is your case, you'll have to match that pattern. For example:
            # lat_str = str(float(f"{lat_clicked:.4f}"))
            # lon_str = str(float(f"{lon_clicked:.4f}"))

            # In any case, adapt so it matches exactly your file naming scheme:
            # "ts_45.75_3.25.bin.xz", etc.

            data_arr = load_time_series(lat_str, lon_str, sentinel, sf)

            if data_arr is None:
                st.warning(f"Pas de série temporelle disponible pour le point ({lat_clicked:.3f}, {lon_clicked:.3f}).")

            else:
                # Figure out the correct "season key"
                s_key = SEASON_MAP[saison_selection]
                # For the timeseries, we do not shift the start year for "Hiver" or "hy" because
                # the date slicing is done in get_season_dates(...) anyway. We'll just pass the user-chosen range
                # i.e. "1980 to 1985", that function knows "djf" means Dec(1979)->Feb(1980).

                with st.spinner(f"Génération de la série temporelle du point ({lat_str}, {lon_str})..."):
                    indices, dates = get_line_indices(s_key, start_year, end_year, min_time)

                    # Build a Pandas time series at hourly resolution
                    series_hr = pd.Series(data_arr[indices], index=dates, name="Prec (hr)")

                    # If "Journalière", sum from 06h to 06h
                    if echelle_selection == "Journalière":
                        # sum every 1D with offset of 6H
                        daily = series_hr.resample("1D", offset="6h").sum()
                        # shift index by 6 hours
                        daily.index = daily.index + pd.Timedelta(hours=6)
                        final_series = daily
                    else:
                        final_series = series_hr

                    # Plot with Plotly
                    fig_ts = go.Figure()
                    fig_ts.add_trace(go.Scatter(
                        x=final_series.index,
                        y=final_series.values,
                        mode='lines',
                        name="Précip."
                    ))
                    fig_ts.update_layout(
                        title=(
                            f"Série temporelle au point ({lat_clicked:.3f}, {lon_clicked:.3f}) du {date_title_map}"
                        ),
                        xaxis_title="Date",
                        yaxis_title=LEGEND_MAP[echelle_selection],
                        template="plotly_white",
                        height=500
                    )
                    st.plotly_chart(fig_ts, use_container_width=True)
        else:
            st.info("Cliquer sur la carte ci-dessus pour afficher une série temporelle.")

    elif onglet_label == "Nationale":
        if stat_key != "date":
            df_agg_year = groupe_by_year(df_filt)

            if stat_key == "sum":
                df_agg_year["mean"] /= 1000
                df_agg_year["std"] /= 1000

            # Définir la plage de l'axe Y
            y_max = (df_agg_year["mean"] + df_agg_year["std"]).max() * 1.05  # Ajout de 5% de marge

            # Création du graphique avec barres d'erreur
            fig_time_series = px.line(
                df_agg_year, 
                x="year", 
                y="mean", 
                error_y="std", 
                markers=True,
                title=f"Moyenne et écart-type de précipitation {STATS_NATIONALE[stat_key]} à l'échelle nationale<br>du {date_title_map}",
                labels={"year": "Année", "mean": cbar_legend},
                width=900,
                height=500
            )

            # Ajustement du style des barres d'erreur
            fig_time_series.update_traces(
                error_y=dict(
                    thickness=0.5,  # Réduction de l'épaisseur
                    width=0,  # Rendre les barres plus courtes horizontalement
                    color="white"  # Couleur 
                )
            )

            # Ajustement de l'axe Y pour commencer à 0 et finir au max ajusté
            fig_time_series.update_layout(
                yaxis=dict(range=[0, y_max])
            )

            # Affichage de la courbe
            st.plotly_chart(fig_time_series)

        elif stat_key == "date":
            # On compte le nombre d'occurrences par date et on trie par date croissante
            count_by_date = df_filt['date'].value_counts().sort_index()
            st.write(f"**Occurence des précipitations maximales nationale du {date_title_map}**")
            st.bar_chart(count_by_date)
