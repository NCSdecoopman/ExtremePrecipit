import pandas as pd
import streamlit as st
import calendar
from huggingface_hub import hf_hub_download
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.utils.config_utils import load_config
from app.utils.menus_utils import menu_statisticals

st.markdown("<h3>Visualisation des précipitations</h3>", unsafe_allow_html=True)

STATS = {
    #"Moyenne": "mean",
    "Maximum": "max",
    "Moyenne des maxima": "mean-max",
    "Cumul": "sum",
    "Date du maximum": "date",
    "Mois comptabilisant le plus de maximas": "month",
    "Jour de pluie": "numday",
}

SEASON = {
    "Année hydrologique": [9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8],
    "Hiver": [12, 1, 2],
    "Printemps": [3, 4, 5],
    "Été": [6, 7, 8],
    "Automne": [9, 10, 11],
}

SCALE = {
    "Horaire": "mm_h",
    "Journalière": "mm_j"
}


@st.cache_data(show_spinner=False)
def load_parquet_from_huggingface_cached(year: int, month: int, repo_id: str, base_path: str) -> pd.DataFrame:
    hf_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{base_path}/{year:04d}/{month:02d}.parquet",
        repo_type="dataset"
    )
    return pd.read_parquet(hf_path)

@st.cache_data(show_spinner=False)
def load_arome_data(min_year_choice: int, max_year_choice: int, months: list[int], config) -> list:
    repo_id = config["repo_id"]
    base_path = config["statisticals"]["modelised"]

    tasks = []

    for year in range(min_year_choice, max_year_choice + 1):
        for month in months:
            if month >= 1 and month <= 8 and month < months[0]:
                actual_year = year + 1
            else:
                actual_year = year
            if actual_year > max_year_choice:
                continue
            tasks.append((actual_year, month))

    dataframes = []
    errors = []

    with st.spinner("Chargement des fichiers..."):
        progress_bar = st.progress(0)
        total = len(tasks)
        completed = 0

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {
            executor.submit(load_parquet_from_huggingface_cached, y, m, repo_id, base_path): (y, m)
            for y, m in tasks
        }

        for future in as_completed(futures):
            y, m = futures[future]
            try:
                df = future.result()
                dataframes.append(df)
            except Exception as e:
                errors.append(f"{y}-{m:02d} : {e}")
            completed += 1
            progress_bar.progress(completed / total)

    progress_bar.empty()  # Efface la barre de progression

    if errors:
        for err in errors:
            st.warning(f"Erreur : {err}")

    if not dataframes:
        raise ValueError("Aucune donnée chargée.")

    return pd.concat(dataframes, ignore_index=True)

def compute_statistic_per_point(df: pd.DataFrame, stat_key: str) -> pd.DataFrame:
    with st.spinner("Chargement des fichiers..."):
        progress_bar = st.progress(0)
    
    if stat_key == "mean":
        # Extraire année et mois
        df["year"] = df["max_date_mm_h"].str[:4].astype(int)
        df["month"] = df["max_date_mm_h"].str[5:7].astype(int)

        # Nombre d'heures dans le mois
        df["nb_hours"] = df.apply(
            lambda row: 24 * calendar.monthrange(row["year"], row["month"])[1],
            axis=1
        )

        # Ajouter une colonne des précipitations horaires totales
        df["mm_h_total"] = df["sum_mm"]
        df["mm_h_mean"] = df["mm_h_total"] / df["nb_hours"]

        # Moyenne réelle des précipitations horaires
        return df.groupby(["lat", "lon"])["mm_h_mean"].mean().reset_index(name="mean_mm_h")

    elif stat_key == "max":
        return df.groupby(["lat", "lon"]).agg(
            max_all_mm_h=("max_mm_h", "max"),
            max_all_mm_j=("max_mm_j", "max")
        ).reset_index()

    elif stat_key == "mean-max":
        return df.groupby(["lat", "lon"]).agg(
            max_mean_mm_h=("max_mm_h", "mean"),
            max_mean_mm_j=("max_mm_j", "mean")
        ).reset_index()

    elif stat_key == "date":
        # Date du maximum horaire et journalier
        idx_h = df.groupby(["lat", "lon"])["max_mm_h"].idxmax()
        idx_j = df.groupby(["lat", "lon"])["max_mm_j"].idxmax()

        df_h = df.loc[idx_h, ["lat", "lon", "max_date_mm_h"]].rename(columns={"max_date_mm_h": "date_max_h"})
        df_j = df.loc[idx_j, ["lat", "lon", "max_date_mm_j"]].rename(columns={"max_date_mm_j": "date_max_j"})

        return pd.merge(df_h, df_j, on=["lat", "lon"])

    elif stat_key == "month":
        # Mois le plus fréquent d'occurrence des maximas horaires et journaliers
        df["mois_max_h"] = df["max_date_mm_h"].str[5:7].astype(int)
        df["mois_max_j"] = df["max_date_mm_j"].str[5:7].astype(int)

        mois_h = df.groupby(["lat", "lon"])["mois_max_h"] \
                .agg(lambda x: x.value_counts().idxmax()) \
                .reset_index(name="mois_pluvieux_h")

        mois_j = df.groupby(["lat", "lon"])["mois_max_j"] \
                .agg(lambda x: x.value_counts().idxmax()) \
                .reset_index(name="mois_pluvieux_j")

        return pd.merge(mois_h, mois_j, on=["lat", "lon"])


    elif stat_key == "numday":
        # Moyenne du nombre de jours de pluie sur la période sélectionnée
        n_years = df["max_date_mm_h"].str[:4].astype(int).nunique()
        return df.groupby(["lat", "lon"])["n_days_gt1mm"] \
                 .sum() \
                 .div(n_years) \
                 .reset_index(name="jours_pluie_moyen")

    else:
        raise ValueError(f"Statistique inconnue : {stat_key}")
    
    progress_bar.empty()  # Efface la barre de progression
    

def get_stat_column_name(stat_key: str, scale_key: str) -> str:
    if stat_key == "mean":
        return f"mean_{scale_key}"
    elif stat_key == "max":
        return f"max_all_{scale_key}"
    elif stat_key == "mean-max":
        return f"max_mean_{scale_key}"
    elif stat_key == "date":
        return f"date_max_{scale_key[-1]}"  # "h" ou "j"
    elif stat_key == "month":
        return f"mois_pluvieux_{scale_key[-1]}"  # "h" ou "j"
    elif stat_key == "numday":
        return "jours_pluie_moyen"
    else:
        raise ValueError(f"Statistique inconnue : {stat_key}")


import plotly.express as px
import pandas as pd

import geopandas as gpd
from shapely.geometry import box

def plot_grid_map(df: pd.DataFrame, column_to_show: str, resolution_km: float = 2.5):
    # Approximation de 1° ≈ 111 km
    delta_deg = resolution_km / 111.0 / 2

    # Création des carrés (polygones) pour chaque point
    polygons = [
        box(lon - delta_deg, lat - delta_deg, lon + delta_deg, lat + delta_deg)
        for lat, lon in zip(df["lat"], df["lon"])
    ]

    gdf = gpd.GeoDataFrame(df[[column_to_show]].copy(), geometry=polygons)
    gdf = gdf.set_crs("EPSG:4326")  # système de coordonnées géographiques

    # Conversion en GeoJSON
    gdf_json = gdf.__geo_interface__

    fig = px.choropleth_mapbox(
        gdf,
        geojson=gdf_json,
        locations=gdf.index,
        color=column_to_show,
        color_continuous_scale="Viridis",
        mapbox_style="carto-positron",
        zoom=5,
        center={"lat": df["lat"].mean(), "lon": df["lon"].mean()},
        opacity=0.8,
        height=600
    )

    fig.update_layout(margin={"r":0, "t":0, "l":0, "b":0})
    return fig


def show(config_path):

    config = load_config(config_path)

    min_years = config["years"]["min"]
    max_years = config["years"]["max"]

    stat_choice, min_year_choice, max_year_choice, season_choice, scale_choice = menu_statisticals(
        min_years,
        max_years,
        STATS,
        SEASON
    )

    stat_choice_key = STATS[stat_choice]
    season_choice_key = SEASON[season_choice]
    scale_choice_key = SCALE[scale_choice]

    try:
        df_all = load_arome_data(
            min_year_choice,
            max_year_choice,
            tuple(season_choice_key),  # tuple hashable
            config  # utilisé pour le hash automatique de cache
        )
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        return


    result_df = compute_statistic_per_point(df_all, stat_choice_key)
    column_to_show = get_stat_column_name(stat_choice_key, scale_choice_key)

    import pandas as pd
    import plotly.graph_objects as go
    from geopy.distance import distance
    import streamlit as st
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import numpy as np

    # Fonction pour générer un carré autour d’un point
    def generate_square(lat, lon, size_km=2.5):
        half_size = size_km / 2
        north = distance(kilometers=half_size).destination((lat, lon), bearing=0)
        south = distance(kilometers=half_size).destination((lat, lon), bearing=180)
        east = distance(kilometers=half_size).destination((lat, lon), bearing=90)
        west = distance(kilometers=half_size).destination((lat, lon), bearing=270)

        square = [
            (north.latitude, west.longitude),
            (north.latitude, east.longitude),
            (south.latitude, east.longitude),
            (south.latitude, west.longitude),
            (north.latitude, west.longitude)
        ]
        return square

    # Normalisation des valeurs et mappage des couleurs
    norm = mcolors.Normalize(vmin=result_df[column_to_show].min(),
                            vmax=result_df[column_to_show].max())
    cmap = cm.get_cmap('viridis')  # tu peux essayer 'plasma', 'inferno', etc.

    # Création de la carte
    fig = go.Figure()

    # Affichage de la progression
    progress_bar = st.progress(0)
    n_total = len(result_df)

    for i, row in result_df.iterrows():
        square_coords = generate_square(row['lat'], row['lon'])
        lats, lons = zip(*square_coords)

        rgba = cmap(norm(row[column_to_show]))
        r, g, b, a = [int(c * 255) if j < 3 else c for j, c in enumerate(rgba)]
        rgba_str = f'rgba({r},{g},{b},{a:.2f})'

        fig.add_trace(go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode='lines',
            fill='toself',
            fillcolor=rgba_str,
            line=dict(color=rgba_str),
            name=f"{row[column_to_show]:.1f}"
        ))

        # Mise à jour de la barre de progression
        if i % 100 == 0 or i == n_total - 1:
            progress_bar.progress(int(i / n_total * 100))

    progress_bar.empty()  # Supprimer la barre une fois terminé


    # Layout
    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            zoom=10,
            center={"lat": result_df['lat'].mean(), "lon": result_df['lon'].mean()}
        ),
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )

    st.plotly_chart(fig)



    # st.write(result_df)
    # if column_to_show in result_df.columns:
    #     st.write(f"Carte de la statistique : `{column_to_show}`")
    #     fig = plot_grid_map(result_df, column_to_show)
    #     st.plotly_chart(fig, use_container_width=True)
    # else:
    #     st.warning(f"La colonne `{column_to_show}` n'est pas disponible dans les résultats.")


