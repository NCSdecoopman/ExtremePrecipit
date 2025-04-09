import streamlit as st
from streamlit_folium import st_folium
import folium

from app.utils.config_utils import *
from app.utils.menus_utils import *
from app.utils.data_utils import *
from app.utils.stats_utils import *
from app.utils.map_utils import *
from app.utils.legends_utils import *
from app.utils.hist_utils import *
from app.utils.scatter_plot_utils import *

import requests

from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.colors as colors
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from branca.colormap import LinearColormap

import plotly.graph_objects as go
import plotly.express as px

import io

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

@st.cache_data
def list_files_and_metadata(echelle, path="data/obs_vs_mod"):
    dir_path = Path(path) / echelle
    files = list(dir_path.glob("lat_*_lon_*_me_*.parquet"))
    data = []
    for f in files:
        match = re.search(r"lat_([-\d.]+)_lon_([-\d.]+)_me_([-\d.]+)", f.name)
        if match:
            lat_str, lon_str, me_str = match.groups()
            try:
                if me_str.endswith("."):
                    me_str = me_str[:-1]
                lat, lon, me = float(lat_str), float(lon_str), float(me_str)
                data.append({"file": f, "lat": lat, "lon": lon, "me": me})
            except ValueError:
                st.warning(f"Impossible de parser le fichier : {f.name}")
    return pd.DataFrame(data)

@st.cache_data
def list_files_and_metadata_hf(echelle: str) -> pd.DataFrame:
    """
    Charge directement la table des fichiers depuis Hugging Face.
    (Beaucoup plus rapide que d'appeler l'API à chaque fois)
    """
    index_url = (
        f"https://huggingface.co/datasets/ncsdecoopman/extreme-precip-binaires/"
        f"resolve/main/data/obs_vs_mod/obs_vs_mod_index_{echelle}.csv"
    )
    return pd.read_csv(index_url)


def import_data(file):
    return pd.read_parquet(file)

def import_data_hf(file: str) -> pd.DataFrame:
    """
    Importe un fichier Parquet depuis une URL distante sur Hugging Face.
    Prend en charge les chemins avec anti-slashs éventuels.
    """
    # Nettoyage du chemin (convertit les \ en /)
    file = file.replace("\\", "/")
    
    # Construire l'URL complète
    url = f"https://huggingface.co/datasets/ncsdecoopman/extreme-precip-binaires/resolve/main/{file}"
    
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"Erreur de téléchargement : {url}")
    
    return pd.read_parquet(io.BytesIO(response.content))


def generate_scatter_plot_interactive(df, x_label, y_label):
    fig = px.scatter(
        df,
        x=x_label,
        y=y_label,
        hover_data=["Date"],  # On affiche la date au survol
        title="",
        opacity=0.6,
        width=500,
        height=500
    )

    min_val = min(df[x_label].min(), df[y_label].min())
    max_val = max(df[x_label].max(), df[y_label].max())
    # Ligne y = x
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='y = x'
        )
    )

    fig.update_layout(
        xaxis=dict(range=[-0.1, df[x_label].max()]),
        yaxis=dict(range=[-0.1, df[y_label].max()])
    )
    return fig



def generate_hexbin_plot_interactive(df, x_label, y_label, gridsize=10):
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from collections import defaultdict

    # Compute 2D histogram bins
    raw_counts, xedges, yedges = np.histogram2d(
        df[x_label], df[y_label], bins=gridsize
    )

    total_points = np.nansum(raw_counts)
    percent_counts = (raw_counts / total_points) * 100  # Convert to percentage

    # Bin centers
    x_centers = 0.5 * (xedges[:-1] + xedges[1:])
    y_centers = 0.5 * (yedges[:-1] + yedges[1:])
    
    # Assign each point to a bin
    x_bin = np.digitize(df[x_label], xedges) - 1
    y_bin = np.digitize(df[y_label], yedges) - 1

    # Filter valid bins
    mask = (x_bin >= 0) & (x_bin < gridsize) & (y_bin >= 0) & (y_bin < gridsize)
    df = df[mask]
    x_bin = x_bin[mask]
    y_bin = y_bin[mask]

    # Create a mapping from bin to date stats
    bin_dates = defaultdict(list)
    for xb, yb, date in zip(x_bin, y_bin, df["Date"]):
        bin_dates[(xb, yb)].append(date)

    # Tooltips with dates + counts + percentage
    date_labels = np.full((gridsize, gridsize), '', dtype=object)
    for (xb, yb), dates in bin_dates.items():
        min_date = pd.to_datetime(dates).min().strftime("%Y-%m-%d")
        max_date = pd.to_datetime(dates).max().strftime("%Y-%m-%d")
        count = len(dates)
        pct = (count / total_points) * 100
        date_labels[yb, xb] = (
            f"Dates: {min_date} → {max_date}<br>"
            f"n = {count}<br>"
            f"{pct:.2f} % des points"
        )

    # Replace zero counts by NaN
    percent_counts = np.where(raw_counts == 0, np.nan, percent_counts)

    fig = go.Figure(data=go.Heatmap(
        z=percent_counts.T,
        x=x_centers,
        y=y_centers,
        colorscale='Inferno',
        colorbar=dict(title='%'),
        hoverongaps=False,
        text=date_labels.T,
        hovertemplate=
            f'{x_label}: %{{x:.2f}}<br>' +
            f'{y_label}: %{{y:.2f}}<br>' +
            '%{text}<extra></extra>'
    ))

    min_val = min(df[x_label].min(), df[y_label].min())
    max_val = max(df[x_label].max(), df[y_label].max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='y = x'
        )
    )
    fig.update_layout(
        title="",
        xaxis_title=x_label,
        yaxis_title=y_label,
        width=500,
        height=500,
        xaxis=dict(range=[-0.1, df[x_label].max()]),
        yaxis=dict(range=[-0.1, df[y_label].max()])
    )

    return fig


def generate_error_histogram(df, model_col="arome", obs_col="obs"):
    df["erreur"] = df[model_col] - df[obs_col]

    fig = px.histogram(
        df,
        x="erreur",
        nbins=200,
        title="",
        labels={"erreur": "Erreur"},
        opacity=0.75
    )

    fig.update_layout(
        xaxis_title="Erreur (mod - obs)",
        yaxis_title="Fréquence",
        bargap=0.1,
        width=500,
        height=500
    )

    return fig


def show(config_path):
    Echelle = st.selectbox("Choix de l'échelle temporelle", ["Horaire"], key="scale_choice") #", "Journalière"
    echelle = Echelle.lower()
    
    #df_files = list_files_and_metadata(echelle)
    #df_files.to_csv("data/obs_vs_mod/obs_vs_mod_index_horaire.csv", index=False)
    df_files = list_files_and_metadata_hf(echelle)

    # Carte Folium
    m = folium.Map(location=[46.9, 1.7], zoom_start=6, tiles="OpenStreetMap")

    # Palette bleu → blanc → rouge centrée sur 0
    vmin = df_files["me"].min()
    vmax = df_files["me"].max()
    vcenter = 0  # centre sur 0

    # Colormap personnalisée avec blanc au centre
    colorscale = LinearSegmentedColormap.from_list(
        "blue_white_red", ["blue", "white", "red"], N=256
    )
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    # Création carte Folium
    m = folium.Map(location=[46.9, 1.7], zoom_start=6, tiles="OpenStreetMap")

    # Ajout des points
    for _, row in df_files.iterrows():
        me = row["me"]
        rgba = colorscale(norm(me))
        hex_color = colors.rgb2hex(rgba)
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4,
            color=hex_color,
            fill=True,
            fill_color=hex_color,
            fill_opacity=0.9,
            tooltip=f"lat: {row['lat']:.2f}, lon: {row['lon']:.2f}, me: {row['me']:.2f}"
        ).add_to(m)


    # Ajout d’une échelle personnalisée (branca)
    colormap = LinearColormap(
        colors=["blue", "white", "red"],
        vmin=vmin,
        vmax=vmax
    )
    colormap.caption = "Erreur moyenne (ME)"
    colormap.add_to(m)

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
            row = df_files[
                (df_files["lat"] == lat_clicked) & (df_files["lon"] == lon_clicked)
            ]
        else:
            row = None

    with col2:
        retrait_0_0 = st.checkbox("Retirer les points où AROME = 0 et OBS = 0 pour une même date (pas de pluie)", value=True)            
        if row is not None and not row.empty:
            file = row.iloc[0]["file"]
            df = import_data_hf(file)

            if retrait_0_0:
                df = df[~((df["pr_mod"] == 0) & (df["pr_obs"] == 0))]
            
            df = pd.DataFrame({
                "arome": df["pr_mod"],
                "obs": df["pr_obs"],
                "Date": df["time"]
            })

            # Calcul des métriques
            rmse = np.sqrt(mean_squared_error(df["obs"], df["arome"]))
            mae = mean_absolute_error(df["obs"], df["arome"])
            bias = np.mean(df["arome"] - df["obs"])
            r2 = r2_score(df["obs"], df["arome"])

            # Affichage
            col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
            with col_metrics1:
                st.metric("RMSE", f"{rmse:.2f}")
            with col_metrics2:
                st.metric("MAE", f"{mae:.2f}")
            with col_metrics3:
                st.metric("Biais", f"{bias:.2f}")
            with col_metrics4:
                st.metric("R²", f"{r2:.3f}")

            col3, col4, col5 = st.columns([2.5, 2.5, 2.5])

            with col3:
                # Exemple d'appel Streamlit
                fig = generate_scatter_plot_interactive(df, "arome", "obs")
                st.plotly_chart(fig, use_container_width=True)

            with col4:
                # Ou :
                fig = generate_hexbin_plot_interactive(df, "arome", "obs")
                st.plotly_chart(fig, use_container_width=True)

            with col5:
                fig = generate_error_histogram(df, "arome", "obs")
                st.plotly_chart(fig, use_container_width=True)

            