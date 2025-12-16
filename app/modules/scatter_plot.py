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

import polars as pl

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
    files = list(dir_path.glob("NUM_POSTE_*_R2_*.parquet"))
    data = []

    for f in files:
        match = re.match(
            r"NUM_POSTE_(\d+)_R2_((-?0\.\d{1,3})|-?1\.0{1,3}|NaN)\.parquet$", f.name
        )
        if match:
            num_poste_str, r2_str = match.groups()[0], match.groups()[1]
            try:
                num_poste = int(num_poste_str)
                r2 = float("nan") if r2_str == "NaN" else float(r2_str)
                if not np.isnan(r2) and not (-1 <= r2 <= 1):
                    continue  # silencieusement ignoré
                data.append({"file": str(f), "NUM_POSTE": num_poste, "R2": r2}) 
            except ValueError:
                continue  # silencieusement ignoré
        # on ignore les autres formats non reconnus
    return pl.DataFrame(data)


def import_data(file):
    return pd.read_parquet(file)




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


def plot_time_series(df, time_col="Date", obs_col="Station", mod_col="AROME"):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df[time_col],
        y=df[obs_col],
        mode="lines+markers",
        name="Observé",
        line=dict(color="blue")
    ))

    fig.add_trace(go.Scatter(
        x=df[time_col],
        y=df[mod_col],
        mode="lines+markers",
        name="Modélisé (AROME)",
        line=dict(color="red")
    ))

    fig.update_layout(
        title="Série temporelle Observé vs AROME",
        xaxis_title="Date",
        yaxis_title="Précipitations (mm)",
        legend=dict(x=0, y=1),
        height=500
    )

    return fig



def show(config_path):
    Echelle = st.selectbox("Choix de l'échelle temporelle", ["Quotidien", "Horaire"], key="scale_choice")
    echelle = Echelle.lower()
    
    df_files = list_files_and_metadata(echelle)
    df_files = add_metadata(df_files, "mm_h" if echelle == "horaire" else "mm_j", type="observed")
    df_files = df_files.to_pandas()
    
    # Carte Folium
    m = folium.Map(location=[46.9, 1.7], zoom_start=6, tiles="OpenStreetMap")

    # Palette bleu → blanc → rouge centrée sur 0
    vmin = df_files["R2"].min()
    vmax = df_files["R2"].max()
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
        r2 = row["R2"]
        rgba = colorscale(norm(r2))
        hex_color = colors.rgb2hex(rgba)
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=0.5,
            color=hex_color,
            fill=True,
            fill_color=hex_color,
            fill_opacity=0.9,
            tooltip=f"lat: {row['lat']:.2f}, lon: {row['lon']:.2f}, R²: {row['R2']:.2f}"
        ).add_to(m)


    # Ajout d’une échelle personnalisée (branca)
    colormap = LinearColormap(
        colors=["blue", "white", "red"],
        vmin=vmin,
        vmax=vmax
    )
    colormap.caption = "R²"
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
        retrait_0_0 = st.checkbox("Sélectionner les cas extrême (seuil à 10 mm/h), dans l’un ou l’autre", value=True)
        value_0_0 =  st.slider(
            "Seuil",
            min_value=0,
            max_value=50,
            value=10,
            step=1,
            key="value_0_0"
        )        
        if row is not None and not row.empty:
            file = row.iloc[0]["file"]
            df = import_data(file)
            
            if df.empty:
                st.warning("Pas de données pour cette station")

            else:
                if retrait_0_0:
                    df = df[(df["pr_obs"] >= value_0_0) | (df["pr_mod"] >= value_0_0)]
                
                df = pd.DataFrame({
                    "AROME": df["pr_mod"],
                    "Station": df["pr_obs"],
                    "Date": df["time"]
                })

                # Calcul des métriques
                rmse = np.sqrt(mean_squared_error(df["Station"], df["AROME"]))
                mae = mean_absolute_error(df["Station"], df["AROME"])
                bias = np.mean(df["AROME"] - df["Station"])
                r2 = r2_score(df["Station"], df["AROME"])

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
                    fig = generate_scatter_plot_interactive(df, "AROME", "Station")
                    st.plotly_chart(fig, use_container_width=True)

                with col4:
                    # Ou :
                    fig = generate_hexbin_plot_interactive(df, "AROME", "Station")
                    st.plotly_chart(fig, use_container_width=True)

                with col5:
                    fig = generate_error_histogram(df, "AROME", "Station")
                    st.plotly_chart(fig, use_container_width=True)

            fig = plot_time_series(df)
            st.plotly_chart(fig, use_container_width=True)

            