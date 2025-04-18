import streamlit as st
from streamlit_folium import st_folium
import folium

from pathlib import Path
from typing import Optional
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
import plotly.graph_objects as go
import pandas as pd


def load_season(year: int, cols: tuple, season_key: str, base_path: str) -> pl.DataFrame:
    filename = f"{base_path}/{year:04d}/{season_key}.parquet"
    return pl.read_parquet(filename, columns=cols)

def load_data(intputdir: str, season: str, echelle: str, cols: tuple, min_year: int, max_year: int) -> pl.DataFrame:
    dataframes = []
    errors = []

    for year in range(min_year, max_year + 1):
        try:
            df = load_season(year, cols, season, intputdir)
            # Ajouter une colonne constante 'year' = 2024
            df = df.with_columns([
                pl.lit(year).alias("year")
            ])
            dataframes.append(df)

        except Exception as e:
            errors.append(f"{year} ({season}) : {e}")

    if errors:
        for err in errors:
            raise ValueError(f"Erreur : {err}")

    if not dataframes:
        raise ValueError("Aucune donnée chargée.")

    return pl.concat(dataframes, how="vertical")


def filter_nan(df: pl.DataFrame):
    return df.drop_nulls(subset=["xi", "mu", "sigma"])

def find_matching_point_fast(df: pl.DataFrame, lat_obs: float, lon_obs: float) -> tuple[float, float]:
    df_with_dist = df.with_columns(
        (
            (pl.col("lat") - lat_obs).pow(2) + (pl.col("lon") - lon_obs).pow(2)
        ).sqrt().alias("distance")
    )
    idx = df_with_dist["distance"].arg_min()
    row = df_with_dist[idx]

    return row["lat"].item(), row["lon"].item()



def plot_gev_return_curve(
    df_mod_ret_pd: pd.DataFrame,
    df_obs_ret_pd: pd.DataFrame,
    maximas_obs: Optional[pd.Series] = None,
    maximas_mod: Optional[pd.Series] = None,
    max_return: int = 100,
    title: str = "",
    height: int = 700
):
    """
    Affiche une courbe interactive de période de retour GEV (modélisé vs observé) + maximas annuels.

    Paramètres
    ----------
    df_mod_ret_pd : pd.DataFrame
        DataFrame avec "return_period" et "return_level" modélisé.
    df_obs_ret_pd : pd.DataFrame
        DataFrame avec "return_period" et "return_level" observé.
    maximas_obs : pd.Series, optionnel
        Série des maximas annuels (bruts) pour afficher les points empiriques observés.
    maximas_mod : pd.Series, optionnel
        Série des maximas annuels (bruts) pour afficher les points empiriques modélisés.
    max_return : int, optionnel
        Période de retour maximale à afficher sur l'axe des x.
    title : str
        Titre du graphique.
    height : int
        Hauteur du graphique en pixels.
    """

    fig = go.Figure()

    # Filtrer les périodes de retour en fonction de la limite max_return
    df_mod_ret_pd = df_mod_ret_pd[df_mod_ret_pd["return_period"] <= max_return]
    df_obs_ret_pd = df_obs_ret_pd[df_obs_ret_pd["return_period"] <= max_return]

    # Ajout du ruban IC modélisé si disponible
    if {"lower", "upper"}.issubset(df_mod_ret_pd.columns):
        fig.add_traces([
            go.Scatter(
                x=pd.concat([df_mod_ret_pd["return_period"], df_mod_ret_pd["return_period"][::-1]]),
                y=pd.concat([df_mod_ret_pd["upper"], df_mod_ret_pd["lower"][::-1]]),
                fill="toself",
                fillcolor="rgba(0, 0, 255, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="IC 95% AROME"
            )
        ])

    # Modélisé
    fig.add_trace(go.Scatter(
        x=df_mod_ret_pd["return_period"],
        y=df_mod_ret_pd["return_level"],
        mode="lines+markers",
        name="AROME",
        marker=dict(symbol="circle", color="blue"),
        hovertemplate="Année : %{x}<br>Valeur : %{y:.1f} mm<extra></extra>"
    ))

    # Maximas annuels modélisés empiriques
    if maximas_mod is not None and len(maximas_mod) > 0:
        maximas_sorted_mod = np.sort(maximas_mod)[::-1]
        n_mod = len(maximas_sorted_mod)
        T_empirical_mod = (n_mod + 1) / np.arange(1, n_mod + 1)
        mask_mod = T_empirical_mod <= max_return

        fig.add_trace(go.Scatter(
            x=T_empirical_mod[mask_mod],
            y=maximas_sorted_mod[mask_mod],
            mode="markers",
            name="Maximas annuels (AROME)",
            marker=dict(symbol="cross", color="blue", size=5),
            hovertemplate="Année : %{x}<br>Valeur : %{y:.1f} mm<extra></extra>"
        ))

    # Ajout du ruban IC observé si dispo
    if {"lower", "upper"}.issubset(df_obs_ret_pd.columns):
        fig.add_traces([
            go.Scatter(
                x=pd.concat([df_obs_ret_pd["return_period"], df_obs_ret_pd["return_period"][::-1]]),
                y=pd.concat([df_obs_ret_pd["upper"], df_obs_ret_pd["lower"][::-1]]),
                fill="toself",
                fillcolor="rgba(255, 0, 0, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="IC 95% Station"
            )
        ])

    # Observé
    fig.add_trace(go.Scatter(
        x=df_obs_ret_pd["return_period"],
        y=df_obs_ret_pd["return_level"],
        mode="lines+markers",
        name="Station",
        marker=dict(symbol="circle", color="red"),
        hovertemplate="Année : %{x}<br>Valeur : %{y:.1f} mm<extra></extra>"
    ))

    # Maximas annuels observés empiriques
    if maximas_obs is not None and len(maximas_obs) > 0:
        maximas_sorted = np.sort(maximas_obs)[::-1]
        n = len(maximas_sorted)
        T_empirical = (n + 1) / np.arange(1, n + 1)
        mask_obs = T_empirical <= max_return

        fig.add_trace(go.Scatter(
            x=T_empirical[mask_obs],
            y=maximas_sorted[mask_obs],
            mode="markers",
            name="Maximas annuels (Station)",
            marker=dict(symbol="cross", color="red", size=5),
            hovertemplate="Année : %{x}<br>Valeur : %{y:.1f} mm<extra></extra>"
        ))

    fig.update_layout(
        title=title,
        xaxis=dict(
            title="Période de retour (années)",
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray"
        ),
        yaxis=dict(
            title="Niveau de précipitation (mm)",
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray"
        ),
        legend=dict(title=None),
        template="simple_white",
        height=height
    )

    st.plotly_chart(fig, use_container_width=True)




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
    
    try:
        df_observed_gev = pl.read_parquet(obs_dir / "gev_param.parquet")
    except Exception as e:
        st.error(f"Erreur lors du chargement des paramètres observés : {e}")
        return
    
    # Chargement des données de correspondances entre NUM_POSTE obs et mod
    df_obs_vs_mod = pl.read_csv(f"data/metadonnees/obs_vs_mod/obs_vs_mod_{echelle}.csv")
    
    # Suppression des NaN
    df_observed_gev = filter_nan(df_observed_gev) 
    df_observed_gev = add_metadata(df_observed_gev, "mm_h" if echelle=="horaire" else "mm_j", type="observed")

    # Carte centrée
    m = folium.Map(location=[46.9, 1.7], zoom_start=5.5, tiles="OpenStreetMap")

    for row in df_observed_gev.iter_rows(named=True):
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=2,
            fill=True,
            fill_color="white",
            fill_opacity=1.0,
            tooltip=f"lat: {row['lat']:.2f}, lon: {row['lon']:.2f}"
        ).add_to(m)

    m.add_child(folium.LatLngPopup())  # Affiche les lat/lon au clic

    col1, col2 = st.columns([2.5, 3])
    with col1:
        map_output = st_folium(m, width=600, height=600)

    with col2:
        if map_output and map_output.get("last_clicked"):
            lat_clicked = map_output["last_clicked"]["lat"]
            lon_clicked = map_output["last_clicked"]["lng"]


            # Récupérer le NUM_POSTE correspondant
            matched_row = df_observed_gev.filter(
                (pl.col("lat") == lat_clicked) & (pl.col("lon") == lon_clicked)
            ).select("NUM_POSTE")

            if matched_row.height == 0:
                st.warning("Aucun point correspondant trouvé.")
                return

            num_poste_obs = int(matched_row["NUM_POSTE"][0])
            

            # Trouver le NUM_POSTE modélisé correspondant
            matched_row = df_obs_vs_mod.filter(
                (pl.col("NUM_POSTE_obs") == num_poste_obs)
            )

            if matched_row.height == 0:
                st.warning("Aucune correspondance modélisée trouvée.")
                return
            
            num_poste_mod = int(matched_row["NUM_POSTE_mod"][0])
      

            # CAST
            df_observed = df_observed.with_columns(pl.col("NUM_POSTE").cast(pl.Int32))
            df_modelised = df_modelised.with_columns(pl.col("NUM_POSTE").cast(pl.Int32))

            # Filtrer les données observées et modélisées
            df_obs_ret = df_observed.filter((pl.col("NUM_POSTE") == num_poste_obs))
            df_mod_ret = df_modelised.filter((pl.col("NUM_POSTE") == num_poste_mod))

            # Extraire les périodes de retour et niveaux associés
            df_obs_ret_pd = df_obs_ret.select(["return_period", "return_level", "lower", "upper"]).to_pandas()
            df_mod_ret_pd = df_mod_ret.select(["return_period", "return_level", "lower", "upper"]).to_pandas()

            # Charger les maximas annuels bruts
            try:
                # Observé
                scale = "max_mm_j" if echelle == "quotidien" else "max_mm_h"
                df_stats_obs = load_data(f"data/statisticals/observed/{echelle}", "hydro", f"{echelle}", ["NUM_POSTE", f"{scale}"], 1960, 2010)
                df_stats_obs = df_stats_obs.with_columns(pl.col("NUM_POSTE").cast(pl.Int32))

                df_stats_point_obs = df_stats_obs.filter((pl.col("NUM_POSTE") == num_poste_obs))
                maximas_annuels_obs = df_stats_point_obs["max_mm_j" if echelle == "quotidien" else "max_mm_h"].to_pandas()

                # Modélisé
                df_stats_mod = load_data("data/statisticals/modelised/horaire", "hydro", f"{echelle}", ["NUM_POSTE", f"{scale}"], 1960, 2010)
                df_stats_mod = df_stats_mod.with_columns(pl.col("NUM_POSTE").cast(pl.Int32))


                df_stats_point_mod = df_stats_mod.filter((pl.col("NUM_POSTE") == num_poste_mod))
                maximas_annuels_mod = df_stats_point_mod["max_mm_j" if echelle == "quotidien" else "max_mm_h"].to_pandas()

            except Exception as e:
                maximas_annuels_obs = None
                maximas_annuels_mod = None
                st.warning(f"Impossible de charger les maximas annuels : {e}")

            # Slider pour limiter la période de retour affichée
            max_return_period = st.slider(
                "Période de retour maximale à afficher (années)", 
                min_value=2, max_value=100, value=80, step=1
            )
            # Affichage de la courbe interactive
            plot_gev_return_curve(df_mod_ret_pd, df_obs_ret_pd, maximas_annuels_obs, maximas_annuels_mod, max_return_period)
