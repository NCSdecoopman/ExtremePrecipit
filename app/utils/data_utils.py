import polars as pl
import numpy as np
import streamlit as st
from scipy.spatial import cKDTree

from app.utils.config_utils import menu_config

def load_season(year: int, season_key: str, base_path: str) -> pl.DataFrame:
    filename = f"{base_path}/{year:04d}/{season_key}.parquet"
    return pl.read_parquet(filename)

def load_data(type_data: str, echelle: str, min_year: int, max_year: int, season: str, config) -> pl.DataFrame:
    _, SEASON, _ = menu_config()
    if season not in SEASON.values():
        raise ValueError(f"Saison inconnue : {season}")

    base_path = f'{config["statisticals"][type_data]}/{echelle}'

    dataframes = []
    errors = []

    for year in range(min_year, max_year + 1):
        try:
            df = load_season(year, season, base_path)

            # Conversion explicite des colonnes dates uniquement si elles existent
            for col in ["max_date_mm_h", "max_date_mm_j"]:
                if col in df.columns:
                    df = df.with_columns(
                        pl.col(col)
                        .cast(pl.Utf8)  # s'assure qu'on peut parser avec str.strptime
                        .str.strptime(pl.Datetime, format="%Y-%m-%d", strict=False)
                        .cast(pl.Utf8)  # retour sous forme de string (comme dans l'ancien code Pandas)
                    )

            dataframes.append(df)

        except Exception as e:
            errors.append(f"{year} ({season}) : {e}")

    if errors:
        for err in errors:
            st.warning(f"Erreur : {err}")

    if not dataframes:
        raise ValueError("Aucune donnée chargée.")

    return pl.concat(dataframes, how="vertical")



def cleaning_data_observed(df: pl.DataFrame, nan_limit: float = 0.1) -> pl.DataFrame:
    # Moyenne du ratio de NaN par station (lat, lon)
    station_counts = (
        df.group_by(["lat", "lon"])
        .agg(pl.col("nan_ratio").mean().alias("nan_ratio"))
    )

    # Stations valides selon le seuil
    valid = station_counts.filter(pl.col("nan_ratio") <= nan_limit)

    # Jointure pour ne garder que les stations valides
    df_filtered = df.join(valid.select(["lat", "lon"]), on=["lat", "lon"], how="inner")

    return df_filtered

def add_alti(df: pl.DataFrame):
    # Charger les altitudes avec Polars
    df_alt = pl.read_csv("data/metadonnees/altitude_model.csv")
    # Harmoniser le type de colonnes AVANT le join
    df_alt = df_alt.with_columns([
        pl.col("lat").cast(pl.Float32),
        pl.col("lon").cast(pl.Float32)
    ])
    # Join avec les données modélisées (lat/lon identiques)
    return df.join(df_alt, on=["lat", "lon"], how="left")


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
