import polars as pl
import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

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
