import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.utils.config_utils import menu_config

def load_season(year: int, season_key: str, base_path: str) -> pd.DataFrame:
    filename=f"{base_path}/{year:04d}/{season_key}.parquet"
    return pd.read_parquet(filename)

def load_data(type_data: str, echelle: str, min_year: int, max_year: int, season: str, config) -> pd.DataFrame:
    _, SEASON, _ = menu_config()
    if season not in SEASON.values():
        raise ValueError(f"Saison inconnue : {season}")

    base_path = f'{config["statisticals"][type_data]}/{echelle}'

    tasks = list(range(min_year, max_year + 1))
    dataframes = []
    errors = []


    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {
            executor.submit(load_season, year, season, base_path): year
            for year in tasks
        }

        for future in as_completed(futures):
            year = futures[future]
            try:
                df = future.result()
                dataframes.append(df)
            except Exception as e:
                errors.append(f"{year} ({season}) : {e}")

    if errors:
        for err in errors:
            st.warning(f"Erreur : {err}")

    if not dataframes:
        raise ValueError("Aucune donnée chargée.")

    return pd.concat(dataframes, ignore_index=True)