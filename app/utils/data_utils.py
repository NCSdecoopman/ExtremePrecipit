import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.utils.config_utils import menu_config

def load_season(year: int, season_key: str, base_path: str) -> pd.DataFrame:
    filename=f"{base_path}/{year:04d}/{season_key}.parquet"
    return pd.read_parquet(filename)

# def load_data(type_data: str, echelle: str, min_year: int, max_year: int, season: str, config) -> pd.DataFrame:
#     _, SEASON, _ = menu_config()
#     if season not in SEASON.values():
#         raise ValueError(f"Saison inconnue : {season}")

#     base_path = f'{config["statisticals"][type_data]}/{echelle}'

#     tasks = list(range(min_year, max_year + 1))
#     dataframes = []
#     errors = []


#     with ThreadPoolExecutor(max_workers=16) as executor:
#         futures = {
#             executor.submit(load_season, year, season, base_path): year
#             for year in tasks
#         }

#         for future in as_completed(futures):
#             year = futures[future]
#             try:
#                 df = future.result()
#                 if not isinstance(df, pd.DataFrame):
#                     st.warning(f"{year} : Objet inattendu de type {type(df)}")
#                 else:
#                     dataframes.append(df)
#             except Exception as e:
#                 errors.append(f"{year} ({season}) : {e}")


#     if errors:
#         for err in errors:
#             st.warning(f"Erreur : {err}")

#     if not dataframes:
#         raise ValueError("Aucune donnée chargée.")

#     return pd.concat(dataframes, ignore_index=True)

def load_data(type_data: str, echelle: str, min_year: int, max_year: int, season: str, config) -> pd.DataFrame:
    _, SEASON, _ = menu_config()
    if season not in SEASON.values():
        raise ValueError(f"Saison inconnue : {season}")

    base_path = f'{config["statisticals"][type_data]}/{echelle}'

    dataframes = []
    errors = []

    for year in range(min_year, max_year + 1):
        try:
            df = load_season(year, season, base_path)
            if not isinstance(df, pd.DataFrame):
                st.warning(f"{year} : Objet inattendu de type {type(df)}")
            else:
                dataframes.append(df)
        except Exception as e:
            errors.append(f"{year} ({season}) : {e}")

    if errors:
        for err in errors:
            st.warning(f"Erreur : {err}")

    if not dataframes:
        raise ValueError("Aucune donnée chargée.")

    return pd.concat(dataframes, ignore_index=True)


def cleaning_data_observed(df, nan_limit: float = 0.1):
    # Agrégation plus rapide sans copy initial
    station_counts = df.groupby(["lat", "lon"], sort=False).agg(
        nan_ratio=("nan_ratio", "mean")
    ).reset_index()

    # Stations valides
    valid = station_counts.loc[station_counts["nan_ratio"] <= nan_limit, ["lat", "lon"]]

    # Utiliser merge avec indicateur pour filtrer plus vite
    df_filtered = df.merge(valid, on=["lat", "lon"], how="inner")

    return df_filtered

