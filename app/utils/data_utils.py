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
    df_observed = df.copy()

    # Agrégation des n_nan et n_total par station
    station_counts = df_observed.groupby(["lat", "lon"], as_index=False).agg(
        n_nan_total=("n_nan", "sum"),
        n_total_total=("n_total", "sum")
    )

    # Calcul du ratio global de NaN
    station_counts["global_nan_ratio"] = station_counts["n_nan_total"] / station_counts["n_total_total"]

    # Filtrage des stations ayant un taux de NaN inférieur au seuil
    good_stations = station_counts[station_counts["global_nan_ratio"] < nan_limit][["lat", "lon"]]

    # Merge pour ne conserver que les bonnes stations
    df_cleaned = df_observed.merge(good_stations, on=["lat", "lon"], how="inner")

    return df_cleaned
