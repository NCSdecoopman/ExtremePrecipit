import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.utils.config_utils import load_config
from app.utils.menus_utils import menu_statisticals

st.markdown("<h3>Visualisation des précipitations</h3>", unsafe_allow_html=True)

STATS = {
    "Moyenne": "mean",
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


def load_arome_data(min_year_choice: int, max_year_choice: int, months: list[int], config) -> list:
    """
    Charge les fichiers Parquet d'une saison donnée entre deux années depuis Hugging Face en parallèle.

    Returns:
        list[pd.DataFrame]: Liste des DataFrames chargés
    """
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
    with ThreadPoolExecutor(max_workers=8) as executor:
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
                st.warning(f"Erreur lors du chargement de {y}-{m:02d} : {e}")

    return dataframes


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

    dfs = load_arome_data(min_year_choice, max_year_choice, season_choice_key, config)

    if not dfs:
        st.error("Aucune donnée n'a pu être chargée.")
        return

    df_all = pd.concat(dfs, ignore_index=True)
    st.success(f"{len(df_all)} lignes chargées.")
    st.dataframe(df_all.head())
