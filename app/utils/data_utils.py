import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download
from concurrent.futures import ThreadPoolExecutor, as_completed

SEASON_TO_FILENAME = {
    "hjf": "hjf.parquet",       # hiver : déc-janv-févr (déc décalé à l'année précédente)
    "mam": "mam.parquet",       # mars-avril-mai
    "jja": "jja.parquet",       # juin-juil-août
    "son": "son.parquet",       # sept-oct-nov
    "hydro": "hydro.parquet",   # sept à août
}

@st.cache_data(show_spinner=False)
def load_season_parquet_from_huggingface(year: int, season: str, repo_id: str, base_path: str) -> pd.DataFrame:
    filename=f"{base_path}/{year:04d}/{SEASON_TO_FILENAME[season]}"
    return pd.read_parquet(filename)

@st.cache_data(show_spinner=False)
def load_seasonal_data(min_year: int, max_year: int, season: str, config) -> pd.DataFrame:
    if season not in SEASON_TO_FILENAME:
        raise ValueError(f"Saison inconnue : {season}")

    repo_id = config["repo_id"]
    base_path = config["statisticals"]["modelised"]

    tasks = list(range(min_year, max_year + 1))
    dataframes = []
    errors = []

    with st.spinner("Chargement des fichiers saisonniers..."):
        progress_bar = st.progress(0)
        total = len(tasks)
        completed = 0

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {
                executor.submit(load_season_parquet_from_huggingface, year, season, repo_id, base_path): year
                for year in tasks
            }

            for future in as_completed(futures):
                year = futures[future]
                try:
                    df = future.result()
                    dataframes.append(df)
                except Exception as e:
                    errors.append(f"{year} ({season}) : {e}")
                completed += 1
                progress_bar.progress(completed / total)

    progress_bar.empty()

    if errors:
        for err in errors:
            st.warning(f"Erreur : {err}")

    if not dataframes:
        raise ValueError("Aucune donnée chargée.")

    return pd.concat(dataframes, ignore_index=True)



# import pandas as pd
# import streamlit as st
# from huggingface_hub import hf_hub_download
# from concurrent.futures import ThreadPoolExecutor, as_completed

# @st.cache_data(show_spinner=False)
# def load_parquet_from_huggingface_cached(year: int, month: int, repo_id: str, base_path: str) -> pd.DataFrame:
#     hf_path = hf_hub_download(
#         repo_id=repo_id,
#         filename=f"{base_path}/{year:04d}/{month:02d}.parquet",
#         repo_type="dataset"
#     )
#     return pd.read_parquet(hf_path)

# @st.cache_data(show_spinner=False)
# def load_arome_data(min_year_choice: int, max_year_choice: int, months: list[int], config) -> list:
#     repo_id = config["repo_id"]
#     base_path = config["statisticals"]["modelised"]

#     tasks = []

#     for year in range(min_year_choice, max_year_choice + 1):
#         for month in months:
#             if month >= 1 and month <= 8 and month < months[0]:
#                 actual_year = year + 1
#             else:
#                 actual_year = year
#             if actual_year > max_year_choice:
#                 continue
#             tasks.append((actual_year, month))

#     dataframes = []
#     errors = []

#     with st.spinner("Chargement des fichiers..."):
#         progress_bar = st.progress(0)
#         total = len(tasks)
#         completed = 0

#     with ThreadPoolExecutor(max_workers=16) as executor:
#         futures = {
#             executor.submit(load_parquet_from_huggingface_cached, y, m, repo_id, base_path): (y, m)
#             for y, m in tasks
#         }

#         for future in as_completed(futures):
#             y, m = futures[future]
#             try:
#                 df = future.result()
#                 dataframes.append(df)
#             except Exception as e:
#                 errors.append(f"{y}-{m:02d} : {e}")
#             completed += 1
#             progress_bar.progress(completed / total)

#     progress_bar.empty()  # Efface la barre de progression

#     if errors:
#         for err in errors:
#             st.warning(f"Erreur : {err}")

#     if not dataframes:
#         raise ValueError("Aucune donnée chargée.")

#     return pd.concat(dataframes, ignore_index=True)