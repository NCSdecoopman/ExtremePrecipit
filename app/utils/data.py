import os
import pandas as pd
import time
import streamlit as st

# Récupération des années disponibles
def get_available_years(output_dir):
    file_path = os.path.join(output_dir, "preanalysis", "nan_resol_temp.csv")
    df = pd.read_csv(file_path)
    return sorted(df["Year"])

# Charger les données sur une période sélectionnée
@st.cache_data
def load_data(start_year, end_year, stat, output_dir):
    all_dfs = []
    total_years = end_year - start_year + 1
    progress_bar = st.progress(0)  # Initialisation de la barre de progression

    for i, year in enumerate(range(start_year, end_year + 1)):
        file_path = os.path.join(output_dir, stat, f"pr_{stat}_grid_{year}.parquet")
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            
            if stat == "p95_global": # Si on charge p95_global, il faut s'assurer que la structure reste correcte
                df = df[["year", "pr_p95_global"]]  # S'assurer que seules ces colonnes existent
            else:
                df["year"] = year  # Ajouter une colonne pour l'année

            all_dfs.append(df)

        # Mettre à jour la progression
        progress_value = int(((i + 1) / total_years) * 100)
        progress_bar.progress(progress_value)
        time.sleep(0.1)  # Pause pour rendre l'animation visible

    time.sleep(0.5)  # Pause finale pour que l'utilisateur voit bien la barre à 100%
    progress_bar.empty()  # Supprimer la barre après chargement complet

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return None