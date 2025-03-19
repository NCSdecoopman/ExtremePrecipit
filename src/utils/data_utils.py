import os
import pandas as pd

# Récupération des années disponibles
def get_available_years(output_dir, stat):
    years = set()
    stat_dir = os.path.join(output_dir, stat)
    if os.path.exists(stat_dir):
        for file in os.listdir(stat_dir):
            if file.endswith(".parquet"):
                year = file.split("_")[-1].split(".")[0]
                years.add(int(year))
    return sorted(years)

# Charger les données sur une période sélectionnée
def load_data(output_dir, start_year, end_year, stat):
    all_dfs = []
    for i, year in enumerate(range(start_year, end_year + 1)):
        file_path = os.path.join(output_dir, stat, f"pr_{stat}_grid_{year}.parquet")
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)            
            df["year"] = year  # Ajouter une colonne pour l'année
            all_dfs.append(df)

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return None