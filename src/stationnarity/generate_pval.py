import os
import numpy as np
import pandas as pd

from multiprocessing import Pool
from tqdm import tqdm

from src.utils.data_utils import get_available_years, load_data
from src.utils.stats_utils import mann_kendall_test, adf_test

def stationarity_tests(group_data):
    (lat, lon), group = group_data  # Déballer les données correctement
    results = []
    group_sorted = group.sort_values('year')
    years = group_sorted["year"].values
    series = group_sorted["pr_max"].values
    
    if len(series) > 10:  # Assurer un nombre suffisant de points
        mk_bool = mann_kendall_test(years, series)
        adf_bool = adf_test(series)
        results.append([lat, lon, mk_bool, adf_bool])

    return pd.DataFrame(results, columns=["lat", "lon", "Mann-Kendall", "ADF"])

def process_group(args):
    """Applique stationarity_tests() sur un groupe (lat, lon)."""
    lat, lon, group = args
    result = stationarity_tests([(lat, lon), group])
    if result is not None:
        return (lat, lon), result
    else:
        return None

if __name__ == "__main__":
    # Définition des répertoires
    OUTPUT_DIR = os.path.join("data", "result")
    stat='max'

    # Récupération des années disponibles
    years = get_available_years(OUTPUT_DIR, stat)

    # Choix de la statistique d'étude et la période d'étude
    start = min(years) ; end = max(years)

    # Chargement des données sous la forme lat, lon, pr_max, year
    df = load_data(OUTPUT_DIR, start, end, stat)

    print(f"Statistique sélectionnée : {stat}")
    print(f"Plage de l'étude : {start} à {end}")
    print(f"Nombre de lignes chargées : {len(df)}")

    # Regroupement des données par point de grille
    grouped = df.groupby(["lat", "lon"])
    print(f"Nombre de groupes (points de grille) : {len(grouped)}")

    num_processes = 16

    with Pool(processes=num_processes) as pool:
        results = []
        for res in tqdm(pool.imap_unordered(process_group, [(lat, lon, group) for (lat, lon), group in grouped]), 
                        total=len(grouped), desc="Traitement des groupes"):
            if res is not None:  # Filtrer les erreurs
                results.append(res)

    # Convertir en DataFrame
    df = pd.concat([r[1] for r in results], ignore_index=True)
 
    # Sauvegarde
    output_dir = "data/result/stationnarity"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "pval_grid.parquet")
    df.to_parquet(file_path)

    print(f"Données enregistrées dans {file_path} :\n{df.head()}")
