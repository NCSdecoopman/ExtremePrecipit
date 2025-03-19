import os
import pandas as pd
from joblib import Parallel, delayed

from src.utils.nc_utils import list_nc_files, generate_year_file_dict, open_dataset
from src.processed.geo_filter import filter_ds_to_france
from src.processed.treatment_precipitation import convert_precipitation

from src.analysis.nc_analysis import generate_time_summary
from src.analysis.pr_analysis import pr_nan

def generate_files(path_file='data/raw'):
    # Lister les .nc à disposition
    files = list_nc_files(path_file)
    # Générer un dictionnaire année - fichier
    dict_files = generate_year_file_dict(files)

    return dict_files

def generate_data(dict_files, year):
    ds = open_dataset(dict_files[year])  # Ouvrir le fichier
    
    try:
        ds_france = filter_ds_to_france(ds) # Filtrer le fichier à la France uniquement
        ds_france = convert_precipitation(ds_france) # Conversion des précipitations
        return ds, ds_france  # Retourne ds aussi si nécessaire
    finally:
        ds.close()  # Fermeture propre même en cas d'erreur

def process_single_year(year, dict_files):
    """
    Traite une année donnée et retourne les informations sous forme de tuple.
    """
    _, ds_france = generate_data(dict_files, year=year)  # Récupérer les données pour l'année donnée
    
    try:
        nb_nan = pr_nan(ds_france)  # % NaN
        time_min, time_max, time_intervals = generate_time_summary(ds_france)  # Extraire les informations temporelles

        return round(nb_nan, 2), year, time_min, time_max, time_intervals
    finally:
        ds_france.close()  # Assure la fermeture après traitement

def process_all_years(dict_files):
    """
    Traite toutes les années du dictionnaire dict_files en parallèle et retourne un DataFrame résumant les informations temporelles et de qualité des données.
    
    :param dict_files: Dictionnaire {année: chemin_fichier}
    :return: DataFrame avec les colonnes ["% NaN", "Year", "Start", "End", "Time intervals"]
    """
    years = sorted(dict_files.keys())
    results = []
    
    for year in years:
        print(year)
        try:
            result = process_single_year(year, dict_files)
            if result:
                results.append(result)
        except Exception as e:
            print(f"Erreur lors du traitement de l'annee {year}: {e}")
    
    df_time_summary = pd.DataFrame(results, columns=["% NaN", "Year", "Start", "End", "Time intervals"])
    return df_time_summary

if __name__ == "__main__":
    # Générer le dictionnaire des fichiers NetCDF
    dict_files = generate_files()

    # Liste des années triées
    years = sorted(dict_files.keys())

    # Traitement parallèle avec 8 cœurs
    print("Début du traitement parallèle")
    results = Parallel(n_jobs=8)(
        delayed(lambda y: (print(y), process_single_year(y, dict_files))[1])(year) for year in years
    )

    # Filtrer les résultats valides (sans erreurs)
    results = [res for res in results if res is not None]

    # Convertir en DataFrame
    df_time_summary = pd.DataFrame(results, columns=["% NaN", "Year", "Start", "End", "Time intervals"])

    # Définir le répertoire et le fichier de sortie
    output_dir = "data/preanalysis"
    output_file = os.path.join(output_dir, "nan_resol_temp.csv")

    # Créer le répertoire s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Sauvegarde du fichier CSV
    df_time_summary.to_csv(output_file, index=False)