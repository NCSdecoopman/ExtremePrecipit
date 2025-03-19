import os
import xarray as xr


def list_nc_files(directory):
    """
    Liste tous les fichiers .nc dans un répertoire donné.
    
    :param directory: Chemin du répertoire à explorer.
    :return: Liste des fichiers .nc trouvés.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError("Le répertoire spécifié n'existe pas.")
    
    nc_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".nc")]
    
    if not nc_files:
        print("Aucun fichier .nc trouvé dans le répertoire spécifié.")
    
    return nc_files

def generate_year_file_dict(files):
    """
    Génère un dictionnaire associant une année à son fichier correspondant.
    
    :param files: Liste des chemins des fichiers.
    :return: Dictionnaire {année: fichier}
    """
    year_file_dict = {}
    
    for file in files:
        filename = os.path.basename(file)
        parts = filename.split('_')
        if parts:
            year_part = parts[-1].split('-')[0]
            if year_part.isdigit():
                year = int(year_part[:4])
                year_file_dict[year] = file
    
    return year_file_dict

def open_dataset(file_path, time_chunk="auto"):
    """
    Ouvre un fichier NetCDF en utilisant xarray avec un découpage sur l'axe temporel.
    
    :param file_path: Chemin du fichier NetCDF.
    :param time_chunk: Taille des chunks sur l'axe temporel.
    :return: xarray.Dataset
    """
    return xr.open_dataset(file_path, chunks={"time": time_chunk})