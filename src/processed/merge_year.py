import os
import numcodecs
from tqdm import tqdm
from dask.diagnostics import ProgressBar
import xarray as xr

from src.preanalysis import generate_files, generate_data
from src.processed.treatment_precipitation import convert_precipitation

def process_year(year, dict_files):
    _, ds_france = generate_data(dict_files, year=year)
    ds_france = ds_france.drop_vars(["x", "y", "lon_bnds", "lat_bnds", "time_bnds"], errors="ignore")
    ds_france = convert_precipitation(ds_france)

    # Suppression du 29 février si l'année est bissextile
    ds_france = ds_france.sel(time=~((ds_france.time.dt.month == 2) & (ds_france.time.dt.day == 29)))

    # Formatage du temps (attention : ici, le temps sera converti en chaînes de caractères)
    ds_france["time"] = ds_france["time"].dt.strftime('%Y-%m-%d %H:%M')

    # On s'assure de ne garder que les variables d'intérêt
    return ds_france[["time", "pr_mm_h", "lat", "lon"]], ds_france.drop_vars("pr_mm_h")

def save_combined_zarr(dict_files, output_zarr='data/processed/pr_fr_mm_h.zarr'):
    """
    Combine les fichiers annuels en un seul fichier Zarr en ne conservant que
    les informations 'lat', 'lon', 'time' et 'pr_mm_h'. Traite les années dans l'ordre
    et affiche une barre d'avancement pour chaque année.
    """
    os.makedirs(os.path.dirname(output_zarr), exist_ok=True)
    
    datasets = []
    years = sorted(dict_files.keys())

    # Traitement séquentiel avec barre de progression
    progress_bar = tqdm(years, desc="Traitement des années")
    for year in progress_bar:
        progress_bar.set_description(f"Traitement de l'année {year}")
        ds_year, _ = process_year(year, dict_files)
        datasets.append(ds_year)

    # Concaténation des datasets sur l'axe du temps
    print("Concaténation des datasets")
    combined_ds = xr.concat(datasets, dim="time")
    
    # Définir 'lat' et 'lon' comme coordonnées
    combined_ds = combined_ds.set_coords(["lat", "lon"])
    
    # Garder uniquement les variables de données souhaitées ('time' et 'pr_mm_h')
    # 'lat' et 'lon' resteront en tant que coordonnées
    combined_ds = combined_ds[["time", "pr_mm_h"]]
    
    # Suppression des métadonnées globales
    combined_ds.attrs = {}
    print(combined_ds)
    
    print("Re-chunking du dataset avant enregistrement")
    chunk_size = {"time": 24}  # Ajuster en fonction de la taille des données
    combined_ds = combined_ds.chunk(chunk_size)
    
    # Configuration de la compression pour chaque variable de données
    compressor = numcodecs.Zstd(level=22)  # Compression maximale avec Zstd
    encoding = {var: {"_FillValue": None, "compressor": compressor} 
                for var in combined_ds.data_vars}
    
    # Enregistrement en Zarr avec suivi de progression
    print("Enregistrement en Zarr après re-chunking avec compression Zstd")
    with ProgressBar():
        combined_ds.to_zarr(output_zarr, mode="w", encoding=encoding)
    
    print(f"Fichier Zarr combiné enregistré : {output_zarr}\n")


if __name__ == "__main__":
    # Utilisation d'un contexte pour assurer une fermeture propre du client
    dict_files = generate_files()
    save_combined_zarr(dict_files)