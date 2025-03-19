import xarray as xr
import pandas as pd
import numpy as np
import os
import lzma
import json
from tqdm import tqdm

SCALE_FACTOR = 10
SENTINEL = -32768
BATCH_SIZE_POINTS = 100  # Ajustable selon la RAM

def ouvrir_zarr(path):
    ds = xr.open_zarr(path)  # On n'impose plus de chunks ici
    ds = ds.assign_coords(time=pd.to_datetime(ds['time'].values, format="%Y-%m-%d %H:%M", errors="coerce"))
    return ds

def ecrire_metadonnees(pr_da, metadata_path):
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    min_time = pd.to_datetime(pr_da.time.values.min())
    time_diff = pd.to_timedelta(pr_da.time.diff(dim="time").values[0]).round("1h")
    delta_hours = int(time_diff / pd.Timedelta(hours=1))

    metadata = {
        "min_time": min_time.strftime("%Y-%m-%d %H:%M:%S"),
        "time_step_hours": delta_hours,
        "num_points": len(pr_da.points),
        "scale_factor": SCALE_FACTOR,
        "sentinel_value": SENTINEL
    }

    with open(metadata_path, "w") as f_json:
        json.dump(metadata, f_json, indent=4)
    
    print(f"Métadonnées écrites dans {metadata_path}.")
    return min_time, delta_hours


def custom_zarr_indexer(zarr_array):
    original_getitem = zarr_array.__getitem__

    def new_getitem(key):
        # Zarr range : (time_slice, point_idx)
        if isinstance(key, tuple) and isinstance(key[1], int):
            # convertit l'index points en float
            key = (key[0], float(key[1]))
        return original_getitem(key)

    zarr_array.__getitem__ = new_getitem
    return zarr_array

def ecrire_fichiers_individuels(pr_da, output_dir_individuel):
    os.makedirs(output_dir_individuel, exist_ok=True)
    lats = pr_da.lat.values
    lons = pr_da.lon.values
    num_points = len(pr_da.points)

    # patch sur le backend_array natif de xarray
    backend_zarr_array = pr_da.encoding.get("backend_array", None)
    if backend_zarr_array is not None:
        custom_zarr_indexer(backend_zarr_array)

    with tqdm(total=num_points, desc="Traitement par batchs de points") as pbar:
        for batch_start in range(0, num_points, BATCH_SIZE_POINTS):
            batch_end = min(batch_start + BATCH_SIZE_POINTS, num_points)

            da_batch = pr_da.isel(points=slice(batch_start, batch_end))

            try:
                da_batch.load()
            except OSError as e:
                print(f"Batch {batch_start}-{batch_end} ignoré : {e}")
                pbar.update(batch_end - batch_start)
                continue

            pr_values_batch = np.nan_to_num(da_batch.values, nan=SENTINEL)  # (time, batch_size)
            pr_values_batch = pr_values_batch.T  # --> (batch_size, time)
            pr_values_batch = np.round(pr_values_batch * SCALE_FACTOR).astype(np.int16)

            for j in range(batch_end - batch_start):
                i = batch_start + j
                lat = lats[i]
                lon = lons[i]
                serie = pr_values_batch[j]

                filename = f"ts_{lat:.4f}_{lon:.4f}.bin.xz"
                filepath = os.path.join(output_dir_individuel, filename)
                with lzma.open(filepath, "wb") as f_indiv:
                    assert serie.shape[0] == n_time, f"Erreur: série de taille {serie.shape[0]}, attendu {n_time}"
                    f_indiv.write(serie.tobytes())

            pbar.update(batch_end - batch_start)

    print("Fichiers individuels (1 fichier = 1 point complet) terminés.")

if __name__ == "__main__":
    input_zarr = "data/processed/pr_fr_mm_h.zarr"
    output_dir_individuel = "data/binaires/individuels"
    metadata_path = "data/binaires/metadata.json"

    print("Ouverture du fichier Zarr...")
    ds = ouvrir_zarr(input_zarr)
    print("Fichier Zarr ouvert.")

    # Ouvrir le zarr
    pr_h = ds['pr_mm_h']

    # Charger la vraie taille totale de "time"
    n_time = pr_h.sizes["time"]

    # Rechunk global
    pr_h = pr_h.chunk({"points": BATCH_SIZE_POINTS, "time": n_time})

    print("Extraction de pr_mm_h et rechunking...")

    # Étape 1 : écrire les métadonnées
    min_time, delta_hours = ecrire_metadonnees(pr_h, metadata_path)

    # Étape 2 : traitement batché performant
    ecrire_fichiers_individuels(pr_h, output_dir_individuel)
    print("Pipeline terminé.")