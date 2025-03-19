import xarray as xr
import pandas as pd
import numpy as np
import os

# Ouvrir le fichier Zarr
ds = xr.open_zarr("data/processed/pr_fr_mm_h.zarr", chunks="auto")
print("Ficher zarr importé")

ds = ds.assign_coords(time=pd.to_datetime(ds['time'].values, format="%Y-%m-%d %H:%M", errors="coerce"))
print(f"Variable 'time' importée et formatée.")

pr_h = ds['pr_mm_h']  # données en mm/h
print("Variable 'pr_mm_h' importée.")

print("Décalage des jours sur 6h-6h")
pr_j = pr_h.resample(time="1D", offset="6h").sum()
pr_j = pr_j.assign_coords(time=pr_j.time + pd.Timedelta(hours=6))
print(f"Décalage terminé avec {pr_j.isnull().sum().values.item()} NaN.")


def add_group_coords(da):
    # Ajouter des coordonnées auxiliaires basées sur le temps (time) à un DataArray
    if 'time' not in da.dims:
        return da
    times = pd.to_datetime(da['time'].values)
    years = times.year  # Ajout de la coordonnée year
    months = times.month
    djf_year = np.where(months == 12, years + 1, years)
    hy_year = np.where(months >= 9, years + 1, years)
    
    season = np.full(len(times), "unknown", dtype=object)
    season[(months == 12) | (months == 1) | (months == 2)] = "djf"
    season[(months >= 3) & (months <= 5)] = "mam"
    season[(months >= 6) & (months <= 8)] = "jja"
    season[(months >= 9) & (months <= 11)] = "son"

    return da.assign_coords(year=("time", years), djf_year=("time", djf_year), hy_year=("time", hy_year), season=("time", season))

if 'time' in ds.dims:
    pr_h = add_group_coords(pr_h)
    pr_j = add_group_coords(pr_j)

def group_by_season(da, season_label):
    if 'time' not in da.dims:
        return None
    if season_label == "hy":
        return da.groupby("hy_year")
    elif season_label == "djf":
        return da.where(da.season == "djf", drop=True).groupby("djf_year")
    elif season_label in ["mam", "jja", "son"]:
        return da.where(da.season == season_label, drop=True).groupby("year")
    else:
        raise ValueError(f"Saison inconnue : {season_label}")

def compute_stats(grouped, scale):
    if grouped is None:
        return None, None, None, None, None
    
    mean_stat = grouped.mean(dim="time")
    max_stat = grouped.max(dim="time")
    date_stat = grouped.map(lambda x: x.idxmax(dim="time").dt.strftime("%Y-%m-%d"))
    
    if scale == "mm_j":
        sum_stat = grouped.sum(dim="time")        
        numday_stat = grouped.map(lambda x: (x >= 1).sum(dim="time")) # nombre de jours avec pr_mm_j >= 1
    else:
        sum_stat, numday_stat = None, None

    return mean_stat, max_stat, sum_stat, date_stat, numday_stat



output_dir = "data/result/preanalysis/stats"
os.makedirs(output_dir, exist_ok=True)

results = {}
for scale, da in [("mm_h", pr_h), ("mm_j", pr_j)]:
    print(f"\n ------------------------------------------ Traitement de {scale}")

    for season in ["hy", "djf", "mam", "jja", "son"]:
        print(f"\n ------------------- Traitement de {season}\n")

        grouped = group_by_season(da, season)
        mean_stat, max_stat, sum_stat, date_stat, numday_stat = compute_stats(grouped, scale)

        for stat, stat_name in zip([mean_stat, max_stat, sum_stat, date_stat, numday_stat],
                                   ["mean", "max", "sum", "date", "numday"]):
            var_name = f"{scale}_{season}_{stat_name}"
            print(f"- {var_name} :")

            if stat is not None:
                # Conversion en DataFrame et réinitialisation de l'index
                # La présence d'un MultiIndex sur 'points' (avec lat et lon) permettra d'obtenir les colonnes souhaitées
                df = stat.to_dataframe(name="pr").reset_index()

                if "points" in df.columns:
                    df = df.drop(columns=["points"])
                
                # Remplacement des éventuels noms de groupe par 'year'
                if "hy_year" in df.columns:
                    df = df.rename(columns={"hy_year": "year"})
                if "djf_year" in df.columns:
                    df = df.rename(columns={"djf_year": "year"})

                print(df.head(3))
                print("\n")
                
                # Sauvegarde en fichier Parquet
                file_path = os.path.join(output_dir, f"{var_name}.parquet")
                df.to_parquet(file_path, engine="pyarrow")  # Utilisation de PyArrow pour un seul fichier     

                results[var_name] = stat
            else:
                print("None\n")