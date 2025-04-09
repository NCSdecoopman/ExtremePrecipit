import pandas as pd
import numpy as np

# Définir le chemin du fichier distant
url = "https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/BASE/HOR/H_74_1990-1999.csv.gz"

# Charger directement le fichier CSV compressé à partir de l'URL
df = pd.read_csv(url, compression="gzip", low_memory=False, sep=";")

# Étape 1 — Trouver le point le plus proche :
target_lat, target_lon = 45.9282, 6.0940

df["distance"] = np.sqrt((df["LAT"] - target_lat)**2 + (df["LON"] - target_lon)**2)
closest_point = df.loc[df["distance"].idxmin()]

lat_closest = closest_point["LAT"]
lon_closest = closest_point["LON"]

# Étape 2 — Filtrer les données de ce point
df_point = df[(df["LAT"] == lat_closest) & (df["LON"] == lon_closest)].copy()
print(df_point[["AAAAMMJJHH", "RR1"]])
# Filtre à la date exacte
print(df_point[df_point["AAAAMMJJHH"] // 100 == 19930629][["RR1", "AAAAMMJJHH"]])