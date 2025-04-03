import pandas as pd

# Chemin vers le fichier
file_path = r"C:\Users\nicod\Documents\GitHub\app\data\statisticals\observed\quotidien\1959\jja.parquet"

# Lecture du fichier Parquet
df = pd.read_parquet(file_path)

print(f"nan_ratio unique : {df['nan_ratio'].unique()}")

# Comptage des couples (lat, lon) uniques
n_unique_points = df[['lat', 'lon']].drop_duplicates().shape[0]

print(f"Nombre de couples (lat, lon) uniques : {n_unique_points}")
