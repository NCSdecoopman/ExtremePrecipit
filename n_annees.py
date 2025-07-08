import polars as pl
import matplotlib.pyplot as plt
import os
import pandas as pd
from collections import defaultdict

# 1. Charger le fichier niveau_retour.parquet
parquet_path = r"C:\Users\nicod\Documents\GitHub\ExtremePrecipit\data\gev\observed\horaire\son\niveau_retour.parquet"
df = pl.read_parquet(parquet_path)

# 2. Compter le nombre d'années de données par station pour la saison 'son'
data_dir = r"C:\Users\nicod\Documents\GitHub\ExtremePrecipit\data\statisticals\observed\horaire"
stations_years = defaultdict(set)

for year in os.listdir(data_dir):
    if not (year.isdigit() and int(year) >= 1990):
        continue
    year_path = os.path.join(data_dir, year)
    son_file = os.path.join(year_path, "son.parquet")
    if os.path.isdir(year_path) and os.path.exists(son_file):
        try:
            df_year = pl.read_parquet(son_file)
            # Détection automatique de la colonne de mesure
            mesure_col = None
            for col in ["max_mm_h"]:
                if col in df_year.columns:
                    mesure_col = col
                    break
            if mesure_col is None:
                print(f"Aucune colonne de mesure trouvée dans {son_file}")
                continue
            # Pour chaque station, vérifier s'il y a au moins une valeur non-NaN
            for num_poste in df_year["NUM_POSTE"].unique().to_list():
                sub = df_year.filter(pl.col("NUM_POSTE") == num_poste)
                if sub[mesure_col].drop_nulls().len() > 0:
                    stations_years[num_poste].add(int(year))
        except Exception as e:
            print(f"Erreur lors de la lecture de {son_file} : {e}")

# 3. Créer un DataFrame avec le nombre d'années par station
years_df = pd.DataFrame([
    {"NUM_POSTE": k, "n_years": len(v)}
    for k, v in stations_years.items()
])

# 4. Fusionner avec le DataFrame principal
df_pd = df.to_pandas()
merged = df_pd.merge(years_df, on="NUM_POSTE")

# 5. Tracer le graphique
plt.figure(figsize=(8, 5))
# Création d'un boxplot de z_T_p par nombre d'années de données
merged.boxplot(column="z_T_p", by="n_years", grid=True)
plt.xlabel("Nombre d'années disponibles")
plt.ylabel("Tendance du niveau de retour (%)")
plt.suptitle("")  # Supprimer le titre automatique ajouté par pandas
plt.show()

# Diagnostic : station avec la valeur absolue la plus élevée de z_T_p
station_extreme = merged.loc[merged['z_T_p'].abs().idxmax()]
print("\n--- Station avec z_T_p extrême ---")
print(station_extreme)