import polars as pl
import matplotlib.pyplot as plt
import numpy as np

# Charger les deux fichiers
df_1990 = pl.read_parquet("data/gev/observed/quotidien/mam/niveau_retour.parquet")
df_1959 = pl.read_parquet("data/gev/observed/quotidien_toute_la_periode/mam/niveau_retour.parquet")

# Stations communes
stations_1990 = set(df_1990["NUM_POSTE"].to_list())
stations_1959 = set(df_1959["NUM_POSTE"].to_list())
stations_communes = list(stations_1990 & stations_1959)
n = len(stations_communes)

# Extraire les tendances pour les stations communes
z_T_p_1990 = df_1990.filter(pl.col("NUM_POSTE").is_in(stations_communes))["z_T_p"].to_numpy()
z_T_p_1959 = df_1959.filter(pl.col("NUM_POSTE").is_in(stations_communes))["z_T_p"].to_numpy()

# Boxplot
plt.figure(figsize=(6,4))
plt.boxplot([z_T_p_1959, z_T_p_1990], labels=["1959-2022", "1990-2022"])
plt.ylabel("Tendance relative (%)")
plt.title("Distribution des tendances relatives (%)\nToutes les stations de chaque période")
plt.grid(axis="y")

# Afficher n sous le boxplot
plt.gca().text(1, -0.15, f"n = {len(stations_communes)}", ha='center', va='top', transform=plt.gca().transAxes, fontsize=11)

# Calculer la différence absolue entre les tendances pour chaque station commune
stations_communes_sorted = sorted(stations_communes)
z_T_p_1959_dict = dict(zip(df_1959["NUM_POSTE"].to_list(), df_1959["z_T_p"].to_list()))
z_T_p_1990_dict = dict(zip(df_1990["NUM_POSTE"].to_list(), df_1990["z_T_p"].to_list()))

# Liste des tuples (station, diff)
diff_list = []
for station in stations_communes_sorted:
    diff = abs(z_T_p_1990_dict[station] - z_T_p_1959_dict[station])
    diff_list.append((station, diff, z_T_p_1959_dict[station], z_T_p_1990_dict[station]))

diff_list_sorted = sorted(diff_list, key=lambda x: x[1], reverse=True)

print("Top 10 des stations avec la plus grande différence de tendance (|1990-2022 - 1959-2022|) :")
for station, diff, t1959, t1990 in diff_list_sorted[:10]:
    print(f"Station {station} : Diff = {diff:.2f}%, 1959-2022 = {t1959:.2f}%, 1990-2022 = {t1990:.2f}%")

plt.tight_layout()
plt.show()