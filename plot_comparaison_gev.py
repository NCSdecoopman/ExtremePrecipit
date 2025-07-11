import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils.data_utils import cleaning_data_observed, years_to_load, load_data
from src.pipelines.pipeline_best_to_niveau_retour import build_x_ttilde, compute_zT_for_years
import random

break_year = 1985


def compute_zT_for_years_VERSION(
    annees_retour: np.ndarray,  # tableau d'années souhaitées
    min_year: int,
    max_year: int,
    df_params: pl.DataFrame,
    df_series: pl.DataFrame,
    T=10
) -> pl.DataFrame:
    """
    Calcule z_T(x) pour chaque station pour deux années de retour (a et b),
    et retourne un DataFrame avec NUM_POSTE, zTpa, zTpb, (zTpb-zTpa)/zTpa.
    """
    rows = []
    for row in df_params.to_dicts():
        data_station = df_series.filter(pl.col("NUM_POSTE") == row["NUM_POSTE"])
        years_obs = data_station["year"].to_numpy()
        t_tilde_obs_raw = (years_obs - min_year) / (max_year - min_year)
        t_min_ret = t_tilde_obs_raw.min()
        t_max_ret = t_tilde_obs_raw.max()
        res0_obs = t_tilde_obs_raw / (t_max_ret - t_min_ret)
        dx = res0_obs.min() + 0.5

        t_tilde_retour = []
        model = row["model"]
        break_model = "_break" in model
        
        for y in annees_retour:
            if break_model and y<=1985:
                t_tilde = 0
            else:
                t_tilde_retour_raw = (y - min_year) / (max_year - min_year)
                res0_ret = t_tilde_retour_raw / (t_max_ret - t_min_ret)
                t_tilde = res0_ret - dx

            t_tilde_retour.append(t_tilde)
        t_tilde_retour = np.array(t_tilde_retour)



        mu0, mu1, sigma0, sigma1, xi = row["mu0"], row["mu1"], row["sigma0"], row["sigma1"], row["xi"]
        CT = ((-np.log(1-1/T))**(-xi) - 1)
        zT = mu0 + mu1*t_tilde_retour + (sigma0 + sigma1*t_tilde_retour)/xi * CT

        rows.append({
            "NUM_POSTE": row["NUM_POSTE"],
            "zT": zT
        })

    return pl.DataFrame(rows)


# Ficher z_T_p
file_1959_2022 = Path('data/gev/observed/quotidien_toute_la_periode/mam/niveau_retour.parquet')
file_1990_2022 = Path('data/gev/observed/quotidien/mam/niveau_retour.parquet')

# Charger les deux tables
zTp_1959_2022 = pl.read_parquet(file_1959_2022)
zTp_1990_2022 = pl.read_parquet(file_1990_2022)

# Fichiers Parquet GEV
file_1959_2022 = Path('data/gev/observed/quotidien_toute_la_periode/mam/gev_param_best_model.parquet')
file_1990_2022 = Path('data/gev/observed/quotidien/mam/gev_param_best_model.parquet')

# Charger les deux tables
gev_1959_2022 = pl.read_parquet(file_1959_2022)
gev_1990_2022 = pl.read_parquet(file_1990_2022)

# Paramètre de chargement des données
input_dir = Path("data/statisticals/observed/quotidien")
mesure = "max_mm_j"
cols = ["NUM_POSTE", mesure, "nan_ratio"]
df_1959_2022 = load_data(input_dir, "mam", "quotidien", cols, 1959, 2022)
df_1990_2022 = load_data(input_dir, "mam", "quotidien", cols, 1990, 2022)

# Selection des stations suivant le NaN max
df_1959_2022 = cleaning_data_observed(df_1959_2022, "quotidien", len_serie=50)
df_1990_2022 = cleaning_data_observed(df_1990_2022, "quotidien", len_serie=25)

# Gestion des NaN
df_1959_2022 = df_1959_2022.drop_nulls(subset=[mesure])
df_1990_2022 = df_1990_2022.drop_nulls(subset=[mesure])

# Normaliser t en t_tilde ou t_tilde* suivant la présence ou non d'un break_point
df_series_1959_2022 = build_x_ttilde(df_1959_2022, 1959, 2022, gev_1959_2022, break_year, mesure)
df_series_1990_2022 = build_x_ttilde(df_1990_2022, 1990, 2022, gev_1959_2022, break_year, mesure)

# Calcul de z_T(1995) et z_T(2022) avec normalisation sur la grille [1995, 2022]
z_levels_1959_2022 = compute_zT_for_years_VERSION(
    np.arange(1959, 2023),
    1959,
    2022    ,
    gev_1959_2022,
    df_series_1959_2022
)

z_levels_1990_2022 = compute_zT_for_years_VERSION(
    np.arange(1990, 2023),
    1990,
    2022    ,
    gev_1990_2022,
    df_series_1990_2022
)

# Filtrer la station
station_id = "6149001"

# Série temporelle observée
df_station = df_1959_2022.filter(pl.col("NUM_POSTE") == station_id)
years = df_station["year"].to_numpy()
values = df_station[mesure].to_numpy()

# Niveaux de retour
z_1959_2022 = z_levels_1959_2022.filter(pl.col("NUM_POSTE") == station_id)["zT"].to_numpy()[0]
z_1990_2022 = z_levels_1990_2022.filter(pl.col("NUM_POSTE") == station_id)["zT"].to_numpy()[0]
years_1959_2022 = np.arange(1959, 2023)
years_1990_2022 = np.arange(1990, 2023)

# Tracé
plt.figure(figsize=(12,6))
plt.plot(years, values, label="Série observée", color="black", marker="o", linestyle="-", alpha=0.5)
plt.plot(years_1959_2022, z_1959_2022, label="Niveau de retour fit 1959-2022", color="blue")
plt.plot(years_1990_2022, z_1990_2022, label="Niveau de retour fit 1990-2022", color="red")
plt.xlabel("Année")
plt.ylabel("Précipitation (mm)")
plt.title(
    f"Station {station_id} (mam, quotidien)"
)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.subplots_adjust(bottom=0.22)  # <-- Ajoute cette ligne pour augmenter la marge

# Affichage des valeurs sur les courbes de niveau de retour
for year, color, z, label in [
    (1995, "blue", z_1959_2022, "fit 1959-2022"),
    (2022, "blue", z_1959_2022, "fit 1959-2022"),
    (1995, "red", z_1990_2022, "fit 1990-2022"),
    (2022, "red", z_1990_2022, "fit 1990-2022"),
]:
    idx = year - (1959 if color == "blue" else 1990)
    value = z[idx]
    plt.annotate(
        f"{value:.1f}",
        (year, value),
        textcoords="offset points",
        xytext=(0, 10 if year == 1990 else -15),
        ha='center',
        color=color,
        fontsize=10,
        fontweight='bold',
        bbox=None  # <-- Enlève le cadre
    )

# Ajout d'un point pour 2022 sur chaque courbe
plt.scatter([2022, 2022], [z_1959_2022[-1], z_1990_2022[-1]], s=40, color=['blue', 'red'], zorder=5)
# Ajout d'un point pour 1995 sur chaque courbe
plt.scatter([1995, 1995], [z_1959_2022[1995-1959], z_1990_2022[1995-1990]], s=40, color=['blue', 'red'], zorder=5)

# Calcul des tendances
zT_1995_1 = z_1959_2022[1995 - 1959]
zT_2022_1 = z_1959_2022[2022 - 1959]
trend_1959 = ((zT_2022_1 - zT_1995_1) / zT_1995_1) * 100

zT_1995_2 = z_1990_2022[1995 - 1990]
zT_2022_2 = z_1990_2022[2022 - 1990]
trend_1990 = ((zT_2022_2 - zT_1995_2) / zT_1995_2) * 100

# Récupération des valeurs z_T_p pour la station
# Pour 1959-2022
val_1959_2022 = zTp_1959_2022.filter(pl.col("NUM_POSTE") == station_id)
print(val_1959_2022)
zTp_1959_2022_val = float(val_1959_2022["z_T_p"][0])

# Pour 1990-2022
val_1990_2022 = zTp_1990_2022.filter(pl.col("NUM_POSTE") == station_id)
print(val_1990_2022)
zTp_1990_2022_val = float(val_1990_2022["z_T_p"][0])

# Affichage sous le graphique
texte_math_1959 = (
    r"$\mathrm{Tendance}_{1959-2022} = \frac{%.2f - %.2f}{%.2f} \times 100 = %.2f\%%$" % (zT_2022_1, zT_1995_1, zT_1995_1, trend_1959)
    + f" (calculée pour les cartes 1959-2022 : {zTp_1959_2022_val:.2f}%)"
)
texte_math_1990 = (
    r"$\mathrm{Tendance}_{1990-2022} = \frac{%.2f - %.2f}{%.2f} \times 100 = %.2f\%%$" % (zT_2022_2, zT_1995_2, zT_1995_2, trend_1990)
    + f" (calculée pour les cartes 1990-2022 : {zTp_1990_2022_val:.2f}%)"
)

plt.figtext(
    0.01, 0.08, texte_math_1959,
    ha='left', va='bottom', fontsize=10, color='black'
)
plt.figtext(
    0.01, 0.02, texte_math_1990,
    ha='left', va='bottom', fontsize=10, color='black'
)

print(texte_math_1959)
print(texte_math_1990)

plt.show()
