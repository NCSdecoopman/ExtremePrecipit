import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

station = '63178001'
echelle = 'horaire'
saison = 'son'
mesure = 'max_mm_h'
data_dir = Path('data/statisticals/observed/horaire')
years = range(1990, 2023)
dfs = []

df_gev = pl.read_parquet('data/gev/observed/horaire/son/niveau_retour.parquet')
params = df_gev.filter(pl.col('NUM_POSTE') == station)
print(params)

for year in years:
    file = data_dir / f'{year:04d}' / f'{saison}.parquet'
    if file.exists():
        df = pl.read_parquet(file, columns=['NUM_POSTE', mesure, 'nan_ratio'])
        df = df.filter(pl.col('NUM_POSTE') == station)
        if df.height > 0:
            df = df.with_columns(pl.lit(year).alias('year'))
            dfs.append(df)

if not dfs:
    raise ValueError("Aucune donnée trouvée pour la station")

df_station = pl.concat(dfs)
df_station = df_station.drop_nulls(subset=[mesure])

years_obs = df_station['year'].to_numpy()
maxs = df_station[mesure].to_numpy()
nan_ratios = df_station['nan_ratio'].to_numpy()

# Paramètres GEV
df_gev = pl.read_parquet('data/gev/observed/horaire/son/gev_param_best_model.parquet')
params = df_gev.filter(pl.col('NUM_POSTE') == station)
print(params)
mu0, mu1, sigma0, sigma1, xi = [params[x].item() for x in ['mu0','mu1','sigma0','sigma1','xi']]
model = params['model'].item()
has_break = '_break_year' in model
min_year = 1990  # Pour l'horaire, comme dans le pipeline
break_year = 1985
val_comparaison = 1995
max_year = 2022  # Harmonisation avec le pipeline

# Calcul du niveau de retour pour chaque année (T=10)
T = 10
CT = ((-np.log(1-1/T))**(-xi) - 1)

# Calcul du niveau de retour pour chaque année (T=20)
T2 = 20
CT2 = ((-np.log(1-1/T2))**(-xi) - 1)

# Calcul t_tilde et niveau de retour pour toutes les années de 1985 à 2022 (T=10)
annees_retour = np.arange(val_comparaison, 2023)

# --- Correction de la normalisation ---
# Calcul sur la période d'observation
if has_break:
    t_tilde_obs_raw = np.array([0.0 if y < break_year else (y - break_year) / (max_year - break_year) for y in years_obs])
    t_tilde_retour_raw = np.array([0.0 if y < break_year else (y - break_year) / (max_year - break_year) for y in annees_retour])
else:
    t_tilde_obs_raw = (years_obs - min_year) / (max_year - min_year)
    t_tilde_retour_raw = (annees_retour - min_year) / (max_year - min_year)

# Normalisation basée uniquement sur la période d'observation
    
t_min_ret = t_tilde_obs_raw.min()
t_max_ret = t_tilde_obs_raw.max()
if t_max_ret == t_min_ret:
    t_tilde_retour = np.full_like(t_tilde_retour_raw, -0.5)
else:
    res0_obs = t_tilde_obs_raw / (t_max_ret - t_min_ret)
    dx_ret = res0_obs.min() + 0.5
    res0_ret = t_tilde_retour_raw / (t_max_ret - t_min_ret)
    t_tilde_retour = res0_ret - dx_ret

z_T_retour = mu0 + mu1*t_tilde_retour + (sigma0 + sigma1*t_tilde_retour)/xi * CT
z_T_retour_20 = mu0 + mu1*t_tilde_retour + (sigma0 + sigma1*t_tilde_retour)/xi * CT2

# Séparation des années pour le tracé pointillé/plein
annee_obs_min = years_obs.min()
mask_extrap = annees_retour <= annee_obs_min
mask_obs = annees_retour >= annee_obs_min

# Tracé
plt.figure(figsize=(6,6))
# Tracé de la courbe des maxima observés
plt.plot(years_obs, maxs, color='blue', linewidth=1, label='Maxima observés')
# Points : orange si au-dessus du niveau de retour, bleu sinon
for i, y in enumerate(years_obs):
    idx = np.where(annees_retour == y)[0]
    if len(idx) > 0:
        z_ref = z_T_retour[idx[0]]
        color = 'orange' if maxs[i] > z_ref else 'blue'
        plt.scatter(y, maxs[i], color=color, zorder=5)
    else:
        plt.scatter(y, maxs[i], color='blue', zorder=5)
# Partie extrapolée (avant observations) T=10
if mask_extrap.any():
    plt.plot(annees_retour[mask_extrap], z_T_retour[mask_extrap], color='red', linestyle='dashed', label='Niveau de retour 10 ans extrapolé')
# Partie avec observations T=10
plt.plot(annees_retour[mask_obs], z_T_retour[mask_obs], color='red', linestyle='-', label='Niveau de retour 10 ans')
# Partie extrapolée (avant observations) T=20
if mask_extrap.any():
    plt.plot(annees_retour[mask_extrap], z_T_retour_20[mask_extrap], color='purple', linestyle='dashed', label='Niveau de retour 20 ans extrapolé')
# Partie avec observations T=20
plt.plot(annees_retour[mask_obs], z_T_retour_20[mask_obs], color='purple', linestyle='-', label='Niveau de retour 20 ans')

# Affichage des valeurs en 1985 et 2022 pour T=10
val_1985 = z_T_retour[annees_retour == val_comparaison][0] if val_comparaison in annees_retour else None
val_2022 = z_T_retour[annees_retour == 2022][0] if 2022 in annees_retour else None
# Affichage des valeurs en 1985 et 2022 pour T=20
val_1985_20 = z_T_retour_20[annees_retour == val_comparaison][0] if val_comparaison in annees_retour else None
val_2022_20 = z_T_retour_20[annees_retour == 2022][0] if 2022 in annees_retour else None

if val_1985 is not None:
    plt.scatter(val_comparaison, val_1985, color='green', zorder=5)
    plt.annotate(f"{val_1985:.1f}", (val_comparaison, val_1985), textcoords="offset points", xytext=(0,10), ha='center', color='green')
    plt.hlines(val_1985, annees_retour.min(), annees_retour.max(), colors='green', linestyles=':', alpha=0.7)
if val_2022 is not None:
    plt.scatter(2022, val_2022, color='green', zorder=5)
    plt.axvline(2022, color='green', linestyle=':', alpha=0.7)
    plt.annotate(f"{val_2022:.1f}", (2022, val_2022), textcoords="offset points", xytext=(0,10), ha='center', color='green')
# Pour T=20
if val_1985_20 is not None:
    plt.scatter(val_comparaison, val_1985_20, color='darkorange', zorder=5)
    plt.annotate(f"{val_1985_20:.1f}", (val_comparaison, val_1985_20), textcoords="offset points", xytext=(0,-15), ha='center', color='darkorange')
    plt.hlines(val_1985_20, annees_retour.min(), annees_retour.max(), colors='darkorange', linestyles=':', alpha=0.7)
if val_2022_20 is not None:
    plt.scatter(2022, val_2022_20, color='darkorange', zorder=5)
    plt.axvline(2022, color='darkorange', linestyle=':', alpha=0.7)
    plt.annotate(f"{val_2022_20:.1f}", (2022, val_2022_20), textcoords="offset points", xytext=(0,-15), ha='center', color='darkorange')

plt.title(f'Station {station} (horaire, SON)')
plt.xlabel('Année')
plt.ylabel('Maximum annuel (mm/h)')
plt.legend()

# Ajout du texte du calcul sous la légende
if (val_1985 is not None) and (val_2022 is not None):
    pourcent = (val_2022 - val_1985) / val_1985 * 100
    texte_calcul = r'$\frac{{{:.1f}-{:.1f}}}{{{:.1f}}} \times 100 = {:.1f}\ \%$'.format(val_2022, val_1985, val_1985, pourcent)
else:
    texte_calcul = ''
if (val_1985_20 is not None) and (val_2022_20 is not None):
    pourcent_20 = (val_2022_20 - val_1985_20) / val_1985_20 * 100
    texte_calcul += '\n' + r'$\frac{{{:.1f}-{:.1f}}}{{{:.1f}}} \times 100 = {:.1f}\ \%$'.format(val_2022_20, val_1985_20, val_1985_20, pourcent_20)
plt.gca().text(
    0.5, -0.20, texte_calcul,
    transform=plt.gca().transAxes, fontsize=12, color='black',
    ha='center', va='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
)

plt.tight_layout()

plt.savefig('C:/Users/nicod/Documents/GitHub/ExtremePrecipit/courbe_niveau_retour_34217001.png')
print('Graphique sauvegardé sous : C:/Users/nicod/Documents/GitHub/ExtremePrecipit/courbe_niveau_retour_34217001.png')
