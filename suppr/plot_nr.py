import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils.data_utils import cleaning_data_observed

station = '38442001'
echelle = 'quotidien'
saison = 'mam'
donnee = 'observed'
data_dir = Path(f'data/statisticals/{donnee}/{echelle}')
# Détermination dynamique de la période d'observation
available_years = [int(d.name) for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit() and len(d.name) == 4]
if not available_years:
    raise ValueError(f"Aucune année trouvée dans {data_dir}")
years = range(min(available_years), max(available_years)+1)
dfs = []

df_gev = pl.read_parquet(f'data/gev/{donnee}/{echelle}/{saison}/niveau_retour.parquet')
params = df_gev.filter(pl.col('NUM_POSTE') == station)
print(params)

mesure = 'max_mm_h' if echelle == "horaire" else 'max_mm_j'
len_serie=25

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
df_station = cleaning_data_observed(df_station, echelle, len_serie)
df_station = df_station.drop_nulls(subset=[mesure])

years_obs = df_station['year'].to_numpy()
maxs = df_station[mesure].to_numpy()
nan_ratios = df_station['nan_ratio'].to_numpy()

# Paramètres GEV
df_gev = pl.read_parquet(f'data/gev/{donnee}/{echelle}/{saison}/gev_param_best_model.parquet')
params = df_gev.filter(pl.col('NUM_POSTE') == station)
print(params)
mu0, mu1, sigma0, sigma1, xi = [params[x].item() for x in ['mu0','mu1','sigma0','sigma1','xi']]
model = params['model'].item()
has_break = '_break_year' in model
min_year = 1959  # Pour l'horaire, comme dans le pipeline
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
annees_retour = np.arange(val_comparaison, max_year+1)

# --- Correction de la normalisation ---
# Fonction de normalisation identique au pipeline

def norm_1delta_0centred(t):
    t = np.asarray(t)
    t_min = t.min()
    t_max = t.max()
    if t_max == t_min:
        return np.full_like(t, -0.5, dtype=float)
    res0 = t / (t_max - t_min)
    dx = res0.min() + 0.5
    return res0 - dx

# 1. Calcul t_tilde_raw pour toutes les années (observées et fictives)
if has_break:
    t_tilde_obs_raw = np.array([0.0 if y < break_year else (y - break_year) / (max_year - break_year) for y in years_obs])
    t_tilde_retour_raw = np.array([0.0 if y < break_year else (y - break_year) / (max_year - break_year) for y in annees_retour])
else:
    t_tilde_obs_raw = (years_obs - min_year) / (max_year - min_year)
    t_tilde_retour_raw = (annees_retour - min_year) / (max_year - min_year)

# 2. Calcul de la normalisation sur la période observée
t_min_obs = t_tilde_obs_raw.min()
t_max_obs = t_tilde_obs_raw.max()
if t_max_obs == t_min_obs:
    t_tilde_obs_norm = np.full_like(t_tilde_obs_raw, -0.5)
    t_tilde_retour = np.full_like(t_tilde_retour_raw, -0.5)
else:
    res0_obs = t_tilde_obs_raw / (t_max_obs - t_min_obs)
    dx = res0_obs.min() + 0.5
    # Appliquer la même normalisation à toutes les années (même hors période observée)
    res0_ret = t_tilde_retour_raw / (t_max_obs - t_min_obs)
    t_tilde_retour = res0_ret - dx

# Calcul des niveaux de retour avec la normalisation corrigée
z_T_retour = mu0 + mu1*t_tilde_retour + (sigma0 + sigma1*t_tilde_retour)/xi * CT
z_T_retour_20 = mu0 + mu1*t_tilde_retour + (sigma0 + sigma1*t_tilde_retour)/xi * CT2

# Séparation des années pour le tracé pointillé/plein
annee_obs_min = years_obs.min()
annee_obs_max = years_obs.max()
mask_extrap = (annees_retour < annee_obs_min) | (annees_retour > annee_obs_max)
mask_obs = (annees_retour >= annee_obs_min) & (annees_retour <= annee_obs_max)

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
# Partie extrapolée (hors observations) T=10
if mask_extrap.any():
    plt.plot(annees_retour[mask_extrap], z_T_retour[mask_extrap], color='red', linestyle='dashed', label='Niveau de retour 10 ans extrapolé')
# Partie avec observations T=10
plt.plot(annees_retour[mask_obs], z_T_retour[mask_obs], color='red', linestyle='-', label='Niveau de retour 10 ans')
# Partie extrapolée (hors observations) T=20
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
print(f"% tendance : {pourcent}")
print(f"[OBS] Période d'observation réelle : {years_obs.min()} - {years_obs.max()} ({len(years_obs)} années)")
print(f"[GEV PLOT] mu0: {mu0}, mu1: {mu1}, sigma0: {sigma0}, sigma1: {sigma1}, xi: {xi}")
plt.savefig('courbe_niveau_retour.png')
print('courbe_niveau_retour.png')

# === DIAGNOSTIC : Comparaison pipeline vs plot ===
# Lecture des valeurs pipeline (niveau_retour.parquet)
parquet_path = f'data/gev/{donnee}/{echelle}/{saison}/niveau_retour.parquet'
df_nr = pl.read_parquet(parquet_path)
row_nr = df_nr.filter(pl.col('NUM_POSTE') == station)
if row_nr.height > 0:
    z_T1_pipeline = row_nr['z_T1'].item()
    z_T_p_pipeline = row_nr['z_T_p'].item()
    # Affichage des valeurs pipeline zTpa et zTpb si elles existent
    if 'zTpa' in row_nr.columns and 'zTpb' in row_nr.columns:
        zTpa_pipeline = row_nr['zTpa'].item()
        zTpb_pipeline = row_nr['zTpb'].item()
        print(f"[PIPELINE] zTpa (1995): {zTpa_pipeline:.6f}, zTpb (2022): {zTpb_pipeline:.6f}")
    else:
        print("[PIPELINE] Colonnes zTpa/zTpb non trouvées dans le Parquet.")
    print(f"[PIPELINE] z_T1 (pente/10ans): {z_T1_pipeline:.6f}")
    print(f"[PIPELINE] z_T_p (tendance %): {z_T_p_pipeline:.6f}")
else:
    print("[PIPELINE] Station non trouvée dans niveau_retour.parquet")

# Calcul pipeline : zTpa et zTpb (1995 et 2022)
# On recalcule t_tilde pour 1995 et 2022 comme dans le pipeline
annees_compare = np.array([1995, 2022])
if has_break:
    t_tilde_compare_raw = np.array([0.0 if y < break_year else (y - break_year) / (max_year - break_year) for y in annees_compare])
else:
    t_tilde_compare_raw = (annees_compare - min_year) / (max_year - min_year)
# Normalisation pipeline sur la période d'observation réelle
t_min_ret = t_tilde_obs_raw.min()
t_max_ret = t_tilde_obs_raw.max()
if t_max_ret == t_min_ret:
    t_tilde_compare = np.full_like(t_tilde_compare_raw, -0.5)
else:
    res0_compare = t_tilde_compare_raw / (t_max_ret - t_min_ret)
    dx = (t_tilde_obs_raw / (t_max_ret - t_min_ret)).min() + 0.5
    t_tilde_compare = res0_compare - dx
zTpa = mu0 + mu1*t_tilde_compare[0] + (sigma0 + sigma1*t_tilde_compare[0])/xi * CT
zTpb = mu0 + mu1*t_tilde_compare[1] + (sigma0 + sigma1*t_tilde_compare[1])/xi * CT
z_T_p_plot = (zTpb - zTpa) / zTpa * 100 if zTpa != 0 else np.nan
print(f"[PLOT] zTpa (1995): {zTpa:.6f}, zTpb (2022): {zTpb:.6f}, z_T_p (tendance %): {z_T_p_plot:.6f}")
print(f"[PLOT] t_tilde 1995: {t_tilde_compare[0]:.6f}, t_tilde 2022: {t_tilde_compare[1]:.6f}")

# Affichage des valeurs t_tilde utilisées pour 1995 et 2022 dans la courbe
idx_1995 = np.where(annees_retour == 1995)[0][0] if 1995 in annees_retour else None
idx_2022 = np.where(annees_retour == 2022)[0][0] if 2022 in annees_retour else None
if idx_1995 is not None and idx_2022 is not None:
    print(f"[COURBE] t_tilde 1995: {t_tilde_retour[idx_1995]:.6f}, t_tilde 2022: {t_tilde_retour[idx_2022]:.6f}")
    print(f"[COURBE] z_T_retour 1995: {z_T_retour[idx_1995]:.6f}, z_T_retour 2022: {z_T_retour[idx_2022]:.6f}")
    pourcent_courbe = (z_T_retour[idx_2022] - z_T_retour[idx_1995]) / z_T_retour[idx_1995] * 100
    print(f"[COURBE] % tendance (courbe): {pourcent_courbe:.6f}")
else:
    print("[COURBE] 1995 ou 2022 non trouvés dans annees_retour")
