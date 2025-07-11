import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import os
from src.utils.data_utils import cleaning_data_observed

# Création du dossier de sortie
output_dir = Path('graph_top')
output_dir.mkdir(exist_ok=True)

# Recherche de tous les fichiers niveau_retour.parquet
base_dir = Path('data/gev/observed')
# Filtrer uniquement les saisons voulues
saisons_voulues = {'mam', 'jja', 'son', 'hydro', 'djf'}
parquet_files = [f for f in base_dir.glob('*/*/niveau_retour.parquet') if f.parts[-2] in saisons_voulues]

# Agrégation des tendances
records = []
for file in parquet_files:
    echelle = file.parts[-3]
    saison = file.parts[-2]
    try:
        df = pl.read_parquet(file)
        if 'z_T_p' in df.columns:
            for row in df.iter_rows(named=True):
                records.append({
                    'station': row['NUM_POSTE'],
                    'z_T_p': row['z_T_p'],
                    'echelle': echelle,
                    'saison': saison,
                    'file': file
                })
    except Exception as e:
        print(f"Erreur lecture {file}: {e}")

# Sélection des 10 tendances les plus extrêmes (valeur absolue)
top10 = sorted(records, key=lambda x: abs(x['z_T_p']), reverse=True)[:10]

# Fonction de génération du graphique (adaptée de plot_nr.py)
def plot_for_station(station, echelle, saison, donnee, output_path):
    data_dir = Path(f'data/statisticals/{donnee}/{echelle}')
    available_years = [int(d.name) for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit() and len(d.name) == 4]
    if not available_years:
        print(f"Aucune année trouvée dans {data_dir}")
        return
    years = range(min(available_years), max(available_years)+1)
    dfs = []
    df_gev = pl.read_parquet(f'data/gev/{donnee}/{echelle}/{saison}/niveau_retour.parquet')
    params = df_gev.filter(pl.col('NUM_POSTE') == station)
    if params.height == 0:
        print(f"Pas de params pour {station} {echelle} {saison}")
        return
    mesure = 'max_mm_h' if echelle == "horaire" else 'max_mm_j'
    for year in years:
        file = data_dir / f'{year:04d}' / f'{saison}.parquet'
        if file.exists():
            df = pl.read_parquet(file, columns=['NUM_POSTE', mesure, 'nan_ratio'])
            df = df.filter(pl.col('NUM_POSTE') == station)
            if df.height > 0:
                df = df.with_columns(pl.lit(year).alias('year'))
                dfs.append(df)
    if not dfs:
        print(f"Aucune donnée trouvée pour la station {station}")
        return
    df_station = pl.concat(dfs)
    df_station = cleaning_data_observed(df_station, echelle)
    df_station = df_station.drop_nulls(subset=[mesure])
    years_obs = df_station['year'].to_numpy()
    maxs = df_station[mesure].to_numpy()
    nan_ratios = df_station['nan_ratio'].to_numpy()
    # Paramètres GEV
    df_gev = pl.read_parquet(f'data/gev/{donnee}/{echelle}/{saison}/gev_param_best_model.parquet')
    params = df_gev.filter(pl.col('NUM_POSTE') == station)
    if params.height == 0:
        print(f"Pas de params best_model pour {station} {echelle} {saison}")
        return
    mu0, mu1, sigma0, sigma1, xi = [params[x].item() for x in ['mu0','mu1','sigma0','sigma1','xi']]
    model = params['model'].item()
    has_break = '_break_year' in model
    min_year = 1990
    break_year = 1985
    val_comparaison = 1995
    max_year = 2022
    T = 10
    CT = ((-np.log(1-1/T))**(-xi) - 1)
    T2 = 20
    CT2 = ((-np.log(1-1/T2))**(-xi) - 1)
    annees_retour = np.arange(val_comparaison, max_year+1)
    # Normalisation
    if has_break:
        t_tilde_obs_raw = np.array([0.0 if y < break_year else (y - break_year) / (max_year - break_year) for y in years_obs])
        t_tilde_retour_raw = np.array([0.0 if y < break_year else (y - break_year) / (max_year - break_year) for y in annees_retour])
    else:
        t_tilde_obs_raw = (years_obs - min_year) / (max_year - min_year)
        t_tilde_retour_raw = (annees_retour - min_year) / (max_year - min_year)
    t_min_obs = t_tilde_obs_raw.min()
    t_max_obs = t_tilde_obs_raw.max()
    if t_max_obs == t_min_obs:
        t_tilde_obs_norm = np.full_like(t_tilde_obs_raw, -0.5)
        t_tilde_retour = np.full_like(t_tilde_retour_raw, -0.5)
    else:
        res0_obs = t_tilde_obs_raw / (t_max_obs - t_min_obs)
        dx = res0_obs.min() + 0.5
        res0_ret = t_tilde_retour_raw / (t_max_obs - t_min_obs)
        t_tilde_retour = res0_ret - dx
    z_T_retour = mu0 + mu1*t_tilde_retour + (sigma0 + sigma1*t_tilde_retour)/xi * CT
    z_T_retour_20 = mu0 + mu1*t_tilde_retour + (sigma0 + sigma1*t_tilde_retour)/xi * CT2
    annee_obs_min = years_obs.min()
    annee_obs_max = years_obs.max()
    mask_extrap = (annees_retour < annee_obs_min) | (annees_retour > annee_obs_max)
    mask_obs = (annees_retour >= annee_obs_min) & (annees_retour <= annee_obs_max)
    plt.figure(figsize=(6,6))
    plt.plot(years_obs, maxs, color='blue', linewidth=1, label='Maxima observés')
    for i, y in enumerate(years_obs):
        idx = np.where(annees_retour == y)[0]
        if len(idx) > 0:
            z_ref = z_T_retour[idx[0]]
            color = 'orange' if maxs[i] > z_ref else 'blue'
            plt.scatter(y, maxs[i], color=color, zorder=5)
        else:
            plt.scatter(y, maxs[i], color='blue', zorder=5)
    if mask_extrap.any():
        plt.plot(annees_retour[mask_extrap], z_T_retour[mask_extrap], color='red', linestyle='dashed', label='Niveau de retour 10 ans extrapolé')
    plt.plot(annees_retour[mask_obs], z_T_retour[mask_obs], color='red', linestyle='-', label='Niveau de retour 10 ans')
    if mask_extrap.any():
        plt.plot(annees_retour[mask_extrap], z_T_retour_20[mask_extrap], color='purple', linestyle='dashed', label='Niveau de retour 20 ans extrapolé')
    plt.plot(annees_retour[mask_obs], z_T_retour_20[mask_obs], color='purple', linestyle='-', label='Niveau de retour 20 ans')
    val_1985 = z_T_retour[annees_retour == val_comparaison][0] if val_comparaison in annees_retour else None
    val_2022 = z_T_retour[annees_retour == 2022][0] if 2022 in annees_retour else None
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
    if val_1985_20 is not None:
        plt.scatter(val_comparaison, val_1985_20, color='darkorange', zorder=5)
        plt.annotate(f"{val_1985_20:.1f}", (val_comparaison, val_1985_20), textcoords="offset points", xytext=(0,-15), ha='center', color='darkorange')
        plt.hlines(val_1985_20, annees_retour.min(), annees_retour.max(), colors='darkorange', linestyles=':', alpha=0.7)
    if val_2022_20 is not None:
        plt.scatter(2022, val_2022_20, color='darkorange', zorder=5)
        plt.axvline(2022, color='darkorange', linestyle=':', alpha=0.7)
        plt.annotate(f"{val_2022_20:.1f}", (2022, val_2022_20), textcoords="offset points", xytext=(0,-15), ha='center', color='darkorange')
    plt.title(f'Station {station} ({echelle}, {saison})')
    plt.xlabel('Année')
    plt.ylabel('Maximum annuel (mm/h)' if echelle=="horaire" else 'Maximum annuel (mm/j)')
    plt.legend()
    y_offset = -0.20
    if (val_1985 is not None) and (val_2022 is not None):
        pourcent = (val_2022 - val_1985) / val_1985 * 100
        texte_calcul_10 = r'\frac{{{:.1f}-{:.1f}}}{{{:.1f}}} \times 100 = {:.1f}\ \%'.format(val_2022, val_1985, val_1985, pourcent)
        plt.gca().text(
            0.5, y_offset, f'${texte_calcul_10}$',
            transform=plt.gca().transAxes, fontsize=12, color='black',
            ha='center', va='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
        )
        y_offset -= 0.08
    if (val_1985_20 is not None) and (val_2022_20 is not None):
        pourcent_20 = (val_2022_20 - val_1985_20) / val_1985_20 * 100
        texte_calcul_20 = r'\frac{{{:.1f}-{:.1f}}}{{{:.1f}}} \times 100 = {:.1f}\ \%'.format(val_2022_20, val_1985_20, val_1985_20, pourcent_20)
        plt.gca().text(
            0.5, y_offset, f'${texte_calcul_20}$',
            transform=plt.gca().transAxes, fontsize=12, color='black',
            ha='center', va='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
        )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Génération des graphiques pour les 10 tendances les plus extrêmes
for i, rec in enumerate(top10, 1):
    print(f"Génération du graphique {i} : Station {rec['station']} {rec['echelle']} {rec['saison']} (z_T_p={rec['z_T_p']:.2f}%)")
    try:
        plot_for_station(
            station=rec['station'],
            echelle=rec['echelle'],
            saison=rec['saison'],
            donnee='observed',
            output_path=output_dir / f'graph_top_{i}.png'
        )
    except Exception as e:
        print(f"Erreur lors de la génération du graphique {i} : {e}")
print("Terminé. Les graphiques sont dans graph_top/") 