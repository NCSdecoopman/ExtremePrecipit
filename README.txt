# Génération du dictionnaire année avec le fichier correspondant

from src.preanalysis import generate_files
dict_files = generate_files()

# Obtention des données pour une année
from src.preanalysis import generate_data
ds, ds_france = generate_data(dict_files, year=2001)

# Affichage des informations du .nc
from src.analysis.nc_analysis import generate_markdown_summary
generate_markdown_summary(ds)

# Sauvegarde chaque année les informations de la France dans un fichier Zarr distinct
# Traitement avec gestion automatique de la mémoire par Dask
python -m src.processed.merge_year

# Sauvegarde le df de NaN et résolution temporelle dans "data/preanalysis/nan_resol_temp.csv"
python -m src.preanalysis

# Génère les statistiques dans "data/result/"
python -m src.analysis.generate_stats

# Génère les paramètres GEV et p values associées de chaque point de grille dans "data/result/gev/param_grid.parquet"
python src/gev/generate_stats.py

# Génère les quantiles de retour et intervalles associés de chaque point de grille dans "data/result/gev/quantiles_grid.parquet"
python src/gev/generate_stats.py