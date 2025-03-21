import os
import concurrent.futures

import pandas as pd
import numpy as np

from src.utils.log_utils import setup_console_logger
from src.utils.file_utils import find_files
from src.utils.stats_utils import gev_stationnaire

logger = setup_console_logger()

def generate_groups(file_path: str) -> None:
    logger.info(f"Traitement de {file_path}")
    
    df = pd.read_parquet(file_path)
    if df.empty:
        logger.warning(f"{file_path} est vide, fichier ignore.")
        return
    
    groups = list(df.groupby(['lat', 'lon']))
    total_groups = len(groups)

    return groups, total_groups

def bootstrap_gev(data, n_bootstrap=100, ci=0.95):
    """Bootstrap des paramètres GEV pour obtenir des intervalles de confiance."""
    boot_params = []

    for _ in range(n_bootstrap):
        sample = data.sample(frac=1, replace=True)
        xi, mu, sigma = gev_stationnaire(sample)
        boot_params.append((xi, mu, sigma))

    boot_params = np.array(boot_params)
    
    lower = (1 - ci) / 2
    upper = 1 - lower
    
    ci_bounds = {
        'xi_ci_lower': np.quantile(boot_params[:, 0], lower),
        'xi_ci_upper': np.quantile(boot_params[:, 0], upper),
        'mu_ci_lower': np.quantile(boot_params[:, 1], lower),
        'mu_ci_upper': np.quantile(boot_params[:, 1], upper),
        'sigma_ci_lower': np.quantile(boot_params[:, 2], lower),
        'sigma_ci_upper': np.quantile(boot_params[:, 2], upper),
    }
    
    xi, mu, sigma = np.mean(boot_params, axis=0)
    return xi, mu, sigma, ci_bounds

def generate_gev(lat, lon, group):
    if group.empty:
        logger.warning(f"Groupe vide pour lat={lat}, lon={lon}")

    NaN = group['pr'].isna().sum()
    if NaN > 0:
        logger.warning(f"NaN pour lat={lat}, lon={lon}, NaN={NaN}")
        data = group['pr'].dropna()  
    else:
        data = group['pr']

    if len(data) < 5:
        logger.warning(f"Trop peu de donnees pour lat={lat}, lon={lon}, n={len(data)}")

    xi, mu, sigma, ci_bounds = bootstrap_gev(data)
    return lat, lon, xi, mu, sigma, ci_bounds

def process_group(group_data):
    lat, lon = group_data[0]
    group = group_data[1]
    lat, lon, xi, mu, sigma, ci_bounds = generate_gev(lat, lon, group)
    return {
        'lat': lat, 'lon': lon, 
        'mu': mu, 'mu_ci_lower': ci_bounds['mu_ci_lower'], 'mu_ci_upper': ci_bounds['mu_ci_upper'],
        'sigma': sigma, 'sigma_ci_lower': ci_bounds['sigma_ci_lower'], 'sigma_ci_upper': ci_bounds['sigma_ci_upper'],
        'xi': xi, 'xi_ci_lower': ci_bounds['xi_ci_lower'], 'xi_ci_upper': ci_bounds['xi_ci_upper']
    }

def save_results(results, file_path, output_dir):
    if results:
        gev_params = pd.DataFrame(results)
        logger.info(f"{len(gev_params)} points valides traites pour {file_path}")
        output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}_gev_stationnaire.parquet")
        gev_params.to_parquet(output_path)
        logger.info(f"Résultat sauvegardé pour {file_path} dans {output_path}")
        return len(gev_params)
    else:
        logger.warning(f"Aucun résultat à sauvegarder pour {file_path}")
        return 0

def main(num_workers=16):
    output_dir = "data/result/gev"
    os.makedirs(output_dir, exist_ok=True)
    
    file_paths = find_files(path="data/result/preanalysis/stats", pattern="mm_*_max.parquet")
    logger.info(f"{len(file_paths)} fichier(s) trouvés à traiter.")

    for file_path in file_paths:
        groups, total_groups = generate_groups(file_path)
        results = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i, group_data in enumerate(groups, start=1):
                futures.append(executor.submit(process_group, group_data))
            logger.info(f"Soumission: {total_groups} groupes envoyes au pool")

            for i, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                if i % 100 == 0 or i == total_groups:
                    logger.info(f"Progression: {i}/{total_groups} groupes traites")
        
        valid_count = save_results(results, file_path, output_dir)
        logger.info(f"{valid_count} points valides sauvegardés pour {file_path}")

if __name__ == "__main__":
    main()
