import os
import numpy as np
import pandas as pd
from scipy.stats import genextreme

# Fonction pour calculer les quantiles GEV
def gev_quantile(mu, sigma, xi, T):
    if xi != 0:
        return mu + (sigma / xi) * ((-np.log(1 - 1 / T))**(-xi) - 1)
    else:  # Cas xi = 0 : loi de Gumbel
        return mu - sigma * np.log(-np.log(1 - 1 / T))

# Calcul des quantiles
def compute_quantiles(row):
    mu, sigma, xi = row['mu'], row['sigma'], row['xi']
    return {T: gev_quantile(mu, sigma, xi, T) for T in return_periods}

# Fonction Bootstrap pour intervalles de confiance
def bootstrap_ci(row, T, n_bootstrap=1000, alpha=0.05):
    mu, sigma, xi = row['mu'], row['sigma'], row['xi']
    boot_samples = [gev_quantile(mu, sigma, xi, T) for _ in range(n_bootstrap)]
    lower_bound = np.percentile(boot_samples, 100 * (alpha / 2))
    upper_bound = np.percentile(boot_samples, 100 * (1 - alpha / 2))
    return lower_bound, upper_bound
  
if __name__ == "__main__":
    # Chargement des données
    data = pd.read_parquet('data/result/gev/param_grid.parquet')

    # Définition des périodes de retour
    return_periods = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    print(f"Périodes de retours étudiées : {return_periods}")

    # Appliquer la fonction à toutes les lignes
    data_quantiles = data.apply(compute_quantiles, axis=1, result_type='expand')

    # Ajouter les résultats au dataframe
    data = pd.concat([data, data_quantiles], axis=1)

    # Calcul des intervalles de confiance
    # print("Calculs des intervalles de confiances")
    # for T in return_periods:
    #     print(f"    pour la periode : {T}")
    #     ci_values = data.apply(lambda row: bootstrap_ci(row, T), axis=1)
    #     data[f'{T}_lower'], data[f'{T}_upper'] = zip(*ci_values)

    # Sauvegarde
    output_dir = "data/result/gev"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "quantiles_grid.parquet")
    data.to_parquet(file_path)
    
    print(f"Données enregistrées dans {file_path} :\n{data.head()}")