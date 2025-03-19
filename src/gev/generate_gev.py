import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import genextreme, kstest
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import kendalltau
from tqdm import tqdm

import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning


def fit_gev(data):
    """ Ajuste une GEV stationnaire et retourne les paramètres """
    if len(data) < 5:  # Vérification du nombre de données
        return np.nan, np.nan, np.nan
    c, loc, scale = genextreme.fit(-data)  # Inversion pour convenir à scipy test sur les maximas
    return c, loc, scale

def ks_test(data, c, loc, scale):
    """ Retourne le test KS """
    if len(data) < 5:  # Vérification du nombre de données
        return np.nan

    # Calcul des CDF théoriques pour le test KS
    cdf_values = genextreme(c, loc=loc, scale=scale).cdf(-data)

    # Test KS : comparaison entre données empiriques et GEV ajustée
    ks_pval = kstest(cdf_values, 'uniform')[1]

    return ks_pval

def test_stationarity(series):
    if len(series) < 5:
        return (np.nan,) * 4

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InterpolationWarning)
        # Test KPSS (Kwiatkowski-Phillips-Schmidt-Shin)
        # Si le p-value est < 0.05, on rejette l'hypothèse nulle ⇒ la série est probablement non stationnaire
        kpss_pval = kpss(series, regression='c')[1] 

    # Test de Dickey-Fuller augmenté (ADF)
    # Si la p-value est < 0.05, on rejette l'hypothèse nulle ⇒ la série est probablement stationnaire
    adf_pval = adfuller(series)[1]
    # Test de Kendall (Mann-Kendall trend test
    # Si p-value < 0.05 (ou un autre seuil classique) on rejette H0 ⇒ Il y a une tendance significative (hausse ou baisse)
    mk_pval = kendalltau(range(len(series)), series)[1] 
    # Test de Ljung-Box
    # Si le p-value est < 0.05, on rejette l’hypothèse nulle d'absence d’autocorrélation ⇒ la série présente de l’autocorrélation
    lb_pval = acorr_ljungbox(series, lags=[10], return_df=True)["lb_pvalue"].iloc[0] 

    return adf_pval, kpss_pval, mk_pval, lb_pval

import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

def process_group(args):
    (lat, lon), group = args
    data = group.set_index("year")["pr"]

    if data.isna().any():
        print(f"Alerte : valeurs manquantes détectées pour le point ({lat}, {lon})")

    c, loc, scale = fit_gev(data)
    adf_pval, kpss_pval, mk_pval, lb_pval = test_stationarity(data)
    ks_pval = ks_test(data, c, loc, scale)
    return [lat, lon, c, loc, scale, adf_pval, kpss_pval, mk_pval, lb_pval, ks_pval]


def process_file(df):
    """ Ajuste la GEV et effectue les tests de stationnarité en parallèle """
    groups = list(df.groupby(["lat", "lon"]))

    with Pool(processes=16) as pool:
        results = list(tqdm(pool.imap(process_group, groups), total=len(groups)))

    return pd.DataFrame(results, columns=[
        "lat", "lon", "c", "loc", "scale", "adf_pval", "kpss_pval", "mk_pval", "lb_pval", "ks_pval"
    ])


if __name__ == "__main__":
    # Définir le répertoire contenant les fichiers
    output_dir = "data/result/gev"
    os.makedirs(output_dir, exist_ok=True)

    # Trouver les fichiers
    file_paths = glob.glob(os.path.join("data/result/preanalysis/stats", "mm_*_max.parquet"))

    # Appliquer le traitement sur chaque fichier
    all_results = []
    for file in file_paths:
        print(f"Traitement de {file}")
        df = pd.read_parquet(file)

        result_df = process_file(df)

        # Affichage
        print(result_df.head(3))
        
        # Récupérer uniquement le nom du fichier sans extension
        file_name = os.path.splitext(os.path.basename(file))[0]
        
        # Sauvegarder dans output_dir
        output_path = os.path.join(output_dir, f"{file_name}_gev.parquet")
        result_df.to_parquet(output_path)
        print(f"Fichier enregistré {output_path}")