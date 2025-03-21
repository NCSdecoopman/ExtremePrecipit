import numpy as np
import pandas as pd
from typing import Tuple

from scipy.stats import kendalltau
from statsmodels.tsa.stattools import adfuller, kpss
from hades_stats import sp_dist

def mann_kendall_test(years, series, alpha=0.05):
    """Teste la tendance d'une série avec Kendall's tau."""
    mask = ~np.isnan(series)
    years_clean = np.array(years)[mask]
    series_clean = np.array(series)[mask]
    
    _, pvalue = kendalltau(years_clean, series_clean)
    
    return pvalue >= alpha # True signifie aucune tendance

def adf_test(series, alpha=0.05):
    """Teste la stationnarité d'une série avec le test ADF."""
    series_clean = series[~np.isnan(series)]
    
    result = adfuller(series_clean)
    pvalue = result[1]
    
    return pvalue < alpha  # True signifie stationnaire

def kpss_test(series, alpha=0.05):
    """Teste la stationnarité d'une série avec le test KPSS."""
    series_clean = series[~np.isnan(series)]
    
    result = kpss(series_clean)
    pvalue = result[1]
    
    return pvalue < alpha

def gev_stationnaire(serie: pd.Series) -> Tuple[float, float, float]:
    """Renvoie xi, mu, sigma pour une GEV stationnaire."""
    return sp_dist('genextreme').fit(serie)