import numpy as np
import polars as pl
from scipy.special import gamma

# --- Quantile GEV ---
# Soit :
#   μ(t)     = μ₀ + μ₁ × t                  # localisation dépendante du temps
#   σ(t)     = σ₀ + σ₁ × t                  # échelle dépendante du temps
#   ξ        = constante                    # forme
#   T        = période de retour (années)
#   p        = 1 − 1 / T                    # probabilité non-excédée associée
# Avec t : année (ou covariable normalisée dans l'intervalle [0; 1]
# t = (annee - min_year) / (max_year - min_year) = (annee - min_year) / delta_year
# Une unité de t (normalisée) = Δa années (max_year - min_year)

# En notant Δa = max_year - min_year et a = annee, on a :
# t = (a − aₘᵢₙ) / Δa  ⇒   a = aₘᵢₙ + t ⋅ Δa

# La quantile notée qᵀ(t) (précipitation pour une période de retour T à l’année t) s’écrit :
#   qᵀ(t) = μ(t) + [σ(t) / ξ] × [ (−log(1 − p))^(−ξ) − 1 ]
#   qᵀ(t) = (μ₀ + μ₁ × t) + [(σ₀ + σ₁ × t) / ξ] × [ (−log(1 − (1/T)))^(−ξ) − 1 ]

# Soit : z_T = [ -log(1 - 1/T) ]^(−ξ) − 1   ← constante pour un T donné
# Donc : qᵀ(t) = μ₀ + μ₁·t + [(σ₀ + σ₁·t) / ξ] · z_T
# Ou : qᵀ(t) = μ(t) + [σ(t) / ξ] · z_T

# En dérivant qᵀ par rapport à t on a :
# dqᵀ/dt = μ₁ + σ₁ / ξ · z_T
# On rappelle :  a = aₘᵢₙ + t ⋅ Δa
# Donc : dt/da = 1 / Δa

# Alors dqᵀ/da = dqᵀ/dt · dt/da = μ₁ + σ₁ / ξ · z_T · 1 / Δa

# LA VARIATION PAR AN de qᵀ :
# dqᵀ/da = 1 / Δa · (μ₁ + σ₁ / ξ · z_T)
# DONC PAR 10 ANS :
# Δqᵀ₁₀ₐₙₛ = (10 / Δa) ⋅ (μ₁ + (σ₁ / ξ) ⋅ zᵀ)



def safe_compute_return_df(df: pl.DataFrame) -> pl.DataFrame:
    REQUIRED_GEV_COLS = ["mu0", "mu1", "sigma0", "sigma1", "xi"]
    for col in REQUIRED_GEV_COLS:
        if col not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias(col))
    df = df.with_columns([
        pl.col(col).fill_null(0.0).fill_nan(0.0) for col in REQUIRED_GEV_COLS
    ])
    return df


def compute_return_levels_ns(params: dict, T: np.ndarray, t_norm: float) -> np.ndarray:
    """
    Calcule les niveaux de retour selon le modèle NS-GEV fourni.
    - params : dictionnaire des paramètres GEV d'un point
    - T : périodes de retour (en années)
    - t_norm : covariable temporelle normalisée (ex : 0 pour année moyenne)
    """    
    mu = params.get("mu0", 0) + params["mu1"] * t_norm if "mu1" in params else params.get("mu0", 0) # μ(t)
    sigma = params.get("sigma0", 0) + params["sigma1"] * t_norm if "sigma1" in params else params.get("sigma0", 0) # σ(t)
    xi = params.get("xi", 0) # xi contant

    if xi != 0:
        qT = mu + (sigma / xi) * ((-np.log(1 - 1 / T))**(-xi) - 1)
    else:
        qT = mu - sigma * np.log(-np.log(1 - 1/T))

    return qT


def delta_qT_X_years(mu1, sigma1, xi, T, year_range, par_X_annees):
    """
    Calcule la variation décennale du quantile de retour qᵀ(t)
    dans un modèle GEV non stationnaire avec t ∈ [0, 1].

    La variation est ramenée à l’échelle des années civiles en tenant compte de la
    durée totale du modèle (year_range = a_max - a_min).
    Si un point de rupture est introduit year_range = a_max - a_rupture,
    avec une Δqᵀ = 0 avant la rupture.

    Δqᵀ = (par_X_annees / year_range) × (μ₁ + (σ₁ / ξ) × z_T)
    avec :
    - z_T = [ -log(1 - 1/T) ]^(-ξ) - 1   si ξ ≠ 0
          = log(-log(1 - 1/T))          si ξ = 0 (Gumbel)

    par_X_annees représente 10, 20, 30 ans dans Δ_10ans qᵀ
    """
    try:
        p = 1 - 1 / T
        if xi == 0:
            z_T = np.log(-np.log(p))
            delta_q = (par_X_annees / year_range) * (mu1 + sigma1 * z_T)
        else:
            z_T = (-np.log(p))**(-xi) - 1
            delta_q = (par_X_annees / year_range) * (mu1 + (sigma1 / xi) * z_T)
        return float(delta_q)
    except Exception:
        return np.nan


def compute_delta_qT(row, T_choice, year_range, par_X_annees):
    return delta_qT_X_years(
        row["mu1"], 
        row["sigma1"], 
        row["xi"], 
        T=T_choice, 
        year_range=year_range,
        par_X_annees=par_X_annees
    )


# --- Espérence, variance, CV de GEV ---

def gev_moments(mu, sigma, xi):
    if xi >= 0.5:
        return np.nan, np.nan, np.nan  # variance indéfinie
    try:
        mean = mu + sigma / xi * (gamma(1 - xi) - 1)
        var = (sigma ** 2) / (xi ** 2) * (gamma(1 - 2 * xi) - gamma(1 - xi) ** 2)
        cv = np.sqrt(var) / mean if mean != 0 else np.nan
        return mean, var, cv
    except Exception:
        return np.nan, np.nan, np.nan


def eval_params_nsgev(mu0, mu1, sigma0, sigma1, xi, t, t0):
    mu_t = mu0 + mu1 * (t - t0)
    sigma_t = sigma0 + sigma1 * (t - t0)
    return gev_moments(mu_t, sigma_t, xi)


def compute_delta_stat(row, stat: str, year_start: int, year_ref: int, year_end: int, par_X_annees: int) -> float:
    """
    Calcule la variation du moment statistique GEV (moyenne, variance, CV)
    exprimée en changement moyen par 10 ans.

    Parameters:
    - row : dictionnaire contenant les paramètres GEV
    - stat : "ΔE", "ΔVar" ou "ΔCV"
    - year_start, year_end : années de début et de fin
    - year_ref : année de référence (t0)

    Returns:
    - Variation du moment sélectionné rapportée à 10 ans
    """
    Δa = year_end - year_start
    if Δa == 0:
        return np.nan  # évite division par zéro

    # Moments aux deux dates
    mean_start, var_start, cv_start = eval_params_nsgev(
        mu0=row["mu0"], mu1=row.get("mu1", 0.0),
        sigma0=row["sigma0"], sigma1=row.get("sigma1", 0.0),
        xi=row["xi"], t=year_start, t0=year_ref
    )

    mean_end, var_end, cv_end = eval_params_nsgev(
        mu0=row["mu0"], mu1=row.get("mu1", 0.0),
        sigma0=row["sigma0"], sigma1=row.get("sigma1", 0.0),
        xi=row["xi"], t=year_end, t0=year_ref
    )

    if stat == "ΔE":
        return (mean_end - mean_start)  * par_X_annees / Δa
    elif stat == "ΔVar":
        return (var_end - var_start) * par_X_annees / Δa
    elif stat == "ΔCV":
        return (cv_end - cv_start) * par_X_annees / Δa
    else:
        return np.nan

