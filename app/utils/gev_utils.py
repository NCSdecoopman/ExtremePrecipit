import numpy as np

# --- Quantile GEV ---
# Soit :
#   μ(t)     = μ₀ + μ₁ × t                  # localisation dépendante du temps
#   σ(t)     = σ₀ + σ₁ × t                  # échelle dépendante du temps
#   ξ        = constante                    # forme
#   T        = période de retour (années)
#   p        = 1 − 1 / T                    # probabilité non-excédée associée

# La quantile notée qᵀ(t) (précipitation pour une période de retour T à l’année t) s’écrit :
#   qᵀ(t) = μ(t) + [σ(t) / ξ] × [ (−log(1 − p))^(−ξ) − 1 ]
#   qᵀ(t) = (μ₀ + μ₁ × t) + [(σ₀ + σ₁ × t) / ξ] × [ (−log(1 − (1/T)))^(−ξ) − 1 ]

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

def safe_compute_return(row, T_array, t_norm):
    """Force les None à 0 et retourne 0 si résultat invalide."""
    params = {
        "mu0": row["mu0"] or 0,
        "mu1": row["mu1"] or 0,
        "sigma0": row["sigma0"] or 0,
        "sigma1": row["sigma1"] or 0,
        "xi": row["xi"] or 0,
    }

    return compute_return_levels_ns(params, T_array, t_norm)