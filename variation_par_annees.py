# On a : qᵀ(t) = μ₀ + μ₁·t + [(σ₀ + σ₁·t) / ξ] · { [ -log(1 - 1/T) ]^(−ξ) − 1 }
# avec :
# μ(t) = μ₀ + μ₁·t : localisation dépendante du temps
# σ(t) = σ₀ + σ₁·t : échelle dépendante du temps
# ξ : forme (fixe)
# T : période de retour
# t : année (ou covariable normalisée)

# Soit : z_T = [ -log(1 - 1/T) ]^(−ξ) − 1   ← constante pour un T donné
# Donc : qᵀ(t) = μ₀ + μ₁·t + [(σ₀ + σ₁·t) / ξ] · z_T

# On cherche : Δqᵀ = qᵀ(t + 10) − qᵀ(t)


# Δqᵀ = [μ₀ + μ₁·(t + 10) + ((σ₀ + σ₁·(t + 10)) / ξ)·z_T]
#      − [μ₀ + μ₁·t       + ((σ₀ + σ₁·t)       / ξ)·z_T]

# Soit :
# Δqᵀ = μ₁·10 + (σ₁·10 / ξ)·z_T


# Donc :
# Δqᵀ (tous les 10 ans) = 10·μ₁ + 10·σ₁·z_T / ξ

import numpy as np

def delta_qT_decennale(mu1, sigma1, xi, T):
    """
    Calcule la variation décennale du quantile de retour qᵀ(t) 
    dans un modèle GEV non stationnaire (μ(t), σ(t)), ξ constant.

    Paramètres :
    ------------
    mu1 : float
        Coefficient de variation temporelle de μ(t) (ex. mm/an)
    sigma1 : float
        Coefficient de variation temporelle de σ(t)
    xi : float
        Paramètre de forme ξ (peut être négatif)
    T : float
        Période de retour (ex: 20, 50, 100)

    Retour :
    --------
    delta_q : float
        Variation du quantile de retour sur 10 ans
    """
    # Calcul de z_T = [-log(1 - 1/T)]^(-ξ) - 1
    log_term = -np.log(1 - 1/T)
    z_T = log_term**(-xi) - 1

    # Variation décennale de qᵀ
    delta_q = 10 * mu1 + 10 * sigma1 * z_T / xi
    return delta_q


# Exemple 
mu1 = 0.2       # mm/an
sigma1 = 0.1    # mm/an
xi = 0.1        # paramètre de forme
T = 20          # période de retour en années

dq = delta_qT_decennale(mu1, sigma1, xi, T)
print(f"Variation décennale de q^{T} : {dq:.2f} mm")
