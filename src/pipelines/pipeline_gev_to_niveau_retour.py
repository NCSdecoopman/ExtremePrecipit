# ┌────────────────────────────────────────────────────────────┐
# │       Estimation classique des paramètres GEV              │
# └────────────────────────────────────────────────────────────┘

# En estimation classique, on ajuste tous les paramètres (μ, σ, ξ)
# pour maximiser la log-vraisemblance ℓ(θ), où θ = (μ, σ, ξ).

# On cherche :
#     θ̂ = arg max_θ ℓ(θ)
# c’est-à-dire :
#     θ̂ = (μ̂, σ̂, ξ̂)

# Cette méthode est appelée estimation du maximum de vraisemblance (MLE).

# ┌────────────────────────────────────────────────────────────────────┐
# │    Formule de la log-vraisemblance ℓ(θ) pour la loi GEV            │
# └────────────────────────────────────────────────────────────────────┘

# Soit θ = (μ, σ, ξ) les paramètres de la loi GEV, et {x₁, ..., xₙ} les données.
# La log-vraisemblance s’écrit :

# Si ξ ≠ 0 :
# ℓ(μ, σ, ξ) = 
#     − n · log(σ)
#     − (1 + 1/ξ) · Σᵢ log(1 + ξ·(xᵢ − μ)/σ)
#     − Σᵢ [1 + ξ·(xᵢ − μ)/σ]^(−1/ξ)

# avec la condition : ∀i, 1 + ξ·(xᵢ − μ)/σ > 0

# Si ξ = 0 (cas Gumbel) :
# ℓ(μ, σ) = 
#     − n · log(σ)
#     − Σᵢ (xᵢ − μ)/σ
#     − Σᵢ exp(−(xᵢ − μ)/σ)

# Cette fonction est utilisée pour estimer les paramètres (μ, σ, ξ)
# par maximum de vraisemblance : θ̂ = arg max_θ ℓ(θ)


# ┌────────────────────────────────────────────────────────────────────┐
# │        Définition du niveau de retour en loi GEV                   │
# └────────────────────────────────────────────────────────────────────┘

# Le niveau de retour (ou quantile d’ordre 1 - 1/T) dans la loi GEV
# correspond à une valeur seuil z_T que l’on dépasse, en moyenne,
# une fois tous les T ans.

# Soit X ~ GEV(μ, σ, ξ), alors :
#     P(X > z_T) = 1 / T
#     P(X ≤ z_T) = 1 - 1 / T
#     z_T = F⁻¹(1 - 1/T), où F⁻¹ est la fonction quantile de la loi GEV

# Formule explicite du quantile z_T = f(μ, σ, ξ) :
# - Si ξ ≠ 0 :
#     z_T = μ + (σ / ξ) * [ ( -log(1 - 1/T) )^( -ξ ) - 1 ]
# - Si ξ = 0 (cas de Gumbel) :
#     z_T = μ - σ * log( -log(1 - 1/T) )

# Interprétation :
# - z₁₀ est la valeur de précipitation (ou autre variable extrême)
#   que l’on s’attend à dépasser une fois tous les 10 ans en moyenne.
# - Cela ne signifie pas qu’on ne peut pas l’observer deux années consécutives :
#   la probabilité de dépassement reste de 1/T chaque année.

# Exemple :
# μ = 20, σ = 5, ξ = 0.2
# T = 50 ans
# Alors :
#     z₅₀ = 20 + (5 / 0.2) * [ ( -log(1 - 1/50) )^(-0.2) - 1 ]

# ┌────────────────────────────────────────────────────────────────────┐
# │         Estimation de z_T et intervalle via vraisemblance profilée │
# └────────────────────────────────────────────────────────────────────┘

# En pratique, z_T dépend des paramètres μ, σ et ξ estimés à partir des données.
# On peut l’évaluer directement en injectant les valeurs estimées :
#     z_T^ = f(μ̂, σ̂, ξ̂)
# Le MLE classique donne un point estimate, mais pas d’intervalle.

# Mais on souhaite aussi connaître l’incertitude autour de cette estimation.
# Pour cela, on utilise la vraisemblance profilée.

# Idée :
# 1. On fixe z (candidat pour z_T) avec les valeurs de μ, σ et ξ estimés par le pipeline de GEV.
# 2. On ajuste les autres paramètres (par exemple σ et ξ) sous la contrainte :
#       z = μ + (σ / ξ) * [ ( -log(1 - 1/T) )^(-ξ) - 1 ]
# 3. On maximise la log-vraisemblance sous cette contrainte.
# 4. On répète cette opération pour différentes valeurs de z autour de l’estimation initiale.
#    Cela donne une **courbe de log-vraisemblance profilée** en fonction de z.

# Le maximum de cette courbe donne l’estimation de z_T.
# Les valeurs de z pour lesquelles la log-vraisemblance reste proche de son maximum
# définissent un **intervalle de confiance** pour z_T.

# En pratique, l’intervalle à 95 % correspond à :
#     2 * [ logL(ẑ_T) - logL(z) ] ≤ χ²(1, 0.95) ≈ 3.84

# → z_T est donc vu comme un **paramètre implicite** du modèle,
#   et la vraisemblance profilée permet de quantifier son incertitude sans recours au bootstrap.

# Cette méthode est très utile dans les modèles non-stationnaires où
# μ et σ varient dans le temps : on peut profiler z_T pour une année cible t*.

# Exemple typique :
#     modèle : μ(t) = μ₀ + μ₁·t    ;    σ(t) = σ₀ + σ₁·t
#     on cherche z_T(t = 2022)
#     on résout :
#         z = μ(t) + (σ(t) / ξ) * [ ( -log(1 - 1/T) )^( -ξ ) - 1 ]
#     puis on maximise la vraisemblance sous cette contrainte,
#     pour chaque valeur candidate de z.
