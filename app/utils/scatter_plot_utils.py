import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.stats import genextreme

def generate_scatter_plot_interactive(df: pl.DataFrame, stat_choice: str, unit_label: str, height: int,
                                      x_label: str = "pr_mod", y_label: str = "pr_obs"):
    df_pd = df.select(["NUM_POSTE_obs", "NUM_POSTE_mod", "lat", "lon", x_label, y_label]).to_pandas()

    fig = px.scatter(
        df_pd,
        x=x_label,
        y=y_label,
        title="",
        opacity=0.5,
        width=height,
        height=height,
        labels={
            x_label: f"{stat_choice} du modèle AROME ({unit_label})",
            y_label: f"{stat_choice} des stations ({unit_label})"
        },
        hover_data={"lat": True, "lon": True}
    )

    precision = ".1f" if unit_label == "mm/j" else ".2f"
    fig.update_traces(
        hovertemplate=
        "Lat: %{customdata[2]:.4f}<br>Lon: %{customdata[3]:.4f}<br>"
        f"{x_label} : %{{x:{precision}}}<br>{y_label} : %{{y:{precision}}}<extra></extra>",
        customdata=df_pd[["NUM_POSTE_obs", "NUM_POSTE_mod", "lat", "lon"]].values
    )

    x_range = [df_pd[x_label].min(), df_pd[x_label].max()]
    y_range = [df_pd[y_label].min(), df_pd[y_label].max()]
    min_diag = min(x_range[0], y_range[0])
    max_diag = min(x_range[1], y_range[1])

    fig.add_trace(
        go.Scatter(
            x=[min_diag, max_diag],
            y=[min_diag, max_diag],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='y = x',
            hoverinfo='skip'
        )
    )

    return fig



def generate_return_period_plot_interactive(
    T, y_obs, y_mod,
    label_obs="Stations", label_mod="AROME",
    unit: str = "mm/j", height: int = 600,
    points_obs: dict | None = None,
    points_mod: dict | None = None
):
    fig = go.Figure()

    # Courbe observations
    fig.add_trace(go.Scatter(
        x=T,
        y=y_obs,
        mode="lines",
        name=label_obs,
        line=dict(color="blue"),
        hovertemplate="Période : %{x:.1f} ans<br>Précipitation : %{y:.1f} " + unit + "<extra></extra>"
    ))

    # Courbe modèle
    fig.add_trace(go.Scatter(
        x=T,
        y=y_mod,
        mode="lines",
        name=label_mod,
        line=dict(color="orange"),
        hovertemplate="Période : %{x:.1f} ans<br>Précipitation : %{y:.1f} " + unit + "<extra></extra>"
    ))

    # Points maximas observés (facultatif)
    if points_obs is not None:
        fig.add_trace(go.Scatter(
            x=points_obs["year"],
            y=points_obs["value"],
            mode="markers",
            name="Maximas mesurés",
            marker=dict(color="blue", size=4, symbol="x"),
            hovertemplate="Période : %{x:.1f} ans<br>Max observé : %{y:.1f} " + unit + "<extra></extra>"
        ))

    # Maximas annuels bruts (facultatif)
    if points_mod is not None:
        fig.add_trace(go.Scatter(
            x=points_mod["year"],
            y=points_mod["value"],
            mode="markers",
            name="Maximas modélisés",
            marker=dict(color="orange", size=4, symbol="x"),
            hovertemplate="Année : %{x:.1f}<br>Max : %{y:.1f} " + unit + "<extra></extra>"
        ))

    fig.update_layout(
        xaxis=dict(
            title="Période de retour (ans)",
            type="log",
            showgrid=True,
            minor=dict(ticklen=4, showgrid=True),
        ),
        yaxis=dict(
            title=f"Précipitation ({unit})",
            showgrid=True,
            minor=dict(ticklen=4, showgrid=True),
        ),
        template="plotly_white",
        height=height
    )

    return fig


def generate_gev_density_comparison_interactive(
    maxima_obs: np.ndarray,
    maxima_mod: np.ndarray,
    params_obs: dict,
    params_mod: dict,
    unit: str = "mm/j",
    height: int = 500,
    t_norm: float = 0.0,  # Covariable normalisée (ex: 0 pour année médiane)
):
    """
    Trace deux courbes de densité GEV (observée et modélisée) superposées, sans histogramme.
    """

    # --- Récupération des paramètres observés ---
    mu_obs = params_obs.get("mu0", 0) + params_obs.get("mu1", 0) * t_norm
    sigma_obs = params_obs.get("sigma0", 1) + params_obs.get("sigma1", 0) * t_norm
    xi_obs = params_obs.get("xi", 0)

    # --- Récupération des paramètres modélisés ---
    mu_mod = params_mod.get("mu0", 0) + params_mod.get("mu1", 0) * t_norm
    sigma_mod = params_mod.get("sigma0", 1) + params_mod.get("sigma1", 0) * t_norm
    xi_mod = params_mod.get("xi", 0)

    # --- Domaine commun pour tracer ---
    minima = min(maxima_obs.min(), maxima_mod.min()) * 0.9
    maxima = max(maxima_obs.max(), maxima_mod.max()) * 1.1
    x = np.linspace(minima, maxima, 500)

    # --- Densités ---
    density_obs = genextreme.pdf(x, c=-xi_obs, loc=mu_obs, scale=sigma_obs)
    density_mod = genextreme.pdf(x, c=-xi_mod, loc=mu_mod, scale=sigma_mod)

    # --- Création figure ---
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x,
        y=density_obs,
        mode="lines",
        name="GEV observée",
        line=dict(color="blue"),
        hovertemplate="Maxima : %{x:.1f} " + unit + "<br>Densité : %{y:.3f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=x,
        y=density_mod,
        mode="lines",
        name="GEV modélisée",
        line=dict(color="orange"),
        hovertemplate="Maxima : %{x:.1f} " + unit + "<br>Densité : %{y:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title="",
        xaxis_title=f"Maximum journalier ({unit})",
        yaxis_title="Densité",
        template="plotly_white",
        height=height,
    )

    return fig
    


import numpy as np
import plotly.graph_objects as go
from scipy.stats import genextreme
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def generate_gev_density_comparison_interactive_3D(
    maxima_obs: np.ndarray,
    maxima_mod: np.ndarray,
    params_obs: dict,
    params_mod: dict,
    unit: str = "mm/j",
    height: int = 500,
    min_year: int = 1960,
    max_year: int = 2015,
):
    """
    Trace deux ensembles de courbes de densité GEV (observée et modélisée) superposées,
    en faisant varier la couleur de violet (min_year) à jaune (max_year).
    """

    # --- Génération des années ---
    years = np.arange(min_year, max_year + 1)

    # --- Couleurs violet -> jaune ---
    cmap = cm.get_cmap('plasma')
    norm = mcolors.Normalize(vmin=min_year, vmax=max_year)
    colors = [mcolors.to_hex(cmap(norm(year))) for year in years]

    # --- Domaine commun pour tracer ---
    minima = min(maxima_obs.min(), maxima_mod.min()) * 0.9
    maxima = max(maxima_obs.max(), maxima_mod.max()) * 1.1
    x = np.linspace(minima, maxima, 500)

    # --- Création de la figure ---
    fig = go.Figure()

    for i, year in enumerate(years):
        t_norm = (year - (min_year + max_year) / 2) / (max_year - min_year)

        # Densité observée
        mu_obs = params_obs.get("mu0", 0) + params_obs.get("mu1", 0) * t_norm
        sigma_obs = params_obs.get("sigma0", 1) + params_obs.get("sigma1", 0) * t_norm
        xi_obs = params_obs.get("xi", 0)

        density_obs = genextreme.pdf(x, c=-xi_obs, loc=mu_obs, scale=sigma_obs)

        fig.add_trace(go.Scatter(
            x=x,
            y=density_obs,
            mode="lines",
            line=dict(color=colors[i]),
            name=f"Obs {year}",
            hovertemplate=f"Obs {year}<br>Maxima : %{{x:.1f}} {unit}<br>Densité : %{{y:.3f}}<extra></extra>",
            showlegend=False,
        ))

        # Densité modélisée
        mu_mod = params_mod.get("mu0", 0) + params_mod.get("mu1", 0) * t_norm
        sigma_mod = params_mod.get("sigma0", 1) + params_mod.get("sigma1", 0) * t_norm
        xi_mod = params_mod.get("xi", 0)

        density_mod = genextreme.pdf(x, c=-xi_mod, loc=mu_mod, scale=sigma_mod)

        fig.add_trace(go.Scatter(
            x=x,
            y=density_mod,
            mode="lines",
            line=dict(color=colors[i]),
            name=f"Mod {year}",
            hovertemplate=f"Mod {year}<br>Maxima : %{{x:.1f}} {unit}<br>Densité : %{{y:.3f}}<extra></extra>",
            showlegend=False,
        ))

    # --- Layout final ---
    fig.update_layout(
        title="",
        xaxis_title=f"Maximum journalier ({unit})",
        yaxis_title="Densité",
        template="plotly_white",
        height=height,
    )

    return fig




def generate_time_series_maxima_interactive(
    years_obs: np.ndarray,
    max_obs: np.ndarray,
    years_mod: np.ndarray,
    max_mod: np.ndarray,
    unit: str = "mm/j",
    height: int = 500,
    nr_year: int = 20,
    return_levels_obs: float | None = None,
    return_levels_mod: float | None = None
):
    fig_time_series = go.Figure()

    # --- Observations (seulement en 'x' sans lignes)
    fig_time_series.add_trace(go.Scatter(
        x=years_obs,
        y=max_obs,
        mode='markers',
        name='Maximas observés',
        marker=dict(symbol='x', size=4, color="blue")
    ))

    # --- Modèle (seulement en 'x' sans lignes)
    fig_time_series.add_trace(go.Scatter(
        x=years_mod,
        y=max_mod,
        mode='markers',
        name='Maximas modélisés',
        marker=dict(symbol='x', size=4, color="orange")
    ))

    # --- Niveau de retour 20 ans observé
    if return_levels_obs is not None:
        fig_time_series.add_trace(go.Scatter(
            x=years_obs,   # ➔ Utilise toutes les années observées !
            y=return_levels_obs,
            mode='lines',
            name=f'NR observé {nr_year} ans',
            line=dict(color='blue', dash='solid')
        ))

    # --- Niveau de retour 20 ans modélisé
    if return_levels_mod is not None:
        fig_time_series.add_trace(go.Scatter(
            x=years_mod,   # ➔ Utilise toutes les années modélisées !
            y=return_levels_mod,
            mode='lines',
            name=f'NR modélisé {nr_year} ans',
            line=dict(color='orange', dash='solid')
        ))

    fig_time_series.update_layout(
        title="",
        xaxis_title="Année",
        yaxis_title=f"Maxima annuel ({unit})",
        height=height,
        template="plotly_white"
    )

    return fig_time_series

import numpy as np
import plotly.graph_objects as go
from scipy.stats import genextreme

def generate_loglikelihood_profile_xi(
    maxima: np.ndarray,
    params: dict,
    unit: str = "mm/j",
    xi_range: float = 3,
    height: int = 500,
    t_norm: float = 0.0
):
    """
    Trace le profil de log-vraisemblance autour de ξ ajusté.
    
    - maxima : valeurs maximales (array)
    - params : dictionnaire des paramètres GEV
    - unit : unité des maxima
    - xi_range : +/- intervalle autour de ξ pour tracer
    - height : hauteur de la figure
    - t_norm : covariable temporelle normalisée
    """

    # Récupération des paramètres (à t_norm donné)
    mu = params.get("mu0", 0) + params.get("mu1", 0) * t_norm
    sigma = params.get("sigma0", 1) + params.get("sigma1", 0) * t_norm
    xi_fit = params.get("xi", 0)

    def compute_nllh(x, mu, sigma, xi):
        if sigma <= 0:
            return np.inf
        try:
            return -np.sum(genextreme.logpdf(x, c=-xi, loc=mu, scale=sigma))
        except Exception:
            return np.inf

    # Points autour du ξ ajusté
    xis = np.linspace(xi_fit - xi_range, xi_fit + xi_range, 200)
    logliks = [-compute_nllh(maxima, mu, sigma, xi) for xi in xis]

    # --- Création figure Plotly ---
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=xis,
        y=logliks,
        mode="lines",
        line=dict(color="blue"),
        name="Log-vraisemblance",
        hovertemplate="ξ : %{x:.3f}<br>Log-likelihood : %{y:.1f}<extra></extra>"
    ))

    # Ligne verticale au ξ ajusté
    fig.add_trace(go.Scatter(
        x=[xi_fit, xi_fit],
        y=[min(logliks), max(logliks)],
        mode="lines",
        line=dict(color="red", dash="dash"),
        name=f"ξ ajusté ({xi_fit:.3f})"
    ))

    fig.update_layout(
        title="",
        xaxis_title="ξ",
        yaxis_title="Log-vraisemblance",
        template="plotly_white",
        height=height
    )

    return fig
