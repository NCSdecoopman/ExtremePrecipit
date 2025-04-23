import streamlit as st

from pathlib import Path
from functools import reduce

from app.utils.config_utils import *
from app.utils.menus_utils import *
from app.utils.data_utils import *
from app.utils.stats_utils import *
from app.utils.map_utils import *
from app.utils.legends_utils import *
from app.utils.hist_utils import *
from app.utils.scatter_plot_utils import *
from app.utils.show_info import show_info_data, show_info_metric

import pydeck as pdk
import polars as pl

ns_param_map = {
    "s_gev": {"mu": "μ", "sigma": "σ", "xi": "ξ"},
    "ns_gev_m1": {"mu0": "μ₀", "mu1": "μ₁", "sigma0": "σ₀", "xi": "ξ"},
    "ns_gev_m2": {"mu0": "μ₀", "sigma0": "σ₀", "sigma1": "σ₁", "xi": "ξ"},
    "ns_gev_m3": {"mu0": "μ₀", "mu1": "μ₁", "sigma0": "σ₀", "sigma1": "σ₁", "xi": "ξ"},
}

# Calcul du ratio de stations valides
def compute_valid_ratio(df: pl.DataFrame, param_list: list[str]) -> float:
    n_total = df.height
    n_valid = df.drop_nulls(subset=param_list).height
    return round(n_valid / n_total, 3) if n_total > 0 else 0.0

# Calcul de la moyenne et de l'écart-type d'une colonne
def loglike_score(df: pl.DataFrame, column: str = "log_likelihood") -> str:
    mean = df[column].mean()
    std = df[column].std()
    return f"{mean:.2f} (± {std:.2f})"

def filter_nan(df: pl.DataFrame, columns: list[str]):
    return df.drop_nulls(subset=columns)

def filter_percentile_all(df: pl.DataFrame, quantile_choice: float, columns: list[str]):
    quantiles = {
        col: df.select(pl.col(col).quantile(quantile_choice, "nearest")).item()
        for col in columns
    }
    filter_expr = reduce(
        lambda acc, col: acc & (pl.col(col) <= quantiles[col]),
        columns[1:],
        pl.col(columns[0]) <= quantiles[columns[0]]
    )
    return df.filter(filter_expr)

def show(config_path):
    col0, col1, col2, col3, col4 = st.columns(5)
    with col0:
        # Liste complète des modèles avec leurs équations explicites
        model_options = {
            "Stationnaire : μ(t) = μ ; σ(t) = σ ; ξ(t) = ξ": "s_gev",
            "Modèle 1 : μ(t) = μ₀ + μ₁·t ; σ(t) = σ₀ ; ξ(t) = ξ": "ns_gev_m1",
            "Modèle 2 : μ(t) = μ₀ ; σ(t) = σ₀ + σ₁·t ; ξ(t) = ξ": "ns_gev_m2",
            "Modèle 3 : μ(t) = μ₀ + μ₁·t ; σ(t) = σ₀ + σ₁·t ; ξ(t) = ξ": "ns_gev_m3"
        }

        # Sélecteur de modèle unique
        model_label = st.selectbox("Modèle GEV", list(model_options.keys()), key="model_type")
        ns_model_choice = model_options[model_label]
        is_non_stationary = ns_model_choice != "s_gev"

    with col1:
        # Choix du paramètre à visualiser (xi, mu, sigma)
        param_choice = st.selectbox(
            "Paramètre GEV à afficher",
            ["μ", "σ", "ξ"],  # Symboles
            index=0,
            key="gev_param_choice"
        )

    with col2:
        Echelle = st.selectbox("Choix de l'échelle temporelle", ["Journalière", "Horaire"], key="scale_choice")
        echelle = "quotidien" if Echelle.lower() == "journalière" else "horaire"
        unit = "mm/j" if echelle == "quotidien" else "mm/h"


    with col3:
        quantile_choice = st.slider(
            "Percentile de retrait",
            min_value=0.950,
            max_value=1.00,
            value=0.999,
            step=0.001,
            format="%.3f",
            key="quantile_choice"
        )


    with col4:
        min_calcul_year = 10 if echelle == "horaire" else 30 # nombre minimale d'année nécessaire pour la GEV
        max_calcul_year = 32 if echelle == "horaire" else 2015 - 1960 + 1 # nombre maximal possible
        value_fixe = 20 if echelle == "horaire" else 55
        
        if is_non_stationary:
            len_serie = st.slider(
                "Nombre d'années minimales",
                min_value=min_calcul_year,
                max_value=max_calcul_year,
                value=value_fixe,
                step=1,
                key="len_serie",
                disabled=True  # Empêche la modification
            )
            len_serie = None
        else:
            len_serie = st.slider(
                "Nombre d'années minimales",
                min_value=min_calcul_year,
                max_value=max_calcul_year,
                value=max_calcul_year if echelle == "quotidien" else 20,
                step=1,
                key="len_serie"
            )
            

    config = load_config(config_path)
    model_key = "gev_non_sta" if is_non_stationary else "gev_sta"
    model_name = ns_model_choice  # modèle utilisé dans le nom de fichier
    mod_dir = Path(config[model_key]["modelised"]) / echelle
    obs_dir = Path(config[model_key]["observed"]) / echelle
    key = f"years_{len_serie}" if len_serie is not None else ""

    try:
        if model_name == "s_gev":
            df_modelised = pl.read_parquet(mod_dir / "gev_param.parquet")
        else:
            df_modelised = pl.read_parquet(mod_dir / f"gev_param_{model_name}.parquet")
    except Exception as e:
        st.error(f"Erreur lors du chargement des paramètres modélisés : {e}")
        return

    try:
        if model_name == "s_gev":
            df_observed = pl.read_parquet(obs_dir / f"gev_param_{key}.parquet")
        else:
            df_observed = pl.read_parquet(obs_dir / f"gev_param_{model_name}.parquet")
    except Exception as e:
        st.error(f"Erreur lors du chargement des paramètres observés : {e}")
        return

    params_gev = ns_param_map[model_name]
    columns_to_filter = list(params_gev.keys())

    # Dans la fonction show(), juste après le chargement des fichiers df_modelised et df_observed
    valid_ratio_model = compute_valid_ratio(df_modelised, columns_to_filter)
    valid_ratio_obs = compute_valid_ratio(df_observed, columns_to_filter)

    # Affichage dans Streamlit (par exemple juste sous les sélecteurs)
    st.write("Convergence du modèle | ", f"AROME : {valid_ratio_model*100:.1f}% - Station : {valid_ratio_obs*100:.1f}%")
    if "log_likelihood" in (df_observed.columns and df_modelised.columns):
        st.write("Vraisemblance moyenne | ", f"AROME : {loglike_score(df_modelised)} - Station : {loglike_score(df_observed)}")

    # Conversion du choix en clés du DataFrame selon le type de modèle
    if param_choice == "μ":
        columns_to_filter = [k for k, v in params_gev.items() if v.startswith("μ")]
        params_gev = {k: v for k, v in params_gev.items() if k in columns_to_filter}
    elif param_choice == "σ":
        columns_to_filter = [k for k, v in params_gev.items() if v.startswith("σ")]
        params_gev = {k: v for k, v in params_gev.items() if k in columns_to_filter}
    else:
        columns_to_filter = ["xi"]
        params_gev = {"xi": "ξ"}


    df_observed = filter_nan(df_observed, columns_to_filter)
    df_modelised = add_metadata(df_modelised, "mm_h" if echelle == "horaire" else "mm_j", type='modelised')    
    df_observed = add_metadata(df_observed, "mm_h" if echelle == "horaire" else "mm_j", type='observed') 

    df_modelised_show = filter_percentile_all(df_modelised, quantile_choice, columns_to_filter)

    colormap = echelle_config("continu", n_colors=15)
    view_state = pdk.ViewState(latitude=46.9, longitude=1.7, zoom=4.6)
    tooltip = create_tooltip("")

    for param, label in params_gev.items():
        vmin_modelised = df_modelised_show[param].min()
        df_modelised_leg, vmin, vmax = formalised_legend(df_modelised_show, param, colormap, vmin=vmin_modelised)
        df_observed_leg, _, _ = formalised_legend(df_observed, param, colormap, vmin, vmax)

        layer = create_layer(df_modelised_leg)
        scatter_layer = create_scatter_layer(df_observed_leg, radius=1500)
        deck = plot_map([layer, scatter_layer], view_state, tooltip)

        df_obs_vs_mod = pl.read_csv(f"data/metadonnees/obs_vs_mod/obs_vs_mod_{echelle}.csv")
        obs_vs_mod = match_and_compare(df_observed, df_modelised, param, df_obs_vs_mod)

        colmap, colplot = st.columns([0.5, 0.5])

        height=450

        with colmap:
            colmapping, collegend = st.columns([0.9, 0.1])
            with colmapping:
                st.markdown(f"<b>Paramètre {label}</b> | AROME : {df_modelised_leg.height} et stations : {df_observed_leg.height}", unsafe_allow_html=True)
                st.pydeck_chart(deck, use_container_width=True, height=height)


            with collegend:
                html_legend = display_vertical_color_legend(height, colormap, vmin, vmax, n_ticks=15, label=label)
                st.markdown(html_legend, unsafe_allow_html=True)

        with colplot:
            if obs_vs_mod is not None and obs_vs_mod.height > 0:
                fig = generate_scatter_plot_interactive(obs_vs_mod, f"{label}", unit if param!="xi" else "sans unité", height)
                me, mae, rmse, r2 = generate_metrics(obs_vs_mod)
                col0m, colm1, colm2, colm3, colm4 = st.columns(5)
                with col0m:
                    st.markdown(f"<b>Points comparés</b><br>{obs_vs_mod.shape[0]}", unsafe_allow_html=True)
                show_info_metric(colm1, "ME", me)
                show_info_metric(colm2, "MAE", mae)
                show_info_metric(colm3, "RMSE", rmse)
                show_info_metric(colm4, "R²", r2)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Aucune donnée disponible pour le scatter plot de {label}.")
