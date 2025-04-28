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
from app.utils.show_info import show_info_metric

import pydeck as pdk
import polars as pl
import numpy as np

from scipy.stats import genextreme

from app.pipelines.import_data import pipeline_data

def standardize_year(year: float, min_year: int, max_year: int) -> float:
    """
    Centre et réduit une année `year` en utilisant min_year et max_year.
    """
    mean = (min_year + max_year) / 2
    std = (max_year - min_year) / 2
    return (year - mean) / std


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


def compute_return_levels_ns(params: dict, model_name: str, T: np.ndarray, t_norm: float) -> np.ndarray:
    """
    Calcule les niveaux de retour selon le modèle NS-GEV fourni.
    - params : dictionnaire des paramètres GEV d'un point
    - model_name : nom du modèle (clé de MODEL_REGISTRY)
    - T : périodes de retour (en années)
    - t_norm : covariable temporelle normalisée (ex : 0 pour année moyenne)
    """    
    mu = params.get("mu0", 0) + params["mu1"] * t_norm if "mu1" in params else params.get("mu0", 0) # μ(t)
    sigma = params.get("sigma0", 0) + params["sigma1"] * t_norm if "sigma1" in params else params.get("sigma0", 0) # σ(t)
    xi = params.get("xi", 0) # xi contant

    return genextreme.ppf(1 - 1/T, c=-xi, loc=mu, scale=sigma)  


ns_param_map = {
    "s_gev": {"mu0": "μ₀", "sigma0": "σ₀", "xi": "ξ"},
    "ns_gev_m1": {"mu0": "μ₀", "mu1": "μ₁", "sigma0": "σ₀", "xi": "ξ"},
    "ns_gev_m2": {"mu0": "μ₀", "sigma0": "σ₀", "sigma1": "σ₁", "xi": "ξ"},
    "ns_gev_m3": {"mu0": "μ₀", "mu1": "μ₁", "sigma0": "σ₀", "sigma1": "σ₁", "xi": "ξ"},
}

# Liste complète des modèles avec leurs équations explicites
model_options = {
    "Stationnaire : μ(t) = μ₀ ; σ(t) = σ₀ ; ξ(t) = ξ": "s_gev",
    "Modèle 1 : μ(t) = μ₀ + μ₁·t ; σ(t) = σ₀ ; ξ(t) = ξ": "ns_gev_m1",
    "Modèle 2 : μ(t) = μ₀ ; σ(t) = σ₀ + σ₁·t ; ξ(t) = ξ": "ns_gev_m2",
    "Modèle 3 : μ(t) = μ₀ + μ₁·t ; σ(t) = σ₀ + σ₁·t ; ξ(t) = ξ": "ns_gev_m3"
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

# Calcul AIC
def mean_aic_score(df: pl.DataFrame, param_list: list[str], col_llh: str = "log_likelihood") -> str:
    k = len(param_list)
    df_valid = df.drop_nulls(subset=param_list + [col_llh])
    mean_llh = df_valid[col_llh].mean()
    aic = 2 * k - 2 * mean_llh
    return f"{aic:.1f}"

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
    config = load_config(config_path)

    col0, col1, col2, col3, col4 = st.columns(5)
    with col0:
        Echelle = st.selectbox("Choix de l'échelle temporelle", ["Journalière", "Horaire"], key="scale_choice")
        echelle = "quotidien" if Echelle.lower() == "journalière" else "horaire"
        unit = "mm/j" if echelle == "quotidien" else "mm/h"

    with col1:
        model_label = st.selectbox(
            "Modèle GEV",
            [None] + list(model_options.keys()),
            format_func=lambda x: "— Choisir un modèle —" if x is None else x,
            key="model_type"
        )

    mod_dir = Path(config["gev"]["modelised"]) / echelle
    obs_dir = Path(config["gev"]["observed"]) / echelle

    if model_label is not None:
        model_name = model_options[model_label]
    else:
        model_name = None

    if model_label is not None:
        with col2:
            # Liste dynamique des paramètres disponibles selon le modèle
            available_params = list(ns_param_map[model_name].values())  # μ₀, μ₁, σ₀, σ₁, ξ...

            param_choice = st.selectbox(
                "Paramètre GEV à afficher",
                available_params,
                index=0,
                key="gev_param_choice"
            )

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
            min_calcul_year = 10 # nombre minimale d'année nécessaire pour la GEV
            max_calcul_year = 56 # nombre maximal possible
            value_fixe = 48 if echelle == "quotidien" else 20
            
            st.slider(
                "Nombre d'années minimales",
                min_value=min_calcul_year,
                max_value=max_calcul_year,
                value=value_fixe,
                step=1,
                key="len_serie",
                disabled=True  # Empêche la modification
            )
        

        # Chargement des données des maximas horaires
        stat_choice_key = "max"
        scale_choice_key = "mm_j" if echelle=="quotidien" else "mm_h"
        min_year_choice = 1960
        max_year_choice = 2015
        season_choice_key = "hydro"
        missing_rate = value_fixe/(max_calcul_year - min_calcul_year + 1)
        params_load = (
            stat_choice_key,
            scale_choice_key,
            min_year_choice,
            max_year_choice,
            season_choice_key,
            missing_rate,
            quantile_choice
        )
        result = pipeline_data(params_load, config)
        df_modelised_load = result["modelised_load"]
        df_observed_load = result["observed_load"]
        
        column_to_show = "max_mm_h" if echelle=="horaire" else "max_mm_j"
        
        # Chargement des paramètres GEV
        try:
            df_modelised = pl.read_parquet(mod_dir / f"gev_param_{model_name}.parquet")
        except Exception as e:
            st.error(f"Erreur lors du chargement des paramètres modélisés : {e}")
            return

        try:
            df_observed = pl.read_parquet(obs_dir / f"gev_param_{model_name}.parquet")
        except Exception as e:
            st.error(f"Erreur lors du chargement des paramètres observés : {e}")
            return

        params_gev = ns_param_map[model_name]
        columns_to_filter = list(params_gev.keys())

        # Cherche la clé correspondant au paramètre choisi
        columns_to_filter = [k for k, v in params_gev.items() if v == param_choice]
        params_gev = {k: v for k, v in params_gev.items() if v == param_choice}


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

            colmap, colplot = st.columns([0.45, 0.55])

            height=450

            with colmap:
                colmapping, collegend = st.columns([0.85, 0.15])
                with colmapping:
                    st.markdown(f"<b>Paramètre {label}</b> | AROME : {df_modelised_leg.height} et stations : {df_observed_leg.height}", unsafe_allow_html=True)
                    st.pydeck_chart(deck, use_container_width=True, height=height)


                with collegend:
                    html_legend = display_vertical_color_legend(height, colormap, vmin, vmax, n_ticks=15, label=label)
                    st.markdown(html_legend, unsafe_allow_html=True)




                if param == "xi":
                    # Ajout d'une nouvelle colonne xi_xi pour le signe (-1, 0, 1)
                    df_modelised_leg_xi = df_modelised.with_columns(
                        pl.when(pl.col(param) < 0).then(pl.lit(-1))
                        .when(pl.col(param) == 0).then(pl.lit(0))
                        .otherwise(pl.lit(1))
                        .alias(f"{param}_xi")
                    )
                    df_observed_leg_xi = df_observed.with_columns(
                        pl.when(pl.col(param) < 0).then(pl.lit(-1))
                        .when(pl.col(param) == 0).then(pl.lit(0))
                        .otherwise(pl.lit(1))
                        .alias(f"{param}_xi")
                    )

                    # Normalise sur 3 classes (-1, 0, +1)
                    vmin_modelised = -1
                    vmax_modelised = 1

                    # Utilise un colormap discret simple
                    discrete_cmap = plt.cm.get_cmap('bwr', 3)  # bleu-blanc-rouge 3 couleurs

                    df_modelised_leg_xi, vmin, vmax = formalised_legend(
                        df_modelised_leg_xi, f"{param}_xi", discrete_cmap, vmin=vmin_modelised, vmax=vmax_modelised
                    )
                    df_observed_leg_xi, _, _ = formalised_legend(
                        df_observed_leg_xi, f"{param}_xi", discrete_cmap, vmin=vmin_modelised, vmax=vmax_modelised
                    )

                    layer_xi = create_layer(df_modelised_leg_xi)
                    scatter_layer_xi = create_scatter_layer(df_observed_leg_xi, radius=1500)
                    deck_xi = plot_map([layer_xi, scatter_layer_xi], view_state, tooltip)

                    with colmapping:
                        st.pydeck_chart(deck_xi, use_container_width=True, height=height)

                    # ➡ Ici affiche la légende discrète rouge-blanc-bleu
                    with collegend:
                        st.write("<br/><br/>", unsafe_allow_html=True)
                        html_legend = """
                        <div style="text-align: left; font-size: 13px; margin-bottom: 4px;">ξ</div>
                        <div style="display: flex; flex-direction: column;">
                            <div style="display: flex; align-items: center; margin-bottom: 6px;">
                                <div style="width: 18px; height: 18px; background-color: blue; margin-right: 6px; border: 1px solid #ccc;"></div>
                                <div style="font-size: 12px;">ξ &lt; 0</div>
                            </div>
                            <div style="display: flex; align-items: center; margin-bottom: 6px;">
                                <div style="width: 18px; height: 18px; background-color: white; margin-right: 6px; border: 1px solid #ccc;"></div>
                                <div style="font-size: 12px;">ξ = 0</div>
                            </div>
                            <div style="display: flex; align-items: center;">
                                <div style="width: 18px; height: 18px; background-color: red; margin-right: 6px; border: 1px solid #ccc;"></div>
                                <div style="font-size: 12px;">ξ &gt; 0</div>
                            </div>
                        </div>
                        """
                        st.markdown(html_legend, unsafe_allow_html=True)




            with colplot:
                if obs_vs_mod is not None and obs_vs_mod.height > 0:
                    fig = generate_scatter_plot_interactive(obs_vs_mod, f"{label}", unit if param!="xi" else "sans unité", height)
                    me, mae, rmse, r2 = generate_metrics(obs_vs_mod)
                    col000, col0m, colm1, colm2, colm3, colm4 = st.columns(6)
                    with col0m:
                        st.markdown(f"<b>Points comparés</b><br>{obs_vs_mod.shape[0]}", unsafe_allow_html=True)
                    show_info_metric(colm1, "ME", me)
                    show_info_metric(colm2, "MAE", mae)
                    show_info_metric(colm3, "RMSE", rmse)
                    show_info_metric(colm4, "R²", r2)
                    
                    # Affiche le graphique avec mode sélection activé
                    event = st.plotly_chart(fig, key=f"scatter_{param}", on_select="rerun")

                    if param == "xi":
                        import pandas as pd

                        # Fonction pour classer en -1, 0 ou 1
                        def classify_sign(x):
                            if x > 0:
                                return 1
                            elif x < 0:
                                return -1
                            else:
                                return 0

                        # Transforme obs_vs_mod avec une colonne struct
                        df_sign = obs_vs_mod.with_columns(
                            pl.struct(["pr_obs", "pr_mod"]).alias("struct")
                        ).select(
                            pl.col("struct").map_elements(lambda row: {
                                "obs_sign": classify_sign(row["pr_obs"]),
                                "mod_sign": classify_sign(row["pr_mod"])
                            })
                        ).unnest(["struct"])

                        # Passe en pandas
                        df_sign_pd = df_sign.to_pandas()

                        # Crosstab sans reindex
                        table_contingence = pd.crosstab(
                            df_sign_pd["obs_sign"], 
                            df_sign_pd["mod_sign"],
                            rownames=[""],
                            colnames=[""]
                        )

                        # ➔ Nettoyage automatique
                        table_contingence.columns.name = None
                        table_contingence.index = table_contingence.index.map({-1: "ξ_obs < 0", 0: "ξ_obs = 0", 1: "ξ_obs > 0"}).dropna()
                        table_contingence.columns = [f"ξ_mod {c}" for c in table_contingence.columns.map({-1: "<0", 0: "=0", 1: ">0"})]

                        # ➔ Convertir en pourcentage
                        total_points = table_contingence.values.sum()  # Sur les vraies cases non nulles
                        table_contingence_pct = (table_contingence / total_points) * 100
                        table_contingence_pct = table_contingence_pct.round(1)

                        st.write("### Tableau de contingence ξ observé vs ξ modélisé (en %)")
                        st.dataframe(table_contingence_pct, use_container_width=True)






            if event and event.selection and "points" in event.selection:
                points = event.selection["points"]

                if points and "customdata" in points[0]:

                    selected = points[0]
                    num_poste_obs = selected["customdata"][0]
                    num_poste_mod = selected["customdata"][1]

                    params_obs = df_observed.filter(pl.col("NUM_POSTE") == num_poste_obs).to_dicts()[0]
                    params_mod = df_modelised.filter(pl.col("NUM_POSTE") == num_poste_mod).to_dicts()[0]

                    col_select_point, col_select_year, col_nr_year = st.columns(3)

                    with col_select_point:
                        st.write(f"Point selectionné : ({params_obs['lat']:.3f}, {params_obs['lon']:.3f})")

                    with col_select_year:
                        year_choix = st.slider(
                            "Choix de t",
                            min_value=min_year_choice,
                            max_value=max_year_choice,
                            value=int((min_year_choice + max_year_choice) / 2),
                            step=1,
                            key="year_choix"
                        )

                    with col_nr_year:
                        nr_year = st.slider(
                            "Choix du NR",
                            min_value=10,
                            max_value=100,
                            value=20,
                            step=10,
                            key="nr_year"
                        )

                    year_choice_norm = standardize_year(year_choix, min_year_choice, max_year_choice)

                    T = np.logspace(np.log10(1.01), np.log10(100), 100)
                    y_obs = compute_return_levels_ns(params_obs, model_name, T, t_norm=year_choice_norm)
                    y_mod = compute_return_levels_ns(params_mod, model_name, T, t_norm=year_choice_norm)

                    # Extraction des maximas annuels bruts
                    df_observed_load = df_observed_load.with_columns(pl.col("NUM_POSTE").cast(pl.Int32))
                    df_observed_load_point = df_observed_load.filter(pl.col("NUM_POSTE") == num_poste_obs)

                    if df_observed_load_point.height > 0:
                        maximas_sorted = np.sort(df_observed_load_point[column_to_show].drop_nulls().to_numpy())[::-1]
                        n = len(maximas_sorted)
                        T_empirical = (n + 1) / np.arange(1, n + 1)
                        points_obs = {
                            "year": T_empirical,
                            "value": maximas_sorted
                        }
                    else:
                        points_obs = None


                    # Extraction des maximas annuels bruts
                    df_modelised_load = df_modelised_load.with_columns(pl.col("NUM_POSTE").cast(pl.Int32))
                    df_modelised_load_point = df_modelised_load.filter(pl.col("NUM_POSTE") == num_poste_mod)

                    if df_modelised_load_point.height > 0:
                        maximas_sorted = np.sort(df_modelised_load_point[column_to_show].drop_nulls().to_numpy())[::-1]
                        n = len(maximas_sorted)
                        T_empirical = (n + 1) / np.arange(1, n + 1)
                        points_mod = {
                            "year": T_empirical,
                            "value": maximas_sorted
                        }
                    else:
                        points_mod = None

                    col_density, col_retour, col_times_series = st.columns(3)

                    with col_density:
                        fig_density_comparison = generate_gev_density_comparison_interactive(
                            points_obs["value"],
                            points_mod["value"],
                            params_obs,
                            params_mod,
                            unit,
                            height=height,
                            t_norm=year_choice_norm
                        )
                        st.plotly_chart(fig_density_comparison, use_container_width=True)

                    with col_retour:
                        fig = generate_return_period_plot_interactive(
                            T, y_obs, y_mod,
                            unit=unit,
                            height=height,
                            points_obs=points_obs,
                            points_mod=points_mod
                        )
                        st.plotly_chart(fig)
                        
                    with col_times_series:
                        years_range = np.arange(min_year_choice, max_year_choice + 1)  # Toutes les années observées
                        years_norm = np.array([standardize_year(y, min_year_choice, max_year_choice) for y in years_range]) # Normalisation

                        # Puis, pour chaque année normalisée, tu appelles compute_return_levels_ns
                        T = np.array([nr_year])  # Niveau de retour 20 ans
                        
                        return_levels_obs = np.array([
                            compute_return_levels_ns(params_obs, model_name, T, t_norm)[0]
                            for t_norm in years_norm
                        ])
                        
                        return_levels_mod = np.array([
                            compute_return_levels_ns(params_mod, model_name, T, t_norm)[0]
                            for t_norm in years_norm
                        ])

                        fig_time_series = generate_time_series_maxima_interactive(
                            years_obs=df_observed_load_point["year"],
                            max_obs=df_observed_load_point[column_to_show].to_numpy(),
                            years_mod=df_modelised_load_point["year"],
                            max_mod=df_modelised_load_point[column_to_show].to_numpy(),
                            unit=unit,
                            height=height,
                            nr_year=nr_year,
                            return_levels_obs=return_levels_obs,
                            return_levels_mod=return_levels_mod
                        )
                        st.plotly_chart(fig_time_series, use_container_width=True)




                            
            else:
                pass


    else:
        col_score, col_rsquared = st.columns(2)

        # --- Colonne gauche : Score AIC moyen ---
        with col_score:
            st.markdown("### Score AIC moyen par modèle")
            for label, model in model_options.items():
                try:
                    df_model = pl.read_parquet(mod_dir / f"gev_param_{model}.parquet")
                    df_obs = pl.read_parquet(obs_dir / f"gev_param_{model}.parquet")
                    param_list = list(ns_param_map[model].keys())

                    aic_model = mean_aic_score(df_model, param_list)
                    aic_obs = mean_aic_score(df_obs, param_list)

                    st.markdown(
                        f"""
                        <div style="margin-bottom: 1rem;">
                            <strong>{label}</strong><br>
                            - AROME : <code>{aic_model}</code><br>
                            - Station : <code>{aic_obs}</code>
                        </div>
                        """, unsafe_allow_html=True
                    )
                except Exception as e:
                    st.warning(f"❌ Modèle {model} : erreur de lecture ({e})")

        # --- Colonne droite : R² par paramètre ---
        with col_rsquared:
            st.markdown("### R² entre stations et modélisation")

            try:
                df_obs_vs_mod_full = pl.read_csv(f"data/metadonnees/obs_vs_mod/obs_vs_mod_{echelle}.csv")
            except Exception as e:
                st.warning(f"Erreur de lecture de `obs_vs_mod_{echelle}.csv`: {e}")
                df_obs_vs_mod_full = None

            if df_obs_vs_mod_full is not None:
                for label, model in model_options.items():
                    try:
                        st.markdown(f"#### {label}")
                        df_model = pl.read_parquet(mod_dir / f"gev_param_{model}.parquet")
                        df_obs = pl.read_parquet(obs_dir / f"gev_param_{model}.parquet")

                        df_model = add_metadata(df_model, "mm_h" if echelle == "horaire" else "mm_j", type="modelised")
                        df_obs = add_metadata(df_obs, "mm_h" if echelle == "horaire" else "mm_j", type="observed")

                        param_map = ns_param_map[model]
                        df_model = df_model.with_columns(pl.col("NUM_POSTE").cast(pl.Int32))
                        df_obs = df_obs.with_columns(pl.col("NUM_POSTE").cast(pl.Int32))

                        for param, symbol in param_map.items():
                            if param not in df_model.columns or param not in df_obs.columns:
                                continue

                            obs_vs_mod = match_and_compare(df_obs, df_model, param, df_obs_vs_mod_full)

                            if obs_vs_mod is not None and obs_vs_mod.height > 0:
                                _, _, _, r2 = generate_metrics(obs_vs_mod)
                                st.markdown(f"• **{symbol}** : R² = `{r2:.3f}`")
                            else:
                                st.markdown(f"• **{symbol}** : données insuffisantes")

                    except Exception as e:
                        st.warning(f"❌ Erreur modèle {model} : {e}")
