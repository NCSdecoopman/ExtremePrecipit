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
from app.utils.gev_utils import compute_return_levels_ns

import pydeck as pdk
import polars as pl
import numpy as np

from app.pipelines.import_data import pipeline_data
from app.pipelines.import_map import pipeline_map

from app.utils.data_utils import standardize_year, filter_nan

ns_param_map = {
    "s_gev": {"mu0": "μ₀", "sigma0": "σ₀", "xi": "ξ"},
    "ns_gev_m1": {"mu0": "μ₀", "mu1": "μ₁", "sigma0": "σ₀", "xi": "ξ"},
    "ns_gev_m2": {"mu0": "μ₀", "sigma0": "σ₀", "sigma1": "σ₁", "xi": "ξ"},
    "ns_gev_m3": {"mu0": "μ₀", "mu1": "μ₁", "sigma0": "σ₀", "sigma1": "σ₁", "xi": "ξ"},
    "ns_gev_m1_break_year": {"mu0": "μ₀", "mu1": "μ₁", "sigma0": "σ₀", "xi": "ξ"},
    "ns_gev_m2_break_year": {"mu0": "μ₀", "sigma0": "σ₀", "sigma1": "σ₁", "xi": "ξ"},
    "ns_gev_m3_break_year": {"mu0": "μ₀", "mu1": "μ₁", "sigma0": "σ₀", "sigma1": "σ₁", "xi": "ξ"},
}

# Liste complète des modèles avec leurs équations explicites
model_options = {
    # Stationnaire
    "M₀(μ₀, σ₀) : μ(t) = μ₀ ; σ(t) = σ₀ ; ξ(t) = ξ": "s_gev",

    # Non stationnaires simples
    "M₁(μ, σ₀) : μ(t) = μ₀ + μ₁·t ; σ(t) = σ₀ ; ξ(t) = ξ": "ns_gev_m1",
    "M₂(μ₀, σ) : μ(t) = μ₀ ; σ(t) = σ₀ + σ₁·t ; ξ(t) = ξ": "ns_gev_m2",
    "M₃(μ, σ) : μ(t) = μ₀ + μ₁·t ; σ(t) = σ₀ + σ₁·t ; ξ(t) = ξ": "ns_gev_m3",

    # Non stationnaires avec rupture
    "M₁⋆(μ, σ₀) : μ(t) = μ₀ + μ₁·t₊ ; σ(t) = σ₀ ; ξ(t) = ξ en notant t₊ = t · 𝟙_{t > t₀} avec t₀ = 1985": "ns_gev_m1_break_year",
    "M₂⋆(μ₀, σ) : μ(t) = μ₀ ; σ(t) = σ₀ + σ₁·t₊ ; ξ(t) = ξ en notant t₊ = t · 𝟙_{t > t₀} avec t₀ = 1985": "ns_gev_m2_break_year",
    "M₃⋆(μ, σ) : μ(t) = μ₀ + μ₁·t₊ ; σ(t) = σ₀ + σ₁·t₊ ; ξ(t) = ξ en notant t₊ = t · 𝟙_{t > t₀} avec t₀ = 1985": "ns_gev_m3_break_year",
}


# Calcul du ratio de stations valides
def compute_valid_ratio(df: pl.DataFrame, param_list: list[str]) -> float:
    n_total = df.height
    n_valid = df.drop_nulls(subset=param_list).height
    return round(n_valid / n_total, 3) if n_total > 0 else 0.0



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

    col0, col1, col2, col3 = st.columns([0.5, 1, 0.5, 0.5])
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
        

        # Chargement des données des maximas horaires
        stat_choice_key = "max"
        scale_choice_key = "mm_j" if echelle=="quotidien" else "mm_h"
        season_choice_key = "hydro"

        if season_choice_key == "hydro":
            min_year_choice = config["years"]["min"]+1
        else:
            min_year_choice = config["years"]["min"]

        max_year_choice = config["years"]["max"]

        missing_rate = 0.15
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
                    show_info_metric(colm4, "r²", r2)
                    
                    # Affiche le graphique avec mode sélection activé
                    event = st.plotly_chart(fig, key=f"scatter_{param}", on_select="rerun")




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

                    y_obs = compute_return_levels_ns(params_obs, T, t_norm=year_choice_norm)
                    y_mod = compute_return_levels_ns(params_mod, T, t_norm=year_choice_norm)

                    # Extraction des maximas annuels bruts
                    df_observed_load = df_observed_load.with_columns(pl.col("NUM_POSTE").cast(pl.Int32))
                    df_observed_load_point = df_observed_load.filter(pl.col("NUM_POSTE") == num_poste_obs)

                    if df_observed_load_point.height > 0:
                        maximas_sorted_obs = np.sort(df_observed_load_point[column_to_show].drop_nulls().to_numpy())[::-1]
                        n = len(maximas_sorted_obs)
                        T_empirical_obs = (n + 1) / np.arange(1, n + 1)
                        points_obs = {
                            "year": T_empirical_obs,
                            "value": maximas_sorted_obs
                        }
                    else:
                        points_obs = None


                    # Extraction des maximas annuels bruts
                    df_modelised_load = df_modelised_load.with_columns(pl.col("NUM_POSTE").cast(pl.Int32))
                    df_modelised_load_point = df_modelised_load.filter(pl.col("NUM_POSTE") == num_poste_mod)

                    if df_modelised_load_point.height > 0:
                        maximas_sorted_mod = np.sort(df_modelised_load_point[column_to_show].drop_nulls().to_numpy())[::-1]
                        n = len(maximas_sorted_mod)
                        T_empirical_mod = (n + 1) / np.arange(1, n + 1)
                        points_mod = {
                            "year": T_empirical_mod,
                            "value": maximas_sorted_mod
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

                        # fig_density_comparison = generate_gev_density_comparison_interactive_3D(
                        #     maxima_obs=points_obs["value"],
                        #     maxima_mod=points_mod["value"],
                        #     params_obs=params_obs,
                        #     params_mod=params_mod,
                        #     unit=unit,
                        #     height=height,
                        #     min_year=min_year_choice,
                        #     max_year=max_year_choice
                        # )

                        # st.plotly_chart(fig_density_comparison, use_container_width=True)


                    with col_retour:
                        fig = generate_return_period_plot_interactive(
                            T=T,
                            y_obs=y_obs,
                            y_mod=y_mod,
                            unit=unit,
                            height=height,
                            points_obs=points_obs,
                            points_mod=points_mod
                        )
                        st.plotly_chart(fig)




                        years_range = np.arange(min_year_choice, max_year_choice + 1)
                        years_norm = np.array([
                            standardize_year(y, min_year_choice, max_year_choice) for y in years_range
                        ])
                        T = np.array([nr_year])  # Période de retour choisie

                        # --- Observations ---
                        if df_observed_load_point.height > 0:
                            max_obs = df_observed_load_point[column_to_show].to_numpy()
                            years_obs = df_observed_load_point["year"].to_numpy()

                            return_levels_obs = np.array([
                                compute_return_levels_ns(params_obs, T, t_norm)
                                for t_norm in years_norm
                            ]).flatten()

                            fig_obs = go.Figure()
                            fig_obs.add_trace(go.Scatter(
                                x=years_obs, y=max_obs,
                                mode="lines",
                                marker=dict(color="blue"),
                                name="Maximas annuels (obs)"
                            ))
                            fig_obs.add_trace(go.Scatter(
                                x=years_range, y=return_levels_obs,
                                mode="lines",
                                line=dict(color="black", dash="dash"),
                                name=f"NR {nr_year} ans (obs)"
                            ))
                            

                        # --- Modélisation ---
                        if df_modelised_load_point.height > 0:
                            max_mod = df_modelised_load_point[column_to_show].to_numpy()
                            years_mod = df_modelised_load_point["year"].to_numpy()

                            return_levels_mod = np.array([
                                compute_return_levels_ns(params_mod, T, t_norm)
                                for t_norm in years_norm
                            ]).flatten()

                            fig_mod = go.Figure()
                            fig_mod.add_trace(go.Scatter(
                                x=years_mod, y=max_mod,
                                mode="lines",
                                marker=dict(color="orange"),
                                name="Maximas annuels (mod)"
                            ))
                            fig_mod.add_trace(go.Scatter(
                                x=years_range, y=return_levels_mod,
                                mode="lines",
                                line=dict(color="black", dash="dash"),
                                name=f"NR {nr_year} ans (mod)"
                            ))



                        ymin, ymax = 0, max(np.nanmax(max_obs), np.nanmax(max_mod))
                    with col_density:
                        fig_obs.update_layout(
                            title="Série temporelle - Observations",
                            xaxis_title="Année",
                            yaxis_title=unit,
                            yaxis=dict(range=[ymin, ymax]),
                            height=height
                        )
                        st.plotly_chart(fig_obs, use_container_width=True)

                    with col_retour:
                        fig_mod.update_layout(
                            title="Série temporelle - Modélisation",
                            xaxis_title="Année",
                            yaxis_title=unit,
                            yaxis=dict(range=[ymin, ymax]),
                            height=height
                        )
                        st.plotly_chart(fig_mod, use_container_width=True)

                        
                    #with col_times_series:
                        # if param == "xi":
                        #     st.write("AROME")
                        #     fig_loglik_profile = generate_loglikelihood_profile_xi(
                        #         maxima=points_mod["value"],  # Les maximas bruts modélisés
                        #         params=params_mod,            # Les paramètres du modèle sélectionné
                        #         unit=unit,
                        #         t_norm=year_choice_norm,
                        #         height=height
                        #     )
                        #     st.plotly_chart(fig_loglik_profile, use_container_width=True)



                        # years_range = np.arange(min_year_choice, max_year_choice + 1)  # Toutes les années observées
                        # years_norm = np.array([standardize_year(y, min_year_choice, max_year_choice) for y in years_range]) # Normalisation

                        # # Puis, pour chaque année normalisée, tu appelles compute_return_levels_ns
                        # T = np.array([nr_year])  # Niveau de retour 20 ans
                        
                        # return_levels_obs = np.array([
                        #     compute_return_levels_ns(params_obs, T, t_norm)
                        #     for t_norm in years_norm
                        # ])
                        
                        # return_levels_mod = np.array([
                        #     compute_return_levels_ns(params_mod, T, t_norm)
                        #     for t_norm in years_norm
                        # ])

                        # fig_time_series = generate_time_series_maxima_interactive(
                        #     years_obs=df_observed_load_point["year"],
                        #     max_obs=df_observed_load_point[column_to_show].to_numpy(),
                        #     years_mod=df_modelised_load_point["year"],
                        #     max_mod=df_modelised_load_point[column_to_show].to_numpy(),
                        #     unit=unit,
                        #     height=height,
                        #     nr_year=nr_year,
                        #     return_levels_obs=return_levels_obs,
                        #     return_levels_mod=return_levels_mod
                        # )
                        # st.plotly_chart(fig_time_series, use_container_width=True)

                            
            else:
                pass


    else:
        col_rsquared, col_map_model = st.columns([0.4, 0.6])

        with col_rsquared:
            st.markdown("### r² entre stations et modélisation")

            try:
                df_obs_vs_mod_full = pl.read_csv(f"data/metadonnees/obs_vs_mod/obs_vs_mod_{echelle}.csv")
            except Exception as e:
                st.warning(f"Erreur de lecture de `obs_vs_mod_{echelle}.csv`: {e}")
                df_obs_vs_mod_full = None

            if df_obs_vs_mod_full is not None:

                model_aic_tables = []
                obs_aic_tables = []

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
                                st.markdown(f"• **{symbol}** : r² = `{r2:.3f}`")
                            else:
                                st.markdown(f"• **{symbol}** : données insuffisantes")

                    except Exception as e:
                        st.warning(f"❌ Erreur modèle {model} : {e}")


        with col_map_model:
            df_model_aic = pl.read_parquet(mod_dir / "gev_param_best_model.parquet")
            df_obs_aic = pl.read_parquet(obs_dir / "gev_param_best_model.parquet")

            df_model_aic = add_metadata(df_model_aic, "mm_h" if echelle == "horaire" else "mm_j", type="modelised")
            df_obs_aic = add_metadata(df_obs_aic, "mm_h" if echelle == "horaire" else "mm_j", type="observed")

            df_obs_aic = filter_nan(df_obs_aic, "xi") # xi est toujours valable


            # liste ordonnée des modèles
            model_names = list(model_options.values())

            # Chargement des affichages graphiques
            height=600
            n_legend = len(model_names)
            
            # Définir l'échelle personnalisée continue ou discrète selon le cas
            colormap = echelle_config("discret", n_colors=n_legend)

            # Normalisation de la légende pour les valeurs modélisées
            df_model_aic, vmin, vmax = formalised_legend(
                df_model_aic,
                column_to_show="model", 
                colormap=colormap,
                is_categorical=True,
                categories=model_names
            )

            # Création du layer modélisé
            layer = create_layer(df_model_aic)

            # Normalisation des points observés avec les mêmes bornes
            df_obs_aic, _, _ = formalised_legend(
                df_obs_aic,
                column_to_show="model", 
                colormap=colormap,
                is_categorical=True,
                categories=model_names
            )
            plot_station = st.checkbox('Afficher les stations', value=False)

            if plot_station:
                scatter_layer = create_scatter_layer(df_obs_aic, radius=1500)
            else:
                scatter_layer = None

            # Tooltip (tu dois t'assurer que unit_label est défini quelque part ou passé en paramètre)
            tooltip = create_tooltip("")

            # View par défaut
            view_state = pdk.ViewState(latitude=46.8, longitude=1.7, zoom=5)

            # Légende vertical
            html_legend = display_vertical_color_legend(
                height,
                colormap,
                vmin,
                vmax,
                n_ticks=n_legend,
                label="Modèle AIC minimal",
                model_labels=model_names
            )
            col1, col2 = st.columns([1, 0.15])

            with col1:
                deck = plot_map([layer, scatter_layer], view_state, tooltip)
                if deck:
                    st.pydeck_chart(deck, use_container_width=True, height=height)

                    param_list = ["mu0", "mu1", "sigma0", "sigma1", "xi"]
                    df_model_aic = df_model_aic.with_columns(pl.col("NUM_POSTE").cast(pl.Int32))
                    df_obs_aic = df_obs_aic.with_columns(pl.col("NUM_POSTE").cast(pl.Int32))

                    for param in param_list:
                        obs_vs_mod = match_and_compare(df_obs_aic, df_model_aic, param, df_obs_vs_mod_full)

                        if obs_vs_mod is not None and obs_vs_mod.height > 0:
                            _, _, _, r2 = generate_metrics(obs_vs_mod)
                            st.markdown(f"• **{param}** : r² = `{r2:.3f}`")
                        else:
                            st.markdown(f"• **{param}** : données insuffisantes")

            with col2:
                st.markdown(html_legend, unsafe_allow_html=True)  
