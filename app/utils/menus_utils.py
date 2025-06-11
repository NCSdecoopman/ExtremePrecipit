import streamlit as st
from pathlib import Path

from app.utils.config_utils import reverse_param_label

def menu_statisticals(min_years: int, max_years: int, STATS, SEASON):
    if "selected_point" not in st.session_state:
        st.session_state["selected_point"] = None

    if "run_analysis" not in st.session_state:
        st.session_state["run_analysis"] = False

    # Crée les colonnes
    col0, col1, col2, col3, col4, col5, col6 = st.columns([0.5, 0.4, 0.4, 0.5, 0.4, 0.4, 0.2])

    with col0:
        st.selectbox("Choix de la statistique étudiée", list(STATS.keys()), key="stat_choice")

    with col1:
        st.slider(
            "Percentile de retrait AROME",
            min_value=0.950,
            max_value=1.00,
            value=0.995,
            step=0.001,
            format="%.3f",
            key="quantile_choice"
        )

    with col2:
        st.selectbox("Choix de la saison", list(SEASON.keys()), key="season_choice")

    with col3:
        season = st.session_state["season_choice"]
        if season in ["Année hydrologique", "Hiver"]:
            st.slider(
                f"Sélection temporelle entre {min_years+1} et {max_years}",
                min_value=min_years+1,
                max_value=max_years,
                value=(min_years+1, max_years),
                key="year_range"
            )
        else:
            st.slider(
                f"Sélection temporelle entre {min_years} et {max_years}",
                min_value=min_years,
                max_value=max_years,
                value=(min_years, max_years),
                key="year_range"
            )

    with col4:
        if st.session_state["stat_choice"] in ["Cumul", "Jour de pluie"]:
            st.selectbox("Choix de l'échelle temporelle", ["Journalière"], key="scale_choice")
        else:
            st.selectbox("Choix de l'échelle temporelle", ["Journalière", "Horaire"], key="scale_choice")

    with col5:
        st.slider(
            "Taux maximal de NaN des stations",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.01,
            key="missing_rate"
        )

    with col6:
        if st.button("Lancer l’analyse"):
            st.session_state["run_analysis"] = True

    if st.session_state["run_analysis"]:
        return (
            st.session_state["stat_choice"],
            st.session_state["quantile_choice"],
            st.session_state["year_range"][0],
            st.session_state["year_range"][1],
            st.session_state["season_choice"],
            st.session_state["scale_choice"],
            st.session_state["missing_rate"]
        )
    else:
        return None



def menu_gev(config: dict, model_options: dict, ns_param_map: dict, SEASON, show_param: bool):
    if "run_analysis" not in st.session_state:
        st.session_state["run_analysis"] = False

    col0, col1, col2, col3, col4, col5, col6, col7 = st.columns([0.6, 0.9, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    # Échelle
    with col0:
        Echelle = st.selectbox("Choix de l'échelle temporelle", ["Journalière", "Horaire"], key="scale_choice")
        st.session_state["echelle"] = "quotidien" if Echelle.lower() == "journalière" else "horaire"
        st.session_state["unit"] = "mm/j" if st.session_state["echelle"] == "quotidien" else "mm/h"

    # Modèle
    with col1:
        selected_model = st.selectbox(
            "Modèle GEV",
            [None] + list(model_options.keys()),
            format_func=lambda x: "— Choisir un modèle —" if x is None else x,
            key="model_type"
        )

    if st.session_state["model_type"] is not None:
        model_name = model_options[st.session_state["model_type"]]
        st.session_state["model_name"] = model_name

        # Quantile
        with col2:
            st.selectbox("Choix de la saison", list(SEASON.keys()), key="season_choice")

        with col3:
            st.slider(
                "Percentile de retrait",
                min_value=0.950,
                max_value=1.000,
                value=1.000,
                step=0.001,
                format="%.3f",
                key="quantile_choice"
            )

        # Paramètre GEV
        with col4:
            if show_param:
                param_map = ns_param_map[model_name]
                available_params = list(param_map.values())  # labels unicode
                selected_label = st.selectbox(
                    "Paramètre GEV à afficher",
                    available_params,
                    index=0,
                    key="gev_param_choice"
                )
                # Conversion propre
                st.session_state["param_choice"] = reverse_param_label(
                    selected_label, model_name, ns_param_map
                )
            else:
                # st.session_state["param_choice"] = "Δqᵀ"
                # selected_label = "Δqᵀ"
                selected_label = st.selectbox(
                    "Quantité à afficher",
                    ["Δqᵀ", "ΔE", "ΔVar", "ΔCV"],
                    index=0,
                    key="delta_param_choice"
                )
                st.session_state["param_choice"] = selected_label

        if selected_label in ["Δqᵀ"]:
            with col5:
                st.slider(
                    "Niveau de retour",
                    min_value=10,
                    max_value=100,
                    value=10,
                    step=10,
                    key="T_choice"
                )
        else:
            st.session_state["T_choice"] = None

        if selected_label in ["Δqᵀ", "ΔE", "ΔVar", "ΔCV"]:
            with col6:
                st.slider(
                    "Delta annees",
                    min_value=1,
                    max_value=60,
                    value=10,
                    step=1,
                    key="par_X_annees"
                )
        else:
            st.session_state["par_X_annees"] = None
        

        # Bouton d’analyse
        with col7:
            if st.button("Lancer l’analyse"):
                st.session_state["run_analysis"] = True

        if st.session_state["run_analysis"]:
            # Valeurs par défaut
            stat_choice_key = "max"
            scale_choice_key = "mm_j" if st.session_state["echelle"] == "quotidien" else "mm_h"
            season_choice_key = SEASON[st.session_state["season_choice"]]
            min_year_choice = config["years"]["min"] + 1 if season_choice_key in ["hydro", "djf"] else config["years"]["min"]
            max_year_choice = config["years"]["max"]
            missing_rate = 0.15
            # Répertoires
            mod_dir = Path(config["gev"]["modelised"]) / st.session_state["echelle"] / season_choice_key
            obs_dir = Path(config["gev"]["observed"]) / st.session_state["echelle"] / season_choice_key

            return {    
                "echelle": st.session_state["echelle"],
                "unit": st.session_state["unit"],
                "model_name": st.session_state["model_name"],
                "model_name_pres": selected_model,
                "param_choice": st.session_state["param_choice"],
                "param_choice_pres": selected_label,
                "quantile_choice": st.session_state["quantile_choice"],
                "stat_choice_key": stat_choice_key,
                "scale_choice_key": scale_choice_key,
                "season_choice_key": season_choice_key,
                "season_choice": st.session_state["season_choice"],
                "min_year_choice": min_year_choice,
                "max_year_choice": max_year_choice,
                "missing_rate": missing_rate,
                "mod_dir": mod_dir,
                "obs_dir": obs_dir,
                "T_choice": st.session_state["T_choice"],
                "par_X_annees": st.session_state["par_X_annees"]
            }

    return None
