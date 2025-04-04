import streamlit as st

def menu_statisticals(min_years: int, max_years: int, STATS, SEASON):
    if "selected_point" not in st.session_state:
        st.session_state["selected_point"] = None

    if "run_analysis" not in st.session_state:
        st.session_state["run_analysis"] = False

    # Crée les colonnes
    col0, col1, col2, col3, col4, col5, col6 = st.columns([0.5, 0.3, 0.4, 0.4, 0.3, 0.6, 0.2])

    with col0:
        st.selectbox("Choix de la statistique étudiée", list(STATS.keys()), key="stat_choice")

    with col1:
        st.slider(
            "Percentile de retrait AROME",
            min_value=0.950,
            max_value=1.00,
            value=0.990,
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
            st.selectbox("Choix de l'échelle temporelle", ["Horaire", "Journalière"], key="scale_choice")

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
