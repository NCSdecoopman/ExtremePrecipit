import streamlit as st

def menu_statisticals(min_years: int, max_years: int, STATS, SEASON):
    if "selected_point" not in st.session_state:
        st.session_state["selected_point"] = None

    # D'abord on définit stat_choice pour l'utiliser plus tard :
    col1, col2, col3, col4 = st.columns([1, 1, 0.75, 0.65])

    with col1:
        stat_choice = st.selectbox("Choix de la statistique étudiée", list(STATS.keys()))

    with col2:
        st.markdown("""
            <style>
            /* Cacher les ticks-bar min et max sous la barre du slider */
            div[data-testid="stSliderTickBarMin"],
            div[data-testid="stSliderTickBarMax"] {
                display: none !important;
            }
            /* Réduire l'espace vertical du slider */
            .stSlider > div[data-baseweb="slider"] {
                margin-bottom: -10px;
            }
            /* Remonter la barre + les poignées */
            .stSlider {
                transform: translateY(-17px);
            }
            </style>
        """, unsafe_allow_html=True)

        min_year_choice, max_year_choice = st.slider(
            f"Sélection temporelle entre {min_years} et {max_years}",
            min_value=min_years,
            max_value=max_years,
            value=(min_years, max_years)
        )

    with col3:
        season_choice = st.selectbox(
            "Choix de la saison",
            list(SEASON.keys())
        )

    with col4:
        if stat_choice in ["Cumul","Jour de pluie"]:
            scale_choice = st.selectbox("Choix de l'échelle temporelle", ["Journalière"])
        else:
            scale_choice = st.selectbox("Choix de l'échelle temporelle", ["Horaire","Journalière"])

    return stat_choice, min_year_choice, max_year_choice, season_choice, scale_choice