import streamlit as st

def menu_statisticals(min_years: int, max_years: int, STATS, SEASON):
    if "selected_point" not in st.session_state:
        st.session_state["selected_point"] = None

    # D'abord on définit stat_choice pour l'utiliser plus tard :
    col0, col1, col2, col3, col4, col5 = st.columns([0.6, 0.3, 0.4, 0.4, 0.3, 0.7])

    with col0:
        stat_choice = st.selectbox("Choix de la statistique étudiée", list(STATS.keys()))

    with col1:
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
        quantile_choice = st.slider(
            "Retrait des percentiles",
            min_value=0.950,
            max_value=1.00,
            value=0.990,
            step=0.001,
            format="%.3f"  # Affichage à 3 décimales
        )

    with col2:
        season_choice = st.selectbox(
            "Choix de la saison",
            list(SEASON.keys())
        )

    with col3:
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

        # On affiche pas la première année en année hydro ou hiver (année incomplète)
        if season_choice in ["Année hydrologique", "Hiver"]:
            min_year_choice, max_year_choice = st.slider(
                f"Sélection temporelle entre {min_years+1} et {max_years}",
                min_value=min_years+1,
                max_value=max_years,
                value=(min_years+1, max_years)
            )

        else:
            min_year_choice, max_year_choice = st.slider(
                f"Sélection temporelle entre {min_years} et {max_years}",
                min_value=min_years,
                max_value=max_years,
                value=(min_years, max_years)
            )

    with col4:
        if stat_choice in ["Cumul","Jour de pluie"]:
            scale_choice = st.selectbox("Choix de l'échelle temporelle", ["Journalière"])
        else:
            scale_choice = st.selectbox("Choix de l'échelle temporelle", ["Horaire","Journalière"])

    with col5:
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

        # On affiche pas la première année en année hydro ou hiver (année incomplète)
        missing_rate = st.slider(
            "Taux maximal de valeurs manquantes pour les données observées",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.01
        )

    return stat_choice, quantile_choice, min_year_choice, max_year_choice, season_choice, scale_choice, missing_rate