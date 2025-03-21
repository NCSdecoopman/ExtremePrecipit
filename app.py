import streamlit as st
st.set_page_config(
    layout="wide", 
    page_title="Visualisation Précipitations",
    page_icon="🌧️"
)

st.markdown("""
    <style>
        /* Empêche les colonnes de dépasser 100% */
        .block-container {
            max-width: 90%;
            margin: auto;
        }
        /* Fixe les colonnes en ligne même sur petits écrans */
        @media (max-width: 1000px) {
            .st-emotion-cache-z5fcl4 {
                flex-direction: row !important;
                flex-wrap: wrap !important;
            }
        }
    </style>
""", unsafe_allow_html=True)



import os
from app.utils.data import get_available_years

from app.modules import series#, gev_sans_bootstrap

import pandas as pd
import plotly.express as px

# A activer
# from app.utils.login import login
# # Vérifie l'authentification avant d'afficher le reste de l'application
# if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
#     login()
#     st.stop()  # Stoppe l'exécution si l'utilisateur n'est pas connecté

# Menu latéral
# st.sidebar.title("Navigation")
# option = st.sidebar.radio(
#     "Données à analyser :", 
#     ["Statistiques descriptives", "GEV sans bootstrap"] #, "Périodes de retours"]
# )
option = "Statistiques descriptives"

# Définition des répertoires
OUTPUT_DIR = os.path.join("data", "result")

years = get_available_years(OUTPUT_DIR)

if not years:
    st.error("Aucune donnée disponible. Vérifie le dossier de sortie.")
    st.stop()

if option == "Statistiques descriptives":
    series.show(OUTPUT_DIR, years)

# elif option == "GEV sans bootstrap":
#     gev_sans_bootstrap.show(OUTPUT_DIR, years)
