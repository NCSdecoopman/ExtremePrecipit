import streamlit as st
st.set_page_config(
    layout="wide", 
    page_title="Visualisation Précipitations",
    page_icon="🌧️"
)

st.markdown("""
    <style>
        /* Applique sur le conteneur global pour éviter le débordement */
        .block-container {
            max-width: 95%;
            margin: auto;
        }

        /* Fixe les colonnes même sur écrans < 1000px */
        @media (max-width: 1000px) {
            .st-emotion-cache-z5fcl4 {
                display: flex !important;
                flex-direction: row !important;
                flex-wrap: wrap !important;
            }
            .st-emotion-cache-z5fcl4 > div {
                flex: 1 1 48% !important; /* Deux colonnes côte à côte sur petits écrans */
                min-width: 48% !important;
                max-width: 48% !important;
                padding: 5px !important;
            }
        }

        /* Encore plus petit (mobile) -> une seule colonne */
        @media (max-width: 600px) {
            .st-emotion-cache-z5fcl4 > div {
                flex: 1 1 100% !important;
                min-width: 100% !important;
                max-width: 100% !important;
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
