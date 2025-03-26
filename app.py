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

st.write("TEST")

# from app.modules import statisticals

# import pandas as pd
# import plotly.express as px

# option = "Statistiques descriptives"

# config_path = "app/config/config.yaml"

# if option == "Statistiques descriptives":
#     statisticals.show(config_path)