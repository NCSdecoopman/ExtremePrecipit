import streamlit as st
from app.modules import statisticals, gev, niveau_retour, temp_stats, scatter_plot, all_max

st.set_page_config(layout="wide", page_title="Visualisation des précipitations", page_icon="🌧️")

st.markdown("""
    <style>
        body {
            font-family: 'Times New Roman', sans-serif !important;
            font-size: 11px !important;
            font-weight: 400 !important;
        }            
        html, body {
            font-size: 11px !important;
        }

        * {
            font-size: 11px !important;
        }
            
        h1, h2, h3, h4, h5, h6, label, button {
            font-size: 11px !important;
        }
            
        section[data-testid="stSidebar"] {
            width: 150px !important;
            min-width: 150px !important;
        }  
                 
        /* Masquer le header de l'app */
        div[class*="stAppHeader"] {
            display: none !important;
        }
            
        div[class*="block-container"] {
            padding-left: 1rem;
            padding-right: 1rem;
            max-width: 98% !important;
        }

        /* Responsive layout des colonnes */
        @media screen and (max-width: 1000px) {
            .element-container:has(> .stColumn) {
                display: flex;
                flex-wrap: wrap;
            }

            .element-container:has(> .stColumn) .stColumn {
                width: 48% !important;
                min-width: 48% !important;
            }
        }

        @media screen and (max-width: 600px) {
            .element-container:has(> .stColumn) .stColumn {
                width: 100% !important;
                min-width: 100% !important;
            }
        }
    </style>
""", unsafe_allow_html=True)

option = "Statistiques descriptives"
st.sidebar.title("Navigation")
option = st.sidebar.selectbox(
    "Navigation",
    ("Statistiques descriptives", "GEV", "Niveau de retour"), #"Scatter plot", "Période de retour", "Temp stats", "Scatter plot", "Tous les max"
    label_visibility="hidden"
)

config_path = "app/config/config.yaml"

if option == "Statistiques descriptives":
    statisticals.show(config_path)

elif option == "GEV":
    gev.show(config_path)

elif option == "Scatter plot":
    scatter_plot.show(config_path)

elif option == "Niveau de retour":
    niveau_retour.show(config_path)

elif option == "Tous les max":
    all_max.show(config_path)

elif option == "Temp stats":
    temp_stats.show(config_path)