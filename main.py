import streamlit as st
from app.modules import statisticals, scatter_plot

st.set_page_config(layout="wide", page_title="Visualisation des précipitations", page_icon="🌧️")

st.markdown("""
    <style>
        html, body, p, div, span {
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

st.sidebar.title("Navigation")
option = st.sidebar.selectbox(
    "",
    ("Statistiques descriptives", "Scatter plot")
)

config_path = "app/config/config.yaml"

if option == "Statistiques descriptives":
    statisticals.show(config_path)

elif option == "Scatter plot":
    scatter_plot.show(config_path)