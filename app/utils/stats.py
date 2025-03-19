import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import kendalltau

def mann_kendall_test(series):
    """ Effectue le test de Mann-Kendall sur une série temporelle. """
    if len(series) < 5:  # On vérifie qu'il y a suffisamment de données
        return np.nan
    _, p_value = kendalltau(series.index, series.values)
    return p_value

import pandas as pd
import streamlit as st

def analyze_stationarity(df, stat="max"):
    """ 
    Charge les données et analyse la stationnarité via le test de Mann-Kendall.
    Renvoie un DataFrame avec les p-valeurs pour chaque point de grille (lat, lon).
    """       
    # Vérification que df est un DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("L'argument 'df' doit être un DataFrame.")

    # Vérification que les colonnes nécessaires existent
    required_columns = ["year", "lat", "lon", f"pr_{stat}"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Les colonnes suivantes sont manquantes dans le DataFrame : {', '.join(required_columns)}")

    # Pivot pour aligner les années en index
    df_pivot = df.pivot(index="year", columns=["lat", "lon"], values=f"pr_{stat}")

    p_values = []

    # Calcul du test de Mann-Kendall avec mise à jour de la progression
    with st.spinner("Analyse de la stationnarité en cours..."):
        for i, col in enumerate(df_pivot.columns):
            p_value = mann_kendall_test(df_pivot[col])
            p_values.append((col, p_value))

    # Création du DataFrame avec les p-values pour chaque point de grille
    p_values_df = pd.DataFrame(p_values, columns=["(lat, lon)", "p_value"])

    # Séparer la colonne '(lat, lon)' en lat et lon
    p_values_df[["lat", "lon"]] = pd.DataFrame(p_values_df["(lat, lon)"].tolist(), index=p_values_df.index)
    p_values_df = p_values_df.drop(columns=["(lat, lon)"])

    return p_values_df
