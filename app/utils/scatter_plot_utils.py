import plotly.graph_objects as go
import pandas as pd
import streamlit as st

def show_scatter_plot(result_df_observed, result_df_modelised):
    # Fusionner les deux datasets par station (lat/lon)
    df_merge = pd.merge(
        result_df_observed[["lat", "lon"]],
        result_df_modelised[["lat", "lon"]],
        left_index=True,
        right_index=True,
        suffixes=('_obs', '_mod')
    )

    st.write(df_merge)
