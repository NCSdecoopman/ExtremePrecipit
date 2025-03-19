import streamlit as st

def plot_map(fig_map, size_max=1, zoom=4.5, title_legend=""):
    fig_map.update_traces(marker=dict(sizeref=size_max))  # Correction pour size_max
    fig_map.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            zoom=zoom  # Correction pour zoom
        ),
        width=1000,
        height=700,
        dragmode=False
    )

    # Vérifier et mettre à jour explicitement coloraxis
    fig_map.update_layout(
        coloraxis_colorbar=dict(
            title=title_legend,  # Utiliser le titre passé en argument
            tickformat=" "
        )
    )

    # Affichage dans Streamlit
    st.plotly_chart(fig_map, use_container_width=True, config={
        "displayModeBar": False,
        "scrollZoom": True,
    })