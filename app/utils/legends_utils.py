import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def get_stat_column_name(stat_key: str, scale_key: str) -> str:
    if stat_key == "mean":
        return f"mean_{scale_key}"
    elif stat_key == "max":
        return f"max_all_{scale_key}"
    elif stat_key == "mean-max":
        return f"max_mean_{scale_key}"
    elif stat_key == "date":
        return f"date_max_{scale_key[-1]}"  # "h" ou "j"
    elif stat_key == "month":
        return f"mois_pluvieux_{scale_key[-1]}"  # "h" ou "j"
    elif stat_key == "numday":
        return "jours_pluie_moyen"
    else:
        raise ValueError(f"Statistique inconnue : {stat_key}")

def get_stat_unit(stat_key: str, scale_key: str) -> str:
    if stat_key in ["mean", "max", "mean-max"]:
        if scale_key == "mm_h":
            return f"mm/h"
        elif scale_key == "mm_j":
            return f"mm/j"
        else:
            return "ERROR"
    elif stat_key == "sum":
        return "mm"
    elif stat_key == "numday":
        return "jours"
    else:
        return ""
    
def formalised_legend(df, column_to_show, colormap):
    vmin, vmax = df[column_to_show].min(), df[column_to_show].max()
    value_norm = (df[column_to_show] - vmin) / (vmax - vmin)
    value_norm = value_norm.clip(0, 1).fillna(0)

    # Convertir en couleurs RGBA (avec alpha fixé à 200)
    df["fill_color"] = value_norm.apply(
        lambda v: [int(255 * c) for c in colormap(v)[:3]] + [200]
    )

    # Format pour affichage dans le tooltip
    df["val_fmt"] = df[column_to_show].map(lambda x: f"{x:.1f}")

    return df, vmin, vmax

def display_vertical_color_legend(height, colormap, vmin, vmax, n_ticks=5, label=""):
    gradient = np.linspace(1, 0, 256).reshape(256, 1)
    fig, ax = plt.subplots(figsize=(1, 5), dpi=100)
    ax.imshow(gradient, aspect='auto', cmap=colormap)
    ax.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)
    base64_img = base64.b64encode(buf.getvalue()).decode()

    ticks = np.linspace(vmax, vmin, n_ticks)

    st.markdown(
        f"""
        <div style="text-align: left; font-size: 13px; margin-top: 8px;">
            <b>{label}</b>
        </div>
        <div style="display: flex; flex-direction: row; align-items: center; height: {height}px; margin-top: 8px;;">
            <img src="data:image/png;base64,{base64_img}" style="height: 100%; width: 20px; border: 1px solid #ccc; border-radius: 5px;"/>
            <div style="display: flex; flex-direction: column; justify-content: space-between; margin-left: 8px; height: 100%; font-size: 12px;">
                {''.join(f'<div>{tick:.1f}</div>' for tick in ticks)}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )