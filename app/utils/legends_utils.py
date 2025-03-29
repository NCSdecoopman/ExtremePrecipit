import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from matplotlib.colors import LinearSegmentedColormap

def convert_custom_colorscale(custom_colorscale):
    positions, colors = zip(*custom_colorscale)
    return LinearSegmentedColormap.from_list("custom_colormap", list(zip(positions, colors)))

def get_stat_column_name(stat_key: str, scale_key: str) -> str:
    if stat_key == "mean":
        return f"mean_all_{scale_key}"
    elif stat_key == "max":
        return f"max_all_{scale_key}"
    elif stat_key == "mean-max":
        return f"max_mean_{scale_key}"
    elif stat_key == "date":
        if scale_key == "Horaire":
            return "date_max_h"
        else:  # Journalière
            return "date_max_j"
    elif stat_key == "month":
        if scale_key == "Horaire":
            return "mois_pluvieux_h"
        else:  # Journalière
            return "mois_pluvieux_j"
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
    
def formalised_legend(df, column_to_show, colormap, vmin=None, vmax=None):
    if column_to_show.startswith("date"):
        # Cas date
        df[column_to_show] = pd.to_datetime(df[column_to_show])
        vmin = pd.to_datetime(df[column_to_show].min()) if vmin is None else pd.to_datetime(vmin)
        vmax = pd.to_datetime(df[column_to_show].max()) if vmax is None else pd.to_datetime(vmax)

        value_norm = (df[column_to_show] - vmin).dt.total_seconds() / (vmax - vmin).total_seconds()
        val_fmt_func = lambda x: x.strftime("%Y-%m-%d")

    elif column_to_show.startswith("mois_pluvieux"):
        # Cas mois pluvieux : valeurs entières de 1 à 12
        vmin = 1
        vmax = 12

        value_norm = (df[column_to_show] - 1) / 11  # Normalise mois de 1 à 12 -> 0 à 1
        val_fmt_func = lambda x: [
            "Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
            "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"
        ][int(x) - 1] if 1 <= int(x) <= 12 else "Inconnu"

    else:
        # Cas numérique continu
        vmin = df[column_to_show].min() if vmin is None else vmin
        vmax = df[column_to_show].max() if vmax is None else vmax

        value_norm = (df[column_to_show] - vmin) / (vmax - vmin)
        val_fmt_func = lambda x: f"{x:.2f}"

    value_norm = value_norm.clip(0, 1).fillna(0)

    df["fill_color"] = value_norm.apply(
        lambda v: [int(255 * c) for c in colormap(v)[:3]] + [255]  # RGBA
    )
    df["val_fmt"] = df[column_to_show].map(val_fmt_func)

    df["lat_fmt"] = df["lat"].map(lambda x: f"{x:.3f}")
    df["lon_fmt"] = df["lon"].map(lambda x: f"{x:.3f}")
    
    return df, vmin, vmax

def display_vertical_color_legend(height, colormap, vmin, vmax, n_ticks=5, label=""):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64
    import streamlit as st

    # --- CAS SPÉCIAL : MOIS (VALEURS DISCRÈTES DE 1 À 12) ---
    if isinstance(vmin, int) and isinstance(vmax, int) and (1 <= vmin <= 12) and (1 <= vmax <= 12):
        mois_labels = [
            "Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
            "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"
        ]

        cmap = colormap  # colormap est déjà passé à la fonction depuis echelle_config

        color_boxes = ""  
        for mois in range(vmin, vmax + 1):
            rgba = cmap((mois - 1) / 11)  # Normalisé entre 0 et 1
            rgb = [int(255 * c) for c in rgba[:3]]
            color = f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"
            label_mois = mois_labels[mois - 1]

            color_boxes += (
                f'<div style="display: flex; align-items: center; margin-bottom: 4px;">'
                f'  <div style="width: 14px; height: 14px; '
                f'              background-color: {color}; '
                f'              border: 1px solid #ccc; '
                f'              margin-right: 6px;">'
                f'  </div>'
                f'  <div style="font-size: 12px;">{label_mois}</div>'
                f'</div>'
            )

        html_mois = (
            f'<div style="text-align: left; font-size: 13px; margin-bottom: 4px;">'
            f'  <b>{label}</b>'
            f'</div>'
            f'<div style="display: flex; flex-direction: column;">'
            f'{color_boxes}'
            f'</div>'
        )
        st.markdown(html_mois, unsafe_allow_html=True)
        return

    # --- CAS NORMAL : LÉGENDE CONTINUE AVEC GRADIENT ---
    gradient = np.linspace(1, 0, 256).reshape(256, 1)
    fig, ax = plt.subplots(figsize=(1, 5), dpi=100)
    ax.imshow(gradient, aspect='auto', cmap=colormap)
    ax.axis('off')

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)
    base64_img = base64.b64encode(buf.getvalue()).decode()

    # Construction des ticks (pour l’échelle continue)
    if isinstance(vmin, pd.Timestamp) and isinstance(vmax, pd.Timestamp):
        ticks_seconds = np.linspace(vmax.timestamp(), vmin.timestamp(), n_ticks)
        ticks = [pd.to_datetime(t, unit='s').strftime("%Y-%m-%d") for t in ticks_seconds]
    else:
        ticks_vals = np.linspace(vmax, vmin, n_ticks)
        ticks = [f"{val:.1f}" for val in ticks_vals]

    st.markdown(
        f"""
        <div style="text-align: left; font-size: 13px;">
            <b>{label}</b>
        </div>
        <div style="display: flex; flex-direction: row; align-items: center; height: {height-30}px;">
            <img src="data:image/png;base64,{base64_img}"
                 style="height: 100%; width: 20px; border: 1px solid #ccc; border-radius: 5px;"/>
            <div style="display: flex; flex-direction: column; justify-content: space-between; 
                        margin-left: 8px; height: 100%; font-size: 12px;">
                {''.join(f'<div>{tick}</div>' for tick in ticks)}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
