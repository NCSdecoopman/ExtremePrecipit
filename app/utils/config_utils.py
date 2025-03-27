import yaml
from matplotlib import colors as mcolors

def menu_config():
    STATS = {
        "Moyenne": "mean",
        "Maximum": "max",
        "Moyenne des maxima": "mean-max",
        "Cumul": "sum",
        "Date du maximum": "date",
        "Mois comptabilisant le plus de maximas": "month",
        "Jour de pluie": "numday",
    }

    SEASON = {
        "Année hydrologique": [9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8],
        "Hiver": [12, 1, 2],
        "Printemps": [3, 4, 5],
        "Été": [6, 7, 8],
        "Automne": [9, 10, 11],
    }

    SCALE = {
        "Horaire": "mm_h",
        "Journalière": "mm_j"
    }

    LEGENDE = {
        "Horaire": "mm/h",
        "Journalière": "mm/j"
    }


    return STATS, SEASON, SCALE


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def echelle_config(type_: str, nombre_label: int = 5):
    couleurs = [
        "white",
        "lightblue",
        "blue",
        "darkblue",
        "green",
        "darkgreen",
        "yellow",
        "orange",
        "red",
        "darkred",
        "grey",
        "black"
    ]

    if type_ == "continu":
        custom_colorscale = [
            [0.0, "white"],  
            [0.01, "lightblue"],
            [0.10, "blue"],
            [0.30, "darkblue"],  
            [0.50, "green"], 
            [0.60, "yellow"],
            [0.70, "red"],  
            [0.80, "darkred"],  
            [1.0, "#654321"]
        ]

    elif type_ == "discret":
        if not (2 <= nombre_label <= len(couleurs)):
            raise ValueError(f"'nombre_label' doit être entre 2 et {len(couleurs)}")
        
        step = len(couleurs) / nombre_label
        indices = [round(i * step) for i in range(nombre_label)]
        indices = list(sorted(set(min(i, len(couleurs)-1) for i in indices)))
        custom_colorscale = [couleurs[i] for i in indices]
                

    else:
        raise ValueError(f"Type d'échelle inconnu : '{type_}'. Utilisez 'continu' ou 'discret'.")

    return mcolors.LinearSegmentedColormap.from_list("custom", custom_colorscale)