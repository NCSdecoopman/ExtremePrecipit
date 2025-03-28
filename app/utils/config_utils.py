import yaml
from matplotlib import colors as mcolors

def menu_config():
    STATS = {
        "Moyenne": "mean",
        "Maximum": "max",
        "Moyenne des maxima": "mean-max",
        "Date du maximum": "date",
        "Mois comptabilisant le plus de maximas": "month",
        "Jour de pluie": "numday",
    }

    SEASON = {
        "Année hydrologique": "hydro",
        "Hiver": "djf",
        "Printemps": "mam",
        "Été": "jja",
        "Automne": "son",
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
    couleurs_vives = [
        "#FFFFFF",  # blanc
        "#00FFFF",  # cyan
        "#0099FF",  # bleu ciel vif
        "#0000FF",  # bleu
        "#00FF00",  # vert fluo
        "#FFFF00",  # jaune vif
        "#FFA500",  # orange
        "#FF4500",  # orange foncé
        "#FF0000",  # rouge
        "#8B0000",  # rouge foncé
        "#800080",  # violet
        "#000000"   # noir
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
        if not (2 <= nombre_label <= len(couleurs_vives)):
            raise ValueError(f"'nombre_label' doit être entre 2 et {len(couleurs_vives)}")
        
        step = len(couleurs_vives) / nombre_label
        indices = [round(i * step) for i in range(nombre_label)]
        indices = list(sorted(set(min(i, len(couleurs_vives)-1) for i in indices)))
        custom_colorscale = [couleurs_vives[i] for i in indices]

    else:
        raise ValueError(f"Type d'échelle inconnu : '{type_}'. Utilisez 'continu' ou 'discret'.")

    return mcolors.LinearSegmentedColormap.from_list("custom", custom_colorscale)
