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
    if type_ == "continu":
        custom_colorscale = [
            (0.0, "#ffffff"),  # blanc
            (0.1, "lightblue"),
            (0.3, "blue"),
            (0.45, "green"),
            (0.6, "yellow"),
            (0.7, "orange"),
            (0.8, "red"),  # rouge
            (1.0, "black"),  # noir
        ]

        return mcolors.LinearSegmentedColormap.from_list("custom", custom_colorscale)

    elif type_ == "discret":
        couleurs_vives = [
            "#e41a1c",  # rouge vif
            "#377eb8",  # bleu soutenu
            "#4daf4a",  # vert
            "#984ea3",  # violet
            "#ff7f00",  # orange
            "#ffff33",  # jaune
            "#a65628",  # brun
            "#f781bf",  # rose
            "#999999",  # gris
            "#66c2a5",  # turquoise
            "#fc8d62",  # corail
            "#8da0cb",  # bleu pastel
        ]
        if not (2 <= nombre_label <= len(couleurs_vives)):
            raise ValueError(f"'nombre_label' doit être entre 2 et {len(couleurs_vives)}")

        # Construire un colorscale interpolé
        step = 1 / (nombre_label - 1)
        custom_colorscale = [[i * step, couleurs_vives[i]] for i in range(nombre_label)]
        return mcolors.LinearSegmentedColormap.from_list("custom_discret", custom_colorscale)

    else:
        raise ValueError(f"Type d'échelle inconnu : '{type_}'. Utilisez 'continu' ou 'discret'.")
