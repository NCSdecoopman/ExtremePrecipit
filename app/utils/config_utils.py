import yaml
from matplotlib import colors as mcolors
from matplotlib.colors import ListedColormap

def menu_config():
    STATS = {
        "Moyenne": "mean",
        "Maximum": "max",
        "Moyenne des maxima": "mean-max",
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

    return STATS, SEASON, SCALE


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def echelle_config(type_: str, n_colors: int = 256):
    if type_ == "continu":
        custom_colorscale = [
            (0.0, "#ffffff"),  # blanc
            (0.1, "lightblue"),
            (0.3, "blue"),
            (0.45, "green"),
            (0.6, "yellow"),
            (0.7, "orange"),
            (0.8, "red"),      # rouge
            (1.0, "black"),    # noir
        ]

        cmap = mcolors.LinearSegmentedColormap.from_list("custom", custom_colorscale)

        if n_colors is not None:
            # Retourne une version discrète avec n couleurs
            return ListedColormap([cmap(i / (n_colors - 1)) for i in range(n_colors)])
        else:
            return cmap

    elif type_ == "discret":
        couleurs_par_mois = [
            "#ffffff",  # Janvier 
            "blue",     # Février 
            "green",    # Mars 
            "red",      # Avril 
            "orange",   # Mai 
            "#00CED1",  # Juin 
            "yellow",   # Juillet 
            "#f781bf",  # Août 
            "purple",   # Septembre 
            "#654321",  # Octobre 
            "darkblue", # Novembre
            "black",    # Décembre 
        ]

        return ListedColormap(couleurs_par_mois)

    else:
        raise ValueError(f"Type d'échelle inconnu : '{type_}'. Utilisez 'continu' ou 'discret'.")

