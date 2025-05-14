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


def echelle_config(type_: bool, echelle: str = None, n_colors: int = 256):
    if type_: # Continu

        if echelle == "diverging_zero_white": # Choix personnalisé
            # Dégradé négatif (bleu) → 0 (blanc) → positif (jaune à rouge)
            custom_colorscale = [
                (0.0, "#08306B"),   # bleu foncé
                (0.1, "#2171B5"),
                (0.2, "#6BAED6"),
                (0.3, "#C6DBEF"),
                (0.49, "#ffffff"),  # blanc à 0
                (0.5, "#ffffff"),
                (0.6, "#ffffb2"),   # jaune clair
                (0.7, "#fecc5c"),
                (0.8, "#fd8d3c"),
                (0.9, "#f03b20"),
                (1.0, "#bd0026"),   # rouge foncé
            ]

            cmap = mcolors.LinearSegmentedColormap.from_list("diverging_zero_white", custom_colorscale)

            if n_colors is not None:
                # Retourne une version discrète avec n couleurs
                return ListedColormap([cmap(i / (n_colors - 1)) for i in range(n_colors)])
            else:
                return cmap
            
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
       
    else: # Discret
      
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

