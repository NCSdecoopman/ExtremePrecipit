import pandas as pd

def generate_markdown_summary(ds):
    """
    Génère un résumé des informations du dataset xarray.
    """
    print("Les attributs globaux :")
    for attr, value in ds.attrs.items():
        print(f"- {attr} : {value}")
    
    print("\n\nDimensions :")
    for dim, size in ds.sizes.items():
        print(f"- {dim} : {size} points de grille")
    
    print("\n\nVariables :")
    for var in ds.data_vars:
        dims = ', '.join(ds[var].dims)
        dtype = str(ds[var].dtype)
        print(f"- {var} ({dims}) : {dtype}")

def generate_time_summary(ds):  
    # Extraire les valeurs temporelles et les convertir en datetime
    time_values = pd.to_datetime(ds.time.values)
    
    # Calculer les différences entre chaque pas de temps
    time_diffs = time_values[1:] - time_values[:-1]
    
    # Rendre unique et trier les valeurs
    unique_intervals = pd.Series(time_diffs).drop_duplicates().sort_values()
    
    # Formater les intervalles d'étude sous forme YYYY-MM-DD
    time_min = time_values.min().strftime('%Y-%m-%d')
    time_max = time_values.max().strftime('%Y-%m-%d')
    # Formater les intervalles d'enregistrement sous forme HH:MM:SS
    formatted_intervals = (str(interval).split(" ")[-1].split(".")[0] for interval in unique_intervals)
    formatted_intervals = list(set(formatted_intervals))

    return time_min, time_max, formatted_intervals