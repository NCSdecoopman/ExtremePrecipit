import os
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

def filter_ds_to_france(ds, ne_directory='data/external/naturalearth'):
    """
    Filtre un dataset xarray pour ne conserver que les points strictement à l'intérieur du polygone de la France.

    Paramètres :
    - ds : xarray.Dataset contenant les variables 'lat' et 'lon' en 2D.
    - ne_directory : chemin vers le dossier contenant le shapefile Natural Earth.

    Retourne :
    - ds_france : xarray.Dataset ne contenant que les points à l'intérieur du polygone de la France.
    """
    # 1) Charger la géométrie de la France en WGS84
    shapefile_path = os.path.join(ne_directory, "ne_10m_admin_0_countries.shp")
    gdf = gpd.read_file(shapefile_path, engine="pyogrio")
    gdf_france = gdf[gdf["ADMIN"] == "France"].to_crs(epsg=4326)
    geom_france = gdf_france.geometry.iloc[0]  # Polygone (ou MultiPolygone) de la France

    # 2) Extraire les grilles de latitude/longitude (2D)
    lat2d = ds["lat"].values  # (ny, nx)
    lon2d = ds["lon"].values  # (ny, nx)
    ny, nx = lat2d.shape

    # 3) Aplatir en vecteurs 1D
    lat_flat = lat2d.ravel()  # (ny * nx,)
    lon_flat = lon2d.ravel()  # (ny * nx,)

    # 4) Construire les points Shapely et tester leur inclusion
    points = [Point(lon, lat) for lon, lat in zip(lon_flat, lat_flat)]
    inside_mask_flat = np.array([geom_france.intersects(pt) for pt in points])
    
    # 5) Selection des bonnes coordonnées
    lat_inside = lat_flat[inside_mask_flat]
    lon_inside = lon_flat[inside_mask_flat]

    # Pour sélectionner dans ds, on a besoin des indices (y, x) correspondant
    # à ces lat/lon à l’intérieur.
    inside_indices = np.where(inside_mask_flat.reshape(ny, nx))  # tuple (array_y, array_x)

    # 6) Sélectionner ces indices dans ds
    #    On crée une nouvelle dimension "points" de taille N_in
    ds_france = ds.isel(
        y=("points", inside_indices[0]),
        x=("points", inside_indices[1])
    )

    # 7) Redéfinir les coordonnées lat/lon pour qu'elles soient 1D
    #    (sinon elles restent 2D et contiennent des valeurs uniques à chaque point)
    ds_france = ds_france.drop_vars(["lat", "lon"])  # enlever l'ancienne version 2D
    ds_france = ds_france.assign_coords({
        "lat": (("points",), lat_inside),
        "lon": (("points",), lon_inside)
    })

    return ds_france