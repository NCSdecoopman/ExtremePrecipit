import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_ds(ds, ax=None):
    """
    Trace une carte des points de grille du dataset `ds`.
    
    Args:
        ds: xarray Dataset contenant les variables 'lat' et 'lon'.
        ax: Matplotlib Axes, si None, un nouvel axe est créé.
    """
    latitudes = ds.lat.values
    longitudes = ds.lon.values

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={'projection': ccrs.PlateCarree()})
    else:
        fig = ax.figure
        ax.set_extent([longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], crs=ccrs.PlateCarree())

    # Ajouter des fonds de carte (côtes, pays)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='dotted')

    # Afficher les points de grille
    ax.scatter(longitudes, latitudes, s=1, color='lightblue', transform=ccrs.PlateCarree(), alpha=0.5)

    # Ajouter labels et titre
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    return fig, ax
