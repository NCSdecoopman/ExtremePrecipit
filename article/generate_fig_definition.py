# FIGURE 0
import xarray as xr
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.ticker import FixedLocator
from pathlib import Path

import matplotlib as mpl
import numpy as np
from rasterio.features import geometry_mask

# --- relief ---
import geopandas as gpd
import rioxarray as rxr


# --- pour les pastilles et la légende ---
from matplotlib.legend_handler import HandlerBase
import matplotlib.patches as mpatches
from matplotlib.text import Text

# -------------------------------------------------------------------
# 0. Dictionnaire des centres (lon, lat) : zones_centers
# -------------------------------------------------------------------
zones_centers = {
    "Rhône Valley": {"coords": (4.60, 44.90)},
    "French Alps": {"coords": (6.39, 44.67)},
    "Massif Central": {"coords": (3.09, 45.03)},
    "Pyrenees": {"coords": (0.7073, 42.8641)},
    "Vosges": {"coords": (6.8943, 48.7324)},
    "Jura": {"coords": (6.0455, 47.0664)},
    "Mediterranean coast": {"coords": (4.80, 43.50)},
    "Paris area": {"coords": (2.2328, 48.6707)},
    "Brittany": {"coords": (-2.76, 48.18)},
}

# -------------------------------------------------------------------
# 1. Domaine AROME
# -------------------------------------------------------------------
nc = Path("../data/raw/modelised/pr_ALPX-3_ERA5_evaluation_r1i1p1f1_CNRM-MF_CNRM-AROME46t1_v1-r1_1hr_195901010030-195912312330.nc")
ds = xr.open_dataset(nc, decode_times=False)

latb = ds["lat_bnds"].values
lonb = ds["lon_bnds"].values
ny, nx, _ = latb.shape

# indices des cellules de bord
top    = [(0, j) for j in range(nx)]
right  = [(i, nx-1) for i in range(ny)]
bottom = [(ny-1, j) for j in range(nx)]
left   = [(i, 0) for i in range(ny)]
border = np.unique(np.array(top + right + bottom + left), axis=0)

polys = []
for i, j in border:
    xs = lonb[i, j, :].astype(float)
    ys = latb[i, j, :].astype(float)
    cx, cy = xs.mean(), ys.mean()
    ang = np.arctan2(ys - cy, xs - cx)
    order = np.argsort(ang)
    ring = [(xs[k], ys[k]) for k in order]
    ring.append(ring[0])
    polys.append(Polygon(ring))

u = unary_union(polys)

proj_data = ccrs.PlateCarree()
proj_map  = ccrs.LambertConformal(
    central_longitude=10,
    central_latitude=47,
    standard_parallels=(30, 60),
)

fig = plt.figure(figsize=(6, 5))
ax = plt.axes(projection=proj_map)

# Aplatir (rasteriser) quasiment tout ce qui est dessiné dans cet axe
# → zorder seuil élevé pour que tout soit rasterisé (courbes, DEM, etc.)
# ax.set_rasterization_zorder(100)

# -------------------------------------------------------------------
# 1bis. DEM altitude en fond (MNT France métropolitaine & DROM)
#      → version re-échantillonnée à ~0.1 km
# -------------------------------------------------------------------
from rasterio.features import geometry_mask

dem_path = Path("../data/external/dem/dem.tif")
dem = rxr.open_rasterio(dem_path).squeeze()  # (y, x)

# 1) Polygone France métropolitaine
countries_path = Path("../data/external/naturalearth/ne_10m_admin_0_countries.shp")
gdf_countries = gpd.read_file(countries_path)
france_poly = gdf_countries[gdf_countries["ADMIN"] == "France"]

france_parts = france_poly.explode(index_parts=True)

# Calcul de l'aire dans un CRS projeté
france_parts_proj = france_parts.to_crs("EPSG:2154")
france_parts_proj["area"] = france_parts_proj.geometry.area
largest = france_parts_proj.sort_values("area", ascending=False).iloc[[0]]

# Retour en WGS84
metropolitan_france = largest.to_crs("EPSG:4326")

# Simplification du contour (accélère fortement le masque)
metropolitan_france_simpl = metropolitan_france.copy()
metropolitan_france_simpl["geometry"] = (
    metropolitan_france_simpl.geometry.simplify(0.01, preserve_topology=True)
)

# 2) Assurer le CRS du DEM
if dem.rio.crs is None:
    dem = dem.rio.write_crs("EPSG:4326")

# 3) Bbox sur la France pour réduire la taille du raster
minx, miny, maxx, maxy = metropolitan_france_simpl.total_bounds
margin = 0.2  # petit débord pour être sûr de tout garder
dem_crop = dem.sel(
    x=slice(minx - margin, maxx + margin),
    y=slice(maxy + margin, miny - margin),  # y décroît
)

# 4) Masque raster (au lieu de clip) → très rapide
geom_union = metropolitan_france_simpl.geometry.union_all()

mask_inside = geometry_mask(
    [geom_union],
    out_shape=dem_crop.shape,
    transform=dem_crop.rio.transform(),
    invert=True,          # True à l'intérieur de la France
)

dem_fr = dem_crop.where(mask_inside)  # → NaN en dehors de la France

# Mask nodata éventuel seulement si ≠ 0
nodata = dem.rio.nodata
if nodata is not None and nodata != 0.0:
    dem_fr = dem_fr.where(dem_fr != nodata)


# 5) Sous-échantillonnage pour obtenir ~2.5 km
dx = float(np.abs(np.diff(dem_fr["x"].values).mean()))
dy = float(np.abs(np.diff(dem_fr["y"].values).mean()))

target_deg = 2.5 / 111.0 
step_x = max(1, int(round(target_deg / dx)))
step_y = max(1, int(round(target_deg / dy)))

dem_plot = dem_fr.isel(
    x=slice(0, None, step_x),
    y=slice(0, None, step_y),
)


# Palette discrète pour l'altitude
from matplotlib.colors import ListedColormap, BoundaryNorm

# Bornes en m
elev_bounds = [0, 100, 200, 500, 1000, 2000, 3000, 3500, 4000, 4500]

# Couleurs (≈ style de ta légende : bas clair, haut vert vif)
elev_colors = [
    "#f7f7f7",  # 0–100
    "#fde0dd",  # 100–200
    "#fcbba1",  # 200–500
    "#fc9272",  # 500–1000
    "#fed976",  # 1000–2000
    "#c2e699",  # 2000–3000
    "#78c679",  # 3000–3500
    "#31a354",  # 3500–4000
    "#006837",  # 4000–4500
    "#004529",  # > 4500 (extension max)
]

elev_cmap = ListedColormap(elev_colors)
elev_norm = BoundaryNorm(elev_bounds, elev_cmap.N, extend="max")

# 6) Grille et tracé
lons = dem_plot["x"].values
lats = dem_plot["y"].values
lon2d, lat2d = np.meshgrid(lons, lats)

im = ax.pcolormesh(
    lon2d,
    lat2d,
    dem_plot.values,
    transform=proj_data,
    cmap=elev_cmap,
    norm=elev_norm,
    shading="auto",
    alpha=0.6,
    zorder=0,
)
im.set_rasterized(True)



# -------------------------------------------------------------------
# Côtes et frontières (au-dessus du DEM)
# -------------------------------------------------------------------
ax.coastlines("10m", linewidth=0.6, zorder=10)
ax.add_feature(
    cfeature.BORDERS.with_scale("10m"),
    linewidth=0.3,
    edgecolor="black",
    zorder=11,
)

# -------------------------------------------------------------------
# 1ter. Courbes de niveau (relief) — filtrées
# -------------------------------------------------------------------
relief_path = Path("../data/external/niveaux/selection_courbes_niveau_france.shp")

relief = (
    gpd.read_file(relief_path)
    .to_crs("EPSG:4326")
)

relief = relief[relief["coordonnees"] == 400].copy()
relief["geometry"] = relief.geometry.simplify(0.01, preserve_topology=True)

relief_feature = cfeature.ShapelyFeature(
    relief.geometry,
    proj_data,
    edgecolor="#000000",
    facecolor="none",
)

ax.add_feature(
    relief_feature,
    linewidth=0.3,
    zorder=9,
)

# -------------------------------------------------------------------
# tracer polygone du domaine
# -------------------------------------------------------------------
def plot_poly(p: Polygon):
    xs, ys = p.exterior.xy
    ax.plot(xs, ys, transform=proj_data, linewidth=2, color="black", zorder=12)
    for interior in p.interiors:
        xi, yi = interior.xy
        ax.plot(xi, yi, transform=proj_data, linewidth=1, color="black", zorder=12)

if isinstance(u, MultiPolygon):
    for g in u.geoms:
        plot_poly(g)
else:
    plot_poly(u)

# emprise
all_x = np.concatenate(
    [np.array(g.exterior.coords)[:, 0]
     for g in (u.geoms if isinstance(u, MultiPolygon) else [u])]
)
all_y = np.concatenate(
    [np.array(g.exterior.coords)[:, 1]
     for g in (u.geoms if isinstance(u, MultiPolygon) else [u])]
)
pad = 1.0
xmin, xmax = all_x.min()-pad, all_x.max()+pad
ymin, ymax = all_y.min()-pad, all_y.max()+pad
ax.set_extent([xmin, xmax, ymin, ymax], crs=proj_data)

# grille et ticks hors carte
dx, dy = 2, 2
xticks = np.arange(np.floor(xmin/dx)*dx, np.ceil(xmax/dx)*dx+dx, dx)
yticks = np.arange(np.floor(ymin/dy)*dy, np.ceil(ymax/dy)*dy+dy, dy)

gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="0.8", linestyle="--")
gl.xlabel_style = {"size": 7}
gl.ylabel_style = {"size": 7}
gl.xlocator = FixedLocator(xticks)
gl.ylocator = FixedLocator(yticks)
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()

gl.top_labels = False
gl.right_labels = False
gl.bottom_labels = True
gl.left_labels = True
gl.x_inline = False
gl.y_inline = False

# cadre autour de la carte
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.0)
    spine.set_edgecolor("black")

# -------------------------------------------------------------------
# 2. Pastilles + lettres + légende
# -------------------------------------------------------------------
entries = []
for name, info in zones_centers.items():
    lon, lat = info["coords"]
    entries.append({"name": name, "lon": lon, "lat": lat})

entries_sorted = sorted(
    entries,
    key=lambda e: (-e["lat"], e["lon"])
)

letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
for i, e in enumerate(entries_sorted):
    e["letter"] = letters[i]

circle_color = "#1f4f7b"

for e in entries_sorted:
    x, y = e["lon"], e["lat"]
    ax.scatter(
        x, y,
        s=70,
        color=circle_color,
        edgecolor="white",
        linewidth=0,
        transform=proj_data,
        zorder=15,
    )
    ax.text(
        x - 0.01, y - 0.025,
        e["letter"],
        fontsize=7,
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
        transform=proj_data,
        zorder=16,
    )

class LegendLetter:
    def __init__(self, letter, color):
        self.letter = letter
        self.color = color

class HandlerLegendLetter(HandlerBase):
    def create_artists(
        self, legend, orig_handle,
        xdescent, ydescent, width, height, fontsize, trans
    ):
        r = min(width, height)
        cx = xdescent + r + width * 0.1
        cy = ydescent + height * 0.5

        circle = mpatches.Circle(
            (cx, cy),
            radius=r,
            facecolor=orig_handle.color,
            edgecolor="white",
            linewidth=1.0,
            transform=trans,
        )

        text = Text(
            cx, cy,
            orig_handle.letter,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=fontsize * 0.9,
            transform=trans,
        )

        return [circle, text]

legend_handles = []
legend_labels = []
for e in entries_sorted:
    legend_handles.append(LegendLetter(e["letter"], circle_color))
    legend_labels.append(e["name"])

ax.legend(
    legend_handles,
    legend_labels,
    handler_map={LegendLetter: HandlerLegendLetter()},
    loc="upper left",          # en haut à droite de l’axe
    bbox_to_anchor=(1.02, 0.98),
    frameon=False,
    fontsize=7,
    handlelength=1.4,
    borderaxespad=0.5,
    labelspacing=0.8,
)


# -------------------------------------------------------------------
# 2bis. Colorbar altitude (verticale, à droite sous la légende)
# -------------------------------------------------------------------
# Position de l'axe principal
box = ax.get_position()

# Axe pour la colorbar : à droite de la carte, en bas (sous la légende)
cax = fig.add_axes([
    box.x1 + 0.04,        # x : un peu à droite de l'axe principal
    box.y0,               # y : bas aligné avec la carte
    0.025,                # largeur de la barre
    box.height * 0.45,    # hauteur : ~45 % de la carte → sous la légende
])

cbar = fig.colorbar(
    im,
    cax=cax,
    orientation="vertical",
    ticks=[100, 200, 500, 1000, 2000, 3000, 3500, 4000, 4500],
)

cbar.set_label("Elevation (m)", fontsize=7)
cbar.ax.tick_params(labelsize=6)


# -------------------------------------------------------------------
# 3. Export
# -------------------------------------------------------------------
plt.savefig("figures/espace_definition.svg", dpi=600, bbox_inches="tight", pad_inches=0)
plt.savefig("figures/espace_definition.pdf", bbox_inches="tight", pad_inches=0)
plt.show()
