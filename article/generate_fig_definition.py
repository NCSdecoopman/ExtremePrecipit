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

# --- pour les pastilles et la légende ---
from matplotlib.legend_handler import HandlerBase
import matplotlib.patches as mpatches
from matplotlib.text import Text

# -------------------------------------------------------------------
# 0. Dictionnaire des centres (lon, lat) : zones_centers
# -------------------------------------------------------------------
zones_centers = {
    "Rhône Valley": {"coords": (4.89, 45.40)},  # plus au nord
    "Cévennes": {"coords": (3.3797, 44.3315)},
    "Mercantour": {"coords": (7.07, 44.19)},
    "Pyrénées-Orientales": {"coords": (2.2291, 42.7603)},

    "French Alps": {"coords": (6.39, 45.67)},
    "Massif Central": {"coords": (3.09, 45.03)},
    "Pyrenees": {"coords": (0.7073, 42.8641)},
    "Vosges": {"coords": (6.8943, 48.7324)},
    "Jura": {"coords": (6.0455, 47.0664)},

    "Grand Ouest": {"coords": (-1.55, 47.22)},
    "Basque Coast": {"coords": (-1.5852, 43.4098)},
    "Mediterranean coast": {"coords": (4.80, 42.95)},      # plus au sud
    "Provence / Côte d'Azur": {"coords": (6.23, 44.09)},
    "Roussillon–Languedoc": {"coords": (3.2629, 43.5714)},
    "Camargue": {"coords": (4.3983, 43.4938)},

    "Paris area": {"coords": (2.2328, 48.6707)},
    "Alsace": {"coords": (7.80, 48.5730)},                 # plus à l'est
    "Brittany": {"coords": (-2.76, 48.18)},
    "Dordogne / Limousin": {"coords": (1.1610, 45.8142)},

    "Ardèche": {"coords": (4.4626, 44.7609)},
    "Var": {"coords": (5.93, 43.12)},
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
proj_map  = ccrs.LambertConformal(central_longitude=10, central_latitude=47,
                                  standard_parallels=(30, 60))

fig = plt.figure(figsize=(6, 5))
ax = plt.axes(projection=proj_map)

ax.coastlines("10m", linewidth=0.6, zorder=10)
ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.3,
               edgecolor="black", zorder=11)

# tracer polygone du domaine
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

# 2.1 Construire une liste (lon, lat, name)
entries = []
for name, info in zones_centers.items():
    lon, lat = info["coords"]
    entries.append({"name": name, "lon": lon, "lat": lat})

# 2.2 Trier N->S puis O->E
entries_sorted = sorted(
    entries,
    key=lambda e: (-e["lat"], e["lon"])   # lat décroissant, lon croissant
)

# 2.3 Assigner les lettres A, B, C...
letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
for i, e in enumerate(entries_sorted):
    e["letter"] = letters[i]

circle_color = "#1f4f7b"

# 2.4 Tracer les pastilles sur la carte
for e in entries_sorted:
    x, y = e["lon"], e["lat"]

    ax.scatter(
        x, y,
        s=70,                     # taille en points^2 (ajuste si besoin)
        color=circle_color,
        edgecolor="white",
        linewidth=0.8,
        transform=proj_data,      # très important avec Cartopy
        zorder=15,
    )

    ax.text(
        x, y,
        e["letter"],
        fontsize=7,
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
        transform=proj_data,
        zorder=16,
    )

# 2.5 Légende à droite avec pastille + lettre
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
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),   # à droite de la carte
    frameon=False,
    fontsize=7,
    handlelength=1.4,
    borderaxespad=0.5,
    labelspacing=0.8,
)

# -------------------------------------------------------------------
# 3. Export
# -------------------------------------------------------------------
plt.savefig("figures/espace_definition.svg", dpi=600, bbox_inches="tight", pad_inches=0)
plt.savefig("figures/espace_definition.pdf", bbox_inches="tight", pad_inches=0)
plt.show()
