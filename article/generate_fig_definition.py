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

nc = Path("../../data/raw/modelised/pr_ALPX-3_ERA5_evaluation_r1i1p1f1_CNRM-MF_CNRM-AROME46t1_v1-r1_1hr_195901010030-195912312330.nc")
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
ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.3, edgecolor="black", zorder=11)

# tracer polygone du domaine
def plot_poly(p: Polygon):
    xs, ys = p.exterior.xy
    ax.plot(xs, ys, transform=proj_data, linewidth=2, color="black", zorder=12)
    for interior in p.interiors:
        xi, yi = interior.xy
        ax.plot(xi, yi, transform=proj_data, linewidth=1, color="black", zorder=12)

if isinstance(u, MultiPolygon):
    for g in u.geoms: plot_poly(g)
else:
    plot_poly(u)

# emprise
all_x = np.concatenate([np.array(g.exterior.coords)[:,0] for g in (u.geoms if isinstance(u, MultiPolygon) else [u])])
all_y = np.concatenate([np.array(g.exterior.coords)[:,1] for g in (u.geoms if isinstance(u, MultiPolygon) else [u])])
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

plt.savefig("espace_definition.svg", dpi=600, bbox_inches="tight", pad_inches=0)
plt.savefig("espace_definition.pdf", bbox_inches="tight", pad_inches=0)