import xarray as xr

ds = xr.open_zarr("data/processed/modelised/zarr/horaire/1959.zarr")
print(ds)
