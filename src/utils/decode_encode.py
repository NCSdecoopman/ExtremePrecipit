import xarray as xr
import numpy as np

def decode_var(arr: xr.DataArray, var_conf: dict) -> xr.DataArray:
    """Returns a float32 DataArray in actual units, with NaN for missing values."""
    arr = arr.astype("float32")

    fill_val = var_conf.get("fill_value", None)
    if fill_val is not None:
        arr = arr.where(arr != fill_val, np.nan)

    scale = var_conf.get("scale_factor", 1.0)
    if scale not in (None, 0, 1):
        arr = arr / scale

    return arr


def encode_var(arr: xr.DataArray, var_conf: dict) -> xr.DataArray:
    """Applies scale, fill_value, and dtype before writing."""
    fill_val = var_conf.get("fill_value", -9999)
    scale = var_conf.get("scale_factor", 1.0)

    arr = arr * scale         # revert to encoded physical steps
    arr = arr.round()
    arr = arr.fillna(fill_val)

    dtype = np.dtype(var_conf.get("dtype", "int16"))
    return arr.astype(dtype)
