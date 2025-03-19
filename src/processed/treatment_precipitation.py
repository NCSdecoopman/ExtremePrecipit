def convert_precipitation(ds):
    """Convertit la précipitation de kg/m²/s en mm/h."""
    ds["pr_mm_h"] = ds["pr"] * 3600
    ds["pr_mm_h"].attrs["units"] = "mm/h"
    return ds