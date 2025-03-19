def pr_nan(ds):
    """
    Calcule le pourcentage de valeurs NaN dans la variable de précipitation 'pr_mm_h'.
    
    Arguments:
    ds -- xarray.Dataset contenant la variable de précipitation
    
    Retourne:
    float -- pourcentage de NaN
    """
    pr = ds["pr_mm_h"]
    nan_percentage = pr.isnull().sum().compute().values / pr.size * 100
    return nan_percentage
