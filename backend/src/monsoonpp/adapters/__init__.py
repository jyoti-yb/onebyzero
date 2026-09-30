from .base import AnalysisAdapter, ForecastAdapter, NativeObsAdapter, ObsAdapter, get_adapter, register  # noqa: F401
from . import synthetic, gfs, imd, era5, ncum, obs_netcdf, imerg  # noqa: F401  (registers adapters)
