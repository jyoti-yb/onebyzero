# Regime-Aware AI Post-Processing of Monsoon Rainfall Forecasts

Initial repository for the SIH meteorological forecasting pipeline.

This stage implements a **7-day smoke-test scaffold only**. It does not train an ML model, download multi-year data, or assume any external dataset URLs. The smoke test is intended to validate that local forecast and observation files can pass through ingestion, GRIB inspection, precipitation accumulation, temporal pairing, grid alignment, quality checks, and basic verification.

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run_smoke_test.py --config config/smoke_test.yaml
```

The command expects local 7-day input files under:

- `data/raw/gfs/` for forecast GRIB files
- `data/raw/imd/` for observation files

To explicitly download one geographically subsetted GFS file from NOAA NOMADS:

```bash
python -m src.download_gfs --date YYYYMMDD --forecast-hour 3
```

The command uses the configured cycle, domain, variables, pressure levels, raw
GFS directory, and manifest directory. It does not run automatically as part of
module import. Existing output files are not overwritten.

To inspect every local GRIB group and print APCP timing metadata:

```bash
python -m src.inspect_grib
```

The inspection is read-only and writes
`reports/smoke_test/02_grib_metadata.csv`.

`config/smoke_test.yaml` is the source of truth for the domain, GFS cycle and
lead, requested variables, rainfall window, and quality thresholds. The runner
validates the full configuration before opening data or creating outputs.

## Pipeline Stages

1. Validate the smoke-test configuration.
2. Inspect forecast GRIB metadata.
3. Build daily precipitation products from forecast accumulations.
4. Load and normalize local IMD observations.
5. Align forecast and observations in time and space.
6. Run quality checks on paired data.
7. Compute basic deterministic verification metrics.

## Design Note

Forecast ingestion is routed through a `ForecastAdapter` protocol in `src/download_gfs.py`. The current `GFSForecastAdapter` supports local discovery and explicit one-file NOMADS downloads; later, an `NCUMForecastAdapter` can implement the same interface without changing downstream processing modules.
