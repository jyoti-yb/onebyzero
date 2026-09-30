# monsoonpp — regime-aware post-processing of monsoon rainfall forecasts

SIH / MoES–NCMRWF problem: identify the prevailing weather regime, then correct raw NWP rainfall accordingly.
Outputs are a bias-corrected rainfall grid, heavy-rain probabilities, a district product and a verification report
(RMSE, ETS, CSI, POD, FAR, FSS).

```
 NWP (GFS dev / NCUM ops) ─┐        IMD rain (truth)      ERA5 (regime labels)
            │ ForecastAdapter            │ ObsAdapter           │ AnalysisAdapter
            └──────────────► canonical schema (lead,time,lat,lon); obs labels interpreted only via a verified obs_time_convention
                                         │
                 ┌───────────────────────┴───────────────────────┐
          REGIME ENGINE (rules, 3 axes)                     FEATURES (forecast-only)
   monsoon state · synoptic system · local forcing      raw rain + neighbourhoods, winds,
                 │                                        moisture, upslope, onshore, terrain
                 └──► REGIME CLASSIFIER (forecast → analysis regime, probs)
                                         │
      L0 raw → L1 clim bias → L2 qmap → L3 GBM → L4 regime-aware GBM → L5 MoE → L6 regime-QM
                                         │                           + calibrated P(>64.5), P(>115.6)
          ┌──────────────────────────────┼───────────────────────────────┐
     VERIFICATION cube              DISTRICT product                EXPLANATIONS
 regime×region×lead×thr×scale   area-weighted, IMD category,     SHAP themes + historical
 + Error Atlas + drift check     warning colour                   analogues + confidence
                                         │
                                   FastAPI  /  CLI
```

## Quick start (offline, synthetic data)

```bash
pip install -e ".[dev]"
monsoonpp all -c configs/synthetic.yaml          # build → train → evaluate (~4 min, 2 cores)
monsoonpp forecast -c configs/synthetic.yaml --date 2024-07-20 --lead 1
monsoonpp serve -c configs/synthetic.yaml --port 8000   # http://localhost:8000/docs
pytest -q                                        # 13 tests, ~40 s
```

## Real data (run on your machine — the build sandbox had no network)

```bash
pip install -e ".[real]"        # herbie-data, cfgrib/eccodes, imdlib, cdsapi
# ~/.cdsapirc with your Copernicus key
python -m monsoonpp.adapters.gfs 2023-07-15      # smoke-test one GFS day first
monsoonpp all -c configs/gfs_imd_era5.yaml
```
Check before trusting any number:
1. **IMD date convention: UNVERIFIED (blocking).** Nobody has confirmed which 24 h period an IMD date label covers. Both real configs ship with `obs_time_convention: UNVERIFIED`. With that setting, building forecast/observation pairs, creating training samples, training and verification all stop with `ObsTimeConventionError`. Reading and decoding IMD files still works. Once the official product documentation has been checked, set `ENDING_03Z` (label D = D-1 03Z → D 03Z) or `STARTING_03Z` (label D = D 03Z → D+1 03Z). Never infer it from filenames, file years, other products or GFS timing. The old `adapter_options.imd_window` key is rejected. Pairing is still by date label, and a label whose observation window differs from the forecast window (for example under `ENDING_03Z`) is rejected, because convention-aware alignment isn't built yet. See `data/obs_time.py`.
2. **GFS daily rain** (`adapters/gfs_precip.py`). The primary construction is `APCP(0-27) - APCP(0-3)`, cell by cell on the native grid, for the 03Z → next-03Z window of a 00Z run. The 6-hourly bucket sum runs only as an independent QA check. Every APCP interval is proven from GRIB2 metadata (PDT 4.8), never from file names or the `.idx` text. Missing, ambiguous or inconsistent messages raise an error (`gfs_strict: true`). Provenance is kept as `tp_*` coordinates and `tp` attributes. The metadata was validated on real data for 00Z D+1 only; other leads use the same checks but are unvalidated.
3. **Atmosphere and geography** — real GFS runs validate PRMSL, PWAT, CAPE, and 850/700/500/200 hPa wind/RH/HGT directly from GRIB2 metadata. Real runs require NOAA ETOPO 2022 elevation-derived static fields and reject synthetic or unproven geography. Run `monsoonpp validate-phase-b -c configs/phase_b_2024_gfs_etopo.yaml` to write the validation report, JSON evidence, NetCDF snapshots, and representative maps.
4. **NCUM** — `adapters/ncum.py` varmap names are placeholders. The generic NetCDF adapter is tested end to end.

## Observation sources (`data/obs_source.py`)

| `obs_source` | Product | Role | is_proxy |
|---|---|---|---|
| `imd` | IMD 0.25° gauge, imdlib `.grd` | FINAL_TRUTH | false |
| `imd_netcdf` | IMD 0.25° gauge, official NetCDF | FINAL_TRUTH | false |
| `imd_ncmrwf_merged` | IMD–NCMRWF 0.25° merged gauge + satellite, NetCDF | OFFICIAL_ALTERNATIVE | false |
| `imerg` (kind `obs_native` only) | NASA GPM IMERG half-hourly HDF5, native 0.1° | SMOKE_TEST_PROXY | true |
| `synthetic` | CI fixture | SYNTHETIC_TEST | true |

- Every observation dataset carries `obs_meta_*` provenance, validated when it is written and again when it is read.
- IMERG can't be labelled IMD or truth, and it can't reach `build_dataset`.
- Every official source still decodes as `UNVERIFIED` and stays blocked until `obs_time_convention` is verified.
- Configure files through `adapter_options.obs: {files: [...], origin: "<url>"}`.

## Pairing and the smoke test
- Pairs are formed on exact `(valid_start, valid_end)` windows (`data/pairing.py`), never on equal labels. Under `ENDING_03Z`, forecast label D pairs with observation label D+1.
- Spatial alignment: same-lattice sources are picked cell for cell. IMERG goes 0.1°→0.25° by first-order conservative (area-weighted) remapping in `data/regrid.py`, and missing source area makes the target cell missing.
- IMERG daily = exactly 48 half-hourly granules whose time bounds are verified, 03Z→03Z. Days with gaps are rejected (`data/imerg_daily.py`).
- `monsoonpp smoke -c configs/smoke_2024w29_imerg.yaml` (or `_imd.yaml` once the convention is verified) writes the report, daily and pooled metrics, the paired NetCDF and maps to `data/<name>/reports/smoke/`. Every output is labelled with the observation role.

## Phase C: regimes before ML
- `monsoonpp phase-c -c configs/phase_c_2024w29_imerg.yaml` runs the objective regime proxies and Regime x Error Atlas without training correction ML.
- `monsoonpp phase-c2 -c configs/phase_c2_202407_imerg.yaml` expands the atlas and adds matched-day, day-block-bootstrap error-distribution comparisons; it still never trains correction ML.
- `monsoonpp phase-c3 -c configs/phase_c3_2024jjas_imerg.yaml` requires exact JJAS pairing, tracks unchanged proxy centres into independent systems, and adds system-block confidence intervals; it never trains correction ML.
- Active/normal/break requires at least 20 years of daily land-rainfall climatology and is reported as `not_assessed` when that evidence is absent.
- Low/depression outputs are explicitly NWP proxies, local forcing outputs are heuristic scores, and WD support is disabled unless a validated upper-level anomaly climatology and upstream track are supplied.
- Atlas rows preserve independent day counts, space-time cell counts, event counts, Bias/MAE/RMSE, POD/FAR/CSI/ETS, FSS, and low-sample warnings.

## Grids (see `grid.py`)
- **Canonical IMD grid:** 0.25°, 6.5–38.5°N × 66.5–100.0°E, 129 lat × 135 lon. It is the `GridConfig` default and is used by `gfs_imd_era5.yaml` and `ncum.yaml`. Only this grid is "the IMD grid".
- **Verification / land mask:** the `land` static variable. On real runs it comes from IMD's valid cells. Ocean and no-data cells stay in the grid and are masked, never cropped away.
- **Runtime domain:** whatever `cfg.grid` is. `grid_role()` returns `imd_canonical`, `imd_crop` (an aligned subset on the 0.25° lattice) or `non_imd`. With `obs_source: imd`, a `non_imd` grid is rejected at build time, because IMD rain must never be interpolated. `synthetic.yaml` is a cropped, coarsened 0.5° development domain (63 × 63), not IMD.

## API

| Endpoint | Returns |
|---|---|
| `GET /forecast/grid?date&lead&model` | corrected + raw grid, P(>64.5), P(>115.6), regime map |
| `GET /forecast/districts?date&lead&warning=` | district table: mean/max rain, IMD category, probabilities, warning colour |
| `GET /forecast/point?lat&lon&date&lead` | all ladder levels, probabilities, drivers, 20 analogues, confidence |
| `GET /regimes?date&lead` | monsoon state, detected lows/depressions, regime shares |
| `GET /verification?section=` | headline / overall / categorical / fss / prob / by_regime / by_region / atlas / drift |
| `GET /atlas?lead&regime` | Regime × Error Atlas rows |
| `GET /model-card`, `/available`, `/health` | registry metadata |

## Layout

```
src/monsoonpp/
  config.py schema.py grid.py
  adapters/   base · synthetic · gfs · imd · era5 · ncum(+generic NetCDF)
  data/       align (IMD window, GFS buckets, regrid) · build (aligned cube)
  regimes/    engine (3-axis rules) · classifier (LightGBM, cross-fitted)
  features.py
  models/     baselines (L0–L2) · gbm (L3–L5, MoE) · restore (L6) · probability (heads)
  verify/     metrics (POD/FAR/CSI/ETS/FSS/Brier) · report · atlas (+drift/PSI)
  products/   district · explain · service
  api/app.py  cli.py  pipeline.py
```

## Design rules enforced in code
- Predictors come **only** from the forecast and static fields. Obs and analysis are used only as targets and labels. There is a test for this, and a parity test proves the serving path (obs stripped) reproduces the offline predictions.
- Splits are **year-blocked**: train 2021–22, val 2023, test 2024. Regime-classifier probabilities for training rows are cross-fitted by year.
- Every level must beat the one below it on the test year, or it doesn't earn its place.
- `nwp_model_version` is stored in the model card. The drift check (PSI + per-regime bias shift) flags when to retrain.
