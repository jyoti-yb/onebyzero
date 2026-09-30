"""GFS APCP daily-rainfall construction: primary, QA, GRIB-metadata proofs, provenance.

Fixtures are genuine GRIB2 messages encoded with ecCodes (PDT 4.8, centre kwbc,
north->south regular_ll grid like GFS pgrb2), so the metadata checks run on real keys.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ec = pytest.importorskip("eccodes")

from monsoonpp.adapters.gfs_precip import (APCPIntervalError, APCPMetadataError, APCPValidationError,
                                           GFSDailyPrecip, Tolerances, _hours, _parse_step_range,
                                           read_apcp_messages, verify_units)
from monsoonpp.data.align import Window, forecast_window, gfs_apcp_primary, gfs_apcp_qa_pieces

INIT = pd.Timestamp("2023-07-15 00:00")
LAT = np.round(np.arange(10.0, 12.001, 0.25), 4)        # ascending (we write north->south)
LON = np.round(np.arange(70.0, 72.501, 0.25), 4)
SHAPE = (LAT.size, LON.size)

# which intervals live in which forecast-hour file (GFS pgrb2 layout: bucket + running total)
LAYOUT = {3: [(0, 3)], 6: [(0, 6)], 9: [(6, 9), (0, 9)], 12: [(6, 12), (0, 12)],
          18: [(12, 18), (0, 18)], 24: [(18, 24), (0, 24)], 27: [(24, 27), (0, 27)]}


def hourly_rates(seed=0):
    rng = np.random.default_rng(seed)
    r = rng.gamma(0.4, 2.0, (28, *SHAPE))          # mm/h, heavy tailed
    r[:, :2, :2] = 0.0                               # permanently dry corner
    r[:, 4, 5] = 28.0                                # extreme cell: 672 mm in 24 h
    return r


def accum(r, a, b):
    return r[a:b].sum(0)


def grib_msg(values, a, b, init=INIT, param=(0, 1, 8), tsp=1, overrides=None) -> bytes:
    g = ec.codes_grib_new_from_samples("GRIB2")
    ec.codes_set(g, "centre", 7)
    for k, v in dict(Ni=LON.size, Nj=LAT.size, latitudeOfFirstGridPointInDegrees=float(LAT[-1]),
                     latitudeOfLastGridPointInDegrees=float(LAT[0]), longitudeOfFirstGridPointInDegrees=float(LON[0]),
                     longitudeOfLastGridPointInDegrees=float(LON[-1]), iDirectionIncrementInDegrees=0.25,
                     jDirectionIncrementInDegrees=0.25).items():
        ec.codes_set(g, k, v)
    ec.codes_set(g, "productDefinitionTemplateNumber", 8)
    ec.codes_set(g, "discipline", param[0]); ec.codes_set(g, "parameterCategory", param[1])
    ec.codes_set(g, "parameterNumber", param[2])
    ec.codes_set(g, "typeOfFirstFixedSurface", 1)
    ec.codes_set(g, "dataDate", int(init.strftime("%Y%m%d"))); ec.codes_set(g, "dataTime", init.hour * 100)
    ec.codes_set(g, "typeOfStatisticalProcessing", tsp)
    ec.codes_set(g, "stepUnits", 1)
    ec.codes_set(g, "startStep", a); ec.codes_set(g, "endStep", b)
    ec.codes_set(g, "bitsPerValue", 16)
    for k, v in (overrides or {}).items():
        ec.codes_set(g, k, v)
    ec.codes_set_values(g, np.ascontiguousarray(values[::-1, :]).ravel())   # north -> south
    m = ec.codes_get_message(g)
    ec.codes_release(g)
    return m


def write_run(d: Path, r, mutate=None) -> dict[int, Path]:
    """mutate(fxx, a, b, values) -> values | None (drop) | list[(a,b,values,kwargs)] (replace)."""
    paths = {}
    for fxx, ivs in LAYOUT.items():
        msgs = [grib_msg(np.full(SHAPE, 1.0), 0, 0, param=(0, 1, 52), tsp=0)]   # non-APCP distractor
        for a, b in ivs:
            vals = accum(r, a, b)
            out = mutate(fxx, a, b, vals) if mutate else vals
            if out is None:
                continue
            if isinstance(out, list):
                msgs += [grib_msg(v, aa, bb, **kw) for v, aa, bb, kw in out]
            else:
                msgs.append(grib_msg(out, a, b))
        p = d / f"gfs.t00z.pgrb2.0p25.f{fxx:03d}.grib2"
        p.write_bytes(b"".join(msgs))
        paths[fxx] = p
    return paths


def builder(paths, **kw):
    return GFSDailyPrecip(lambda init, fxx: paths[fxx], **kw)


W1 = forecast_window(pd.Timestamp("2023-07-15"), 1, "starting")


# ------------------------------------------------------------------- happy path
def test_window_formulas():
    assert W1 == Window(INIT, 3, 27)
    assert gfs_apcp_primary(W1) == [(27, 0, 27, +1), (3, 0, 3, -1)]
    assert gfs_apcp_qa_pieces(W1) == [(6, 0, 6, +1), (3, 0, 3, -1), (12, 6, 12, +1), (18, 12, 18, +1),
                                      (24, 18, 24, +1), (27, 24, 27, +1)]


def test_primary_matches_truth_cell_by_cell_and_qa(tmp_path):
    r = hourly_rates()
    res = builder(write_run(tmp_path, r)).build(W1, 1)
    truth = accum(r, 3, 27)
    q = res.provenance["packing_quantum_mm"]
    assert res.field.shape == SHAPE and np.allclose(res.field.values, truth, atol=4 * q)
    assert float(res.field.min()) >= 0.0 and float(res.field.max()) > 600
    assert res.qa["meaningful_negative_cells"] == 0
    assert res.qa["monotonic_violation_frac"] == 0.0
    assert res.qa["monotonic_chain"] == ["0-3", "0-6", "0-12", "0-18", "0-24", "0-27"]
    assert res.qa["qa_max_abs_mm"] < 0.5 and res.qa["qa_frac_below_max"] == 1.0
    assert res.qa["qa_sources"] == ["+0-6", "-0-3", "+6-12", "+12-18", "+18-24", "+24-27"]


def test_provenance(tmp_path):
    res = builder(write_run(tmp_path, hourly_rates())).build(W1, 1)
    p, a = res.provenance, res.field.attrs
    assert p["model"] == "GFS" and p["cycle"] == "00Z" and p["forecast_lead_days"] == 1
    assert p["init_time"] == "2023-07-15 00:00:00"
    assert p["valid_start"] == "2023-07-15 03:00:00" and p["valid_end"] == "2023-07-16 03:00:00"
    assert p["source_apcp_step_ranges"] == "+0-27 -0-3"
    assert p["source_apcp_grib_stepRange"] == "0-27|0-3"
    assert p["source_units"] == "kg m**-2" and p["output_units"] == "mm" and a["units"] == "mm"
    assert p["construction_method"].startswith("APCP(0-27) - APCP(0-3)")
    assert "APCP(0-6) - APCP(0-3)" in p["qa_method"]
    for k in ("model", "cycle", "init_time", "forecast_lead_days", "valid_start", "valid_end",
              "source_apcp_step_ranges", "source_units", "output_units", "construction_method"):
        assert k in a


def test_non_apcp_messages_ignored_and_metadata_decoded(tmp_path):
    paths = write_run(tmp_path, hourly_rates())
    msgs = read_apcp_messages(paths[27])
    assert sorted(m.interval for m in msgs) == ["0-27", "24-27"]      # distractor (0/1/52) skipped
    m = [m for m in msgs if m.interval == "0-27"][0]
    assert m.short_name == "tp" and m.param == (0, 1, 8) and m.step_type == "accum"
    assert m.init == INIT and m.valid_end == pd.Timestamp("2023-07-16 03:00") and m.units == "kg m**-2"
    assert m.lat[0] < m.lat[-1]                                        # flipped to ascending


# ----------------------------------------------------------- fail-loud metadata
def test_interval_proven_by_grib_not_filename(tmp_path):
    # the file labelled f027 does not contain a 0-27 message (it holds 0-24 instead)
    def mut(fxx, a, b, v):
        if (fxx, a, b) == (27, 0, 27):
            return [(v, 0, 24, {})]
        return v
    with pytest.raises(APCPIntervalError, match="no APCP 0-27"):
        builder(write_run(tmp_path, hourly_rates(), mut)).build(W1, 1)


def test_missing_interval(tmp_path):
    mut = lambda fxx, a, b, v: None if (a, b) == (0, 3) else v
    with pytest.raises(APCPIntervalError, match="no APCP 0-3"):
        builder(write_run(tmp_path, hourly_rates(), mut)).build(W1, 1)


def test_ambiguous_duplicate(tmp_path):
    def mut(fxx, a, b, v):
        if (a, b) == (0, 27):
            return [(v, 0, 27, {}), (v + 5.0, 0, 27, {})]
        return v
    with pytest.raises(APCPIntervalError, match="ambiguous"):
        builder(write_run(tmp_path, hourly_rates(), mut)).build(W1, 1)


def test_wrong_step_type_rejected(tmp_path):
    def mut(fxx, a, b, v):
        if (a, b) == (0, 27):
            return [(v, 0, 27, {"tsp": 0})]          # average, not accumulation
        return v
    with pytest.raises(APCPMetadataError, match="stepType|typeOfStatisticalProcessing"):
        builder(write_run(tmp_path, hourly_rates(), mut)).build(W1, 1)


def test_init_time_mismatch(tmp_path):
    def mut(fxx, a, b, v):
        if (a, b) == (0, 3):
            return [(v, 0, 3, {"init": INIT - pd.Timedelta(days=1)})]
        return v
    with pytest.raises(APCPMetadataError, match="init"):
        builder(write_run(tmp_path, hourly_rates(), mut)).build(W1, 1)


def test_reference_time_significance_checked(tmp_path):
    def mut(fxx, a, b, v):
        if (a, b) == (0, 27):
            return [(v, 0, 27, {"overrides": {"significanceOfReferenceTime": 0}})]   # analysis, not forecast start
        return v
    with pytest.raises(APCPMetadataError, match="reference time"):
        builder(write_run(tmp_path, hourly_rates(), mut)).build(W1, 1)


def test_units_and_time_units():
    assert verify_units("kg m**-2") == "kg m**-2"
    with pytest.raises(APCPMetadataError):
        verify_units("kg m**-2 s**-1")
    assert _hours(4, 10, "x") == 12 and _hours(1, 2, "x") == 24 and _hours(180, 0, "x") == 3
    with pytest.raises(APCPMetadataError):
        _hours(1, 99, "x")
    with pytest.raises(APCPMetadataError):
        _hours(20, 0, "x")                     # 20 min: not whole hours
    assert _parse_step_range("0-27", 1) == (0, 27) and _parse_step_range("0-27h", 1) == (0, 27)
    with pytest.raises(APCPMetadataError):
        _parse_step_range("27", 1)             # instantaneous step, not a range


# -------------------------------------------------------- physical / QA checks
def test_meaningful_negative_or_nonmonotonic_rejected(tmp_path):
    def mut(fxx, a, b, v):
        if (a, b) == (0, 3):
            v = v.copy(); v[6, 6] += 50.0          # 0-3 total larger than 0-27 in one cell
        return v
    with pytest.raises(APCPValidationError, match="decreases|< -"):
        builder(write_run(tmp_path, hourly_rates(), mut)).build(W1, 1)


def test_negative_check_independent_of_monotonic_chain(tmp_path):
    # tolerances wide for monotonicity so the negative-rain check is what fires
    def mut(fxx, a, b, v):
        if (a, b) == (0, 3):
            v = v.copy(); v[6, 6] += 50.0
        return v
    b = builder(write_run(tmp_path, hourly_rates(), mut), tol=Tolerances(monotonic_mm=1e9), run_qa=False)
    with pytest.raises(APCPValidationError, match="daily rain < -"):
        b.build(W1, 1)


def test_qa_disagreement_rejected(tmp_path):
    def mut(fxx, a, b, v):
        if (a, b) == (12, 18):
            v = v.copy(); v[3, 3] += 3.0           # bucket inconsistent with running totals
        return v
    with pytest.raises(APCPValidationError, match="primary vs QA"):
        builder(write_run(tmp_path, hourly_rates(), mut)).build(W1, 1)


def test_window_must_be_03z_to_next_03z(tmp_path):
    paths = write_run(tmp_path, hourly_rates())
    with pytest.raises(APCPValidationError, match="cycle 06Z"):
        builder(paths).check_window(Window(INIT + pd.Timedelta(hours=6), 3, 27),
                                    INIT + pd.Timedelta(hours=9), INIT + pd.Timedelta(hours=33))
    with pytest.raises(APCPValidationError, match="03Z"):
        builder(paths).check_window(W1, INIT + pd.Timedelta(hours=6), INIT + pd.Timedelta(hours=30))
    with pytest.raises(ValueError):
        gfs_apcp_qa_pieces(Window(INIT, 6, 30))


# ------------------------------------------------------------ adapter integration
def fake_atmos_day():
    import xarray as xr
    from monsoonpp.adapters.gfs import DERIVED_ATMOS_FIELDS
    from monsoonpp.adapters.gfs_atmos import FIELD_SPECS
    names = [*FIELD_SPECS, *DERIVED_ATMOS_FIELDS]
    ds = xr.Dataset({name: (("lat", "lon"), np.zeros(SHAPE, dtype="float32")) for name in names},
                    coords={"lat": LAT, "lon": LON})
    return ds, {"source_files": ["fixture.grib2"], "sample_valid_times": ["2023-07-15 06:00:00"],
                "field_validation": []}


def test_adapter_uses_primary_and_carries_provenance(tmp_path, monkeypatch):
    from monsoonpp.adapters.gfs import GFSAdapter
    from monsoonpp.config import load_config
    r = hourly_rates()
    paths = write_run(tmp_path, r)
    cfg = load_config(None, forecast_source="gfs", raw_dir=str(tmp_path / "raw"))
    cfg.grid.lat_min, cfg.grid.lat_max, cfg.grid.lon_min, cfg.grid.lon_max = 10.0, 12.0, 70.0, 72.5
    ad = GFSAdapter(cfg)
    monkeypatch.setattr(ad, "_fetch_apcp_file", lambda init, fxx: paths[fxx])
    monkeypatch.setattr(ad, "_load_atmos_day", lambda window: fake_atmos_day())
    ds = ad.load(pd.DatetimeIndex(["2023-07-15"]), [1])
    tp = ds.tp.sel(lead=1).isel(time=0).values
    assert np.allclose(tp, accum(r, 3, 27), atol=0.05)
    assert ds.tp_valid_start.values[0, 0] == "2023-07-15 03:00:00"
    assert ds.tp_valid_end.values[0, 0] == "2023-07-16 03:00:00"
    assert ds.tp_source_apcp_step_ranges.values[0, 0] == "+0-27 -0-3"
    assert float(ds.tp_qa_max_abs_mm.values[0, 0]) < 0.5
    assert ds.tp.attrs["model"] == "GFS" and ds.tp.attrs["cycle"] == "00Z"
    assert ds.tp.attrs["source_units"] == "kg m**-2" and ds.tp.attrs["output_units"] == "mm"
    assert ds.tp.attrs["construction_method"].startswith("APCP(0-f_end) - APCP(0-f_start)")


def test_adapter_fails_loudly_on_bad_metadata(tmp_path, monkeypatch):
    from monsoonpp.adapters.gfs import GFSAdapter
    from monsoonpp.config import load_config
    paths = write_run(tmp_path, hourly_rates(), lambda fxx, a, b, v: None if (a, b) == (0, 27) else v)
    cfg = load_config(None, forecast_source="gfs", raw_dir=str(tmp_path / "raw"))
    cfg.grid.lat_min, cfg.grid.lat_max, cfg.grid.lon_min, cfg.grid.lon_max = 10.0, 12.0, 70.0, 72.5
    ad = GFSAdapter(cfg)
    monkeypatch.setattr(ad, "_fetch_apcp_file", lambda init, fxx: paths[fxx])
    monkeypatch.setattr(ad, "_load_atmos_day", lambda window: fake_atmos_day())
    with pytest.raises(APCPIntervalError):
        ad.load(pd.DatetimeIndex(["2023-07-15"]), [1])
