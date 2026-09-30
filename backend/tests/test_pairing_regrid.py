"""Window pairing, conservative regridding, IMERG 03Z->03Z construction (unit level)."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monsoonpp.data.imerg_daily import IMERGDailyError, imerg_daily_03z
from monsoonpp.data.obs_source import ObsSourceMeta, ProductRole, attach, read
from monsoonpp.data.obs_time import ObsTimeConventionError
from monsoonpp.data.pairing import PairingError, align_to_windows, match_windows, obs_labels_for_windows
from monsoonpp.data.regrid import RegridError, conservative_regrid, overlap_matrix, regular_edges, strict_select

T = pd.Timestamp
H = pd.Timedelta(hours=1)


# ------------------------------------------------------------------ pairing
def test_obs_labels_for_windows_both_conventions():
    s = pd.DatetimeIndex(["2024-07-15 03:00", "2024-07-16 03:00"]); e = s + 24 * H
    assert list(obs_labels_for_windows(s, e, "STARTING_03Z").strftime("%Y-%m-%d")) == ["2024-07-15", "2024-07-16"]
    assert list(obs_labels_for_windows(s, e, "ENDING_03Z").strftime("%Y-%m-%d")) == ["2024-07-16", "2024-07-17"]
    with pytest.raises(ObsTimeConventionError):
        obs_labels_for_windows(s, e, "UNVERIFIED")
    with pytest.raises(PairingError, match="not representable"):
        obs_labels_for_windows(s + 3 * H, e + 3 * H, "ENDING_03Z")        # 06Z windows


def test_match_windows_exact_and_duplicates():
    cs = pd.DatetimeIndex(["2024-07-15 03:00", "2024-07-16 03:00"])
    assert list(match_windows(cs[::-1], cs[::-1] + 24 * H, cs, cs + 24 * H)) == [1, 0]
    assert list(match_windows(cs, cs + 23 * H, cs, cs + 24 * H)) == [-1, -1]       # end must match too
    with pytest.raises(PairingError, match="identical window"):
        match_windows(cs, cs + 24 * H, cs.append(cs[:1]), (cs + 24 * H).append(cs[:1] + 24 * H))


def test_align_never_uses_equal_labels():
    """obs labelled 15,16,17 under ENDING_03Z cover windows 14/15/16 -> forecast label 16 pairs obs label 17."""
    labels = pd.DatetimeIndex(["2024-07-15", "2024-07-16", "2024-07-17"])
    obs = xr.Dataset({"rain": (("time",), [1.0, 2.0, 3.0])},
                     coords={"time": labels, "source_date_label": ("time", np.array(list(labels.strftime("%Y-%m-%d")))),
                             "valid_start": ("time", (labels - 21 * H).values), "valid_end": ("time", (labels + 3 * H).values)})
    f_lab = pd.DatetimeIndex(["2024-07-15", "2024-07-16", "2024-07-17"])
    al, ok = align_to_windows(obs, obs.valid_start.values, obs.valid_end.values, f_lab + 3 * H, f_lab + 27 * H, f_lab)
    assert list(ok) == [True, True, False]
    assert al.rain.values[0] == 2.0 and str(al.source_date_label.values[0]) == "2024-07-16"
    assert al.rain.values[1] == 3.0 and str(al.source_date_label.values[1]) == "2024-07-17"
    assert np.isnan(al.rain.values[2]) and str(al.source_date_label.values[2]) == ""   # missing stays missing


# ------------------------------------------------------------------ regrid
SRC_LAT = np.round(np.arange(10.05, 12.5, 0.1), 3)      # edges 10.0 .. 12.5
SRC_LON = np.round(np.arange(70.05, 72.5, 0.1), 3)
DST_LAT = np.arange(10.125, 12.4, 0.25)                 # edges 10.0 .. 12.5
DST_LON = np.arange(70.125, 12.4 + 60, 0.25)


def src_field(seed=0):
    rng = np.random.default_rng(seed)
    return xr.DataArray(rng.gamma(0.8, 10, (2, SRC_LAT.size, SRC_LON.size)), dims=("time", "lat", "lon"),
                        coords={"time": [0, 1], "lat": SRC_LAT, "lon": SRC_LON})


def cell_area(lat, lon):
    el, en = regular_edges(lat), regular_edges(lon)
    return np.outer(np.diff(np.sin(np.deg2rad(el))), np.diff(en))


def test_constant_preserved_and_integral_conserved():
    d = src_field()
    dst = conservative_regrid(d, DST_LAT, DST_LON)
    assert dst.shape == (2, DST_LAT.size, DST_LON.size) and np.isfinite(dst.values).all()
    for t in range(2):
        a = np.sum(d.values[t] * cell_area(SRC_LAT, SRC_LON))
        b = np.sum(dst.values[t] * cell_area(DST_LAT, DST_LON))
        assert b == pytest.approx(a, rel=1e-5)
    c = xr.full_like(d, 7.5)
    assert np.allclose(conservative_regrid(c, DST_LAT, DST_LON).values, 7.5)


def test_regular_edges_accepts_float32_quantized_lattice_only():
    imerg_like = np.linspace(-179.95, 179.95, 3600).astype("float32")
    edges = regular_edges(imerg_like)
    assert edges.size == imerg_like.size + 1
    distorted = imerg_like.astype("float64")
    distorted[100] += 1e-3
    with pytest.raises(RegridError, match="uniform spacing"):
        regular_edges(distorted)


def test_partial_overlap_weights_by_hand():
    d = src_field(1).isel(time=0)
    dst = conservative_regrid(d, DST_LAT, DST_LON).values
    # dst cell (0,0): lat [10.0,10.25], lon [70.0,70.25]; src cells 0,1 full, cell 2 half
    wl = np.array([np.sin(np.deg2rad(b)) - np.sin(np.deg2rad(a)) for a, b in ((10.0, 10.1), (10.1, 10.2), (10.2, 10.25))])
    wn = np.array([0.1, 0.1, 0.05])
    w = np.outer(wl, wn)
    exp = np.sum(w * d.values[:3, :3]) / w.sum()
    assert dst[0, 0] == pytest.approx(exp, rel=1e-5)
    assert overlap_matrix(np.array([0, .1, .2, .3]), np.array([0, .25]))[0].tolist() == pytest.approx([.1, .1, .05])


def test_missing_stays_missing_or_partial_coverage():
    d = src_field().isel(time=0).copy()
    d.values[0, 0] = np.nan
    strict = conservative_regrid(d, DST_LAT, DST_LON)
    assert np.isnan(strict.values[0, 0]) and np.isfinite(strict.values[1:, 1:]).all()
    loose = conservative_regrid(d, DST_LAT, DST_LON, min_coverage=0.5)
    assert np.isfinite(loose.values[0, 0])
    wider = conservative_regrid(d, np.append(DST_LAT, 12.625), DST_LON)          # footprint beyond source extent
    assert np.isnan(wider.values[-1]).all()


def test_strict_select_refuses_interpolation():
    d = xr.DataArray(np.zeros((3, 3)), dims=("lat", "lon"), coords={"lat": [10, 10.25, 10.5], "lon": [70, 70.25, 70.5]})
    assert strict_select(d, np.array([10.25]), np.array([70.5])).shape == (1, 1)
    with pytest.raises(RegridError):
        strict_select(d, np.array([10.125]), np.array([70.0]))


# -------------------------------------------------------- IMERG daily build
def native(start="2024-07-15 03:00", n=48, rate=lambda k: 1.0 + k, skip=(), nan_at=None, shift=pd.Timedelta(0)):
    st = pd.DatetimeIndex([T(start) + pd.Timedelta(minutes=30 * k) + shift for k in range(n) if k not in skip])
    ks = [k for k in range(n) if k not in skip]
    data = np.stack([np.full((4, 5), rate(k), "f4") for k in ks])
    if nan_at is not None:
        data[nan_at[0], nan_at[1], nan_at[2]] = np.nan
    ds = xr.Dataset({"precipitation_rate": (("time", "lat", "lon"), data)},
                    coords={"time": st, "lat": np.arange(10.05, 10.45, 0.1), "lon": np.arange(70.05, 70.55, 0.1),
                            "source_interval_start": ("time", st.values),
                            "source_interval_end": ("time", (st + pd.Timedelta(minutes=30)).values)})
    return attach(ds, ObsSourceMeta("NASA-GPM-IMERG", "NASA GPM IMERG half-hourly", ProductRole.SMOKE_TEST_PROXY,
                                    "f", "o", "0.1 deg", "g", "mm/hr", "EXPLICIT_SOURCE_INTERVALS", True))


def test_exact_48_granule_sum():
    d, rej = imerg_daily_03z(native(), [T("2024-07-15 03:00")])
    assert rej == [] and int(d.n_granules.values[0]) == 48
    assert np.allclose(d.rain.values[0], sum((1.0 + k) * 0.5 for k in range(48)))
    assert T(d.valid_start.values[0]) == T("2024-07-15 03:00") and T(d.valid_end.values[0]) == T("2024-07-16 03:00")
    m = read(d)
    assert m.product_role is ProductRole.SMOKE_TEST_PROXY and m.units == "mm" and "IMD" not in m.product_name


def test_gap_rejects_window_and_nan_propagates():
    with pytest.raises(IMERGDailyError, match="no complete"):
        imerg_daily_03z(native(skip=(17,)), [T("2024-07-15 03:00")])
    two = xr.concat([native(), native("2024-07-16 03:00", skip=(5,))], "time")
    two = two.assign_attrs(native().attrs)
    d, rej = imerg_daily_03z(two, [T("2024-07-15 03:00"), T("2024-07-16 03:00")])
    assert d.sizes["time"] == 1 and rej[0]["granules_found"] == 47 and "gap" in rej[0]["reason"]
    d2, _ = imerg_daily_03z(native(nan_at=(10, 2, 3)), [T("2024-07-15 03:00")])
    assert np.isnan(d2.rain.values[0, 2, 3]) and np.isfinite(d2.rain.values[0, 0, 0])


def test_window_must_start_03z_and_granules_on_lattice():
    with pytest.raises(IMERGDailyError, match="not 03:00"):
        imerg_daily_03z(native(), [T("2024-07-15 00:00")])
    with pytest.raises(IMERGDailyError, match="not aligned"):
        imerg_daily_03z(native(shift=pd.Timedelta(minutes=15)), [T("2024-07-15 03:00")])
