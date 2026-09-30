import pandas as pd
import pytest

from monsoonpp.tools.fetch_imerg import granule_query_window


def test_month_query_covers_exact_03z_windows():
    first, last, count = granule_query_window("2024-07-01", "2024-07-31")
    assert first == pd.Timestamp("2024-07-01 03:00:00")
    assert last == pd.Timestamp("2024-08-01 02:59:59")
    assert count == 31 * 48


def test_query_rejects_reversed_dates():
    with pytest.raises(ValueError, match="end must not precede start"):
        granule_query_window("2024-07-02", "2024-07-01")
