import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

from src.inspect_grib import (
    _metadata_rows_for_group,
    _write_csv_report,
    print_apcp_metadata,
)


class FakeCoordinate:
    def __init__(self, values: object) -> None:
        self.values = values


class FakeVariable:
    def __init__(self) -> None:
        self.dims = ("isobaricInhPa", "latitude", "longitude")
        self.attrs = {
            "GRIB_shortName": "APCP",
            "GRIB_units": "kg m-2",
            "GRIB_stepType": "accum",
            "GRIB_stepRange": "0-3",
            "GRIB_typeOfLevel": "isobaricInhPa",
        }


class FakeDataset:
    def __init__(self) -> None:
        self.sizes = {"isobaricInhPa": 1, "latitude": 2, "longitude": 2}
        self.coords = {
            "time": FakeCoordinate(datetime(2026, 9, 26, 0, 0)),
            "step": FakeCoordinate(timedelta(hours=3)),
            "valid_time": FakeCoordinate(datetime(2026, 9, 26, 3, 0)),
            "isobaricInhPa": FakeCoordinate(850),
            "latitude": FakeCoordinate([6.5, 6.75]),
            "longitude": FakeCoordinate([66.5, 66.75]),
        }
        self.data_vars = {"tp": FakeVariable()}
        self.attrs = {"GRIB_edition": 2}


def test_mocked_apcp_metadata_is_extracted(tmp_path: Path, capsys) -> None:
    rows = _metadata_rows_for_group(Path("sample.grib2"), 0, FakeDataset())

    assert len(rows) == 1
    row = rows[0]
    assert row["variable"] == "tp"
    assert row["GRIB_shortName"] == "APCP"
    assert row["GRIB_units"] == "kg m-2"
    assert row["GRIB_stepType"] == "accum"
    assert row["GRIB_stepRange"] == "0-3"
    assert json.loads(row["forecast_reference_time"]) == ["2026-09-26T00:00:00"]
    assert json.loads(row["valid_time"]) == ["2026-09-26T03:00:00"]
    assert json.loads(row["forecast_hour"]) == [3]
    assert json.loads(row["pressure_level_hpa"]) == [850]

    report_path = _write_csv_report(rows, tmp_path / "02_grib_metadata.csv")
    with report_path.open(newline="", encoding="utf-8") as handle:
        written_rows = list(csv.DictReader(handle))
    assert written_rows[0]["GRIB_stepRange"] == "0-3"

    print_apcp_metadata(rows)
    output = capsys.readouterr().out
    assert "stepType: accum" in output
    assert "stepRange: 0-3" in output
    assert "forecast time:" in output
