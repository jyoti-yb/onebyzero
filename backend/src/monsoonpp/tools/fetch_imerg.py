"""Authenticated IMERG Final granule fetcher for an exact range of 03Z daily windows."""
from __future__ import annotations

import argparse
from pathlib import Path

import earthaccess
import pandas as pd


def granule_query_window(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    first = pd.Timestamp(start).normalize() + pd.Timedelta(hours=3)
    stop = pd.Timestamp(end).normalize() + pd.Timedelta(days=1, hours=3)
    days = int((pd.Timestamp(end).normalize() - pd.Timestamp(start).normalize()).days) + 1
    if days < 1:
        raise ValueError("end must not precede start")
    return first, stop - pd.Timedelta(seconds=1), days * 48


def main(argv=None) -> None:
    parser = argparse.ArgumentParser("fetch-imerg")
    parser.add_argument("--start", required=True, help="first 03Z daily-window label")
    parser.add_argument("--end", required=True, help="last 03Z daily-window label")
    parser.add_argument("--out", default="data/raw/imerg")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)

    first, last, expected = granule_query_window(args.start, args.end)
    auth = earthaccess.login(strategy="environment")
    if not auth.authenticated:
        raise RuntimeError("Earthdata authentication failed")
    granules = earthaccess.search_data(
        short_name="GPM_3IMERGHH", version="07",
        temporal=(first.isoformat(), last.isoformat()), count=-1,
    )
    if len(granules) != expected:
        raise RuntimeError(f"CMR returned {len(granules)} granules; expected exactly {expected}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = earthaccess.download(granules, out, threads=args.threads, show_progress=True)
    unique = {Path(path).name for path in paths}
    if len(unique) != expected:
        raise RuntimeError(f"download resolved {len(unique)} unique files; expected exactly {expected}")
    bad = []
    for path in paths:
        file = Path(path)
        signature = b""
        if file.exists() and file.stat().st_size >= 8:
            with file.open("rb") as stream:
                signature = stream.read(8)
        if signature != b"\x89HDF\r\n\x1a\n":
            bad.append(file.name)
    if bad:
        raise RuntimeError(f"{len(bad)} downloaded files failed the HDF5 signature check")
    print(f"IMERG Final V07 granules ready: {expected} ({first} through {last})")


if __name__ == "__main__":
    main()
