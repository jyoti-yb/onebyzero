"""Audit cached Phase-B GFS atmospheric subsets with the strict GRIB decoder."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..adapters.gfs_atmos import REGIME_FIELDS, decode_gfs_atmos_file


def audit(root: Path, month: str, bbox: list[float], profile: str) -> tuple[list[Path], list[tuple[Path, str]]]:
    good, bad = [], []
    suffix, required = ("04b2", None) if profile == "phase_b_full" else ("0921", REGIME_FIELDS)
    pattern = f"{month}*/subset_*{suffix}__gfs.t00z.pgrb2.0p25.f*"
    for path in sorted(root.glob(pattern)):
        try:
            fxx = int(path.name.rsplit(".f", 1)[1])
            decode_gfs_atmos_file(
                path, pd.Timestamp(path.parent.name), fxx, bbox=bbox, required_fields=required
            )
            good.append(path)
        except Exception as exc:
            bad.append((path, f"{type(exc).__name__}: {exc}"))
    return good, bad


def main(argv=None) -> None:
    parser = argparse.ArgumentParser("audit-gfs-atmos-cache")
    parser.add_argument("--root", default="data/raw/gfs/gfs")
    parser.add_argument("--month", default="202407")
    parser.add_argument("--profile", choices=("phase_b_full", "regime_required"), default="phase_b_full")
    parser.add_argument("--remove-invalid", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    workspace = Path.cwd().resolve()
    if workspace not in root.parents:
        raise RuntimeError(f"cache root must remain inside workspace: {root}")
    good, bad = audit(root, args.month, [6.5, 38.5, 66.5, 100.0], args.profile)
    print(f"atmosphere cache profile={args.profile}: good={len(good)} invalid={len(bad)}")
    for path, reason in bad:
        print(f"INVALID {path} ({path.stat().st_size} bytes): {reason[:180]}")
    if args.remove_invalid:
        for path, _ in bad:
            path.unlink()
        print(f"removed_invalid={len(bad)}")


if __name__ == "__main__":
    main()
