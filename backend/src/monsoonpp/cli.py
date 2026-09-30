"""Command line.

  monsoonpp build     -c configs/synthetic.yaml     # adapters -> aligned cube
  monsoonpp train     -c ...                        # regimes, ladder L0-L6, prob heads
  monsoonpp evaluate  -c ...                        # verification report + atlas + drift
  monsoonpp forecast  -c ... --date 2024-07-20 --lead 1   # district table for one day
  monsoonpp all       -c ...
  monsoonpp serve     -c ... --port 8000
  monsoonpp inspect-obs PATH [--out DIR]           # read-only observation-file inspection
  monsoonpp smoke -c configs/smoke_2024w29_imerg.yaml   # 7-day real-data smoke test
  monsoonpp validate-phase-b -c configs/phase_b_2024_gfs_etopo.yaml
  monsoonpp phase-c -c configs/phase_c_2024w29_imerg.yaml
  monsoonpp phase-c2 -c configs/phase_c2_202407_imerg.yaml
  monsoonpp phase-c3 -c configs/phase_c3_2024jjas_imerg.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time


def main(argv=None):
    ap = argparse.ArgumentParser("monsoonpp")
    ap.add_argument("cmd", choices=["build", "train", "evaluate", "forecast", "all", "serve", "inspect-obs",
                                    "smoke", "validate-phase-b", "phase-c", "phase-c2", "phase-c3"])
    ap.add_argument("path", nargs="?", help="file for inspect-obs")
    ap.add_argument("--out", help="output directory for inspect-obs reports")
    ap.add_argument("-c", "--config", default=None)
    ap.add_argument("--date")
    ap.add_argument("--lead", type=int, default=1)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose or a.cmd in ("train", "all") else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    if a.cmd == "inspect-obs":            # read-only; needs no run config
        from .tools.inspect_obs import run
        if not a.path:
            sys.exit("usage: monsoonpp inspect-obs PATH [--out DIR]")
        rep = run(a.path, a.out)
        v = rep["time_convention_verdict"]
        print(f"[inspect-obs] {rep['source_filename']}: {v['status']} -> obs_time_convention {v['obs_time_convention']}")
        print("\n".join(rep["_written"]))
        return

    from .config import load_config
    cfg = load_config(a.config)
    t0 = time.time()

    if a.cmd in ("build", "all"):
        from .data.build import build_dataset
        ds = build_dataset(cfg)
        print(f"[build] {cfg.dataset_path}  days={ds.sizes['time']} leads={list(ds.lead.values)} grid={ds.sizes['lat']}x{ds.sizes['lon']}")
    if a.cmd in ("train", "all"):
        from .pipeline import run_train
        meta = run_train(cfg)
        print(f"[train] models -> {cfg.model_dir}  ({meta['train_seconds']}s)")
    if a.cmd in ("evaluate", "all"):
        from .verify.report import headline, write_report
        res = write_report(cfg)
        print(headline(res).round(3).to_string(index=False))
        print(f"[evaluate] report -> {cfg.report_dir / 'report_test.md'}")
    if a.cmd == "forecast":
        from .products.service import ForecastService
        if not a.date:
            sys.exit("--date required")
        s = ForecastService(cfg)
        print(json.dumps(s.regimes(a.date, a.lead), indent=1))
        print(s.districts(a.date, a.lead).head(25).round(2).to_string(index=False))
    if a.cmd == "smoke":
        from .smoke import run_smoke
        summ = run_smoke(cfg)
        print("\n".join(summ["banner"]))
        print(f"[smoke] paired {summ['windows_paired']}/{summ['windows_requested']} windows; outputs:")
        print("\n".join(summ["outputs"]))
    if a.cmd == "validate-phase-b":
        from .phase_b import run_phase_b
        result = run_phase_b(cfg)
        print(f"[phase-b] {result['status']}")
        print("\n".join(result["outputs"]))
    if a.cmd == "phase-c":
        from .phase_c import run_phase_c
        result = run_phase_c(cfg)
        print(f"[phase-c] {result['status']}; move_to_ml={str(result['move_to_ml']).lower()}")
        print("\n".join(result["outputs"]))
    if a.cmd == "phase-c2":
        from .phase_c2 import run_phase_c2
        result = run_phase_c2(cfg)
        print(f"[phase-c2] {result['status']}; move_to_ml={str(result['move_to_ml']).lower()}")
        print("\n".join(result["outputs"]))
    if a.cmd == "phase-c3":
        from .phase_c3 import run_phase_c3
        result = run_phase_c3(cfg)
        print(f"[phase-c3] {result['decision']}")
        print("\n".join(result["outputs"]))
    if a.cmd == "serve":
        import uvicorn
        from .api.app import create_app
        uvicorn.run(create_app(a.config), host="0.0.0.0", port=a.port)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
