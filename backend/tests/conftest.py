import pytest

from monsoonpp.config import load_config


@pytest.fixture(scope="session")
def tiny_cfg(tmp_path_factory):
    d = tmp_path_factory.mktemp("run")
    cfg = load_config(None, name="tiny", years=[2020, 2021, 2022, 2023], leads=[1, 2],
                      season_start="06-15", season_end="08-31", data_dir=str(d),
                      obs_time_convention="STARTING_03Z")   # synthetic fixture: explicit
    # Coarse 1 deg TEST domain (cropped, non-IMD) — explicit so tests don't depend on
    # GridConfig defaults, which are the canonical IMD 0.25 deg grid.
    cfg.grid.lat_min, cfg.grid.lat_max, cfg.grid.lon_min, cfg.grid.lon_max = 6.5, 37.5, 66.5, 97.5
    cfg.grid.res = 1.0
    cfg.split.train_years, cfg.split.val_years, cfg.split.test_years = [2020, 2021], [2022], [2023]
    cfg.model.n_estimators = 150
    cfg.model.learning_rate = 0.15
    cfg.model.min_expert_samples = 1000
    cfg.model.moe_shrinkage_k = 2000
    return cfg


@pytest.fixture(scope="session")
def trained(tiny_cfg):
    from monsoonpp.data.build import build_dataset
    from monsoonpp.pipeline import run_train
    from monsoonpp.verify.report import write_report
    build_dataset(tiny_cfg)
    run_train(tiny_cfg)
    res = write_report(tiny_cfg)
    return tiny_cfg, res
