from math import isnan
from pathlib import Path
from tomllib import load
from types import SimpleNamespace
from typing import Any


def get_experiment_config(
    model_name: str,
    dataset_name: str,
    config_filepath: Path,
    local_config_filepath: Path | None = None,
) -> SimpleNamespace:

    with config_filepath.open("rb") as c:
        conf = load(c)

    _nan2none(conf)

    if local_config_filepath is not None:
        with local_config_filepath.open("rb") as lc:
            localconf = load(lc)

        _nan2none(localconf)
        _override_conf(conf, localconf)

    assert "common" in conf
    assert model_name in conf
    assert dataset_name in conf[model_name]

    return SimpleNamespace(conf["common"] | conf[model_name][dataset_name])


def _nan2none(conf: dict[str, Any]):
    for k, v in conf.items():
        if isinstance(v, dict):
            _nan2none(conf[k])
        elif isinstance(v, float):
            if isnan(v):
                conf[k] = None


def _override_conf(conf: dict[str, Any], localconf: dict[str, Any]):
    for k, v in localconf.items():
        if k in conf:
            if isinstance(v, dict):
                _override_conf(conf[k], localconf[k])
            else:
                conf[k] = v
        else:
            conf[k] = v
