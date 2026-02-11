import sys
from pathlib import Path

modules = Path(__file__).resolve().parent.parent.parent
sys.path.append(modules.as_posix())

from spectre_vit.configs.toml_parser import get_experiment_config

config = modules / "spectre_vit" / "configs" / "experiments.toml"
localconfig = config.parent / "experiments.local.toml"

c = get_experiment_config(
    "spectre_vit",
    "imagenet1k",
    config,
    localconfig if localconfig.exists() else None,
)

print(c)
