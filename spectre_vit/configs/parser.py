import importlib
from types import SimpleNamespace


def module_to_dict(module):
    return {k: getattr(module, k) for k in dir(module) if not k.startswith("_") or k == "__base__"}


def parse_config(config_path: str) -> dict:
    """Parser for python config model files

    Args:
        config_path (str): _description_

    Returns:
        dict: _description_
    """
    config_path = config_path.replace("/", ".").replace(".py", "")
    mod = module_to_dict(importlib.import_module(config_path))

    if "__base__" in mod:
        default_config_path = ".".join(config_path.split(".")[:-1])
        default_config_path = f"{default_config_path}.{mod['__base__']}"
        base_mod = module_to_dict(importlib.import_module(default_config_path.replace(".py", "")))
        mod |= base_mod

    return SimpleNamespace(mod)
