import importlib
import os


def parse_config(config_path: str) -> dict:
    """Parser for python config model files

    Args:
        config_path (str): _description_

    Returns:
        dict: _description_
    """
    config_name = os.path.basename(config_path).replace(".py", "")
    config_path = config_path.replace("/", ".")
    mod = importlib.import_module(config_path.replace(".py", ""))
    return mod
    # return getattr(getattr(mod, "configs"), config_name)
