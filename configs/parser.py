import os


def parse_config(config_path: str) -> dict:
    """Parser for python config model files

    Args:
        config_path (str): _description_

    Returns:
        dict: _description_
    """
    config_name = os.path.basename(config_path).replace('.py', '')
    mod = __import__(config_path.replace('.py', '').replace('/', '.'))
    return getattr(mod, config_name)
