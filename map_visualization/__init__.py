import configparser as cp
import typing as tg


def map_config_loader(f: tg.TextIO) -> tg.Dict:
    config = cp.ConfigParser()
    return config.read_file(f)