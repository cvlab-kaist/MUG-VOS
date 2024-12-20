from .data_engine import EngineV1

def get_engine(config):
    if config['engine']['version'] == 1:
        return EngineV1(config)
    else:
        raise ValueError('Invalid engine version')