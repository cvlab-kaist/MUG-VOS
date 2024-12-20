from .davis import DAVISLoader
from .visa import ViSALoader
from .ytvos import YTVOSLoader

def get_loader(config):
    if config['loader']['dataset'] == 'DAVIS':
        return DAVISLoader(config)
    elif config['loader']['dataset'] == 'ViSA':
        return ViSALoader(config)
    elif config['loader']['dataset'] == 'YTVOS':
        return YTVOSLoader(config)
    else:
        raise ValueError('Invalid dataset')