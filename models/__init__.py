import logging
import importlib
logger = logging.getLogger('base')

def create_model(opt):
    model_opt = opt['model']
    try:
        net = importlib.import_module("models.{}".format(model_opt['which_model'])).Model(opt)
        logger.info('Model [{:s}] is created.'.format(net.__class__.__name__))
    except:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model_opt['which_model']))
    return net