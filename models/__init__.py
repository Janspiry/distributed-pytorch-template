import logging
import importlib
logger = logging.getLogger('base')

def create_model(opt):
    model_opt = opt['model']
    try:
        ''' loading Model() class from given file's name '''
        net = importlib.import_module("models.{}".format(model_opt['which_model'])).Model(opt)
        if opt['global_rank']==0:
            logger.info('Model [{:s}- {:s}] is created.'.format(net.name(), model_opt['which_model']))
    except:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model_opt['which_model']))
    return net