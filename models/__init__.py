import torch
from core.praser import init_obj

''' create_model '''
def create_model(**cfg_model):
    opt = cfg_model['opt']
    logger = cfg_model['logger']

    model_opt = opt['model']['which_model']
    model_opt['args'].update(cfg_model)
    model = init_obj(model_opt, logger, default_file_name='models.model', init_type='Model')

    return model

''' define_network '''
def define_network(logger, opt, network_opt):
    net = init_obj(network_opt, logger, default_file_name='models.network', init_type='Network')
    if opt['phase'] == 'train' and opt['path']['resume_state'] is None:
        logger.info('Network weights initialize using [{:s}] method.'.format(network_opt['args'].get('init_type', 'default')))
        net.init_weights()
    else:
        ''' loading from checkpoint, which define in model initialization part '''
        pass
    return net

''' define metric '''
def define_loss(logger, loss_opt):
    return init_obj(loss_opt, logger, default_file_name='models.loss', init_type='Loss')

''' define metric '''
def define_metric(logger, metric_opt):
    return init_obj(metric_opt, logger, default_file_name='models.metric', init_type='Metric')

def define_optimizer(logger, optimizer_opt):
    return init_obj(optimizer_opt, logger, given_module=torch.optim, init_type='Optimizer')

def define_scheduler(logger, scheduler_opt):
    return init_obj(scheduler_opt, logger, given_module=torch.optim.lr_scheduler, init_type='Scheduler')