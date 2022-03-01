import torch
from core.praser import init_objs

def create_model(**cfg_model):
    """ create_model """
    opt = cfg_model['opt']
    logger = cfg_model['logger']

    model_opt = opt['model']['which_model']
    model_opt['args'].update(cfg_model)
    model = init_objs(model_opt, logger, default_file_name='models.model', init_type='Model')

    return model

def define_network(logger, opt, network_opt):
    """ define network with weights initialization """
    net = init_objs(network_opt, logger, default_file_name='models.network', init_type='Network')

    if opt['phase'] == 'train' and opt['path']['resume_state'] is None:
        if isinstance(net, list):
            for net_idx in range(len(net)):
                logger.info('Network weights initialize using [{:s}] method.'.format(network_opt[net_idx]['args'].get('init_type', 'default')))
                net[net_idx].init_weights()
        else:
            logger.info('Network weights initialize using [{:s}] method.'.format(network_opt['args'].get('init_type', 'default')))
            net.init_weights()
    return net


def define_loss(logger, loss_opt):
    return init_objs(loss_opt, logger, default_file_name='models.loss', init_type='Loss')

def define_metric(logger, metric_opt):
    return init_objs(metric_opt, logger, default_file_name='models.metric', init_type='Metric')

def define_optimizer(logger, optimizer_opt):
    return init_objs(optimizer_opt, logger, given_module=torch.optim, init_type='Optimizer')

def define_scheduler(logger, scheduler_opt):
    return init_objs(scheduler_opt, logger, given_module=torch.optim.lr_scheduler, init_type='Scheduler')