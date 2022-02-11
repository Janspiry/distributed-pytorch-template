import functools
import logging
import torch.nn as nn
from torch.nn import init
from torch.nn.parallel import DistributedDataParallel as DDP

import importlib

import core.util as Util
logger = logging.getLogger('base')
####################
# initialize
####################


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(
            weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [{:s}] not implemented'.format(init_type))


####################
# define network
####################
def define_network(opt, network_name, init_type="kaiming"):
    try:
        ''' loading Network() class from given file's name '''
        net = importlib.import_module("models.network.{}".format(network_name)).Network(opt)
        if opt['global_rank']==0:
            logger.info('Network [{:s}- {:s}] is created.'.format(net.name(), network_name))
    except:
        raise NotImplementedError('Network [{:s}] not recognized.'.format(network_name))
    
    if opt['phase'] == 'train' and opt['path']['resume_state'] is None:
        init_weights(net, init_type, scale=0.1)
    else:
        pass
        # loading from checkpoint, which define in model initialization part
    
    net = Util.set_device(net)
    if opt['distributed']:
        net = DDP(net, device_ids=[opt['global_rank']], output_device=opt['global_rank'], 
                      broadcast_buffers=True, find_unused_parameters=False)
    return net

def define_networks(opt):
    model_opt = opt['model']
    network_num = len(model_opt['which_networks'])
    init_types_num = len(model_opt['init_types'])
    nets = []
    for idx in range(network_num):
        if idx<init_types_num:
            nets.append(define_network(opt, model_opt['which_networks'][idx], model_opt['init_types'][idx]))
        else:
            nets.append(define_network(opt, model_opt['which_networks'][idx]))
    return nets 
    