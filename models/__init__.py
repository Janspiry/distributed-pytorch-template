import logging
import importlib
from torch.nn.parallel import DistributedDataParallel as DDP

import core.util as Util
logger = logging.getLogger('base')

''' create_model '''
def create_model(opt):
    model_opt = opt['model']['which_model']
    model_file_name, model_class_name = "models.{}".format(model_opt["name"][0]), model_opt["name"][1]
    try:
        ''' set base model input '''
        model_args={
            "networks": define_networks(opt),
            "phase": opt['phase'],
            "save_dir": opt['path']['checkpoint'],
            "resume_dir": opt['path']['resume_state'],
            "rank": opt['global_rank'],
            "finetune_norm": opt['finetune_norm']
        }
        ''' loading Model() class from given file's name '''
        model_args.update(model_opt['args'])
        model = getattr(importlib.import_module(model_file_name), model_class_name)(**model_args)
        if opt['global_rank']==0:
            logger.info('Model [{:s} from {:s}] is created.'.format(model_class_name, model_file_name))
    except:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model_class_name))
    return model


''' define_network '''
def define_network(opt, network_opt):
    net_file_name, net_class_name = "models.network.{}".format(network_opt['name'][0]), network_opt['name'][1]
    net_args = network_opt['args']
    try:
        ''' loading Network() class from given file's name '''
        net = getattr(importlib.import_module(net_file_name), net_class_name)(**net_args)
        if opt['global_rank']==0:
            logger.info('Network [{:s} form {:s}] is created.'.format(net_class_name, net_file_name))
    except:
        raise NotImplementedError('Network [{:s}] not recognized.'.format(net_class_name))
    
    if opt['phase'] == 'train' and opt['path']['resume_state'] is None:
        if opt['global_rank']==0:
            logger.info('Network [{:s}] weights initialize using [{:s}] method.'.format(net_args.get('init_type', 'default'), net_class_name))
        net.init_weights()
    else:
        ''' loading from checkpoint, which define in model initialization part '''
        pass
    
    net = Util.set_device(net)
    if opt['distributed']:
        net = DDP(net, device_ids=[opt['global_rank']], output_device=opt['global_rank'], 
                      broadcast_buffers=True, find_unused_parameters=True)
    return net

''' define_networks, which returns a network list '''
def define_networks(opt):
    networks = []
    for net_opt in opt['model']['which_networks']:
        networks.append(define_network(opt, net_opt))
    return networks