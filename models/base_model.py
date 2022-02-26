import os
import torch
import torch.nn as nn
import logging
import collections
from collections import OrderedDict
import core.util as Util
logger = logging.getLogger('base')
class BaseModel():
    def __init__(self, networks, phase, rank, save_dir, resume_dir=None, finetune_norm=False):
        self.phase = phase
        self.rank = rank
        self.finetune_norm = finetune_norm

        ''' process record '''
        self.save_dir = save_dir
        self.resume_dir = resume_dir
        self.epoch = 0
        self.iter = 0

        ''' networks and optimizers '''
        self.networks = networks
        self.schedulers = []
        self.optimizers = []

        ''' log and visual result dict '''
        self.log_dict = OrderedDict()
        self.visuals_dict = OrderedDict()

        CustomResult = collections.namedtuple('CustomResult', 'name result')
        self.results_dict = CustomResult([],[]) # {"name":[], "result":[]}


    def set_input(self, input):
        self.input = Util.set_device(input)

    def forward(self):
        pass
    
    '''used in test time, no backprop'''
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def update_hyperparams(self):
        pass
    
    def get_current_iters(self):
        return self.epoch, self.iter

    ''' return tensor dict to show on tensorboard, key can be arbitrary '''
    def get_current_visuals(self):
        return self.visuals_dict

    ''' return information dict to save on logging file '''
    def get_current_log(self):
        return self.log_dict

    ''' return tensor dict to save on given result path, key must contains name and result '''
    def save_current_results(self):
        return self.results_dict

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def get_current_learning_rate(self):
        return self.schedulers[0].get_lr()[0]
    
    ''' save pretrained model and training state, which only do on GPU 0 '''
    def save(self):
        pass 

    ''' load pretrained model and training state '''
    def load(self):
        pass
    

    def print_network(self, network):
        if self.rank!=0:
            return
        s, n = self.get_network_description(network)
        if isinstance(network, nn.DataParallel):
            net_struc_str = '{} - {}'.format(network.__class__.__name__, network.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(network.__class__.__name__)
        logger.info('Network structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, network, network_label, iter_step):
        if self.rank!=0:
            return
        save_filename = '{}_{}.pth'.format(iter_step, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load_network(self, network, network_label, strict=True):
        if self.resume_dir is None:
            return 
        model_path = "{}_{}.pth".format(self.resume_dir, network_label)
        logger.info('Loading pretrained model for [{:s}] ...'.format(model_path))
        if isinstance(network, nn.DataParallel):
            network = network.module
        network.load_state_dict(torch.load(model_path, map_location = lambda storage, loc: Util.set_device(storage)), strict=strict)

    ''' saves training state during training '''
    def save_training_state(self, epoch, iter_step):
        state = {'epoch': epoch, 'iter': iter_step, 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        save_filename = '{}.state'.format(iter_step)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(state, save_path)

    ''' resume the optimizers and schedulers for training '''
    def resume_training(self):
        if self.phase!='train' or self.resume_dir is None:
            return
        state_path = "{}.state".format(self.resume_dir)
        logger.info('Loading training state for [{:s}] ...'.format(state_path))
        resume_state = torch.load(state_path, map_location = lambda storage, loc: Util.set_device(storage))
        
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

        self.epoch = resume_state['epoch']
        self.iter = resume_state['iter']

    ''' get the string and total parameters of the network'''
    def get_network_description(self, network):
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    