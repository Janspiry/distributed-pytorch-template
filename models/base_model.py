import os
import torch
import torch.nn as nn
import logging
from collections import OrderedDict
logger = logging.getLogger('base')
class BaseModel():
    def name(self):
        return 'BaseModel'

    def __init__(self, opt):
        self.opt = opt
        self.phase = opt['phase']
        self.gpu_ids = opt['gpu_ids']
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = opt['path']['checkpoint']
        self.epoch = 0
        self.iter = 0
        self.schedulers = []
        self.optimizers = []

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    '''used in test time, no backprop'''
    def val(self):
        pass

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

    def get_current_visuals(self):
        return self.input

    def get_current_log(self):
        return {}

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def get_current_learning_rate(self):
        return self.schedulers[0].get_lr()[0]
    
    def save_network(self, network, network_label, iter_step):
        save_filename = '{}_{}.pth'.format(iter_step, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel):
            network = network.module
        network.load_state_dict(torch.load(load_path), strict=strict)

    def save_training_state(self, epoch, iter_step):
        '''Saves training state during training, which will be used for resuming'''
        state = {'epoch': epoch, 'iter': iter_step, 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        save_filename = '{}.state'.format(iter_step)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(state, save_path)

    def resume_training(self, load_path):
        '''Resume the optimizers and schedulers for training'''
        resume_state = torch.load(load_path)
        self.epoch = resume_state['epoch']
        self.iter = resume_state['iter']
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

    def load(self):
        load_path = self.opt['path']['resume_state']
        model_path = "{}_{}.pth".format(load_path, "net")
        state_path = "{}.state".format(load_path)
        if model_path is not None:
            logger.info('Loading pretrained model for G [{:s}] ...'.format(model_path))
            if self.opt['finetune_norm']:
                self.load_network(model_path, self.net, strict=False)
            else:
                self.load_network(model_path, self.net)
            self.resume_training()
        if self.phase=='train' and state_path is not None:
            logger.info('Loading training state for [{:s}] ...'.format(state_path))
            self.resume_training(state_path)
        
    def save(self, total_iters, total_epoch):
        self.save_network(self.net, 'net', total_iters)
        self.save_training_state(total_epoch, total_iters)


    '''Get the string and total parameters of the network'''
    def get_network_description(self, network):
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info('Network structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)