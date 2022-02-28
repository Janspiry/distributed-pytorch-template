import os
from abc import abstractmethod
from functools import partial
import collections

import torch
import torch.nn as nn

import core.util as Util
from core.logger import LogTracker
CustomResult = collections.namedtuple('CustomResult', 'name result')
class BaseModel():
    def __init__(self, opt, networks, phase_loader, val_loader, losses, metrics, logger, writer):
        self.opt = opt
        self.phase = opt['phase']
        self.finetune = opt['finetune_norm']
        self.set_device = partial(Util.set_device, rank=opt['global_rank'])

        ''' process record '''
        self.batch_size = self.opt['datasets']['train']['dataloader']['args']['batch_size']
        self.epoch = 0
        self.iter = 0 

        ''' networks and optimizers '''
        self.networks = networks
        self.phase_loader = phase_loader
        self.val_loader = val_loader
        self.losses = losses
        self.metrics = metrics
        self.schedulers = []
        self.optimizers = []

        ''' log and visual result dict '''
        self.logger = logger
        self.writer = writer

        self.train_metrics = LogTracker(*[m.__name__ for m in self.losses], writer=self.writer, phase='train')
        self.val_metrics = LogTracker(*[m.__name__ for m in self.losses], *[m.__name__ for m in self.metrics], writer=self.writer, phase='val')

        self.results_dict = CustomResult([],[]) # {"name":[], "result":[]}

    def train(self):
        while self.epoch <= self.opt['train']['n_epoch']:
            self.epoch += 1
            if self.opt['distributed']:
                self.phase_loader.sampler.set_epoch(self.epoch) #  Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas use a different random ordering for each epoch

            train_log = self.train_step()

            # save logged informations into log dict
            train_log.update({'epoch': self.epoch, 'iters': self.iter})

            # print logged informations to the screen
            for key, value in train_log.items():
                self.logger.info('{:5s}: {}\t'.format(str(key), value))
            
            if self.epoch % self.opt['train']['save_checkpoint_epoch'] == 0:
                self.logger.info('Saving the self at the end of epoch {:.0f}'.format(self.epoch))
                self.save_everything()

            if self.epoch % self.opt['train']['val_epoch'] == 0:
                self.logger.info("\n\n\n------------------------------Validation Start------------------------------")
                if self.val_loader is None:
                    self.logger.info('Validation stop where dataloader is None, Skip it.')
                else:
                    val_log = self.val_step()
                    for key, value in val_log.items():
                        self.logger.info('{:5s}: {}\t'.format(str(key), value))
                self.logger.info("\n------------------------------Validation End------------------------------\n\n")
        self.logger.info('Number of Epochs has reached the limit, End.')

    def test(self):
        pass

    @abstractmethod
    def train_step(self):
        raise NotImplementedError

    def val_step(self):
        pass

    def test_step(self):
        pass

    ''' save pretrained model and training state, which only do on GPU 0 '''
    def save_everything(self):
        pass 

    ''' load pretrained model and training state '''
    def load_everything(self):
        pass
    
    ''' get the string and total parameters of the network'''
    def get_network_description(self, network):
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n
    
    def print_network(self, network):
        if self.opt['global_rank'] !=0:
            return
        s, n = self.get_network_description(network)
        if isinstance(network, nn.DataParallel):
            net_struc_str = '{} - {}'.format(network.__class__.__name__, network.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(network.__class__.__name__)
        self.logger.info('Network structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        self.logger.info(s)

    def save_network(self, network, network_label):
        if self.opt['global_rank'] !=0:
            return
        save_filename = '{}_{}.pth'.format(self.epoch, network_label)
        save_path = os.path.join(self.opt['path']['checkpoint'], save_filename)
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load_network(self, network, network_label, strict=True):
        if self. opt['path']['resume_state'] is None:
            return 
        model_path = "{}_{}.pth".format(self. opt['path']['resume_state'], network_label)
        self.logger.info('Loading pretrained model for [{:s}] ...'.format(model_path))
        if isinstance(network, nn.DataParallel):
            network = network.module
        network.load_state_dict(torch.load(model_path, map_location = lambda storage, loc: Util.set_device(storage)), strict=strict)

    ''' saves training state during training '''
    def save_training_state(self):
        if self.opt['global_rank'] !=0:
            return
        state = {'epoch': self.epoch, 'iter': self.iter, 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        save_filename = '{}.state'.format(self.epoch)
        save_path = os.path.join(self.opt['path']['checkpoint'], save_filename)
        torch.save(state, save_path)

    ''' resume the optimizers and schedulers for training '''
    def resume_training(self):
        if self.phase!='train' or self. opt['path']['resume_state'] is None:
            return
        state_path = "{}.state".format(self. opt['path']['resume_state'])
        self.logger.info('Loading training state for [{:s}] ...'.format(state_path))
        resume_state = torch.load(state_path, map_location = lambda storage, loc: self.set_device(storage))
        
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

   