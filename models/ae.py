import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from collections import OrderedDict
import logging

from .base_model import BaseModel
from . import networks
logger = logging.getLogger('base')

class Model(BaseModel):
    def __init__(self, opt):
        super(Model, self).__init__(opt)
        
        # define network and load pretrained models
        self.net = networks.define_network(opt)
        self.load()

        if self.phase=='train':
            self.net.train()

            # loss
            self.loss_fn = nn.L1Loss()
        
            # find the parameters to optimize
            if opt['finetune_norm']:
                optim_params = []
                for k, v in self.net.named_parameters():
                    v.requires_grad = True
                    if k.find('backbone') >= 0:
                        v.requires_grad = False
                    else:
                        optim_params.append(v)
                        logger.info('Params [{:s}] will optimize.'.format(k))
            else:
                optim_params = list(self.net.parameters())

            # optimizers
            self.optimizer = torch.optim.Adam(optim_params, lr=1e-4, weight_decay=0)
            self.optimizers.append(self.optimizer)

            # schedulers, not sued now
            for optimizer in self.optimizers:
                pass
                self.schedulers.append(torch.optim.lr_scheduler.CyclicLR(
                    optimizer,
                    base_lr=1e-7,
                    max_lr=1e-4,
                    gamma=0.99994,
                    cycle_momentum=False)
                )
            self.log_dict = OrderedDict()
        # print network
        self.print_network()

    def set_input(self, data):
        self.input = data['input']
        self.path = data['path']

    def get_image_paths(self):
        return self.path
        
    def optimize_parameters(self, step):
        self.optimizer.zero_grad()
        self.output = self.net(self.input)
        l_pix = self.loss_fn(self.output, self.input)
        l_pix.backward()
        self.optimizer.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def val(self):
        self.net.eval()
        with torch.no_grad():
            self.output = self.net(self.input)
            l_pix = self.loss_fn(self.output, self.input)
            self.log_dict['l_pix'] = l_pix.item()
        self.net.train()

    def test(self):
        self.net.eval()
        with torch.no_grad():
            self.output = self.net(self.input)
        self.net.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['Input'] = self.input.detach()[0].float().cpu()
        out_dict['Output'] = self.output.detach()[0].float().cpu()
        return out_dict

    