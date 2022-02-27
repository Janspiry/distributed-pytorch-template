import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import tqdm
from .base_model import BaseModel

class Model(BaseModel):
    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)
        
        ''' networks are a list'''
        self.netG = self.set_device(self.networks[0]) # get the defined network

        ''' define parameters, include loss, optimizers, schedulers, etc.''' 
        if self.phase != 'test':
            self.netG.train()

            ''' loss, import munual loss using self.set_device '''
            self.loss_fn = nn.L1Loss()
        
            ''' find the parameters to optimize '''
            optim_params = list(filter(lambda p: p.requires_grad, self.netG.parameters()))
           
            ''' optimizers '''
            self.optimizer = torch.optim.Adam(optim_params, lr=1e-4, weight_decay=0)
            self.optimizers.append(self.optimizer)

            ''' schedulers, not sued now '''
            for optimizer in self.optimizers:
                pass
                self.schedulers.append(
                    torch.optim.lr_scheduler.CyclicLR(
                    optimizer,
                    base_lr=1e-7, max_lr=1e-4, gamma=0.99994, cycle_momentum=False)
                )
        ''' load pretrained models and print network '''
        self.load() 
        self.print_network(self.netG)
    
        
    def set_input(self, data):
        self.input = self.set_device(data['input']) # you must use set_device in tensor
        self.path = data['path']
    
    def get_current_visuals(self):
        dict = {}
        dict['input'] = self.input.detach()[0].float().cpu()
        dict['output'] = self.output.detach()[0].float().cpu()
        return dict

    def save_current_results(self):
        self.results_dict = self.results_dict._replace(name=self.path, result=self.output)
        return self.results_dict._asdict()

    def _train_epoch(self):
        self.netG.train()
        self.train_metrics.reset()
        for train_data in tqdm.tqdm(self.phase_loader):
            self.set_input(train_data)
            self.optimizer.zero_grad()
            self.output = self.netG(self.input)
            loss = self.loss_fn(self.output, self.input)
            loss.backward()
            self.optimizer.step()

            self.iter += self.batch_size
            self.writer.set_iter(self.iter, phase='train')
            self.train_metrics.update('loss', loss.item())
            if self.iter % self.opt['train']['log_iter'] == 0:
                for key, value in self.train_metrics.result().items():
                    self.logger.info('{:5s}: {}\t'.format(str(key), value))
                for key, value in self.get_current_visuals().items():
                    self.writer.add_image(key, value)

        for scheduler in self.schedulers:
            scheduler.step()
        return self.train_metrics.result()
    
    def _val_epoch(self):
        self.netG.eval()
        self.val_metrics.reset()
        with torch.no_grad():
            for val_data in tqdm.tqdm(self.val_loader):
                self.set_input(val_data)
                self.output = self.netG(self.input)
                loss = self.loss_fn(self.output, self.input)
                
                self.iter += self.batch_size
                self.writer.set_iter(self.iter, phase='val')
                self.val_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.val_metrics.update(met.__name__, met(self.input, self.output))
                for key, value in self.get_current_visuals().items():
                    self.writer.add_image(key, value)
                self.writer.save_images(self.save_current_results())

        return self.val_metrics.result()

    def _test(self):
        pass

    def load(self):
        self.load_network(network=self.netG, network_label="netG")
        self.resume_training()
    
    def save(self):
        self.save_network(network=self.netG, network_label='netG')
        self.save_training_state()

