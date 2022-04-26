import torch
import tqdm
from core.base_model import BaseModel
from core.logger import LogTracker
import copy
class EMA():
    def __init__(self, beta=0.9999):
        super().__init__()
        self.beta = beta
    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Model(BaseModel):
    def __init__(self, networks, optimizers, lr_schedulers, losses, ema_scheduler=None, **kwargs):
        ''' must to init BaseModel with kwargs '''
        super(Model, self).__init__(**kwargs)

        ''' networks, dataloder, optimizers, losses, etc. '''
        self.netG = networks[0]
        if ema_scheduler is not None:
            self.ema_scheduler = ema_scheduler
            self.netG_EMA = copy.deepcopy(self.netG)
            self.EMA = EMA(beta=self.ema_scheduler['ema_decay'])
        else:
            self.ema_scheduler = None
        ''' ddp '''
        self.netG = self.set_device(self.netG, distributed=self.opt['distributed'])
        if self.ema_scheduler is not None:
            self.netG_EMA = self.set_device(self.netG_EMA, distributed=self.opt['distributed'])
        
        self.loss_fn = losses[0]
        self.schedulers = lr_schedulers
        self.optG = optimizers[0]

        ''' networks can be a list, and must convers by self.set_device function if using multiple GPU. '''
        self.load_everything()

        ''' can rewrite in inherited class for more informations logging '''
        self.train_metrics = LogTracker(*[m.__name__ for m in losses], phase='train')
        self.val_metrics = LogTracker(*[m.__name__ for m in losses], *[m.__name__ for m in self.metrics], phase='val')
        self.test_metrics = LogTracker(*[m.__name__ for m in losses], *[m.__name__ for m in self.metrics], phase='test')

    def set_input(self, data):
        ''' must use set_device in tensor '''
        self.input = self.set_device(data['input'])
        self.path = data['path']
    
    def get_current_visuals(self):
        dict = {
            'input': self.input.detach()[0].float().cpu()
            ,'output': self.output.detach()[0].float().cpu()
        }
        return dict

    def save_current_results(self):
        self.results_dict = self.results_dict._replace(name=self.path, result=self.output.detach().float().cpu())
        return self.results_dict._asdict()

    def train_step(self):
        self.netG.train()
        self.train_metrics.reset()
        for train_data in tqdm.tqdm(self.phase_loader):
            self.set_input(train_data)
            self.optG.zero_grad()
            self.output = self.netG(self.input)
            loss = self.loss_fn(self.output, self.input)
            loss.backward()
            self.optG.step()

            self.iter += self.batch_size
            self.writer.set_iter(self.epoch, self.iter, phase='train')
            self.train_metrics.update(self.loss_fn.__name__, loss.item())
            self.writer.add_scalar(self.loss_fn.__name__, loss.item())
            if self.iter % self.opt['train']['log_iter'] == 0:
                for key, value in self.train_metrics.result().items():
                    self.logger.info('{:5s}: {}\t'.format(str(key), value))
                for key, value in self.get_current_visuals().items():
                    self.writer.add_image(key, value)
            if self.ema_scheduler is not None:
                if self.iter % self.ema_scheduler['ema_iter'] == 0 and self.iter > self.ema_scheduler['ema_start']:
                    self.logger.info('Update the EMA  model at the iter {:.0f}'.format(self.iter))
                    self.EMA.update_model_average(self.netG_EMA, self.netG)

        for scheduler in self.schedulers:
            scheduler.step()
        return self.train_metrics.result()
    
    def val_step(self):
        self.netG.eval()
        self.val_metrics.reset()
        with torch.no_grad():
            for val_data in tqdm.tqdm(self.val_loader):
                self.set_input(val_data)
                self.output = self.netG(self.input)
                loss = self.loss_fn(self.output, self.input)
                
                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='val')
                self.val_metrics.update(self.loss_fn.__name__, loss.item())
                self.writer.add_scalar(self.loss_fn.__name__, loss.item())
                for met in self.metrics:
                    key, value = met.__name__, met(self.input, self.output)
                    self.writer.add_scalar(key, value)
                    self.val_metrics.update(key, value)
                for key, value in self.get_current_visuals().items():
                    self.writer.add_image(key, value)
                self.writer.save_images(self.save_current_results())

        return self.val_metrics.result()

    def load_everything(self):
        """ save pretrained model and training state, which only do on GPU 0. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.load_network(network=self.netG, network_label=netG_label, strict=False)
        if self.ema_scheduler is not None:
            self.load_network(network=self.netG_EMA, network_label=netG_label+'_ema', strict=False)
        self.resume_training([self.optG], self.schedulers) 

    def save_everything(self):
        """ load pretrained model and training state, optimizers and schedulers must be a list. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.save_network(network=self.netG, network_label=netG_label)
        if self.ema_scheduler is not None:
            self.save_network(network=self.netG_EMA, network_label=netG_label+'_ema')
        self.save_training_state([self.optG], self.schedulers)
