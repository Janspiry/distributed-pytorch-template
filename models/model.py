import torch
import tqdm
from core.base_model import BaseModel
from core.logger import LogTracker
class Model(BaseModel):
    def __init__(self, networks, optimizers, lr_schedulers, losses, **kwargs):
        ''' must to init BaseModel with kwargs '''
        super(Model, self).__init__(**kwargs)

        ''' networks, dataloder, optimizers, losses, etc. '''
        self.netG = self.set_device(networks[0], distributed=self.opt['distributed']) # get the defined network
        self.loss_fn = losses[0]
        self.schedulers = lr_schedulers
        self.optG = optimizers[0]

        ''' networks can be a list, and must convers by self.set_device function if using multiple GPU. '''
        self.load_everything()

        ''' can rewrite in inherited class for more informations logging '''
        self.train_metrics = LogTracker(*[m.__name__ for m in losses], writer=self.writer, phase='train')
        self.val_metrics = LogTracker(*[m.__name__ for m in losses], *[m.__name__ for m in self.metrics], writer=self.writer, phase='val')

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
        self.results_dict = self.results_dict._replace(name=self.path, result=self.output)
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
            if self.iter % self.opt['train']['log_iter'] == 0:
                for key, value in self.train_metrics.result().items():
                    self.logger.info('{:5s}: {}\t'.format(str(key), value))
                for key, value in self.get_current_visuals().items():
                    self.writer.add_image(key, value)

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
                for met in self.metrics:
                    self.val_metrics.update(met.__name__, met(self.input, self.output))
                for key, value in self.get_current_visuals().items():
                    self.writer.add_image(key, value)
                self.writer.save_images(self.save_current_results())

        return self.val_metrics.result()

    def load_everything(self):
        """ save pretrained model and training state, which only do on GPU 0. """
        self.load_network(network=self.netG, network_label=self.netG.__class__.__name__)
        self.resume_training(self.schedulers, [self.optG]) 

    def save_everything(self):
        """ load pretrained model and training state, optimizers and schedulers must be a list. """
        self.save_network(network=self.netG, network_label=self.netG.__class__.__name__)
        self.save_training_state(self.schedulers, [self.optG])
