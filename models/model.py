import torch
import tqdm
from core.base_model import BaseModel

class Model(BaseModel):
    def __init__(self, lr, weight_decay, **kwargs):
        ''' must to init BaseModel with kwargs '''
        super(Model, self).__init__(**kwargs)

        ''' networks can be a list, and must convers by self.set_device function if using multiple GPU. '''
        self.netG = self.set_device(self.networks[0], distributed=self.opt['distributed']) # get the defined network
        self.print_network(self.netG)

        if self.phase != 'test':
            self.loss_fn = self.losses[0]

            optim_params = list(filter(lambda p: p.requires_grad, self.netG.parameters()))
            self.optimizer = torch.optim.Adam(optim_params, lr=lr, weight_decay=weight_decay)
            self.optimizers.append(self.optimizer)

            ''' schedulers, not sued now '''
            for optimizer in self.optimizers:
                pass 
                self.schedulers.append(torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-7, max_lr=1e-4, gamma=0.99994, cycle_momentum=False))

        self.load_everything() 

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
            self.optimizer.zero_grad()
            self.output = self.netG(self.input)
            loss = self.loss_fn(self.output, self.input)
            loss.backward()
            self.optimizer.step()

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

    def test_step(self):
        pass

    def load_everything(self):
        self.load_network(network=self.netG, network_label="netG")
        self.resume_training()
    
    def save_everything(self):
        self.save_network(network=self.netG, network_label='netG')
        self.save_training_state()

