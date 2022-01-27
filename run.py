import torch
import argparse
import logging
import core.logger as Logger
import core.praser as Praser
from tensorboardX import SummaryWriter
import os
import numpy as np
import time
from data.data_loader import create_dataloader
from models.networks import define_network

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Praser.parse(args)
    
    # logger set
    Logger.init_logger(opt=opt)
    base_logger = logging.getLogger('base')
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])
    base_logger.info(Praser.dict2str(opt))
    
    # set cuda environment
    gpu_str = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    base_logger.info('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # select model and dataset
    dataset = create_dataloader(opt, phase='train')    
    model = define_network(opt)
    total_steps = 0

    for epoch in range(opt['resume_training']['begin_iter'] + 1, opt['training']['n_iter'] + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_steps += opt['batch_size']
            epoch_iter += opt['batch_size']
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt['training']['display_freq'] == 0:
                visualizer.display_current_results(model.get_current_visuals(), epoch)

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt['batch_size']
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
            (epoch, opt.niter, time.time() - epoch_start_time))

        model.update_hyperparams(epoch)
