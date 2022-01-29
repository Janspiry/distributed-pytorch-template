import torch
import argparse
import logging
import tqdm.tqdm as tqdm
import core.logger as Logger
import core.praser as Praser
import os
import numpy as np
import time
from data import create_dataloader
from models import create_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/train.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(validation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')

    # parser configs
    args = parser.parse_args()
    opt = Praser.parse(args)
    
    # set logger 
    Logger.init_logger(opt=opt)
    base_logger = logging.getLogger('base')
    base_logger.info(Praser.dict2str(opt))
    
    # set cuda environment
    gpu_str = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    base_logger.info('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # set model and dataset
    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_loader =  create_dataloader(opt, phase='train') 
            base_logger.info('Initial Training Dataset') 
        elif phase == 'val':
            val_loader = create_dataloader(opt, phase='val')
            base_logger.info('Initial Validation Dataset') 
    base_logger.info('Initial Dataset Finished') 
    # model
    model = create_model(opt)
    base_logger.info('Initial Model Finished') 

    total_epoch, total_iters = model.get_current_iters()
    if opt['phase'] == 'train':
        base_logger.info('Begin Model Training.')
        while True:
            epoch_start_time = time.time()
            total_epoch += 1
            if total_epoch >= opt['training']['n_epoch']: 
                break

            train_pbar = tqdm(train_loader)
            for train_data in train_pbar:
                if total_iters >= opt['training']['n_iter']: 
                    break
                total_iters += opt['datasets']['train']['batch_size']
                model.set_input(train_data)
                model.optimize_parameters()

                if total_iters % opt['training']['display_freq'] == 0:
                    Logger.display_current_results(total_iters, total_epoch, model.get_current_visuals(), phase='train')

                if total_iters % opt['training']['print_freq'] == 0:
                    logs = model.get_current_log()
                    Logger.print_current_logs(total_iters, total_epoch, logs)
                    Logger.display_current_logs(total_iters, total_epoch, logs)

                if total_iters % opt['training']['save_checkpoint_freq'] == 0:
                    base_logger.info('Saving the model at the end of epoch %d, iters %d' % (total_epoch, total_iters))
                    model.save(total_iters, total_epoch)
                
                if total_iters % opt['training']['val_freq'] == 0:
                    try:
                        # val_loader can be None
                        val_pbar = tqdm(val_loader)
                        for val_data in val_pbar:
                            model.set_input(val_data)
                            model.val()
                            Logger.display_current_results(total_iters, total_epoch, model.get_current_visuals(), phase='val')
                            Logger.print_current_logs(total_iters, total_epoch, model.get_current_log(), phase='val')
                    except:
                        base_logger.info('Validation dataloader maybe not exist, Skip validation.')
                        pass
                   
            base_logger.info('End of epoch %d / %d \t Time Taken: %d sec' %
                (total_epoch, opt['training']['max_epoch'], time.time() - epoch_start_time))
    elif opt['phase'] == 'val':
        base_logger.info('Begin Model Validation.')
        val_pbar = tqdm(val_loader)
        for val_data in val_pbar:
            model.set_input(val_data)
            model.val()
            Logger.display_current_results(total_iters, total_epoch, model.get_current_visuals(), phase='val')
            Logger.print_current_logs(total_iters, total_epoch, model.get_current_log(), phase='val')
