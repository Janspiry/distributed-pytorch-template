import argparse
from email.policy import default
import logging
from cv2 import phase
import tqdm
import time
import os
import warnings

import torch
import torch.multiprocessing as mp

import core.logger as Logger
import core.praser as Praser
import core.util as Util
from data import create_dataloader
from models import create_model

''' init_process_group '''
def main_worker(gpu, ngpus_per_node, opt):
    if 'local_rank' not in opt:
        opt['local_rank'] = opt['global_rank'] = gpu
    if opt['distributed']:
        torch.cuda.set_device(int(opt['local_rank']))
        print('using GPU {} for training'.format(int(opt['local_rank'])))
        torch.distributed.init_process_group(backend = 'nccl', 
            init_method = opt['init_method'],
            world_size = opt['world_size'], 
            rank = opt['global_rank'],
            group_name='mtorch'
        )
    '''set seed and and cuDNN environment '''
    torch.backends.cudnn.enabled = True
    warnings.warn('You have chosen to use cudnn for accleration. torch.backends.cudnn.enabled=True')
    Util.set_seed(opt['seed'])

    ''' set logger '''
    Logger.init_logger(opt=opt)
    phase_logger = logging.getLogger(opt['phase'])

    '''set model and dataset'''
    if opt['phase'] == 'train':
        train_loader, val_loader = create_dataloader(opt, phase='train')
    else:
        test_loader = create_dataloader(opt, phase=opt['phase']) 
    model = create_model(opt)

    total_epoch, total_iters = model.get_current_iters()
    phase_logger.info('Begin model {}.'.format(opt['phase']))
    if opt['phase'] == 'train':
        while True:
            epoch_start_time = time.time()
            total_epoch += 1
            train_loader.sampler.set_epoch(total_epoch) #  Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas use a different random ordering for each epoch
            if total_epoch >= opt['train']['n_epoch']: 
                phase_logger.info('Number of Epochs has reached the limit, End.')
                break

            for train_data in tqdm.tqdm(train_loader):
                if total_iters >= opt['train']['n_iter']: 
                    break
                total_iters += opt['datasets']['train']['dataloader_args']['batch_size']
                model.set_input(train_data)
                model.optimize_parameters()
                
                if opt['global_rank']==0:
                    if total_iters % opt['train']['display_freq'] == 0:
                        Logger.display_current_results(total_epoch, total_iters, model.get_current_visuals(), phase='train')

                    if total_iters % opt['train']['print_freq'] == 0:
                        logs = model.get_current_log()
                        Logger.print_current_logs(total_epoch, total_iters, logs, phase='train')
                        Logger.display_current_logs(total_epoch, total_iters, logs, phase='train')

                    if total_iters % opt['train']['save_checkpoint_freq'] == 0:
                        phase_logger.info('Saving the model at the end of epoch {:.0f}, iters {:.0f}'.format(total_epoch, total_iters))
                        model.save(total_iters, total_epoch)
                    
                    if total_iters % opt['train']['val_freq'] == 0:
                        try:
                            # val_loader can be None
                            for val_data in tqdm.tqdm(val_loader):
                                model.set_input(val_data)
                                model.val()
                                Logger.display_current_results(total_epoch, total_iters, model.get_current_visuals(), phase='val')
                                Logger.save_current_results(total_epoch, total_iters, model.save_current_results(), phase='val')
                                Logger.print_current_logs(total_epoch, total_iters, model.get_current_log(), phase='val')
                        except:
                            phase_logger.info('Validation error where dataloader maybe not exist, Skip it.')
            if opt['global_rank']==0:
                phase_logger.info('End of epoch {:.0f}/{:.0f}\t Time Taken: {:.2f} sec'.format(total_epoch, opt['train']['n_epoch'], time.time() - epoch_start_time))
    else:
        for data in tqdm.tqdm(test_loader):
            model.set_input(data)
            model.test()
            Logger.save_current_results(total_epoch, total_iters, model.save_current_results(), phase=opt['phase'])
            Logger.print_current_logs(total_epoch, total_iters, model.get_current_log(), phase=opt['phase'])
            Logger.print_current_logs(total_epoch, total_iters, model.get_current_log(), phase=opt['phase'])
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/base.json', help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train','test'], help='Run train or test', default='train')
    parser.add_argument('-b', '--batch', type=int, default=None, help='Batch size in every gpu')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-P', '--port', default='21012', type=str)

    ''' parser configs '''
    args = parser.parse_args()
    opt = Praser.parse(args)
    
    ''' cuda devices '''
    gpu_str = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    print('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))

    ''' use DistributedDataParallel(DDP) and multiprocessing for multi-gpu training'''
    # [Todo]: multi GPU on multi machine
    if opt['distributed']:
        ngpus_per_node = len(opt['gpu_ids']) # or torch.cuda.device_count()
        opt['world_size'] = ngpus_per_node
        opt['init_method'] = 'tcp://127.0.0.1:'+ args.port 
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        opt['world_size'] = 1 
        main_worker(0, 1, opt)