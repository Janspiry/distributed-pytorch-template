import torch.utils.data
import importlib
import logging
from functools import partial

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import core.util as Util
logger = logging.getLogger('base')
def create_dataloader(opt, phase='train'):
    '''create dataset and set random seed'''
    worker_init_fn = partial(Util.set_seed, base=opt['seed'])
    dataset_opt = opt['datasets'][phase]
    dataset = create_dataset(dataset_opt, phase)

    '''create datasampler'''
    data_sampler = None
    if opt['distributed']:
        data_sampler = DistributedSampler(dataset, 
            num_replicas=opt['world_size'], rank=opt['global_rank'])

    '''create dataloader'''
    dataloader = DataLoader(
        dataset,
        batch_size=dataset_opt['batch_size']//opt['world_size'],
        num_workers=dataset_opt['num_workers'],
        shuffle=(phase=='train') and dataset_opt['use_shuffle'],
        pin_memory=dataset_opt['pin_memory'],
        sampler=data_sampler, 
        worker_init_fn=worker_init_fn
    )
    return dataloader

def create_dataset(dataset_opt, phase):
    '''create dataset, loading Dataset() class from given file's name '''
    dataset_name = 'data.'+dataset_opt['name']
    dataset = importlib.import_module(dataset_name).Dataset(dataset_opt, phase=phase)
    
    logger.info('Dataset [{:s} - {:s}] is created. Size is {}. Phase is {}'.format(dataset.name(),
                                                        dataset_opt['name'], len(dataset), phase))
    return dataset

