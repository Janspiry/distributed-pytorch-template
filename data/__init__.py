import importlib
import logging
from functools import partial

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import core.util as Util
logger = logging.getLogger('base')

''' create dataloader '''
def create_dataloader(opt, phase='train'):
    '''create dataset and set random seed'''
    worker_init_fn = partial(Util.set_seed, base=opt['seed'])
    dataset_opt = opt['datasets'][phase]
    dataset = create_dataset(opt, dataset_opt, phase)

    '''create datasampler'''
    data_sampler = None
    if opt['distributed']:
        data_sampler = DistributedSampler(dataset, 
            num_replicas=opt['world_size'], rank=opt['global_rank'])

    dataloader = DataLoader(
        dataset,
        batch_size=dataset_opt['batch_size'],
        num_workers=dataset_opt['num_workers'],
        shuffle=(data_sampler is None) and (phase=='train') and dataset_opt['use_shuffle'],
        pin_memory=dataset_opt['pin_memory'],
        sampler=data_sampler, 
        worker_init_fn=worker_init_fn
    )
    return dataloader


''' create dataset '''
def create_dataset(opt, dataset_opt, phase):
    dataset_name = 'data.'+dataset_opt['name']
    ''' loading Dataset() class from given file's name '''
    dataset = importlib.import_module(dataset_name).Dataset(dataset_opt, phase=phase)
    if opt['global_rank']==0:
        logger.info('Dataset [{:s} - {:s}] is created. Size is {}. Phase is {}'.format(dataset.name(),
                                                            dataset_opt['name'], len(dataset), phase))
    return dataset

