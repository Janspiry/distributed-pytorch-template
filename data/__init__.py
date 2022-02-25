import importlib
import logging
from functools import partial
from sklearn.utils import shuffle

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch import Generator
from torch.utils.data import random_split

import core.util as Util
logger = logging.getLogger('base')

''' create dataloader '''
def create_dataloader(opt, phase='train', **dataloader_args):
    '''create dataset and set random seed'''
    worker_init_fn = partial(Util.set_seed, gl_seed=opt['seed'])
    dataset_opt = opt['datasets'][phase]
    if phase == 'train':
        dataset, val_dataset = create_dataset(opt, dataset_opt, phase)
    else:
        dataset = create_dataset(opt, dataset_opt, phase)
    
    '''create datasampler'''
    data_sampler = None
    if opt['distributed']:
        data_sampler = DistributedSampler(dataset, shuffle=dataloader_args.get('shuffle', False), num_replicas=opt['world_size'], rank=opt['global_rank'])
        dataloader_args.update({'shuffle':False}) # sampler option is mutually exclusive with shuffle 
    dataloader = DataLoader(dataset, sampler=data_sampler, worker_init_fn=worker_init_fn, **dataloader_args)

    if phase == 'train':
        val_dataloader = DataLoader(val_dataset, worker_init_fn=worker_init_fn, **dataloader_args)
        return dataloader, val_dataloader
    return dataloader


''' create dataset '''
def create_dataset(opt, dataset_opt, phase):
    dataset_name = 'data.'+dataset_opt['name']
    ''' loading Dataset() class from given file's name '''
    dataset = importlib.import_module(dataset_name).Dataset(dataset_opt, phase=phase)
    data_len = len(dataset)

    if opt['global_rank']==0:
        logger.info('Dataset [{:s} - {:s}] is created. Size is {}. Phase is {}'.format(dataset.name(),
                                                            dataset_opt['name'], data_len, phase))
    if phase == 'train':
        split = dataset_opt['validation_split']
        if split == 0.0:
            return dataset, None
        if isinstance(split, int):
            assert split > 0
            assert split < data_len, "Validation set size is configured to be larger than entire dataset."
            valid_len = split
        else:
            valid_len = int(data_len * split)
        train_dataset, val_dataset = random_split(dataset=dataset, lengths=[data_len-valid_len, valid_len], generator=Generator().manual_seed(opt['seed']))
        return train_dataset, val_dataset

    return dataset

