import importlib
import logging
from functools import partial
import numpy as np

from torch.utils.data.distributed import DistributedSampler
from torch import Generator, randperm
from torch.utils.data import DataLoader, Subset

import core.util as Util
logger = logging.getLogger('base')

''' create dataloader '''
def create_dataloader(opt):
    '''create dataset and set random seed'''
    dataloader_args = opt['datasets'][opt['phase']]['dataloader']['args']
    worker_init_fn = partial(Util.set_seed, gl_seed=opt['seed'])

    phase_dataset, val_dataset = create_dataset(opt)

    '''create datasampler'''
    data_sampler = None
    if opt['distributed']:
        data_sampler = DistributedSampler(phase_dataset, shuffle=dataloader_args.get('shuffle', False), num_replicas=opt['world_size'], rank=opt['global_rank'])
        dataloader_args.update({'shuffle':False}) # sampler option is mutually exclusive with shuffle 
    
    ''' create dataloader and validation dataloader '''
    dataloader = DataLoader(phase_dataset, sampler=data_sampler, worker_init_fn=worker_init_fn, **dataloader_args)
    if val_dataset is not None:
        val_dataloader = DataLoader(val_dataset, worker_init_fn=worker_init_fn, **dataloader_args) # val_dataloader don't use DistributedSampler to run only GPU 0!
    else:
        val_dataloader = None
    return dataloader, val_dataloader


''' create dataset '''
def create_dataset(opt):
    ''' loading Dataset() class from given file's name '''
    dataset_opt = opt['datasets'][opt['phase']]['which_dataset']
    dataset_file_name, dataset_class_name = 'data.'+dataset_opt['name'][0], dataset_opt['name'][1]
    phase_dataset = getattr(importlib.import_module(dataset_file_name), dataset_class_name)(**dataset_opt['args'])
    val_dataset = None

    valid_len = 0
    if 'debug' in opt['name']:
        data_len = opt['debug']['data_len']
    else:
        data_len = len(phase_dataset)
    split = dataset_opt.get('validation_split', 0)    
    
    ''' divide validation dataset '''
    if split > 0.0 or 'debug' in opt['name']: # split==0 when phase is test or validation_split is 0.
        if isinstance(split, int):
            assert split < data_len, "Validation set size is configured to be larger than entire dataset."
            valid_len = split
        else:
            valid_len = int(data_len * split)
        data_len -= valid_len
        phase_dataset, val_dataset = subset_split(dataset=phase_dataset, lengths=[data_len, valid_len], generator=Generator().manual_seed(opt['seed']))
    
    if opt['global_rank']==0:
        logger.info('Dataset [{:s} from {:s}] is created. Size is {} and Phase is {}.'.format(dataset_class_name, dataset_file_name, data_len, opt['phase']))
        if opt['phase']=='train':
            logger.info('Dataset [{:s} from {:s}] is created. Size is {} and Phase is {}.'.format(dataset_class_name, dataset_file_name, valid_len, 'val'))   
    return phase_dataset, val_dataset

''' split a dataset into non-overlapping new datasets of given lengths. main code is from random_split function in pytorch '''
def subset_split(dataset, lengths, generator):
    indices = randperm(sum(lengths), generator=generator).tolist()
    Subsets = []
    for offset, length in zip(np.add.accumulate(lengths), lengths):
        if length == 0:
            Subsets.append(None)
        else:
            Subsets.append(Subset(dataset, indices[offset - length : offset]))
    return Subsets
