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
    dataloader = DataLoader(phase_dataset, sampler=data_sampler, worker_init_fn=worker_init_fn, **dataloader_args)
    val_dataloader = DataLoader(val_dataset, worker_init_fn=worker_init_fn, **dataloader_args)

    return dataloader, val_dataloader


''' create dataset '''
def create_dataset(opt):
    ''' loading Dataset() class from given file's name '''
    dataset_opt = opt['datasets'][opt['phase']]['which_dataset']
    dataset_file_name, dataset_class_name = 'data.'+dataset_opt['name'][0], dataset_opt['name'][1]

    phase_dataset = getattr(importlib.import_module(dataset_file_name), dataset_class_name)(**dataset_opt['args'])
    
    data_len = len(phase_dataset)
    split = dataset_opt.get('validation_split', 0)
    if split == 0.0: # phase is test or validation_split is 0.
        return phase_dataset, None
    
    if isinstance(split, int):
        assert split > 0
        assert split < data_len, "Validation set size is configured to be larger than entire dataset."
        valid_len = split
    else:
        valid_len = int(data_len * split)
    phase_dataset, val_dataset = random_split(dataset=phase_dataset, lengths=[data_len-valid_len, valid_len], generator=Generator().manual_seed(opt['seed']))

    if opt['global_rank']==0:
        logger.info('Dataset {:s} from {:s} is created. Size is {}. Phase is {}'.format(dataset_class_name, dataset_file_name, data_len, opt['phase']))
    else:
        # return phase_dataset, None  # return splited dataset to ensure validation step only executes on GPU 0, not used now 
        pass
    return phase_dataset, val_dataset

