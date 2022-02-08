import torch.utils.data
import importlib
import logging
logger = logging.getLogger('base')
def create_dataloader(opt, phase='train'):
    '''create dataloader'''
    dataset_opt = opt['datasets'][phase]
    dataset = create_dataset(dataset_opt, phase)
    if phase == 'train':
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            num_workers=dataset_opt['num_workers'],
            shuffle=dataset_opt['use_shuffle'],
            pin_memory=dataset_opt['pin_memory']
        )
    else:
        dataloader =  torch.utils.data.DataLoader(
            dataset, 
            batch_size=dataset_opt['batch_size'],
            shuffle=False, 
            num_workers=dataset_opt['num_workers'], 
            pin_memory=dataset_opt['pin_memory']
        )
    if opt['distributed']:
        pass
    elif len(opt['gpu_ids'])>0:
        pass
    return dataloader

def create_dataset(dataset_opt, phase):
    '''create dataset, loading Dataset() class from given file's name '''
    dataset_name = 'data.'+dataset_opt['name']
    dataset = importlib.import_module(dataset_name).Dataset(dataset_opt, phase=phase)
    
    logger.info('Dataset [{:s} - {:s}] is created. Size is {}. Phase is {}'.format(dataset.name(),
                                                        dataset_opt['name'], len(dataset), phase))
    return dataset

