import random
import numpy as np
import torch
''' set random seed '''
def set_seed(seed, base=0, is_set=True):
  seed += base
  assert seed >=0, '{} >= {}'.format(seed, 0)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)
  ''' change the deterministic and benchmark maybe cause uncertain convolution behavior. '''
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True

''' set parameter to gpu or cpu '''
def set_device(args):
  if torch.cuda.is_available():
    if isinstance(args, list):
      return (item.cuda() for item in args)
    elif isinstance(args, dict):
      for key, item in args.items():
        if item is not None:
            args[key] = item.cuda()
    else:
      return args.cuda()
  return args



