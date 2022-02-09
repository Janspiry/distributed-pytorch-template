import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist

def set_seed(seed, base=0, is_set=True):
  ''' set random seed '''
  seed += base
  assert seed >=0, '{} >= {}'.format(seed, 0)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True


def set_device(args):
  ''' set parameter to gpu or cpu '''
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



