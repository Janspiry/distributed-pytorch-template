import random
import numpy as np
import torch
''' change pytorch[-1, 1] tensor to numpy '''
def postprocess(img):
    img = (img+1)/2*255
    img = img.permute(0,2,3,1)
    img = img.int().cpu().numpy().astype(np.uint8)
    return img


''' set random seed, gl_seed used in worker_init_fn function '''
def set_seed(seed, gl_seed=0):
  if seed >=0 and gl_seed>=0:
    seed += gl_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

  ''' change the deterministic and benchmark maybe cause uncertain convolution behavior. '''
  # speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
  if seed >=0 and gl_seed>=0:  # slower, more reproducible
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False
  else:  # faster, less reproducible
      torch.backends.cudnn.deterministic = False
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



