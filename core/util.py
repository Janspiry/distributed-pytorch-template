import random
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid
import math

def tensor2img(tensor, in_type='pt', out_type=np.uint8):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    in_type: zc(zero center)[-1, 1], pt(pytorch)[0,1]
    '''
    if in_type == 'pt':
        tensor = tensor.squeeze().clamp_(*[0, 1])  # clamp
    elif in_type == 'zc':
        tensor = tensor.squeeze().clamp_(*[-1, 1])  # clamp
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError('Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if in_type == 'pt':
        img_np = img_np * 255
    elif in_type == 'zc':
        img_np = (img_np+1) * 127.5
    return img_np.round().astype(out_type)

def postprocess(images, in_type='pt', out_type=np.uint8):
    return [tensor2img(image, in_type=in_type, out_type=out_type) for image in images]


def set_seed(seed, gl_seed=0):
    """  set random seed, gl_seed used in worker_init_fn function """
    if seed >=0 and gl_seed>=0:
        seed += gl_seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    ''' change the deterministic and benchmark maybe cause uncertain convolution behavior. 
        speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html '''
    if seed >=0 and gl_seed>=0:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def set_gpu(args, distributed=False, rank=0):
    """ set parameter to gpu or ddp """
    if distributed and isinstance(args, torch.nn.Module):
        return DDP(args.cuda(), device_ids=[rank], output_device=rank, broadcast_buffers=True, find_unused_parameters=True)
    else:
        return args.cuda()
        
def set_device(args, distributed=False, rank=0):
    """ set parameter to gpu or cpu """
    if torch.cuda.is_available():
        if isinstance(args, list):
            return (set_gpu(item, distributed, rank) for item in args)
        elif isinstance(args, dict):
            return {key:set_gpu(args[key], distributed, rank) for key in args}
        else:
            args = set_gpu(args, distributed, rank)
    return args



