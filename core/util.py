import random
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

def postprocess(img):
	""" change pytorch[-1, 1] tensor to numpy """
	img = (img+1)/2*255
	img = img.permute(0,2,3,1)
	img = img.int().cpu().numpy().astype(np.uint8)
	return img


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



