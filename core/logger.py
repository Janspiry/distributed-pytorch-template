from genericpath import exists
import os
import os.path as osp
import numpy as np
from PIL import Image
import logging
# from torch.utils.tensorboard import SummaryWriter\
from tensorboardX import SummaryWriter

tb_logger = None
gl_opt = None
def init_logger(opt, phase='base'):
    global gl_opt 
    gl_opt = opt
    setup_logger(None, opt['path']['log'], 'base', level=logging.INFO, screen=False)
    setup_logger(phase, opt['path']['log'], phase, level=logging.INFO, screen=False)
    if phase=='train':
        setup_logger('val', opt['path']['log'], 'val', level=logging.INFO, screen=False)
    setup_tblogger(log_dir=opt['path']['tb_logger'])    

def setup_tblogger(log_dir):
    global tb_logger
    tb_logger = SummaryWriter(log_dir)

def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
    '''set up logger'''
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    log_file = os.path.join(root, '{}.log'.format(phase))
    fh = logging.FileHandler(log_file, mode='a+')
    fh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        l.addHandler(sh)

def print_current_logs(epoch, i, errors, phase='train'):
    message = '<epoch: {:.0f}, iters: {:.0f}>'.format(epoch, i)
    for k, v in errors.items():
        message += k + ': ' + '{:.4f}'.format(v) + ' |'
    logger =  logging.getLogger(phase)
    logger.info(message)

def display_current_logs(epoch, i, errors, phase='train'):
    global tb_logger
    for k, v in errors.items():
        tb_logger.add_scalar(phase+"/"+str(k), v, i)

def display_current_results(epoch, i, results, phase='val'):
    global tb_logger
    for k, v in results.items():
        tb_logger.add_image(phase+"/"+str(k), v, i)


def postprocess(img):
    '''save results'''
    img = (img+1)/2*255
    img = img.permute(0,2,3,1)
    img = img.int().cpu().numpy().astype(np.uint8)
    return img

def save_current_results(epoch, i, results, phase='val'):
    global gl_opt
    result_path = os.path.join(gl_opt['path']['results'], phase)
    os.makedirs(result_path, exist_ok=True)
    result_path = os.path.join(result_path, str(i))
    os.makedirs(result_path, exist_ok=True)

    ''' get names and corresponding images from results[OrderedDict] '''
    try:
        names = results['name']
        outputs = postprocess(results['result'])
    except:
        raise NotImplementedError('You must specify the context of name and result in save_current_results functions of model.')
    for i in range(len(names)): 
        Image.fromarray(outputs[i]).save(os.path.join(result_path, names[i]))


