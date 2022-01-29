import os
import os.path as osp
import logging
from torch.utils.tensorboard import SummaryWriter

tb_logger = None
def init_logger(opt):
    setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    setup_logger('train', opt['path']['log'], 'train', level=logging.INFO, screen=True)
    setup_logger('val', opt['path']['log'], 'val', level=logging.INFO, screen=False)
    setup_logger('test', opt['path']['log'], 'test', level=logging.INFO, screen=False)
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
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        l.addHandler(sh)

def print_current_logs(self, epoch, i, errors, phase='base'):
    message = '<epoch: %d, iters: %d>'.format(epoch, i)
    for k, v in errors.items():
        v = ['%.4f' % iv for iv in v]
        message += k + ': ' + ', '.join(v) + ' | '
    logger =  logging.getLogger(phase)
    logger.info(message)

def display_current_logs(self, epoch, i, errors, phase='base'):
    global tb_logger
    for k, v in errors.items():
        tb_logger.add_scalar(phase+":"+str(k), v, i)

def display_current_results(self, epoch, i, results, phase='base'):
    global tb_logger
    for k, v in results.items():
        tb_logger.add_image(phase+":"+str(k), v, i)