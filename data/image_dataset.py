import torch.utils.data as data
from torchvision import transforms
from .auto_augment import AutoAugment, ImageNetAutoAugment
from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')

class Dataset(data.Dataset):
    def name(self):
        return 'ImageDataset'
    def __init__(self, opt, phase='train', image_size=[256, 256], loader=pil_loader):
        root = opt['data_root']
        imgs = make_dataset(root)
        self.imgs = imgs
        if phase == 'train':
            self.tfs = transforms.Compose([
                 transforms.Resize((image_size[0], image_size[1])),
                 ImageNetAutoAugment(),
                 transforms.ToTensor()
            ])
        else:
            self.tfs = transforms.Compose([
                 transforms.Resize((image_size[0], image_size[1])),
                 transforms.ToTensor()
            ])
        
        self.loader = loader

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.loader(path)
        img = self.tfs(img)
        ret['input'] = img
        ret['path'] = path.rsplit("/")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)
