import os 
from PIL import Image

path = '/data/jlw/datasets/comofod'

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


for item in os.listdir(path):
    print(item)
    if is_image_file(item):
        Image.open(os.path.join(path,item)).convert("RGB")