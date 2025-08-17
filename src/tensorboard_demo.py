from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter(log_dir='logs')
img_array = np.array(Image.open('dataset/train/bees_image/16838648_415acd9e3f.jpg'))

writer.add_image('test', img_array, 2, dataformats='HWC')

for i in range(100):
    writer.add_scalar('y=3x', 3*i, i)

writer.close()