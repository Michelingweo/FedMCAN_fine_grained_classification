import h5py
import numpy as np
from sklearn.decomposition import PCA
import torch
import matplotlib.pyplot as plt

import os
import json

from utils.setting import args_parser
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

from models.model import *
import time
from PIL import Image


args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')


project_path = '.'

meta_path = os.path.join(project_path,'data/metadata.pth')

metadata = torch.load(meta_path)

a = '034'
print(int(a))


# for i in range(len(metadata['img_ids'])):
#     metadata['img_ids'][i] =






