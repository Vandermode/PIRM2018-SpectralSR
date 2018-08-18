# There are functions for creating a train and validation iterator.
import torch
import torchvision
import random
import cv2

try: 
    from .util import *
except:
    from util import *

from torchvision.transforms import Compose, ToPILImage, ToTensor, RandomHorizontalFlip, RandomChoice
from torch.utils.data import DataLoader, Dataset
from torchnet.dataset import TransformDataset, SplitDataset, TensorDataset, ResampleDataset

from PIL import Image


class HSI2Tensor(object):
    """
    Transform a numpy array with shape (C, H, W)
    into torch 4D Tensor (1, C, H, W) or (C, H, W)
    """
    def __init__(self, use_2dconv):
        self.use_2dconv = use_2dconv

    def __call__(self, hsi):
        if self.use_2dconv:
            img = torch.from_numpy(hsi)
        else:
            img = torch.from_numpy(hsi[None])
        # for ch in range(hsi.shape[0]):
        #     hsi[ch, ...] = minmax_normalize(hsi[ch, ...])
        # img = torch.from_numpy(hsi)        
        return img.float()


class LoadMatHSI(object):
    def __init__(self, input_key, gt_key=None, transform=None, input_transform=None, use_2dconv=True, with_filename=False):
        self.gt_key = gt_key
        self.input_key = input_key
        self.transform = transform
        self.input_transform = input_transform
        self.use_2dconv = use_2dconv
        self.with_filename = with_filename
    
    def __call__(self, inputs):
        def __transform(mat, key, is_input):            
            if self.transform:
                res = self.transform(mat[key][:].transpose((2,0,1)))                
            else:
                res = mat[key][:].transpose((2,0,1))
            
            if is_input and self.input_transform is not None:
                res = self.input_transform(res)

            if self.use_2dconv:
                res = torch.from_numpy(res).float()
            else:
                res = torch.from_numpy(res[None]).float()  # for 3D net
            return res 

        if self.with_filename:
            mat, filename = inputs
        else:
            mat, filename = inputs, -1
        
        input = __transform(mat, self.input_key, True)
        if self.gt_key is not None:
            gt = __transform(mat, self.gt_key, False)
        else:
            gt = -1
            
        return input, gt, filename


class LoadMatKey(object):
    def __init__(self, key, withfile_name=False):
        self.key = key
        self.with_filename = withfile_name
    
    def __call__(self, inputs):
        if self.with_filename:
            mat, filename = inputs
        else:
            mat = inputs
        
        item = mat[self.key][:].transpose((2,0,1))        
        return item

# Define Datasets
class DatasetFromFolder(Dataset):
    """Wrap data from image folder"""
    def __init__(self, data_dir, suffix='png'):
        super(DatasetFromFolder, self).__init__()
        self.filenames = [
            os.path.join(data_dir, fn) 
            for fn in os.listdir(data_dir) 
            if fn.endswith(suffix)
        ]

    def __getitem__(self, index):
        img = Image.open(self.filenames[index]).convert('L')
        return img

    def __len__(self):
        return len(self.filenames)


class MatDataFromFolder(Dataset):
    """Wrap mat data from folder"""
    def __init__(self, data_dir, load=loadmat, suffix='mat', size=None, with_filename=False):
        super(MatDataFromFolder, self).__init__()
        self.filenames = [
            os.path.join(data_dir, fn) 
            for fn in os.listdir(data_dir)
            if fn.endswith(suffix)
        ]
        self.load = load
        self.with_filename = with_filename

        if size and size <= len(self.filenames):
            self.filenames = self.filenames[:size]

    def __getitem__(self, index):
        filename = self.filenames[index]
        mat = self.load(filename)
        if self.with_filename:
            return mat, os.path.basename(filename).split('.')[0]
        else:
            return mat

    def __len__(self):
        return len(self.filenames)


class PIRMDataset(Dataset):
    def __init__(self, hr_dataset, lr_dataset, size=None, upsample=False, transform=None):
        super(PIRMDataset, self).__init__()
        self.hr_dataset = hr_dataset
        self.lr_dataset = lr_dataset
        self.transform = transform
        self.size = size
        self.upsample = upsample

    def __len__(self):
        return self.size or len(self.hr_dataset)

    def __getitem__(self, idx):        
        hr = self.hr_dataset[idx]
        lr = self.lr_dataset[idx]
        if self.upsample:
            lr = cv2.resize(lr.transpose((1, 2, 0)), (96, 96)).transpose((2, 0, 1))
        # mode = random.randint(0, 7)
        hr = self.transform(hr)
        lr = self.transform(lr)

        return lr, hr
