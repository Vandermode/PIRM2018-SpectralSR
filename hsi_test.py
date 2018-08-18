import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import models

from utility import *
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, LambdaLR

from hsi_setup import Engine, train_options


if __name__ == '__main__':
    """Training settings"""
    parser = argparse.ArgumentParser(
        description='Hyperspectral Image Super-Resolution')
    opt = train_options(parser)
    print(opt)
    scale_factor = opt.sf
    cuda = not opt.no_cuda

    """Setup Engine"""
    engine = Engine(opt)
    prefix = opt.prefix

    HSI2Tensor = partial(HSI2Tensor, use_2dconv=engine.get_net().use_2dconv)

    print('==> Preparing data..')

    mat_dataset = MatDataFromFolder('/data/zhangtao/pirm2018/validation')

    transform = Compose([
        lambda x: x/ 65535.,
    ])

    mat_transform = Compose([
        LoadMatHSI(input_key='lr'+str(scale_factor), gt_key='hr', 
        transform=transform, use_2dconv=engine.get_net().use_2dconv)
    ])
    
    mat_dataset = TransformDataset(mat_dataset, mat_transform)

    pirm_VL = DataLoader(
                    mat_dataset,
                    batch_size=1, shuffle=False,
                    num_workers=0, pin_memory=cuda
                )
    
    test_dataset = MatDataFromFolder('/data/zhangtao/pirm2018/testing_lr', with_filename=True)

    test_transform = Compose([
        LoadMatHSI(input_key='lr'+str(scale_factor), gt_key=None, with_filename=True,
        transform=transform, use_2dconv=engine.get_net().use_2dconv)
    ])

    test_dataset = TransformDataset(test_dataset, test_transform)
    
    pirm_Test = DataLoader(
                    test_dataset,
                    batch_size=1, shuffle=False,
                    num_workers=0, pin_memory=cuda        
                )

    engine.validate(pirm_VL, save=False)
    # engine.test(pirm_Test, savedir=os.path.join('/data/zhangtao/pirm2018/testing_lr/result/',opt.arch,opt.prefix))
    
