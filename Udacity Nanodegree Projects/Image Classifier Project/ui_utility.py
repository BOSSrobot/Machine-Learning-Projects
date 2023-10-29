import argparse
import torch 
from torchvision import models

def parse_args(args_list):
    parser = argparse.ArgumentParser()
   
    for arg in args_list: 
        n = arg.pop('dest')
        parser.add_argument(n, **arg)
            
    return parser.parse_args()

def get_device(gpu=True):
    return torch.device("cuda" if (torch.cuda.is_available() and gpu) else "cpu")

def preload_model(name):
    loader = {"vgg19": models.vgg19, "vgg16": models.vgg16, 'resnet34': models.resnet34}
    return loader[name](pretrained=True)    
    