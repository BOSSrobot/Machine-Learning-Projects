import ui_utility as uu
import train_utility as tu

import torch
from collections import OrderedDict


args = [{'dest': 'data_dir',         'help': 'The directory containing the images'}, 
        {'dest': '--save_dir',       'help': 'The directory to save checkpoints. Enter none to save in current directory.'},
        {'dest': '--arch',           'help': 'Model base architecture. Must be vgg19, vgg16, or resnet34', 'default': 'vgg19'}, 
        {'dest': '--learning_rate',  'help': 'The learning rate', 'default': 0.003, 'type': float},
        {'dest': '--hidden_units',   'help': 'Number of hidden units', 'default': 4096, 'type': int},
        {'dest': '--epochs',         'help': 'The Number of epochs', 'default': 10, 'type': int},
        {'dest': '--gpu',            'help': 'Flag to use turn on gpu use', 'action': 'store_true', 'default': False}]

def main():
    
    in_arg = uu.parse_args(args)
    device = uu.get_device(in_arg.gpu)
    
    dataloaders, class_to_idx = tu.get_dataloaders(in_arg.data_dir, bs = 64)
    model, criterion, optimizer = tu.get_model(in_arg.arch, in_arg.hidden_units, in_arg.learning_rate, device, class_to_idx)
    
    print("Loaded Devices!!! Starting training...\n")
    tu.train(model, criterion, optimizer, dataloaders, in_arg.epochs, device)
    print("=====Finished training the model=====\n")
    
    if (in_arg.save_dir):
        # Allow user to also save in current directory by entering none
        if in_arg.save_dir != "none":
            in_arg.save_dir = in_arg.save_dir + "\\"
        else: 
            in_arg.save_dir = ""
            
        tu.save(model, optimizer, in_arg.epochs, in_arg.hidden_units, in_arg.save_dir)    
        print("Saved model!")
        
if __name__ == "__main__":
    main()
    
    