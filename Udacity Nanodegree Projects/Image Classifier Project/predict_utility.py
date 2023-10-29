from PIL import Image
import json
from ui_utility import preload_model
from collections import OrderedDict

import torch
from torch import nn
import numpy as np

def get_class_name_wrapper(category_names):
    if category_names:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        
        return lambda cat_id: cat_to_name[cat_id]
    
    else: 
        return lambda cat_id: cat_id

def get_model(path, device):
    
    if device == 'gpu' and torch.cuda.is_available():
        map_location= lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    
    checkpoint = torch.load(path + ".pth", map_location=map_location)
    
    # Rebuild model
    model = preload_model(checkpoint['base'])
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(checkpoint['classifier'])
    model.classifier = classifier
    
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['mapping']
    
    model.eval()
    model.to(device)

    return model

def get_np_image(path):
    
    image = Image.open(path)
    image = image.resize((256, 256))

    width, height = image.size
    left, right = width//2 - 112, width//2 + 112
    up, down = height//2 - 112, height//2 + 112
    image = image.crop((left, up, right, down))
        
        
    arr = np.array(image).astype(np.float32)
    arr /= 255.0
    arr -= np.array([0.485, 0.456, 0.406])
    arr /= np.array([[0.229, 0.224, 0.225]])
    
    arr = arr.transpose((2, 0, 1))
    
    return arr

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    ax.set_title(title)
    
    return ax

def predict(image, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file    
    features = torch.unsqueeze(image, dim=0)
    features = features.to(device)
    
    ps = torch.exp(model.forward(features))
    
    probs, classes = ps.topk(topk)
    idx_to_class = {i : c for c, i in model.class_to_idx.items()}
    classes = np.vectorize(lambda key : idx_to_class[key])(classes.detach())
    
    return probs.detach(), classes