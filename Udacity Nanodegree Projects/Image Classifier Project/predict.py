import ui_utility as uu
import predict_utility as pu
from PIL import Image
import torch

args = [{'dest': 'path',             'help': 'The filepath of the image'}, 
        {'dest': 'checkpoint',       'help': 'The filename of the checkpoint. Ex: checkpoint'},
        {'dest': '--topk',           'help': 'Set the amount of likely classes to return', 'default': 1, 'type': int}, 
        {'dest': '--category_names', 'help': 'The name of the file with the mapping from category to real name'},
        {'dest': '--gpu',            'help': 'Flag to use turn on gpu use', 'action': 'store_true', 'default': False}]


def main():

    in_arg = uu.parse_args(args)
    device = uu.get_device(in_arg.gpu)
   
    get_class_name = pu.get_class_name_wrapper(in_arg.category_names)    
    model = pu.get_model(in_arg.checkpoint, device)
    
    np_image = pu.get_np_image(in_arg.path)
    features = torch.from_numpy(np_image)
    
    probs, classes = pu.predict(features, model, device, in_arg.topk)
    
    probs, classes = probs.reshape(-1), classes.reshape(-1)     
    top_index = probs.argmax()
   
    print(f"The most likely predicted class is {get_class_name(classes[top_index])} with probability {probs[top_index]:.4f}")
    if in_arg.topk > 1: 
        print(f"The top {in_arg.topk} classes are show below:")
        for c, p in zip(classes, probs): 
            print(f"Class: {get_class_name(c) :<40} Probability: {p:.4f}")    
        
if __name__ == "__main__":
    main()
    
    