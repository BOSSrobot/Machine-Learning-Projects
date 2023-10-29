import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from collections import OrderedDict

from workspace_utils import active_session
from ui_utility import preload_model

def get_dataloaders(data_dir, bs = 64): 
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    training_transforms = transforms.Compose([transforms.RandomRotation(20), 
                                         transforms.Resize(255),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    testing_transforms = transforms.Compose([transforms.Resize(255), 
                                         transforms.CenterCrop(224), 
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])


    data_transforms = {'training' : training_transforms, 'testing' : testing_transforms}
    
    image_datasets = {'training' : datasets.ImageFolder(train_dir, transform=data_transforms['training']),
                  'validation' : datasets.ImageFolder(train_dir, transform=data_transforms['testing']),
                  'testing' : datasets.ImageFolder(test_dir, transform=data_transforms['testing'])}

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {'training' : torch.utils.data.DataLoader(image_datasets['training'], batch_size=bs, shuffle=True),
                   'validation' : torch.utils.data.DataLoader(image_datasets['validation'], batch_size=bs),
                   'testing' : torch.utils.data.DataLoader(image_datasets['testing'], batch_size=bs)}
    
    return dataloaders, image_datasets['training'].class_to_idx

def get_classifer_structure(in_units, num_hidden_units):
    return OrderedDict([
            ('fc1', nn.Linear(in_units, num_hidden_units)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(0.2)),
            ('fc2', nn.Linear(num_hidden_units, 1024)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(0.2)),
            ('fc3', nn.Linear(1024, 102)),
            ('output', nn.LogSoftmax(dim=1))])

def get_model(arch_type, num_hidden_units, learning_rate, device, class_to_idx):

    model = preload_model(arch_type)
    for param in model.parameters():
        param.requires_grad = False


    classifier_structure = get_classifer_structure(model.classifier[0].in_features, num_hidden_units)
    classifier = nn.Sequential(classifier_structure)
    model.classifier = classifier
    model.class_to_idx = class_to_idx
    model.base = arch_type

    model.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    return model, criterion, optimizer

def train(model, criterion, optimizer, dataloaders, epochs, device):
    with active_session():
    
        num_val_batches = len(dataloaders['validation'])
        num_train_batches = len(dataloaders['training'])

        for epoch in range(1, epochs+1):

            # Training
            model.train()
            running_loss = 0

            for inputs, labels in dataloaders['training']:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                out = model.forward(inputs)
                loss = criterion(out, labels)

                running_loss += loss.item()
                
                loss.backward()
                optimizer.step()

            running_loss /= num_train_batches

            # Evaluation 
            model.eval()
            accuracy, loss = 0, 0

            for inputs, labels in dataloaders['validation']:

                inputs, labels = inputs.to(device), labels.to(device)

                ps = model.forward(inputs)
                pred = torch.exp(ps)
                _, out = pred.topk(1, dim = 1)

                bool_acc = labels.view(out.shape) == out
                accuracy += torch.mean(bool_acc.type(torch.FloatTensor)).item()

                batch_loss = criterion(ps, labels)
                loss += batch_loss.item()

            accuracy /= num_val_batches
            loss /= num_val_batches

            print(f"Epoch {epoch}/{epochs}: Training loss\t\t {running_loss:.3f}")
            print(f"Epoch {epoch}/{epochs}: Validation accuracy\t {accuracy:.3f}")
            print(f"Epoch {epoch}/{epochs}: Validation loss\t\t {loss:.3f}")
            
def save(model, optimizer, epochs, num_hidden_units, save_dir):
    
    classifier_structure = get_classifer_structure(model.classifier[0].in_features, num_hidden_units)
    
    checkpoint = {'state_dict' : model.state_dict(),
              'optim_dict' : optimizer.state_dict(),
              'epochs_done' : epochs,
              'classifier' : classifier_structure,
              'mapping' : model.class_to_idx, 
              'base' : model.base}

    torch.save(checkpoint, save_dir + 'checkpoint.pth')