
import torch
import os


from torch.utils.data import DataLoader,Dataset
import torch.optim as optim 
import torch.nn as nn

from myModels import  myLeNet, myResnet
from myDatasets import  get_cifar10_train_val_set,get_cifar10_unlabeled_set,get_labeleddata
from tool import load_parameters, train, fixed_seed
from torch.utils.data import ConcatDataset

# Modify config if you are conducting different models
from cfg import Resnet_cfg as cfg
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from tqdm import tqdm
from torchsummary import summary
def train_interface():
    
    """ input argumnet """
   
    data_root = cfg['data_root']
    model_type = cfg['model_type']
    num_out = cfg['num_out']
    num_epoch = cfg['num_epoch']
    split_ratio = cfg['split_ratio']
    seed = cfg['seed']
    
    # fixed random seed
    fixed_seed(seed)
    

    os.makedirs( os.path.join('./acc_log',  model_type), exist_ok=True)
    os.makedirs( os.path.join('./save_dir', model_type), exist_ok=True)    
    log_path = os.path.join('./acc_log', model_type, 'acc_' + model_type + '_.log')
    save_path = os.path.join('./save_dir', model_type)


    with open(log_path, 'w'):
        pass
    
    ## training setting ##
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu') 
    
    
    """ training hyperparameter """
    lr = cfg['lr']
    batch_size = cfg['batch_size']
    milestones = cfg['milestones']
    
    
    ## Modify here if you want to change your model ##
    # model = myLeNet(num_out=num_out)

    # model = myResnet()
    # model.fc = nn.Linear(512, num_out)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, num_out)

    # print model's architecture
    print(model)

    # Get your training Data 
    ## TO DO ##
    # You need to define your cifar10_dataset yourself to get images and labels for earch data
    # Check myDatasets.py 
    
    train_set, val_set =  get_cifar10_train_val_set(root=data_root, ratio=split_ratio)    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
    
    # define your loss function and optimizer to unpdate the model's parameters.
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,weight_decay=1e-6, nesterov=True)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=milestones, gamma=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    
    # We often apply crossentropyloss for classification problem. Check it on pytorch if interested
    criterion = nn.CrossEntropyLoss()
    
    # Put model's parameters on your device
    model = model.to(device)
    ### TO DO ### 
    # Complete the function train
    # Check tool.py
    train(model=model, train_loader=train_loader, val_loader=val_loader, 
          num_epoch=num_epoch, log_path=log_path, save_path=save_path,
          device=device, criterion=criterion, optimizer=optimizer, scheduler=scheduler)

"""
    #label unlabeled data
    unlabeled_dataset = get_cifar10_unlabeled_set('./p2_data/')
    unlabeled_loader = DataLoader(unlabeled_dataset,batch_size=128,num_workers=2)
    model.cpu()
    load_parameters(model,'./save_dir/ResNet/best_model.pt')
    model.eval()
    model.to(device)
    softmax = nn.Softmax(dim=1)
    labels = []
    remaining_idxs = []
    for batch_idx, (data, _ ) in enumerate(tqdm(unlabeled_loader)):
        with torch.no_grad():
            data = data.to(device) 
            output = model(data)
            prob, label = softmax(output).max(dim=1)
            idxs = torch.arange(data.shape[0]) + batch_idx * 128
            mask = (prob >= 0.7)
            labels.extend(label[mask].tolist())
            remaining_idxs.append(idxs[mask])
            del output
    remaining_idxs = torch.cat(remaining_idxs).tolist()
    new_images = [unlabeled_dataset.images[i] for i in remaining_idxs]
    unlabeled_dataset.images = new_images
    unlabeled_dataset.labels = labels

    
    
    new_model = models.resnet18(pretrained=True)
    new_model.fc = nn.Linear(512, num_out)
    print(new_model)


    new_train_set = ConcatDataset([train_set,unlabeled_dataset])
    new_label_set = val_set # ConcatDataset([val_set,label_val_set])
    new_train_loader = DataLoader(new_train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    new_val_loader = DataLoader(new_label_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
    optimizer = optim.SGD(new_model.parameters(), lr=lr, momentum=0.9,weight_decay=1e-6, nesterov=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    criterion = nn.CrossEntropyLoss()
    new_model = new_model.to(device)
    train(model=new_model, train_loader=new_train_loader, val_loader=new_val_loader, 
          num_epoch=num_epoch, log_path=log_path, save_path=save_path,
          device=device, criterion=criterion, optimizer=optimizer, scheduler=scheduler)
    summary(model,(3,32,32))
"""
if __name__ == '__main__':
    train_interface()




    