import argparse
import os
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from workspace_utils import active_session
from PIL import Image
import numpy as np
import json
import re

def input_args():
    
    parser=argparse.ArgumentParser(description='Training')
    parser.add_argument('data_dir',type=str,help='Path to the image directory')
    parser.add_argument('--arch',type=str,default='vgg16',help='Architecture eg: vgg16 or alexnet')
    parser.add_argument('--learning_rate',type=float,default=0.001,help='Learning Rate')
    
    parser.add_argument('--hidden_units',type=int,help='Hidden Units for the model')
    parser.add_argument('--epochs',type=int,default=7,help='Number of Epochs')
    parser.add_argument('--save_dir',type=str,default='.',\
                        help='Path to the directory where the checkpoint is saved')
    parser.add_argument('--gpu', action='store_true', help='True: gpu, False: cpu')
    
    return parser.parse_args()

def dataseperation(data_dir):
    
    #Seperating Data into train,validation and test datasets
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    #Defining transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])    
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    
    train_dataset = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    valid_dataset = datasets.ImageFolder(data_dir + '/valid', transform=valid_transforms)
    test_dataset = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
    
    #Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=25, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=25, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=25)
    
    dataloaders={'train':trainloader,'valid':validloader,'test':testloader}
    datatransforms={'train':train_transforms,'valid':valid_transforms,'test':test_transforms}
    datacollection={'train':train_dataset,'valid':valid_dataset,'test':test_dataset}
    
    dataiter=iter(trainloader)
    images,labels=dataiter.next()
    print(type(images))
    print(images.shape)
    return dataloaders,datatransforms,datacollection

    
def torch_device(inp_device):
   
    device = torch.device('cuda' if inp_device and torch.cuda.is_available() else 'cpu')
    
    return device      

def create_model(device,args):
    # Citation Mentor Prasun S in Ask a Mentor 
    print('Args:',args.arch)
    
    if re.search('vgg16',args.arch):       
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad=False
        if args.hidden_units==None:
            args.hidden_units=1568
        model.classifier=nn.Sequential(nn.Linear(25088,6272),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.2),
                                       nn.Linear(6272,args.hidden_units),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.2),
                                       nn.Linear(args.hidden_units,102),
                                       nn.LogSoftmax(dim=1))
        
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), args.learning_rate)
  
    elif args.arch == 'alexnet':
        model=models.alexnet(pretrained=True)
        for param in model.parameters():
            param.requires_grad=False
        if args.hidden_units==None:
            args.hidden_units=2304
        model.classifier=nn.Sequential(nn.Linear(9216,4608),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.2),
                                       nn.Linear(4608,args.hidden_units),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.2),
                                       nn.Linear(args.hidden_units,102),
                                       nn.LogSoftmax(dim=1)) 
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), args.learning_rate)

    try:
        return model,criterion,optimizer
    except Exception as err:
        raise Exception("Architecture {} not supported , please choose vgg16 or alexnet".format(args.arch))

def training(model,criterion,optimizer,args,device,dataloaders):
    
    model.to(device)
    steps=0
    running_loss=0
    print_every=60
    
    trainloader=dataloaders['train']
    dataiter=iter(trainloader)
    images,labels=dataiter.next()
    print(type(images))
    print(images.shape)
    
    validloader=dataloaders['valid']
    with active_session():
        for epoch in range(args.epochs):
            #print('training')
            for i, (images, labels) in enumerate (trainloader):
                #print('inside trainloader')
                steps+=1
                images,labels= images.to(device),labels.to(device)
                optimizer.zero_grad()

                #forward pass
                logits=model.forward(images)
                loss=criterion(logits,labels)

                #backpropagation
                loss.backward()
                #updating weights
                optimizer.step()

                running_loss+=loss.item()
                if steps % print_every == 0 :
                    model.eval()
                    valid_loss=0
                    accuracy=0
                    #turning backprop off
                    with torch.no_grad():
                        for images,labels in validloader:
                            #print('validation')
                            images,labels= images.to(device),labels.to(device)

                            #forward pass
                            logps=model.forward(images)
                            batch_loss=criterion(logps,labels)
                            valid_loss+=batch_loss.item()

                            #Calculate the accuracy
                            ps=torch.exp(logps)
                            top_ps,top_class= ps.topk(1,dim=1)
                            equality= top_class ==labels.view(*top_class.shape)
                            accuracy+=torch.mean(equality.type(torch.FloatTensor)).item()


                    print(f"Epoch: {epoch+1}/{args.epochs}.."
                          f"Training_loss: {running_loss/(print_every):.3f}.."
                          f"Validation loss: {valid_loss/len(validloader):.3f}.."
                          f"Validation Accuracy: {accuracy/len(validloader):.3f}..")
                    running_loss=0
                    model.train()
    return model  

def testing(device,model,criterion,dataloaders):
    
    model.eval()
    testloader=dataloaders['test']
    test_loss=0
    test_accuracy=0
    with torch.no_grad():
        for images,labels in testloader:
            #print('test')
            images,labels= images.to(device),labels.to(device)
            #forward pass
            logps=model.forward(images)
            batch_loss=criterion(logps,labels)
            test_loss+=batch_loss.item()

            #Calculate the accuracy
            ps=torch.exp(logps)
            top_ps,top_class= ps.topk(1,dim=1)
            equality= top_class ==labels.view(*top_class.shape)
            test_accuracy+=torch.mean(equality.type(torch.FloatTensor))
        print("Test loss: {:.3f} Test Accuracy:{:.3f}".format(test_loss/len(testloader),test_accuracy/len(testloader)))                               
    return  test_loss/len(testloader),  test_accuracy

def save_checkpoint(model,args,optimizer,datacollection):
    
   
    train_dataset=datacollection['train']
    model.class_to_idx = train_dataset.class_to_idx
    
    #Citation Mentor Tilak D in Ask a Mentor
    if re.search('vgg16',args.arch):
        filename='trained_model_vgg16.pth' 
        checkpoint = {
            'input size': 25088,
            'output size': 102,
            'hidden units':args.hidden_units,
            'state_dict': model.state_dict(),
            'epochs': args.epochs,
            'arch': args.arch,
            'classifier': model.classifier,
            'optimizer': optimizer.state_dict(),
            'class_to_idx': model.class_to_idx,
            'lr':args.learning_rate
             }
        

    elif args.arch == 'alexnet':
        filename='trained_model_alexnet.pth'
        checkpoint = {
            'input size': 9216,
            'output size': 102,
            'hidden units':args.hidden_units,
            'state_dict': model.state_dict(),
            'epochs': args.epochs,
            'arch':args.arch,
            'classifier': model.classifier,
            'optimizer': optimizer.state_dict(),
            'class_to_idx': model.class_to_idx,
            'lr':args.learning_rate
             }   
    torch.save(checkpoint,(args.save_dir +'/'+filename))
    
def main():
    
    args=input_args()
    print("Args:",args)

    #Creating Datasets
    dataloaders,datatransforms,datacollection=dataseperation(args.data_dir)
    
    #CPU or GPU
    device=torch_device(args.gpu)

    try :
        
        #Creating Model
        model,criterion,optimizer=create_model(device,args)
        
        #Training Model
        model=training(model,criterion,optimizer,args,device,dataloaders)
        
        #Testing Model with test dataset
        test_loss,test_loader=testing(device,model,criterion,dataloaders)
    
        #Saving Checkpoint
        save_checkpoint(model,args,optimizer,datacollection)
        
    except  Exception as err: 
        print(err)
      

        

if __name__ == "__main__":
    main()