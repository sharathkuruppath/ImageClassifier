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

# Parsing Arguments
def input_args():

    parser=argparse.ArgumentParser(description='Prediction')
    parser.add_argument('image_path',type=str,help='Path to the image directory eg : test/23/image_03382.jpg')
    parser.add_argument('--category_names',type=str,default='cat_to_name.json',help='Path to the json file')
    parser.add_argument('checkpoint',type=str,help='saved model e.g. trained_model_vgg.pth')
    parser.add_argument('--topk',type=int,default=5,help='display top k probabilities')
    parser.add_argument('--gpu', action='store_true', help='True: gpu, False: cpu')
    
    return parser.parse_args()

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(args,device):
    
    if re.search('vgg16',args.checkpoint):
        checkpoint = torch.load('trained_model_vgg16.pth')
        model = models.vgg16(pretrained=True)
    elif re.search('alexnet',args.checkpoint):
        checkpoint = torch.load('trained_model_alexnet.pth')
        model = models.alexnet(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False

    model.class_to_idx = checkpoint['class_to_idx']     
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])  
    model.to(device)
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    ''' 

    pil_image = Image.open(image)
    # Citation Udacity Mentor Survesh answer in Ask a Mentor
    width,height=pil_image.size
    if (width > height):
        asp_ratio=width/height
        pil_image = pil_image.resize((int(asp_ratio*256),256))
    elif (height > width):
        asp_ratio=height/width
        pil_image = pil_image.resize((256,int(asp_ratio*256)))
        
    new_height=224
    new_width=224            
        
    width,height=pil_image.size
    left =(width-new_width)/2
    right = (width+new_width)/2
    top=(height-new_height)/2
    bottom=(height+new_height)/2

    pil_image=pil_image.crop((left,top,right,bottom))
    np_image=np.array(pil_image)
    np_image=np_image/255
    
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image=(np_image-mean)/std

    np_image=np_image.transpose((2,0,1))

    return np_image

def predict(np_image, model, device,topk):
    
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Citation Udacity Mentor Eban answer in Ask a Mentor
    processed_img=torch.from_numpy(np_image).type(torch.FloatTensor).unsqueeze_(0)    
    with torch.no_grad():
        processed_img=processed_img.to(device)
        
        logps=model.forward(processed_img)
        ps = torch.exp(logps)
        # Citation Udacity Mentor Shibin M answer in Ask a Mentor
        probs, classes = torch.topk(ps,int(topk))
        top_p=probs.tolist()[0]
        classes=np.array(classes)
        index_to_class = {value: key for key, value in model.class_to_idx.items()}            
        top_class = [index_to_class[idx] for idx in classes[0]]
        
    return top_p, top_class

def flower_category(cat_to_name):
    
    with open(cat_to_name, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def torch_device(inp_device):
   
    device = torch.device('cuda' if inp_device and torch.cuda.is_available() else 'cpu')
    
    return device 

def main():

    #Parsing Input Args
    args=input_args()
    
    #CPU or GPU
    device=torch_device(args.gpu)
    
    #Load Checkpoint
    model=load_checkpoint(args,device)
    
    #Process Image
    np_image=process_image(args.image_path)
    
    #Predict Classes
    top_probs, top_classes = predict(np_image, model,device,args.topk) 
    print(top_probs,top_classes)
    
    cat_to_name= flower_category(args.category_names)
    flower_names=[]    
    for classes in top_classes:
        flower_names.append(cat_to_name[str(classes)])
    
    for i ,j in enumerate(zip(flower_names,top_probs)):
        print(i+1, "Flower: {}, Probability : {:.2f}%".format(j[0],j[1]*100))

if __name__ == "__main__":
    main()