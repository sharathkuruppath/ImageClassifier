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
    ''' 
        Description :Predict the class (or classes) of an image using a trained deep learning model.
        params : none 
        returns : Parsed arguments from the command line
    '''
    
    parser=argparse.ArgumentParser(description='Prediction')
    parser.add_argument('--image_path',type=str,help='Path to the image directory eg : test/23/image_03382.jpg')
    parser.add_argument('--cat_to_name',type=str,default='cat_to_name.json',help='Path to the json file')
    parser.add_argument('--check_point',type=str,default='trained_model_vgg.pth',help='saved model e.g. trained_model_vgg.pth')
    
    parser.add_argument('--topk',type=int,default=5,help='display top k probabilities')
    parser.add_argument('--gpu', type=bool, default='True', help='True: gpu, False: cpu')
    args=parser.parse_args()
    return args

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(args,device):
    ''' 
        Description :loades the saved model from the checkpoint.
        params : filepath - checkpoint file
        returns : trained model
    '''
    
    if re.search('vgg',args.check_point):
        checkpoint = torch.load('trained_model_vgg.pth')
        model = models.vgg16(pretrained=True)
    elif re.search('alexnet',args.check_point):
        checkpoint = torch.load('trained_model_alexnet.pth')
        model = models.alexnet(pretrained=True)       
    #Citation Mentor Javier J in Ask a Mentor
    #model = getattr(models, checkpoint['arch'])(pretrained=True)
    model.class_to_idx = checkpoint['class_to_idx']     
    model.classifier = checkpoint['classifier']
    #model.input_size=checkpoint['input size']
    #model.output_size=checkpoint['output size']
    #model.hidden_units= checkpoint['hidden units']

    model.load_state_dict(checkpoint['state_dict'])  
    #optimizer = checkpoint['optimizer']
    #epochs = checkpoint['epochs']

    
   

            
    
   
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    return model

def process_image(image):
    ''' 
        Description :Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
        params : image - Image file along with its absolute path
          
        returns : Numpy image
    '''


    #print(image)
    pil_image = Image.open(image)

    # TODO: Process a PIL image for use in a PyTorch model
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
    
    #print(pil_image.size)
    np_image=np.array(pil_image)
    np_image=np_image/255
    
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image=(np_image-mean)/std
    #print ("Image:",np_image)
    np_image=np_image.transpose((2,0,1))
    #print ("Transposed image:",np_image)
    return np_image

def predict(np_image, model, device,topk):
    ''' 
        Description :Predict the class (or classes) of an image using a trained deep learning model.
        params : image_path - Image file along with its absolute path
                 model - saved model after training
                 topk - top 'k' probabilities which predicts the image correctly 
        returns : topk probabilities and their classes
    '''
    
    # TODO: Implement the code to predict the class from an image file
    #img= Image.open(image_path)
    # Citation Udacity Mentor Eban answer in Ask a Mentor
    processed_img=torch.from_numpy(np_image).type(torch.FloatTensor).unsqueeze_(0)
    #imshow(processed_img)
    with torch.no_grad():
        processed_img=processed_img.to(device)
        
        logps=model.forward(processed_img)
        ps = torch.exp(logps)
        # Citation Udacity Mentor Shibin M answer in Ask a Mentor
        probs, classes = torch.topk(ps,int(topk))
        top_p=probs.tolist()[0]
        classes=np.array(classes)
        index_to_class = {value: key for key, value in model.class_to_idx.items()}
        #idx_to_class=dict()
        #for key ,value in model.class_to_idx.items():
        #    idx_to_class.value=key
        #    idx_to_class.key=value
            
        top_class = [index_to_class[idx] for idx in classes[0]]
    return top_p, top_class

def flower_category(cat_to_name):
    ''' 
        Description :load the json file containing the labels and the names of the flowers.
        params : json file 
        returns : dictionary containing labels to the loaded image
    '''
    with open(cat_to_name, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def main():
    #parser=argparse.ArgumentParser()
    #args=parser.parse_args()
    args=input_args()
    #print("args:",args)
    device = torch.device("cuda" if torch.cuda.is_available()else "cpu")
    model=load_checkpoint(args,device)
    if args.image_path==None:
        file_path='test/1/image_06752.jpg'
    else:
        file_path=args.image_path
        
    home_path='./flowers/'
    image_path= home_path+file_path   
    
    np_image=process_image(image_path)
    top_probs, top_classes = predict(np_image, model,device,args.topk) 
    print(top_probs,top_classes)
    cat_to_name= flower_category(args.cat_to_name)
    flower_names=[]
    for classes in top_classes:
        flower_names.append(cat_to_name[str(classes)])
    
    for i ,j in enumerate(zip(flower_names,top_probs)):
        print(i+1, "Flower: {}, Probability : {:.2f}%".format(j[0],j[1]*100))
    
   
    #print(model)   
    

   
    

if __name__ == "__main__":
    main()