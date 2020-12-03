import re
import pandas as pd
from numpy.random import RandomState
from torchnlp.word_to_vector import FastText
from torch.utils import data
import spacy
import pickle
import numpy as np
import math
from efficientnet_pytorch import EfficientNet
from PIL import Image
import random
from collections import Counter
import nltk
import os
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pickle
from utils import *

MAX_NUM_IMGS = 9   # maximum images allowed for making decision

nltk.download("punkt")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define model
class Eff_Classifier(nn.Module):
    """
    image classifier based on efficient net
    """
    def __init__(self,n_classes):
        super(Eff_Classifier, self).__init__()
        self.effnet =  EfficientNet.from_pretrained('efficientnet-b5')
        self.l1 = nn.Linear(1000 , 256)
        self.dropout = nn.Dropout(0.5)
        self.l2 = nn.Linear(256,n_classes) # 6 is number of classes
        self.relu = nn.LeakyReLU()
    def forward(self, input):
        x = self.effnet(input)
        x = x.view(x.size(0),-1)
        x = self.dropout(self.relu(self.l1(x)))
        x = self.l2(x)
        return x


class Image_Classifier(nn.Module):
    def __init__(self,n_classes):
        super(Image_Classifier, self).__init__()
        self.eff_net = Eff_Classifier(n_classes)
        checkpoint = torch.load('models/eff_b5.pt', map_location=torch.device('cpu'))
        self.eff_net.load_state_dict(checkpoint['model'])
        with open('models/idx2label_image.pickle', 'rb') as f:
            self.idx2label = pickle.load(f)

    def collate_fn(self,batch):
        """
        takes in a list of image path and output the transformed image tensor
        :param batch: string. A list of image path
        :return: the transformed image tensor for the given image path
        """
        # batch::str --> a list of image path
        try:
            transpose = transforms.Compose(
                [transforms.Resize((456, 456)), transforms.ToTensor(),
                 transforms.Normalize((0.4302, 0.4575, 0.4539), (0.2361, 0.2347, 0.2432))])
            imgs = []
            for i in batch[:30]: # takes at most 30 images for memory limit
                try:
                    pic = Image.open(i)
                    if pic.mode != 'RGB':
                        pic = pic.convert('RGB')
                    # if pic.mode != 'RGB':
                    #     print(f'pic:{i} not in RGB format')
                    #     continue
                    imgs.append(transpose(pic).unsqueeze(0))
                except:
                    continue
            imgs = torch.cat(imgs, dim=0)
            return imgs
        except:
            print(batch[:2])
            print ('error processing image')
            return -1,-1  # if some broken image, output -1,-1


    def draw_images(self,img_dir):
        """
        wrapper function for collate_fn
        """
        imgs = sorted(list(os.listdir(img_dir)))[:MAX_NUM_IMGS]
        imgs = self.collate_fn([img_dir+i for i in imgs])
        return imgs

    def model_infer(self,img_dir):
        """
        takes in a list of image path, output the classification result from image classifier.
        :param img_dir: a list of image path
        :return: the classification label for the input images and the dataframe for the output results
        """
        idx2label = self.idx2label
        model = self.eff_net
        model.to(device)
        inputs = self.draw_images(img_dir)
        inputs = inputs.to(device)
        with torch.no_grad():
            model.eval()
            outputs = model(inputs)
            outputs = torch.softmax(outputs,dim=-1)
            max_val, preds = torch.max(outputs, 1)
        out =  list(zip([idx2label[i] for i in preds.cpu().numpy()],list(max_val.cpu().numpy())))
        out = pd.DataFrame(out,columns=['name','prob'])
        # groupby the label name, get the indices for max_count labels and max_prob labels
        df = out.groupby(['name'],as_index = False).agg({'prob':['max','count']})
        max_prob_indice = df[('prob','max')].values.argmax()
        max_count_indice = df[('prob','count')].values.argmax()
        # if max count = max prob, output the label
        if max_prob_indice==max_count_indice:
            label = df.iloc[max_prob_indice][('name','')]
            return label,df
        # if max_count number > 3* max_prob number, output the label with max count
        elif df.iloc[max_count_indice][('prob','count')]>3*df.iloc[max_prob_indice][('prob','count')]:
            label = df.iloc[max_count_indice][('name','')]
            return label,df
        # if max_count label is more than 75% of the total counts, output max_count label
        elif df.iloc[max_count_indice][('prob','count')]>= 0.75*np.sum(df[('prob','count')]):
            label = df.iloc[max_count_indice][('name','')]
            return label,df
        else:
            label = df.iloc[max_prob_indice][('name','')]
            return label,df

