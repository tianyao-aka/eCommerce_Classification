import re
import pandas as pd
from numpy.random import RandomState
from torchnlp.word_to_vector import FastText
from torch.utils import data
from nltk.tokenize import word_tokenize
import spacy
import pickle
from sklearn.metrics import *
import numpy as np
import math
import pickle
from efficientnet_pytorch import EfficientNet
from PIL import Image
import random
from collections import Counter
import nltk
from torch.utils.data import TensorDataset,DataLoader,WeightedRandomSampler
import os
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from FastText_Vector import *


nltk.download("punkt")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Text_NN(nn.Module):
    """
    a nn module for text classification
    """
    def __init__(self, fasttext_vec,num_class, max_features=20000):
        super(Text_NN, self).__init__()
        # load pretrained embedding in embedding layer.
        self.fasttext = fasttext_vec

        # Convolutional Layers with different window size kernels
        # Dropout layer
        self.linear_layer = nn.Linear(300,1024)
        self.linear_layer2 = nn.Linear(1024,512)
        self.linear_layer3 = nn.Linear(512,256)
        self.linear_layer4 = nn.Linear(256,num_class)
#         self.wide_model = nn.Linear(max_features,num_class,bias = False)
        self.relu1 = nn.ReLU()
        # FC layer
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, input):
        """
        forward func
        :param input: a embedded text with fasttext word vector, a sentence is represented by the average embedding of all words.
        :return:
        """
        title_emb = input
#         titles_count_vec = input[1]
#         print(title_emb.shape)
#         print (titles_count_vec.shape)
        h = self.relu3(self.linear_layer3(self.dropout2(self.relu2(self.linear_layer2(self.dropout1(self.relu1(self.linear_layer(title_emb))))))))
        logit_deep = self.linear_layer4(h)
#         logit_wide = self.wide_model(titles_count_vec)
        return logit_deep


class Text_classifier(object):
    def __init__(self):
        self.vectors = FastText_Vector()
        self.text_model = Text_NN(None, 28, max_features=10)
        temp = torch.load('models/best_fasttext_deep_valid_model_0.94443.params')
        self.text_model.load_state_dict(temp)
        with open('models/idx2label.pickle', 'rb') as f:
            self.idx2label = pickle.load(f)

    def _model_infer(self,texts,num_inference= 300):
        idx2label = self.idx2label
        model = self.text_model
        total_out_probs = []
        with torch.no_grad():
            model.to(device)
            transformed_texts, post_text = transform_text(self.vectors,texts) # get the text embedding and processed text
            # forward the model for #num_inference times, then get the average prob and variance of the prob
            # without model.eval(), equivalent to a bayesian nn with bernoulli distribution for each neuron
            for _ in range(num_inference):
                out = model(transformed_texts)
                out = torch.softmax(out, dim=-1)
                total_out_probs.append(out.unsqueeze(1))
            total_out_probs = torch.cat(total_out_probs, dim=1)
            total_out_probs_mean = total_out_probs.mean(dim=1)
            max_val, indices = torch.max(total_out_probs_mean, dim=1)
            max_val = max_val.cpu().numpy()
            indices = indices.cpu().numpy()
            prob_variance = total_out_probs[np.arange(len(total_out_probs)), :, indices].std(dim=-1).cpu().numpy() # fancy indexing to retrieve prob variance for the index of max_prob_val
        return list(zip(texts, post_text, [idx2label[i] for i in indices], max_val, prob_variance))


    def inference(self,df_products):
        """
        inference method using bayesian text classifier. applicable both for collection page and product page.
        :param df_products: dataframe for products.
        :return: a dataframe holding classification results. columns are:['title','processed_title','predict_label','probability','variance']
        """
        texts = df_products['title'].tolist()
        N = len(texts)
        positions_for_colleciton_pages = record_collection_page_positions(texts)
        if not positions_for_colleciton_pages: # if there is no collection page,only product page
            product_texts = [i for indice,i in enumerate(texts) if indice not in positions_for_colleciton_pages]
            classification_results = self._model_infer(product_texts)
            df_ret = pd.DataFrame(classification_results,columns=['title','post_title','predict','probability','variance'])
            df_ret = df_ret.reset_index()
            return df_ret
        else: # both collection page and product page exists in the ad text
            classification_results = []
            product_texts = [i for indice,i in enumerate(texts) if indice not in positions_for_colleciton_pages]
            collection_texts = [i for indice,i in enumerate(texts) if indice in positions_for_colleciton_pages]
            product_classification_results = self._model_infer(product_texts) # get classification result for product pages
            collection_pages_classification_results = [get_label_per_collection_page(self._model_infer(collection_page)) for collection_page in collection_texts] # get classification result for collection pages
            assert len(collection_pages_classification_results) == len(positions_for_colleciton_pages)
            assert N == len(collection_pages_classification_results) + len(product_classification_results)
            # merge all the predictions, both for product pages and collection pages, into a new list
            for i in range(N):
                if i in positions_for_colleciton_pages:
                    classification_results.append(collection_pages_classification_results.pop(0))
                else:
                    classification_results.append(product_classification_results.pop(0))
        df_ret = pd.DataFrame(classification_results,columns=['title','post_title','predict','probability','variance'])
        df_ret = df_ret.reset_index()
        return df_ret  # return dataframe of model results.


