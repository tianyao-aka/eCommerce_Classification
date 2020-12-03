import re
import pandas as pd
from numpy.random import RandomState
from torchnlp.word_to_vector import FastText
from nltk.tokenize import word_tokenize
import spacy
import numpy as np
import math
import random
import nltk
import os
from collections import Counter
import torch


Clothing_Footwear_Threshold = 0.85
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

nltk.download("punkt")

def remove_punct2(text):
    if len(text.split()) == 1 and '-' in text:
        text = ' '.join(text.split('-'))
    remove_chars = '[!"#$%\*+,./:;<=>?，。?★、…《》？“”‘’！\\^_`{|}~]+'
    text = re.sub(remove_chars, ' ', text)
#     remove_chars = "\-[^\-]*\-"
#     text = re.sub(remove_chars, '', text)
    return text.replace("'s", "").replace("&", ' and ').replace('@', 'at')


def remove_punct(text):
    text = text.replace("【", "(")
    text = text.replace("】", ")")
    text = text.replace("[", "(")
    text = text.replace("]", ")")
    text = text.replace("（", "(")
    text = text.replace("）", ")")
    remove_chars = "\([^\(\)]*\)"
    text = re.sub(remove_chars, '', text)
    remove_chars = "[0-9]+[a-z]+"
    text = re.sub(remove_chars, '', text)
    remove_chars = "[0-9]+[A-Z]+"
    text = re.sub(remove_chars, '', text)
    # remove_chars = "[0-9]+"
    # text = re.sub(remove_chars, '', text)
    return text


def process_text(data, remove_punc2=True):
#     if len(data.split()) == 1 and '-' in data:
#         temp = data.split('-')
#         return ' '.join(temp)
    data = str(data)
    data = data.lower()
    if remove_punc2:
        data = remove_punct2(data)

    data = remove_punct(data)
    out = [i.lower().replace('-', ' ') for i in word_tokenize(data)]
    return " ".join(out)

def transform_text(vectors,titles):
  titles = [process_text(i) for i in titles]
  titles_emb = torch.cat([get_single_sent_emb(vectors,i) for i in titles],dim=0)
#   titles_count_vectorized = tfidf_model.transform(titles).toarray()
#   titles_count_vectorized = torch.tensor(titles_count_vectorized).float()

  return titles_emb.to(device),titles


def get_single_sent_emb(vectors,sent):
    """
    using fasttext vector to obtain sentence embedding.
    :param vectors: word vectors
    :param sent: input sentence, should be a product title
    :return: averaged embedding for this given sentence
    """
    try:
        emb = [vectors[[x]] for x in sent.split()]
        emb = torch.mean(torch.cat(emb, dim=0), dim=0).view(1, -1)
        return emb.to(device)
    except:
        return vectors[['000']].to(device) # all zeros returned


def record_collection_page_positions(text_list):
    # text_list:: a List of titles for product pages or list of titles for collection pages
    """
    record the index of collection pages
    return: the indices of collection pages
    """
    positions = [i for i,j in enumerate(text_list) if isinstance(j,list)]
    return set(positions)


def get_label_per_collection_page(model_infer_results,use_max_count = True):
    # model_infer_results: a list of tuple from model inference outcome -->  (text,postprocessed_text,label,max_prob,variance)
    # print ('model infer results:',model_infer_results)
    labels = [i[2] for i in model_infer_results]
    most_common_label,most_common_count = Counter(labels).most_common(1)[0]
    max_prob = max([i[3] for i in model_infer_results])
    max_prob_results = [('-1','-1',i,j,k) for _,_,i,j,k in model_infer_results if j==max_prob][0]
    if use_max_count:
        return ('-1','-1',most_common_label,most_common_count,-1)
    else:
        return max_prob_results


def match_product_category(label,prob,thres = Clothing_Footwear_Threshold):
    if "Women's Footwear" in label or "Men's Footwear" in label or "Women's Clothing" in label or "Men's Clothing" in label:
        if prob<thres:
            return True
    return False

