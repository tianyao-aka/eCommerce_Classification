import re
import pandas as pd
from numpy.random import RandomState
from nltk.tokenize import word_tokenize
import shutil
import pickle
import numpy as np
import math
import random
import nltk
import os
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from text_model import *
from image_model import *
from utils import *
import requests
import json
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from pandarallel import pandarallel
pandarallel.initialize()


nltk.download("punkt")
OUT_IMG_DIR = 'temp_images/'   # temp diretory for saving temporary images for classification
TEMP_PREDICT_DIR = 'predict_summary/' # optional directory for saving summary files during classification process
PROB_THRESHOLD = 0.505  # threshold for classification
VAR_THRESHOLD = 0.3 # variance threshold for text classifier

text_model = Text_classifier()
image_model = Image_Classifier(31)

file_list = os.listdir('./')

try:
    shutil.rmtree(OUT_IMG_DIR,ignore_errors=True)
    os.mkdir(OUT_IMG_DIR)
except:
    pass

try:
    shutil.rmtree(TEMP_PREDICT_DIR,ignore_errors=True)
    os.mkdir(TEMP_PREDICT_DIR)
except:
    pass


def requests_retry_session(
    retries=2,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 504),
    session=None,
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def download_images(ds):
    try:
        urls = ds['urls']
        index = ds['index']
        if urls==-1:
            return
        for url in urls:
            name = url.split('?')[0].split('/')[-1]

            response = requests_retry_session().get(url, timeout=4)
            if str(index) not in os.listdir(OUT_IMG_DIR):
                os.mkdir(OUT_IMG_DIR + str(index))
            with open(OUT_IMG_DIR + str(index) + '/' + name, 'wb') as f:
                f.write(response.content)

    except:
        return



def preprocess_data(product_urls):
    # this method crawl product info, parse title and img_urls and finally download images
    # return a dataframe
    crawl_api = 'https://fb-algo.mobvista.com/crawler/crawl_product_info'
    parsed_info = []
    print (f'total urls:{len(product_urls)}')
    count = 1
    for link in product_urls:
        if count%10==0:
            print (f'current crawling progress: {count}')
        count += 1
        response = requests.post(crawl_api, json.dumps({'link': link,
                                                        'num_product': 30,
                                                        'fields': ['title', 'img_urls']}))
        data = json.loads(response.text)
        if data['code']!=0 or 'products' not in data:
            parsed_info.append(('failed',-1))
        else:
            if len(data['products'])==1 and data['products'][0]['code']==0: # product page
                # print ('product page')
                # title = data['products'][0]['title'] if 'title' in data['products'][0] else 'failed'
                if 'title' in data['products'][0]:
                    title = data['products'][0]['title']
                else:
                    if 'title' not in data['products'][0] and 'img_urls' in data['products'][0] and len(data['products'][0]['img_urls'])>2:
                        title = 'none'
                    else:
                        title = 'failed'
                img_urls = data['products'][0]['img_urls'] if 'img_urls' in data['products'][0] else 'failed'
                parsed_info.append([title,img_urls])
            elif len(data['products'])>1: # collection page
                # print ('collection page')
                titles = [i['title'] for i in data['products'] if i['code']==0 and 'title' in i]
                if not titles:
                    parsed_info.append(('failed', -1))
                img_urls = [i['img_urls'] for i in data['products'] if i['code']==0 if 'img_urls' in i]
                img_urls = [j for i in img_urls for j in i][:40]  # flatten list and take maximum of 40 images for collection pages
                parsed_info.append([titles, img_urls])
    df = pd.DataFrame(parsed_info,columns = ['title','urls'])
    df = df.reset_index()
    # _ = df.parallel_apply(download_images, axis=1)
    return df


def model_inference(product_urls,output_predict_summary= False,bypass_crawling_and_text_processing = False,download_imgs = True):
    global OUT_IMG_DIR,TEMP_PREDICT_DIR,text_model,image_model
    
    try:
        shutil.rmtree(OUT_IMG_DIR, ignore_errors=True)
        os.mkdir(OUT_IMG_DIR)
    except:
        pass

    try:
        shutil.rmtree(TEMP_PREDICT_DIR, ignore_errors=True)
        os.mkdir(TEMP_PREDICT_DIR)
    except:
        pass
    
    if not bypass_crawling_and_text_processing: # text classification if no classification results present
        print ('starting crawling and parsing data.....')
        df_products = preprocess_data(product_urls)
        print ('starting text classification process.....')
        df_classification_result = text_model.inference(df_products)
#         df_classification_result.to_excel('results.xlsx',index = False)
    else:  # directly load the result from excel, normally won't use
        df_classification_result = pd.read_excel('results.xlsx',index = False)
    print ('text classification finished, starting downloading images.....')
    if download_imgs:
        to_download_imgs = []
        df = df_classification_result
        for i in range(df.shape[0]):
            pred = df.iloc[i]['predict']
            prob = df.iloc[i]['probability']
            variance = df.iloc[i]['variance']
            title = df.iloc[i]['title']
            # if condition doesn't satisfy, put this product into a list for further downloading
            if match_product_category(pred,prob) or prob<0.505 or variance>0.3 or title=='-1': # title=='-1' means the title is okay, but not shown as the record is a collection page with many titles
                to_download_imgs.append(True)
            else:
                to_download_imgs.append(False)
        df_imgs_to_classified = df_products[to_download_imgs]
        print(f'there are {df_imgs_to_classified.shape[0]} product images to download for re-classification')
        print (df_imgs_to_classified)
        _ = df_imgs_to_classified.parallel_apply(download_images,axis=1)
        print ('end downloading images')
    #return df_classification_result
    img_set = list(os.listdir(OUT_IMG_DIR))
    img_set = set([int(i) for i in img_set])
    img_preds = []
    df = df_classification_result
    for i in range(df.shape[0]):
        # for every link, start the inference step
        # try:
            pred = df.iloc[i]['predict']
            prob = df.iloc[i]['probability']
            variance = df.iloc[i]['variance']
            title = df.iloc[i]['title']
            if title=='failed':
                img_preds.append('Other(Crawling failed)')
            elif i not in img_set:
                img_preds.append(pred) # if not in img_set, then use text classification result directly
            elif match_product_category(pred,prob): # if pred_label in some subset and not certain, need to use img prediction
                if OUT_IMG_DIR[-1]!='/':
                    OUT_IMG_DIR += '/'
                img_dir = OUT_IMG_DIR+str(i)+ '/'
#                 imgs = draw_images(OUT_IMG_DIR+str(i)+ '/')
#                 print ('total image count:',imgs.shape[0])
                label,df_pred = image_model.model_infer(img_dir) # predict using image model
                if 'Women' not in label and 'Men' not in label and 'Baby' not in label and 'Kid' not in label and 'Sport' not in label:
                    img_preds.append(pred)
                    if output_predict_summary:
                        df_pred.to_excel(TEMP_PREDICT_DIR+str(i)+'.xlsx')
                    continue
                else:
                    if 'Women' in label or 'Men' in label:
                        suffix = pred.split(' ')[-1]
                        prefix = label.split(' ')[0]
                        img_preds.append(prefix + ' ' + suffix)
                        df_pred.to_excel(str(i)+'.xlsx')
                        continue
                    else:
                        img_preds.append(label)
                        if output_predict_summary:
                            df_pred.to_excel(TEMP_PREDICT_DIR + str(i) + '.xlsx')
                        continue
            elif title=='-1' and prob>22:  # if prob >1 , prob refers to count, if prob>22, then most of the items are belong to the same prediction
                img_preds.append(pred)
                continue
            elif prob<PROB_THRESHOLD or variance>VAR_THRESHOLD or title=='-1':
                if pred=='Automotive' and prob>0.6 and title!='-1':
                    img_preds.append(pred)
                    continue
                # if prob too small or variance too high,use image classifier
                img_dir = OUT_IMG_DIR+str(i)+ '/'
                label,df_pred = image_model.model_infer(img_dir)
                img_preds.append(label)
                if output_predict_summary:
                    df_pred.to_excel(TEMP_PREDICT_DIR + str(i) + '.xlsx')
            else:
                img_preds.append(pred+ '_same_as_text')
    df['final_preds'] = img_preds
    return df



if __name__=='__main__':
    urls = pd.read_excel('links_with_title.xlsx')
    urls = urls['link'].tolist()
    b_size = 10
    N = len(urls)//b_size + 1
    predictions = []
    for i in range(N):
        if len(urls[i*b_size:(i+1)*b_size])==0:
            break
        df = model_inference(urls[i*b_size:(i+1)*b_size],False)
        predictions.append(df)
    df_predictions = pd.concat(predictions,axis=0)

