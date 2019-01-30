# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 03:38:31 2018

@author: s.prabhakar.daley
"""
import tensorflow as tf
import pandas as pd
import numpy as np
import page,words
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2
#import editdistance
from DataLoader import DataLoader, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess

def predictor(imgpath,nlines, decoder = 1):
    
    tf.reset_default_graph()
    
    if decoder == 0:
        decoderType = DecoderType.BestPath
    elif decoder == 1:
        decoderType = DecoderType.BeamSearch
    elif decoder == 2:
        decoderType = DecoderType.WordBeamSearch
        
    fnCharList = '../model/charList.txt'
    def idxImage(index):
        """ Getting next image from the array """
        if index < len(Box1):
            b = bBoxes[index]
            x1, y1, x2, y2 = b
            img = image[y1:y2, x1:x2] 
            cv2.imwrite('temp.jpg',img)
            temp = preprocess(cv2.imread('temp.jpg',cv2.IMREAD_GRAYSCALE), Model.imgSize) 
            batch = Batch(None, [temp] * Model.batchSize) # fill all batch elements with same input image
            recognized = model.inferBatch(batch)          
        return recognized[0]
    
    model = Model(open(fnCharList).read(), decoderType, mustRestore=True)
    # Increase size of images
    
    plt.rcParams['figure.figsize'] = (9.0, 7.0)
    IMG = 'text5'    # 1, 2, 3
    image = cv2.cvtColor(cv2.imread(imgpath ), cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    # Crop image and get bounding boxes
    crop = page.detection(image)
    cv2.imwrite('crop.jpg', crop)
    bBoxes = words.detection(crop)
    
    Box1 = pd.DataFrame(bBoxes)
    Box1 = Box1.rename(index = str, columns = {0:'x1',1:'y1',2: 'x2',3: 'y2'})
    Box1.sort_values(by = ['y1','x1'])
    #plt.hist(Box1['y1'],bins=250)
    #plt.xticks(np.arange(0,4000,120), rotation = 90)
    #plt.figure(figsize = (5000,5000))
    #plt.show()
    
    ###Clustering###
    
    X = np.array(Box1['y1'])
    X = np.reshape(X, (-1, 1))
    kmeans = KMeans(n_clusters = nlines)
    kmeans.fit(X)
    #kmeans.labels_
    #kmeans.cluster_centers_
    #np.unique(kmeans.cluster_centers_)
    #np.unique(kmeans.labels_)
    #plt.scatter(np.unique(kmeans.cluster_centers_),np.unique(kmeans.labels_))
    Box1['cluster']= kmeans.labels_
    
    sentactual = ''
    sentarray = Box1[['y1','cluster']].groupby('cluster').agg(np.min).sort_values('y1').index
    
    for sent in sentarray:
        wordarray = Box1[Box1['cluster']==sent].sort_values('x1').index
        for word in wordarray:
            #print(Box1[Box1.index == word])
                
            wordactual = idxImage(int(word))
            sentactual = sentactual+' '+wordactual
        docactual = sentactual+'\n'
#    print(docactual)
    return docactual


doc = predictor('C:/Users/s.prabhakar.daley/Desktop/faltu/IAM docs/HW imgs/a01-003u-HW.png',11)