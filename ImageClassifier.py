# -*- coding: utf-8 -*-
"""
Created on Sat May  7 14:04:07 2022

@author: mert_
"""
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
import numpy as np
import cv2
import os
from os import listdir
from datetime import datetime
import pandas as pd
import seaborn as sn
import shutil
from skimage import io
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing,metrics
import os.path
from sklearn.metrics import classification_report
from collections import Counter
from sklearn.metrics import confusion_matrix
import seaborn as sns


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_idx = np.argsort(distances)[: self.k]
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
        mc = Counter(k_neighbor_labels).mc(1)
        return mc[0][0]
  
def featureExtraction(file):
    container = []
    
    for img in file:

      
        img = cv2.resize(img,(128,128))

        # Reducing noise
        img = cv2.GaussianBlur(img,(5,5), 0)
        bins=np.array([0,51,102,153,204,255])
        img[:,:,:] = np.digitize(img[:,:,:],bins,right=True)*51
     

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     

        ret,background = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
     

        background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)

        ret,foreground = cv2.threshold(img_gray,0,255,cv2.THRESH_TOZERO_INV+cv2.THRESH_OTSU)  
        foreground = cv2.bitwise_and(img,img, mask=foreground)  
     
        finalimage = background+foreground        
    
        img=np.array(finalimage).flatten()
                    
        container.append(img)
    
    return container

    
def process_and_load(path):
    
   
    img_container=[]

        
    label_container=[]
  
    paths = glob(path+"\\"+"*")
    
    for path in paths:
        img_directory=glob(path+"/*.jpg")
        label=os.path.basename(path)
        for i in img_directory:
            img = cv2.imread(i)
            label_container.append(label)
            img_container.append(img)
            

    return img_container,label_container

def KNN(k,training_images,training_label,test_images,test_label):
    knn= KNeighborsClassifier(k)
    knn.fit(training_images,training_label)
    y_pred=knn.predict(test_images)
    report=classification_report(test_label,y_pred)
    print(report)
    confusion = confusion_matrix(y_true = test_label,y_pred = y_pred)    
    s= sns.heatmap(confusion,annot=True ,cmap='nipy_spectral_r')
    s.set_xlabel('Predicted');s.set_ylabel('Desired'); 
    s.xaxis.set_ticklabels(tags); s.yaxis.set_ticklabels(tags);
    s.tick_params(axis='both', which='major', labelsize=6)  # Adjust to fit
    s.set_title("Confusion Matrix For Image Classification")
    accuracy=knn.score(test_images,test_label)
    print("Accuracy -> ")
    print(accuracy)
    

if __name__ == "__main__":
    rootdir = os.path.dirname(os.path.abspath(__file__)) #bulundu
    tags =["airplanes", "bonsai", "chair","ewer","faces","flamingo","guitar","leopards","motorbikes","starfish"]  
  
    validation=0
    test_images =[]
    training_images = []
    validation_images = []
    
    test_label = []
    training_label=[]
    validation_label=[]
    
    
    test_path = rootdir + r"\Database" + r"\TestSet"
    training_path = rootdir + r"\Database" + r"\TrainingSet"
    validation_path = rootdir + r"\Database" + r"\ValidationSet"
    
    

    test_images,test_label = process_and_load(test_path)
    test_images = featureExtraction(test_images)
  
    training_images,training_label = process_and_load(training_path)
    training_images = featureExtraction(training_images) 
  
    validation_images,validation_label = process_and_load(validation_path)
    validation_images = featureExtraction(validation_images)
   
    if(validation==1):
        
        KNN(3,training_images,training_label,validation_images,validation_label)
    else:
        KNN(1,training_images,training_label,test_images,test_label)