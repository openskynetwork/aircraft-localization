# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 13:06:38 2020

@author: Param
"""

import os
import sys
# For running from cluster terminal
if len(sys.argv)>1:
    os.chdir(os.path.abspath(os.path.join(__file__ ,"../../..")))
    sys.path.append(os.getcwd())


import pdb
import ast
import numpy as np
import pickle
import pandas as pd
from src.helper.visualize import *
from src.helper.utils import *
from itertools import combinations
from sklearn.linear_model import LinearRegression

def modelsensordrift(df):
    assert len(set(df.serial)) == 1
    
    idx=df.serial.iloc[0]
    df=df.dropna()
    Xorig=df.timeAtServer.values.reshape(-1,1)
    yorig=df.drift.values.reshape(-1,1)
    
    #set thresholds
    errorthresh=[1e10,1e9,1e8,1e7,1e6,1e5,1e4]
    X=Xorig.copy()
    y=yorig.copy()
    
    #remove outliers
    for thresh in errorthresh:
        reg=LinearRegression().fit(X, y)           
        ypred= reg.predict(X)
        errors=pd.DataFrame(np.hstack((X,y,ypred,np.abs(y-ypred))))
        X=errors[errors[3]<thresh][0].values.reshape(-1,1)
        y=errors[errors[3]<thresh][1].values.reshape(-1,1)
        ypred=errors[errors[3]<thresh][2].values.reshape(-1,1)

    errors=pd.DataFrame(np.hstack((X,y,ypred,y-ypred))) 
    #Do insane checks
    if errors.shape[0]/df.shape[0]>0.9 and errors.shape[0]>1000 and np.mean(errors[3])<100:
        return reg
    else:
        return None

#-----------------------LOAD DATA---------------------------------------------
folderpath=os.getcwd()
#df=pickle.load(open(os.path.join(folderpath,'data','comp_train_sub.pkl'),'rb'))
df=pd.read_csv(os.path.join(folderpath,'data','sensordrifts.csv')).iloc[:,1:]
# sensordf=pd.read_csv(os.path.join(folderpath,'data','round2_sensors.csv'))

# sensortype_dict= dict(zip(sensordf.serial, sensordf.good))
# goodseriallaggy=[474.0,460.0,470.0,550.0,414.0]
# goodserial=[key for (key, value) in sensortype_dict.items() if value if key not in goodseriallaggy]
#------------------------------------------------------------------------------


##
serials=set(df.serial)
driftmodels={}

for idx in serials:
    dfidx=df[df.serial==idx]
    driftmodels[idx]=modelsensordrift(dfidx)
