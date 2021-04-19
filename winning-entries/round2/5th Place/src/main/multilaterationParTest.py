# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 12:52:51 2020

@author: Param
"""

import os
import sys
# Go to root dir
os.chdir(os.path.abspath(os.path.join(__file__ ,"../../..")))
sys.path.append(os.getcwd())
    
import pdb
import numpy as np
import pickle
import pandas as pd
from src.helper.visualize import *
from src.helper.utils import *
from src.helper.myMLAT import DoMLAT
from src.helper.scipyMLAT import scipyMLAT,smartScipyMLAT
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool


def parallelize_dataframe(df, func):
    cores=mp.cpu_count()
    df_split = np.array_split(df, cores)
    pool = Pool(cores)
    df = pd.concat(pool.map(func,df_split))
    pool.close()
    pool.join()
    return df

def fixSensors(df):
    
    sensors=np.asarray(ast.literal_eval(df.measurements.iloc[0]))
    keepidx=[]
    
    for i in range(sensors.shape[0]):
        if sensors[i,0] in goodserial:
            keepidx.append(i)
        elif sensors[i,0] in goodseriallaggy:
            sensors[i,1]+=lag
            keepidx.append(i)
        elif sensors[i,0] in driftdict.keys():
            if driftdict[sensors[i,0]] is not None:
                drift=driftdict[sensors[i,0]].predict(np.asarray(df.timeAtServer.iloc[0]).reshape(1,1))[0][0]
                sensors[i,1]-=drift
                keepidx.append(i)
            
            
    sensors=sensors[keepidx,:]
    df.numMeasurements.iloc[0]=sensors.shape[0]
    sensors=np.array2string(sensors,separator=',').replace('\n','')
    df.measurements.iloc[0]=sensors
    return df



def MLATfunc(df):
    df_res=df.copy()
    df_res['error']=np.NaN
    df_res['cost']=np.NaN
    for i in range(df.shape[0]):
        if i%1000==0:
          print(i)
        try:
          dfin=fixSensors(pd.DataFrame(df.iloc[i,:]).T)
          df_res.iloc[i,:-2]=dfin.iloc[0,:]
          if dfin.numMeasurements.iloc[0]>=3:
            # res=scipyMLAT(dfin,sensordf)
            # df_res.latitude.iloc[i]=res['lat']
            # df_res.longitude.iloc[i]=res['lon']
            res=DoMLAT(dfin,sensordf)
            if res is not None:
                df_res.latitude.iloc[i]=res[0]
                df_res.longitude.iloc[i]=res[1]               
        except:
          print('Exception')
    return df_res

      
    

#-----------------------LOAD DATA---------------------------------------------
folderpath=os.getcwd()
#df=pickle.load(open(os.path.join(folderpath,'data','comp_train_sub.pkl'),'rb'))
df=pd.read_csv(os.path.join(folderpath,'data','round2_competition.csv'))
df=df[df.latitude.isnull()]
sensordf=pd.read_csv(os.path.join(folderpath,'data','round2_sensors.csv'))
driftdict=pickle.load(open(os.path.join(folderpath,'data','driftmodels.pkl'),'rb'))

sensortype_dict= dict(zip(sensordf.serial, sensordf.good))
goodseriallaggy=[474.0,460.0,470.0,550.0,414.0]
lag=120620 
goodserial=[key for (key, value) in sensortype_dict.items() if value if key not in goodseriallaggy]

del sensortype_dict
#------------------------------------------------------------------------------

#-----------------------DO MULTILATERATION ROW-WISE----------------------------
df_out=parallelize_dataframe(df,MLATfunc)
df_out.to_csv(os.path.join(folderpath,'data','mlat_results_test_myMLAT.csv'))

#dfnew=df[0:200]
#df_out=MLATfunc(dfnew)
#df_out=parallelize_dataframe(df,MLATfunc)
#df_out.to_csv(os.path.join(folderpath,'data','mlat_results_all0_smart.csv'))
#------------------------------------------------------------------------------