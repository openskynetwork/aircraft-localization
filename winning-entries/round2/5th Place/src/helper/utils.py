# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 11:16:18 2020
Utility Functions
@author: Param
"""

import pickle
import ast
import os
import numpy as np
from datetime import timezone
from datetime import datetime
from datetime import timedelta
import pandas as pd
pd.set_option('mode.chained_assignment', None)
import pdb
import pyproj


def RMSE_2D(gt_df,test_df):#gt_df,test_df
    
    assert gt_df.shape[0]==test_df.shape[0]
     
    gt_df=gt_df[gt_df.index.isin(test_df.index) & ~test_df.latitude.isnull()]
    test_df=test_df[~test_df.latitude.isnull()]
    
    
    X_pred=test_df.iloc[:,[3,4]]
    X=gt_df.iloc[:,[3,4]]


    def vectorize_radians(x):
        return np.vectorize(np.radians)(x)

    X = vectorize_radians(X)
    X_pred = vectorize_radians(X_pred)

    R = 6373000

    lat1 = X[:,0]
    lat2 = X_pred[:,0]
    dlat = X[:,0]-X_pred[:,0]
    dlon = X[:,1]-X_pred[:,1]

    a = np.power(np.sin(dlat / 2), 2) + np.cos(lat1) * np.cos(lat2) * np.power(np.sin(dlon / 2), 2)
    c = 2 * np.arctan2(np.sqrt(a),np.sqrt(1-a))

    vector_errors = R * c
    val=np.percentile(vector_errors,90)
    vector_errors=vector_errors[vector_errors<=val]
    sum_errors = np.sum(np.power(vector_errors, 2))
    return np.sqrt(sum_errors/len(vector_errors))





def fromlatlon(lat,lon,alt):
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84') 
    x,y,z=pyproj.transform(lla, ecef, lon, lat,alt,radians=False)
    return x,y,z


def fromxy(x,y,z):
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84') 
    lon,lat,alt=pyproj.transform(ecef, lla, x, y,z,radians=False)
    return lat,lon,alt




#extract xyz from lat lon alt and append to df
def addxyz(df,order=[0,1,2]): # order : (lla)
    tmp_xyz=[]
    for i in range(df.shape[0]):
        tmp_xyz.append(fromlatlon(df.iloc[i,order[0]],df.iloc[i,order[1]],df.iloc[i,order[2]]))
    
    tmp_xyz=np.asarray(tmp_xyz)
    df['x']=tmp_xyz[:,0]
    df['y']=tmp_xyz[:,1]
    df['z']=tmp_xyz[:,2]
    
    return df

#extract sensor measurements into seperate columns
def sensor_measurements(df):
    
    sen_arr=np.empty((df.shape[0],max(df.numMeasurements)*3))
    sen_arr[:] = np.NaN    
    for i in range(df.shape[0]):
        measurement=np.fromstring(df.iloc[i,-1].replace('[','').replace(']',''),sep=',')
        measurement=np.reshape(measurement,(-1,3))
        measurement=measurement[measurement[:,0].argsort()]
        measurement=measurement.flatten()
        sen_arr[i,:measurement.shape[0]]=measurement
        
    return sen_arr


def split_sensors(df):
    sen_arr=[]
    for i in range(df.shape[0]):
        sen_arr.append(np.fromstring(df.iloc[i,-1].replace('[','').replace(']',''),sep=','))
    sen_arr=np.asarray(sen_arr)
    
    for i in range(sen_arr.shape[1]):
        df[i]=sen_arr[0,i]
        
    return df


def addXYtoSenDf(sensordf):
    # add xyz to sensordf csv which is read from sensor file
    sensordf['X']=0
    sensordf['Y']=0
    sensordf['Z']=0
    
    for i in range(sensordf.shape[0]):
        senXY=fromlatlon(sensordf.iloc[i,1],sensordf.iloc[i,2],sensordf.iloc[i,3])
        sensordf.iloc[i,-3]=senXY[0];sensordf.iloc[i,-2]=senXY[1];sensordf.iloc[i,-1]=senXY[2]
        
    return sensordf
    