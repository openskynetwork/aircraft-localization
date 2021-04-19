# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 10:27:27 2020

@author: Param
"""

from src.helper.utils import *
import numpy as np
import pandas as pd
import warnings
from scipy.optimize import minimize,LinearConstraint
warnings.simplefilter("ignore", UserWarning)
from itertools import combinations,product
import random
  

def cost_func_lla(X,sensors, timestamps):
    #X= (3,)  lla
    #sensors n*3 xyz, times n*3
    c=299792458 / 1.0003
    cost=0
    numSen=sensors.shape[0]
    
    Xecef=fromlatlon(X[0],X[1],X[2])
    
    for i0,i1 in combinations(range(numSen),2):
        dist_sub=(np.linalg.norm(Xecef-sensors[i0,:])-np.linalg.norm(Xecef-sensors[i1,:]))
        time_sub=(timestamps[i1]-timestamps[i0])*c
        cost+=np.abs(dist_sub+time_sub)
        
    return cost




def scipyMLAT(df,sensordf,bad_sensors=[],gt=None):
    
    df=df.iloc[:,:9]
    assert df.shape[0]==1
    #extract sensors
    df=split_sensors(df)
    key=list(df.iloc[0,list(range(9,df.shape[1],3))])

    #Extract only timestamps
    group_times=df.iloc[0,list(range(10,len(list(df.columns)),3))]        
    
    #extract the sensor data
    sensor_list=[sensordf.loc[sensordf['serial'] == val] for val in key ] 
    
    
    if len(sensor_list)>=3:
        
        # Calculate and add X,Y,Z for each sensor
        sensor_list=pd.concat([item for item in sensor_list]) 
        sensor_list=addxyz(sensor_list,order=[1,2,3])
    
    
        #Important variables for opt
        sensor_arr=np.asarray(sensor_list.iloc[:,-3:]) #xyz
        (meanlat,meanlon)=np.mean(sensor_list.iloc[:,[1,2]],axis=0)
        X0=np.asarray([meanlat,meanlon,df.iloc[0,5]+150])
        
        group_times=np.asarray(group_times).squeeze()/1e9
        
        #get constraints (bounds)
        width=3
        bounds=(X0-[width,width,200],X0+[width,width,200])
        linear_constraint = LinearConstraint(np.eye(3),bounds[0],bounds[1])
        
        
        result={}
        #trust-constr
       
        res=minimize(cost_func_lla, X0, args=(sensor_arr,group_times), method='trust-constr',constraints=linear_constraint)
        if res is not None:
            result['lat']=res['x'][0]
            result['lon']=res['x'][1]

            if gt is not None:            
                res['x']=np.asarray(fromlatlon(res['x'][0],res['x'][1],df.iloc[0,5]+150))
                #print("error: %f cost : %f" %(np.linalg.norm(res['x']-fromlatlon(gt[0],gt[1],df.iloc[0,5]+150)),res['fun']))
                result['error']=np.linalg.norm(res['x']-fromlatlon(gt[0],gt[1],df.iloc[0,5]+150))
                result['cost']=res['fun']
            else:
                result['error']=np.NaN
                result['cost']=np.NaN                 

        else:
            result['lat']=np.NaN
            result['lon']=np.NaN            
            result['error']=np.NaN
            result['cost']=np.NaN 
                    
        return result
            

def scipyMLAT_only2(df,sensordf,bad_sensors=[],gt=None):
    
    df=df.iloc[:,:9]
    assert df.shape[0]==1
    #extract sensors
    df=split_sensors(df)
    key=list(df.iloc[0,list(range(9,df.shape[1],3))])

    #Extract only timestamps
    group_times=df.iloc[0,list(range(10,len(list(df.columns)),3))]        
    
    #extract the sensor data
    sensor_list=[sensordf.loc[sensordf['serial'] == val] for val in key ] 
    
    
    if len(sensor_list)>=2:
        
        # Calculate and add X,Y,Z for each sensor
        sensor_list=pd.concat([item for item in sensor_list]) 
        sensor_list=addxyz(sensor_list,order=[1,2,3])
    
    
        #Important variables for opt
        sensor_arr=np.asarray(sensor_list.iloc[:,-3:]) #xyz
        (meanlat,meanlon)=np.mean(sensor_list.iloc[:,[1,2]],axis=0)
        X0=np.asarray([meanlat,meanlon,df.iloc[0,5]+150])
        
        group_times=np.asarray(group_times).squeeze()/1e9
        
        #get constraints (bounds)
        width=3
        bounds=(X0-[width,width,200],X0+[width,width,200])
        linear_constraint = LinearConstraint(np.eye(3),bounds[0],bounds[1])
        
        
        result={}
        #trust-constr
       
        res=minimize(cost_func_lla, X0, args=(sensor_arr,group_times), method='trust-constr',constraints=linear_constraint)
        if res is not None:
            result['lat']=res['x'][0]
            result['lon']=res['x'][1]

            if gt is not None:            
                res['x']=np.asarray(fromlatlon(res['x'][0],res['x'][1],df.iloc[0,5]+150))
                #print("error: %f cost : %f" %(np.linalg.norm(res['x']-fromlatlon(gt[0],gt[1],df.iloc[0,5]+150)),res['fun']))
                result['error']=np.linalg.norm(res['x']-fromlatlon(gt[0],gt[1],df.iloc[0,5]+150))
                result['cost']=res['fun']
            else:
                result['error']=np.NaN
                result['cost']=np.NaN                 

        else:
            result['lat']=np.NaN
            result['lon']=np.NaN            
            result['error']=np.NaN
            result['cost']=np.NaN 
                    
        return result
        



