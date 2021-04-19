# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 16:30:45 2020

@author: Param
"""

from src.helper.utils import *
import numpy
from itertools import combinations


def costgrid(arr,alt,sensor,times):
    """
    Calculates costs at different grid locations of lat lon from given sensors and timestamps

    Parameters
    ----------
    arr : array (2*k*k)
        k size grid array for lat lon
    alt : float
        altitude.
    sensor : array (n*3)
        ECEF xyz for n sensors
    times : array (n)
        n corresponding timestamps

    Returns
    -------
    TYPE
        minidx: grid index with minimum cost
        cost: cost corresponding to the index 

    """
    
    #speed of light 
    c=299792458 / 1.0003
    
    costdict={i:{j:[] for j in range(11)} for i in range(11)}
    costArr=np.zeros((11,11))
    for i in range(11):
        for j in range(11):
                X=np.asarray(fromlatlon(arr[0,i,j],arr[1,i,j],alt))
                cost=0
                numSen=sensor.shape[0]
                
                for i0,i1 in combinations(range(numSen),2):
                    dist_sub=(np.linalg.norm(X-sensor[i0,:])-np.linalg.norm(X-sensor[i1,:]))
                    time_sub=(times[i1]-times[i0])*c
                    costdict[i][j].append(np.abs(dist_sub+time_sub))
                    cost+=np.abs(dist_sub+time_sub)
                    
                costArr[i,j]=cost
                                    
    #print(np.min(costArr))
    minidx=np.unravel_index(np.argmin(costArr),costArr.shape)
    return minidx,np.max(np.asarray(costdict[minidx[0]][minidx[1]])) 

           

def MLAT_iter(ts,time,alt,center=None,length=3,gt=None):
    """
    Performs Mulitlateration 

    Parameters
    ----------
    ts : array (n*3)
        ECEF xyz for n sensors.
    time : array (n)
        n correspoinding timestamps.
    alt : float
        altitude.
    center : array (2), optional
        Centre of the lat lon grid . The default is None.
    length : float, optional
        total length of the grid. The default is 3.
    gt : array (2), optional
        ground truth lat lon position . The default is None.

    Returns
    -------
    TYPE  : array (2) or None
        lat lon location or None.

    """
    #ts : n*3 time: n , gt: list/1d array
    
    #make grid
    lat=np.linspace(center[0]-length,center[0]+length,num=11)
    lon=np.linspace(center[1]-length,center[1]+length,num=11)
    arr=np.asarray(np.meshgrid(lat,lon))


    #find cost
    minidx,cost=costgrid(arr,alt,ts,time)
    
    #return
    if length<0.001:
        if cost<200:
            if gt is not None:
                error=np.linalg.norm(np.asarray(fromlatlon(gt[0],gt[1],alt))\
                     -np.asarray(fromlatlon(arr[0,minidx[0],minidx[1]],arr[1,minidx[0],minidx[1]],alt)))
                print("Residual error :", error)
                return center,error,cost
            else:
                return center            
        else:
            return None
    
    #recursively go in until the length is small enough            
    ans=MLAT_iter(ts,time,alt,center=arr[:,minidx[0],minidx[1]],length=length*0.4,gt=gt)          
    
    return ans
    

def DoMLAT(df,sensordf,sensorMeta=[],bad_sensors=[],gt=None):
    """
    Preprocessing for Mulitlateration 

    Parameters
    ----------
    df : DataFrame (1*9)
        single row , one data instance.
    sensordf : DataFrame
        Info about sensors (from sensors.csv).
    sensorMeta : list, optional
        Information about good sensor combinations. The default is [].
    bad_sensors : list, optional
        Bad sensor IDs. The default is [].
    gt : array (2), optional
        lat lon ground truth. The default is None.

    Returns
    -------
    TYPE: array (2) or None
        lat lon location or None.
    conf : float
        Confidence value 

    """
    
    conf = 0
    df=df.iloc[:,:9]
    assert df.shape[0]==1
    #extract sensors
    df=split_sensors(df)
    key=list(df.iloc[0,list(range(9,df.shape[1],3))])
    #Extract only timestamps
    times=df.iloc[0,list(range(10,len(list(df.columns)),3))].values    
    
    #Remove Bad Sensors
    keytimes=[item for item in zip(key,times) if item[0] not in bad_sensors]
    
    if keytimes:
        key,times=zip(*keytimes)
 
        
        if len(key)>2:    
            num=[i for i in range(3,min(10,len(key)+1))] #total number measurements
            keylist=[]
            for i in num:
                 keylist+=[tuple(sorted(item)) for item in combinations(key,i)]
                 
            
            keylist=[item for item in keylist if item not in sensorMeta['bad'].keys()]
            
            if keylist:
                #All possible key combinations and extracted the match with good with max freq
                keydict={key:sensorMeta['good'][key] for key in keylist if key in sensorMeta['good'].keys()}
                if keydict:
                    key=max(keydict, key=keydict.get)
                    conf=0.8
                else:
                    #else choose one with max sensors
                    key=max(keylist,key=lambda k: len(k))
                    conf=0.3
                
                
                #Extract sensors and timestamps
                keytimes=sorted(keytimes)
                times=[item[1] for item in keytimes if item[0] in key]    
                #extract the sensor data
                sensor_list=[sensordf.loc[sensordf['serial'] == val] for val in key ]
                sensor_list=pd.concat([item for item in sensor_list])    
        
        
                if sensor_list.shape[0]>=3:
                    
                    # Calculate and add X,Y,Z for each sensor
                    sensor_list=addxyz(sensor_list,order=[1,2,3])
                
                
                    #Important variables for opt
                    center=np.mean(np.asarray(sensor_list.iloc[:,1:3]),0) 
                    sensor_arr=np.asarray(sensor_list.iloc[:,-3:]) #xyz
                    group_times=np.asarray(times).squeeze()/1e9
                    alt=df.iloc[0,5]+150
                    
                    
                    res=MLAT_iter(sensor_arr,group_times,alt,center=center,gt=gt)
            
                    return res,conf

    return None,conf