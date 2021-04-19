# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 16:30:45 2020

@author: Param
"""

from src.helper.utils import *
import math
import numpy
import scipy
from scipy.optimize import minimize
from itertools import combinations
import datetime



def costgrid(arr,alt,sensor,times):
    
    
    #arr : 2*10*10 (xyz,x,y) latlon
    #sensor n*3 xyz, times n*3
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
    #ts : n*3 time: n , gt: list/1d array
    
    #make grid
    lat=np.linspace(center[0]-length,center[0]+length,num=11)
    lon=np.linspace(center[1]-length,center[1]+length,num=11)
    arr=np.asarray(np.meshgrid(lat,lon))


    #find cost
    minidx,cost=costgrid(arr,alt,ts,time)
    
    #return
    if length<0.005:
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
                
    ans=MLAT_iter(ts,time,alt,center=arr[:,minidx[0],minidx[1]],length=length*0.4,gt=gt)          
    
    return ans
    

def DoMLAT(df,sensordf,bad_sensors=[],gt=None):
    
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
        center=np.mean(np.asarray(sensor_list.iloc[:,1:3]),0) 
        sensor_arr=np.asarray(sensor_list.iloc[:,-3:]) #xyz
        group_times=np.asarray(group_times).squeeze()/1e9
        alt=df.iloc[0,5]+150
        
        res=MLAT_iter(sensor_arr,group_times,alt,center=center,gt=gt)

    

        return res
    
    else:
        
        return None


if __name__ == "__main__":

    #-------------------------LOAD DATA-------------------------------------------
    
    DataFilepath=os.path.join(os.getcwd(),'data','round1_competition.csv')
    test_df=pd.read_csv(DataFilepath)
    test_df=test_df[test_df.isnull().any(1)]
      
    # #Result df
    # DataFilepath=os.path.join(os.getcwd(),'data','training_1_round_1','training_1_category_1_result.csv')
    # gt_df=pd.read_csv(DataFilepath)    
    
    SensorFilepath=os.path.join(os.getcwd(),'data',"sensors.csv")
    sensordf=pd.read_csv(SensorFilepath)
    
    bad_sensors=[131,142,208,414,460,470,550,474]
    
    del DataFilepath,SensorFilepath
    #-----------------------------------------------------------------------------
    
    # test_df=test_df.iloc[:1000,:]
    # start_time=datetime.datetime.now()
    count=0
    idx=[]
    for i in range(test_df.shape[0]):
        tmp_df=pd.DataFrame(test_df.iloc[i,:]).T
        X=DoMLAT(tmp_df,sensordf,bad_sensors)
        if X is not None:
            test_df.iloc[i,3]=X[0]
            test_df.iloc[i,4]=X[1]
            print(i)
            count+=1
    
    # stop_time=datetime.datetime.now()
    # print(stop_time-start_time)        
    pickle.dump(test_df,open(os.path.join(os.getcwd(),'data','res_df1.pckl'),'wb'))
