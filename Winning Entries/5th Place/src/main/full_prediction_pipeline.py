# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 10:11:57 2020

@author: Param
"""

import pickle
import pandas as pd
import numpy as np 
import os
from src.helper.utils import *
from sklearn.preprocessing import MinMaxScaler
from numpy import array, linalg, matrix
from scipy.special import comb as nOk
from scipy.signal import medfilt
import scipy
import pwlf
        

def extracttime(df):
    """
    Replaces the "TimeatServer" field by the average of sensor timestamps 

    Parameters
    ----------
    df : DataFrame
        Subset for each trajectory.

    Returns
    -------
    times : array
        modified timestamps.

    """
    times=[]
    for i in range(df.shape[0]):
        tmp_time=np.fromstring(df.iloc[i,8].replace('[','').replace(']',''),sep=',')
        tmp_time=np.mean(tmp_time[range(1,tmp_time.shape[0],3)])/1e9
        times.append(tmp_time)
    return np.asarray(times)


def bezfit(group,pred_group):
    """
    Estimating trajectory from localized points and interpolating inbetween points

    Steps:
        1) REMOVE OUTLIERS BY BEZIER CURVE FITTING
        2) TRAJECTORY ESTIMATION BY PIECEWISE LINEAR FIT
        3) PREDICTION
        
    Parameters
    ----------
    group : DataFrame consisting of a few localized points
        Other points are Nan.
    pred_group : DataFrame, all points are NaN
        placeholder.

    Returns
    -------
    pred_group : DataFrame
        All points are predicted from estimated trajectory

    """
    
#---------------------REMOVE OUTLIERS BY BEZIER CURVE FITTING------------------    
    #Bezier Curve parameters
    deg=4
    Mtk = lambda n, t, k: t**(k)*(1-t)**(n-k)*nOk(n,k)
    bezierM = lambda ts: matrix([[Mtk(deg-1,t,k) for k in range(deg)] for t in ts])
        
    #Calculate t parameter
    scaler = MinMaxScaler()
    t=np.asarray(group.iloc[:,1])
    scaler.fit(t.reshape(-1,1))
    t=scaler.transform(t.reshape(-1,1)).squeeze()
    
    err_thresh=[0.7,0.3,0.1]
    for i in range(len(err_thresh)):
        #Iteration starts

        
        #intial fit
        M=bezierM(t)
        M_ = linalg.pinv(M)
        X=np.asarray(M_*np.asarray(group.iloc[:,[3,4]]))
                            
        #calculate outliers
        tmp_res=M*X
        tmp_res=np.stack(tmp_res).astype(None)
        diff=np.linalg.norm(tmp_res-np.stack(np.asarray(group.iloc[:,[3,4]])).astype(None),axis=1)
        idxs=np.where(diff<err_thresh[i]) #inliers
        #remove outliers
        group=group.iloc[idxs[0].tolist(),:]
        
        #recompute t
        t=np.asarray(group.iloc[:,1]).reshape(-1,1)
        scaler.fit(t)
        t=scaler.transform(t).squeeze()
        
    
    # initialize piecewise linear fit with your x and y data
    my_pwlfx = pwlf.PiecewiseLinFit(np.stack(group.iloc[:,1]).astype(None), np.stack(group.iloc[:,3]).astype(None),degree=2,weights= np.stack(group.iloc[:,9]).astype(None))
    my_pwlfy = pwlf.PiecewiseLinFit(np.stack(group.iloc[:,1]).astype(None), np.stack(group.iloc[:,4]).astype(None),degree=2,weights= np.stack(group.iloc[:,9]).astype(None))
    
    # fit the data for seven line segments
    res = my_pwlfx.fit(2)
    res = my_pwlfy.fit(2)
    
    #remove more outliers
    tmp_resx=my_pwlfx.predict(group.iloc[:,1])
    tmp_resy=my_pwlfy.predict(group.iloc[:,1])   
    tmp_arr=np.stack(np.asarray(group.iloc[:,[3,4]])).astype(None)
    tmp_err=np.linalg.norm(np.asarray([tmp_resx,tmp_resy]).T-tmp_arr,axis=1)
    group['tmp_err']=tmp_err
    group=group[group['tmp_err']<np.quantile(tmp_err,0.9)]
    
#-----------------------------------------------------------------------------


#---------------------PIECEWISE LINEAR FIT------------------------------------    
    
    #fit again
    # initialize piecewise linear fit with your x and y data
    my_pwlfx = pwlf.PiecewiseLinFit(np.stack(group.iloc[:,1]).astype(None), np.stack(group.iloc[:,3]).astype(None),weights=np.stack(group.iloc[:,9]).astype(None),degree=2)
    my_pwlfy = pwlf.PiecewiseLinFit(np.stack(group.iloc[:,1]).astype(None), np.stack(group.iloc[:,4]).astype(None),weights=np.stack(group.iloc[:,9]).astype(None),degree=2)
    
    # fit the data for seven line segments
    res = my_pwlfx.fit(6)
    res = my_pwlfy.fit(6)
#----------------------------------------------------------------------------- 

#-------------------------------PREDICTION------------------------------------      
    pred_tmp=pred_group.copy()
    pred_tmp=pred_tmp[(pred_tmp['timeAtServer']>=min(group.iloc[:,1])) & (pred_tmp['timeAtServer']<=max(group.iloc[:,1]))] 
    pred_tmp.iloc[:,3]=my_pwlfx.predict(pred_tmp.iloc[:,1])
    pred_tmp.iloc[:,4]=my_pwlfy.predict(pred_tmp.iloc[:,1])      
    
    for i,idx in enumerate(pred_tmp.index):
        pred_group.at[idx,'latitude']=pred_tmp.iloc[i,3]
        pred_group.at[idx,'longitude']=pred_tmp.iloc[i,4]
    
    return pred_group

def tsplit(group,binTresh=3):
    """
    Splitting and cutting trajectory into regions with high localized points. Regions 
    with low number of points are rejected.
    

    Parameters
    ----------
    group : dataFrame
        dataFrame of the trajectory
    binTresh : float, optional
        count threshold. The default is 3.

    Returns
    -------
    groups : list of dataframes
        list of parts of the trajectory.

    """
    
    if group.shape[0]==0:
        return []
    
    #only conf=0.8 points are considered 
    grouphigh=group[group.conf==0.9]        
    t=np.asarray(grouphigh.iloc[:,1])
    hist,bins=np.histogram(t,bins=np.ceil(((max(t)+0.1)-min(t))/65).astype(int))
    hist[hist<binTresh]=0
    hist[hist>=binTresh]=1
    hist=scipy.ndimage.morphology.binary_closing(hist).astype(int)
    
    #split and find bins
    subidx=[i for i in range(1,len(hist)) if hist[i]!=hist[i-1]]
    subidx=[0]+subidx+[len(hist)]
    sublists=[bins[subidx[i]:subidx[i+1]+1] for i in range(len(subidx[:-1])) if all(hist[subidx[i]:subidx[i+1]])==1]
    sublists=[item for item in sublists if len(item)>2]
    
    #subgroups
    groups=[]
    for sub in sublists:
        groups.append(group[(group['timeAtServer']>=min(sub)) & (group['timeAtServer']<=max(sub))])
        
          
    return groups
    

def curvefitting(df,gt_df=None):
    """
    Preprocessing for trajectory estimation 
    
    Input:
    df : input dataframe (from round_competition.csv)
    gt_df : ground truth dataframe (if trainni)
    
    Output:
        
    df : with predicted points    
    """

    # group by trajectory (process each trajectory seperately)
    for j,(name,group) in enumerate(df.groupby('aircraft')):
        group['timeAtServer']=extracttime(group)
        group=group.sort_values(by='timeAtServer')
        
        group['conf']=group['conf'].map({0.8:0.9,0.3:0.05})
        print(j)
        #Prediction  group
        pred_group=group.copy()
        pred_group['latitude']=np.NaN
        pred_group['longitude']=np.NaN
        
        #Localized points
        group=group.dropna(subset=['latitude'])
        
        #Splitting trajectory into high points regions
        groups=tsplit(group)
        
        #Trajectory estimation on each region 
        for k,group in enumerate(groups):
            if group.shape[0]>1:
                pred_group=bezfit(group,pred_group)        
        
        
        #assign group to df
        for i,idx in enumerate(pred_group.index):
            df.at[idx,'latitude']=pred_group.iloc[i,3]
            df.at[idx,'longitude']=pred_group.iloc[i,4]
        
    return df
                
            
#-------------------------LOAD DATA-------------------------------------------

DataFilepath=os.path.join(os.getcwd(),'data','round1_competition.csv')
data=pd.read_csv(DataFilepath)
data=data[data.isnull().any(1)]

DataFilepath=os.path.join(os.getcwd(),'data','round1_sample_empty.csv')
res_df=pd.read_csv(DataFilepath)

del DataFilepath
#-----------------------------------------------------------------------------


#---------------------------LOAD MODEL PREDICTION-----------------------------

test_df=pickle.load(open(os.path.join(os.getcwd(),'data','res_test_MLAT3_001.pckl'),'rb'))
test_df=pd.DataFrame(test_df)
test_df.columns=list(data.columns)+['conf']
#-----------------------------------------------------------------------------



#---------------------------CURVE FITTING OPTIMIZATION-------------------------

test_df_op1=curvefitting(test_df)
coverage=1-(test_df_op1['latitude'].isna().sum()/test_df_op1.shape[0])
print("coverage: " ,coverage)

#-----------------------------------------------------------------------------



#----------------------FILL RES DF--------------------------------------------

res_df.iloc[:,1:]=test_df_op1.iloc[:,[3,4,5]].values
entryNum=0
res_df.to_csv(os.path.join(os.getcwd(),'results','entry_%d.csv'%entryNum),index=False)

