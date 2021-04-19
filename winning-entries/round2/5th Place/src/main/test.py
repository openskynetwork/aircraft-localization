# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 15:46:26 2021

@author: Param
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 10:11:57 2020

@author: Param
"""
import os
import sys

# Go to root dir
os.chdir(os.path.abspath(os.path.join(__file__ ,"../../..")))
sys.path.append(os.getcwd())
    
print(os.getcwd())
    
import pickle
import pandas as pd
import numpy as np 
import os
# from src.helper.utils import *
from sklearn.preprocessing import MinMaxScaler
from numpy import array, linalg, matrix
from scipy.special import comb as nOk
from scipy.signal import medfilt
import scipy
import pwlf
        

def bezfit(group,pred_group,j):
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
    deg=5
    Mtk = lambda n, t, k: t**(k)*(1-t)**(n-k)*nOk(n,k)
    bezierM = lambda ts: matrix([[Mtk(deg-1,t,k) for k in range(deg)] for t in ts])
        
    #Calculate t parameter
    scaler = MinMaxScaler()
    t=np.asarray(group.iloc[:,1])
    scaler.fit(t.reshape(-1,1))
    t=scaler.transform(t.reshape(-1,1)).squeeze()
    
    err_thresh=[0.7,0.3,0.1,0.05]
    for i in range(len(err_thresh)):
        #Iteration starts

        
        #intial fit
        M=bezierM(t)
        M_ = linalg.pinv(M)
        X=np.asarray(M_*np.asarray([group.predLat,group.predLon]).T)
                            
        #calculate outliers
        tmp_res=M*X
        tmp_res=np.stack(tmp_res).astype(None)
        diff=np.linalg.norm(tmp_res-np.stack(np.asarray([group.predLat,group.predLon]).T).astype(None),axis=1)
        idxs=np.where(diff<err_thresh[i]) #inliers
        #remove outliers
        group=group.iloc[idxs[0].tolist(),:]
        
        #recompute t
        t=np.asarray(group.iloc[:,1]).reshape(-1,1)
        scaler.fit(t)
        t=scaler.transform(t).squeeze()
                
    
    # initialize piecewise linear fit with your x and y data
    my_pwlfx = pwlf.PiecewiseLinFit(np.stack(group.iloc[:,1]).astype(None), np.stack(group.predLat).astype(None),degree=1)
    my_pwlfy = pwlf.PiecewiseLinFit(np.stack(group.iloc[:,1]).astype(None), np.stack(group.predLon).astype(None),degree=1)
    
    # fit the data for seven line segments
    res = my_pwlfx.fit(5)
    res = my_pwlfy.fit(5)
    
    #remove more outliers
    tmp_resx=my_pwlfx.predict(group.iloc[:,1])
    tmp_resy=my_pwlfy.predict(group.iloc[:,1])   
    tmp_arr=np.stack(np.asarray([group.predLat,group.predLon]).T).astype(None)
    tmp_err=np.linalg.norm(np.asarray([tmp_resx,tmp_resy]).T-tmp_arr,axis=1)
    group['tmp_err']=tmp_err
    group=group[group['tmp_err']<np.quantile(tmp_err,0.9)]

# #-----------------------------------------------------------------------------


# #---------------------PIECEWISE LINEAR FIT------------------------------------    
    
    #fit again

    # initialize piecewise linear fit with your x and y data
    my_pwlfx = pwlf.PiecewiseLinFit(np.stack(group.iloc[:,1]).astype(None), np.stack(group.predLat).astype(None),degree=2)
    my_pwlfy = pwlf.PiecewiseLinFit(np.stack(group.iloc[:,1]).astype(None), np.stack(group.predLon).astype(None),degree=2)
    
    # fit the data for seven line segments
    res = my_pwlfx.fit(5)
    res = my_pwlfy.fit(5)
#----------------------------------------------------------------------------- 

#-------------------------------PREDICTION------------------------------------      
    pred_tmp=pred_group.copy()
    pred_tmp=pred_tmp[(pred_tmp['timeAtServer']>=min(group.iloc[:,1])) & (pred_tmp['timeAtServer']<=max(group.iloc[:,1]))] 
    pred_tmp.iloc[:,3]=my_pwlfx.predict(pred_tmp.iloc[:,1])
    pred_tmp.iloc[:,4]=my_pwlfy.predict(pred_tmp.iloc[:,1])    


    
    #visualize_adsb_save(gt=[np.asarray([pred_tmp.iloc[:,3].values,pred_tmp.iloc[:,4].values]).T],pred=[np.asarray([group.predLat.values,group.predLon.values]).T],save_path="%d.html"%j)
    
    for i,idx in enumerate(pred_tmp.index):
        pred_group.at[idx,'predLat']=pred_tmp.iloc[i,3]
        pred_group.at[idx,'predLon']=pred_tmp.iloc[i,4]
    
    return pred_group

    

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
            #group['timeAtServer']=extracttime(group)
            group=group.sort_values(by='timeAtServer')
            print('%d of 300'%j)
            #Prediction  group
            pred_group=group.copy()
            pred_group['predLat']=np.NaN
            pred_group['predLon']=np.NaN
            
            #Localized points
            group=group.dropna(subset=['predLat'])
            
            #Splitting trajectory into high points regions
            groups=[group]
            
            #Trajectory estimation on each region 
            for k,group in enumerate(groups):
                if group.shape[0]>1:
                    try:
                      pred_group=bezfit(group,pred_group,j)  
                    except:
                      print('Exception')    
            
            #assign group to df
            for i,idx in enumerate(pred_group.index):
                df.at[idx,'predLat']=pred_group.predLat.iloc[i]
                df.at[idx,'predLon']=pred_group.predLon.iloc[i]
        
    return df
                
            
#-------------------------LOAD DATA-------------------------------------------

DataFilepath=os.path.join(os.getcwd(),'data','mlat_results_test.csv')
df=pd.read_csv(DataFilepath).iloc[:,1:]
df.index=df.iloc[:,0]
df['predLat']=df.latitude
df['predLon']=df.longitude
DataFilepath=os.path.join(os.getcwd(),'data','round2_sample_empty.csv')
out_df=pd.read_csv(DataFilepath)

del DataFilepath
#-----------------------------------------------------------------------------


# 

#---------------------------CURVE FITTING OPTIMIZATION-------------------------

df=curvefitting(df)
coverage=1-(df['predLat'].isna().sum()/df.shape[0])
print("coverage: " ,coverage)

#-----------------------------------------------------------------------------

#---------------------------RMSE----------------------------------

dfnew=df[['id','predLat','predLon','geoAltitude']]
dfnew.columns=['id','latitude','longitude','geoAltitude']    
df.to_csv('result_submission_out.csv')

