import pickle
import pandas as pd
import numpy as np 
import os
from src.helper.utils import *
from src.helper.MLAT3_1 import DoMLAT

#-------------------------LOAD DATA-------------------------------------------

    
#testing
DataFilepath=os.path.join(os.getcwd(),'data','round1_competition.csv')
test_df=pd.read_csv(DataFilepath)
test_df=test_df[test_df.isnull().any(1)]

# Sensor Information
SensorFilepath=os.path.join(os.getcwd(),'data',"sensors.csv")
sensordf=pd.read_csv(SensorFilepath)

sensorMetapath=os.path.join(os.getcwd(),'data',"sensor_meta_filtered.pckl")
sensorMeta=pickle.load(open(sensorMetapath,'rb'))


# Bad sensors (learned from training data)
bad_sensors=[142,414,460,470,550]

del DataFilepath,SensorFilepath,sensorMetapath
#-----------------------------------------------------------------------------


#-----------------------------IF TESTING-----------------------------------
test_df['conf']=np.NaN
idx=[]
for i in range(test_df.shape[0]):
    tmp_df=pd.DataFrame(test_df.iloc[i,:]).T
    X,conf=DoMLAT(tmp_df,sensordf,sensorMeta,bad_sensors)
    if X is not None:
        test_df.iloc[i,3]=X[0]
        test_df.iloc[i,4]=X[1]
        test_df.iloc[i,-1]=conf
        print(i)        
pickle.dump(test_df,open(os.path.join(os.getcwd(),'data','res_test_MLAT3_001.pckl'),'wb'))

#-----------------------------------------------------------------------------