import pandas as pd
import matplotlib.pyplot as plt
import pyproj
import torch
import pytorch_lightning as pl
import random
import itertools
from collections import namedtuple
from datautils import TensorDataLoader, Infinitecycle, DataTraj
import os
import pickle
from common import torchmapf, load_sensors, freeze, haversine_distance, loadall, lla2ecef, get_close_sensors, rmse90, rmse50, rmse, conv, compute_multilateration_error
from torch.nn import functional as F
import numpy as np
import time


class Test(pl.LightningModule):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--batch_size', type=int, default=100000000000)
        parser.add_argument('--load_sensorsparams', type=str,default="")
        parser.add_argument('--save_aircraftpos', type=str,default="")
        parser.add_argument('--pbname', type=str, required=True)
        parser.add_argument('--ts', action='store_true',default=False)
        parser.add_argument('--lr', type=float, default=1e-1)
        parser.add_argument('--close_sensor', type=float, default=15000.)
        parser.add_argument('--continuity', type=int, default=1)
        parser.add_argument('--steps_per_epoch', type=int, default=1000)
        parser.add_argument('--speed_limit', type=float,default=300.)

    def __init__(self,hparams):
        super().__init__()
        self.hparams = hparams

    def prepare_data(self):
        # load data
        self.pbname = self.hparams.pbname
        fdf,edf,m=loadall(self.pbname)
        # prepare data input for pytorch
        self.dgroup = {}
        self.dgroup['baroalt']=['baroAltitude']
        self.dgroup['times']=['stimestamp'+str(i) for i in range(m)]
        self.dgroup['timeAtServer']=['timeAtServer']
        self.dgroup['sensors']=['sensor'+str(i) for i in range(m)]
        self.dgroup['id']=['id']
        self.dgroup['samesensors'] = ['samesensor'+str(i) for i in range(m-1)]
        self.Batch = namedtuple('Batch', list(self.dgroup))
        self.dico = {'sensors':torch.LongTensor,'id':torch.LongTensor}
        if self.hparams.ts: # if training set
            self.nfts = edf.query('longitude==longitude').sort_values(by=["aircraft","timeAtServer"]).reset_index(drop=True)
        else: # if test set
            self.nfts = edf.query('longitude!=longitude').sort_values(by=["aircraft","timeAtServer"]).reset_index(drop=True)
        del edf
        print("#aircraft",self.nfts.aircraft.nunique())
        print("self.nfts.head()",self.nfts.head())
        print(self.nfts.numMeasurements.describe())
        # detect repeated and discard repeated measurements with same timeAtServer
        self.norepeat = np.concatenate((np.diff(self.nfts.timeAtServer.values)>0.,np.array([True])))
        print("detected repeated measurements",self.norepeat.shape[0]-self.norepeat.sum())
        self.nfts = self.nfts.iloc[self.norepeat].reset_index(drop=True)
        assert (not np.any(np.diff(self.nfts.timeAtServer.values)==0.))
        # compute same sensors indicator
        for j in range(1,m):
            self.nfts.loc[:,"samesensor"+str(j-1)] = self.nfts.loc[:,"sensor"+str(j-1)].values != self.nfts.loc[:,"sensor"+str(j)].values

            #np.logical_or(self.nfts.loc[:,"sensor"+str(j-1)].values != self.nfts.loc[:,"sensor"+str(j)].values,self.nfts.loc[:,"stimestamp"+str(j-1)].values != self.nfts.loc[:,"stimestamp"+str(j)].values)
        # load sensorsparams
        loc = load_sensors(self.pbname,self.hparams.load_sensorsparams)
        self.loc_sensors = loc["loc"].cpu()
        self.alt_sensors = loc["alt"].cpu()
        self.shift_sensors = loc["shift"].cpu()
        self.C = loc['C']
        # freeze them
        freeze(self.loc_sensors)
        freeze(self.alt_sensors)
        freeze(self.shift_sensors)
        self.C.cpu()
        self.C.requires_grad=False
        # detect close sensors
        lclose_sensors = get_close_sensors(self.loc_sensors,self.hparams.close_sensor,fdf.sensor.unique())
        # define aircraft positions parameters
        self.latlon = torch.nn.Embedding(int(self.nfts.id.max())+1, 2)
        def count(dataset):
            c = 1
            m= int(dataset.numMeasurements.max())
            for i in range(m-1):
                c += dataset.loc[:,"sensor"+str(i)].values != dataset.loc[:,"sensor"+str(i+1)].values
            return c
        # measure count
        self.nfts.loc[:,"countmeasure"] = count(self.nfts)
        self.nfts.loc[:,"countmeasurecorrected"] = self.nfts.loc[:,"countmeasure"].values
        def isamongsensor(dataset,s):
            c=0
            for i in range(m-1):
                c = np.maximum(c,dataset.loc[:,"sensor"+str(i)].values == s)
            return c
        # update measure count by substracting close sensors
        for (i,j) in lclose_sensors:
            self.nfts.loc[:,"countmeasurecorrected"] = self.nfts.loc[:,"countmeasurecorrected"].values - isamongsensor(self.nfts,i)*isamongsensor(self.nfts,j)
        # only estimate the aicraft positions that have enough measurements
        self.nfts = self.nfts.query("countmeasurecorrected>=4").reset_index(drop=True)
        # initialize aircraft positions with sensors barycenters
        def init_weights():
            prevpt = None
            for i,line in self.nfts.iterrows():
                if line.countmeasure != 0:
                    mean = tuple(self.loc_sensors.weight[int(line["sensor"+str(i)]),:] for i in range(line.countmeasure))
                    prevpt = sum(mean)/len(mean)
                else:
                    assert (prevpt is not None)
                self.latlon.weight[int(line.id),:]= prevpt
        with torch.no_grad():
            init_weights()

    def forward(self, batch):
        return self.latlon(torch.cat(batch.id))

    def train_dataloader(self):
        print("train_dataloader")
        self.ts = DataTraj(self.nfts, conv(self.dgroup,self.dico, self.Batch._fields),idtraj="aircraft")
        tsload = TensorDataLoader(self.ts, batch_size = self.hparams.batch_size,shuffle=False,pin_memory=True)
        print("len(train_dataloader)",len(tsload))
        return Infinitecycle(tsload, self.hparams.steps_per_epoch)


    def compute_error(self, batch):
        batch = self.Batch(*batch)
        latlon = self.forward(batch)
        sensors = torch.cat(batch.sensors)
        loc = self.loc_sensors(sensors)
        h = self.alt_sensors(sensors)
        shift = self.shift_sensors(sensors)
        baroalt = torch.cat(batch.baroalt)[:,0]
        xyzpos = lla2ecef(latlon[:,0,0],latlon[:,0,1],baroalt)
        xyzloc = lla2ecef(loc[:,:,0],loc[:,:,1],h[:,:,0])
        dists = torch.norm(xyzloc-xyzpos.unsqueeze(1),p=2,dim=-1)
        times = torch.cat(batch.times) + shift[:,:,0]
        times = [times[:,i] for i in range(sensors.shape[-1])]
        dists = [dists[:,i]for i in range(sensors.shape[-1])]
        samesensors = torch.cat(batch.samesensors)
        samesensors = [None] + [samesensors[:,i] for i in range(samesensors.shape[-1])]
        r = compute_multilateration_error(dists, times, samesensors, self.C)
        def compute_continuity(s):
            res=0
            ldxy=[]
            k=0
            for timeAtServer in batch.timeAtServer:
                n=timeAtServer.shape[0]
                ldxy.append(xyzpos_surface[k+s:k+n] - xyzpos_surface[k:k+n-s])
                k+=n
            dt = torch.cat([timeAtServer[s:,0] - timeAtServer[:-s,0]  for timeAtServer in batch.timeAtServer])
            dxy = torch.cat(ldxy)
            dxy = torch.norm(dxy,p=2,dim=-1)
            return F.relu(dxy - self.hparams.speed_limit * dt).sum()
        tempco=0.
        if self.hparams.continuity>=1:
            xyzpos_surface = lla2ecef(latlon[:,0,0], latlon[:,0,1], torch.zeros_like(latlon[:,0,1]))
        for i in range(1,self.hparams.continuity+1):
            tempco = tempco + compute_continuity(i) / self.hparams.continuity
        return r,tempco

    def training_step(self, batch, batch_idx):
        r,tempcomean = self.compute_error(batch)
        rm = r.sum()
        rate = 1000
        c = (batch_idx+1000*self.current_epoch)/rate
        ratio = max(1/(c+1),0.5)
        loss = rm*ratio + (1-ratio)*tempcomean
        print(rm,tempcomean)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss}

    def configure_optimizers(self):
        params=[{'params':self.latlon.parameters(),'lr':self.hparams.lr}]
        optimizer = torch.optim.Adam(params)
        scheduler = {
            'scheduler':torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=100, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=100, min_lr=0, eps=1e-80),
            'interval':'step',
            'monitor':'loss',
            }
        return [optimizer], [scheduler]

    def after_train(self):
        self.cuda()
        self.eval()
        def predict(dataset):
            vs =  DataTraj(dataset, conv(self.dgroup,self.dico, self.Batch._fields),idtraj = "aircraft")
            val_data = TensorDataLoader(vs, batch_size = self.hparams.batch_size,shuffle=False,pin_memory=True)
            def prepare_batch(batch):
                return self.Batch(*tuple([xi.cuda() for xi in x] for x in batch))
            pred = torchmapf(lambda x:self.forward(x).squeeze(1),prepare_batch,val_data)
            pred = pred.cpu().numpy()
            r = torchmapf(lambda x:self.compute_error(x)[0],prepare_batch,val_data).cpu().numpy()
            r = r/(dataset.countmeasure.values*(dataset.countmeasure.values-1)/2)
            dataset.loc[:,"nnpredlatitude"]=pred[:,[0]]
            dataset.loc[:,"nnpredlongitude"]=pred[:,[1]]
            dataset.loc[:,"error"]=r
            return r
        r = predict(self.nfts)
        if self.hparams.ts:
            suffix=""
        else:
            vdf = pd.read_csv("./Data/{}_result/{}_result.csv".format(self.pbname,self.pbname))
            self.nfts=self.nfts.merge(vdf,on='id',suffixes=('','true'))
            suffix="true"

        # statistics on estimated vs true aircraft position
        dist = haversine_distance(self.nfts.loc[:,"nnpredlatitude"].values, self.nfts.loc[:,"nnpredlongitude"].values, self.nfts.loc[:,"latitude"+suffix].values, self.nfts.loc[:,"longitude"+suffix].values)
        print("dist90",type(dist),rmse90(dist))
        rthresh = np.sort(r,axis=None)[r.shape[0]//2+1]
        print("rthresh",rthresh)
        print("dist90filter50",type(dist),rmse90(dist[r<=rthresh]))
        print("dist50",type(dist),rmse50(dist))

        # statistics on multilateration error
        print(self.nfts.loc[r.argmax()])
        print(self.nfts.loc[r.argmax(),["sensor"+str(i) for i in range(12)]].values)
        print(self.nfts.loc[r.argmax(),["stimestamp"+str(i) for i in range(12)]].values)
        print(self.nfts.countmeasure.describe())
        print(self.nfts.error.describe())
        if self.hparams.save_aircraftpos != "":
            pickle.dump(self.nfts,open(self.hparams.save_aircraftpos,'wb'))
