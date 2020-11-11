import pandas as pd
import torch
import pytorch_lightning as pl
import random
import itertools
from collections import namedtuple
from datautils import TensorDataLoader, Infinitecycle, DataPoint
import os
import pickle
from common import torchmapf, get_sensors_embedding, loadall, load_sensors, C, conv, compute_multilateration_error, lla2ecef, addxyz
import numpy as np
import time

class Test(pl.LightningModule):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--batch_size', type=int, default=10000)
        parser.add_argument('--save_sensorsparams', type=str,default="")
        parser.add_argument('--lr', type=float, default=5e2)
        parser.add_argument('--gamma', type=float, default=0.9)
        parser.add_argument('--pbname', type=str, required=True)
        parser.add_argument('--load_sensorsparams', type=str,default="")
        parser.add_argument('--steps_per_epoch', type=int, default=1000)

    def __init__(self,hparams):
        super().__init__()
        self.hparams = hparams

    def prepare_data(self):
        # load data
        pbname = self.hparams.pbname
        fdf, edf, m = loadall(self.hparams.pbname)
        m= fdf.numMeasurements.max()
        # substract min timestamp in order to have a lossless conversion to float32
        for i in range(m-1,-1,-1):
            edf.loc[:,"stimestamp"+str(i)]= edf.loc[:,"stimestamp"+str(i)].values - edf.loc[:,"stimestamp0"].values
        # define data used as input of pytorch
        self.dgroup = {}
        self.dgroup['xyz']=['x','y','z']
        self.dgroup['times']=['stimestamp'+str(i) for i in range(m)]
        self.dgroup['sensors']=['sensor'+str(i) for i in range(m)]
        self.dgroup['samesensors'] = ['samesensor'+str(i) for i in range(m-1)]
        self.Batch = namedtuple('Batch', list(self.dgroup))
        self.dico = {'sensors':torch.LongTensor}
        # keep known acft postiions
        self.nfts = edf.query('longitude==longitude').reset_index(drop=True)
        del edf
        # add x, y, z variable from lat lon alt variable
        self.nfts = addxyz(self.nfts, "geoAltitude")
        # compute same sensors indicator
        for j in range(1,m):
            self.nfts.loc[:,"samesensor"+str(j-1)] = self.nfts.loc[:,"sensor"+str(j-1)].values != self.nfts.loc[:,"sensor"+str(j)].values
        # define sensors params to be estimated
        loc = load_sensors(self.hparams.pbname,self.hparams.load_sensorsparams)
        self.loc_sensors = loc["loc"]
        self.alt_sensors = loc["alt"]
        self.shift_sensors = loc["shift"]
        self.C = loc["C"]
        print("initial values")
        self.print_sensorsparams()

    def print_sensorsparams(self):
        print("C {:.12f}".format(self.C.item()))
#        print("lat,lon",self.loc_sensors.weight)
#        print("alt",self.alt_sensors.weight)
#        print("time shift",self.shift_sensors.weight)

    def forward(self, batch):
        loc = self.loc_sensors(batch.sensors)
        h = self.alt_sensors(batch.sensors)
        shift = self.shift_sensors(batch.sensors)
        xyzloc = lla2ecef(loc[:,:,0], loc[:,:,1], h[:,:,0])
        def dist(i):
            return torch.norm(xyzloc[:,i]-batch.xyz,p=2,dim=-1)#torch.sqrt( ((xyzloc[:,i]-batch.xyz)**2).sum(-1))
        dists=[dist(i) for i in range(xyzloc.shape[1])]
        same_sensors = [None]+[batch.samesensors[:,i] for i in range(batch.samesensors.shape[-1])]
        #[torch.logical_or(batch.sensors[:,j] != batch.sensors[:,j+1],batch.times[:,j] != batch.times[:,j+1]) for j in range(batch.sensors.shape[-1]-1)]
        times = [batch.times[:,i] + shift[:,i,0] for i in range(batch.sensors.shape[-1])]
        return compute_multilateration_error(dists, times, same_sensors,self.C)

    def train_dataloader(self):
        print("train_dataloader")
        self.ts = DataPoint(self.nfts, conv(self.dgroup,self.dico, self.Batch._fields))
        tsload = TensorDataLoader(self.ts, batch_size = self.hparams.batch_size,shuffle=True,pin_memory=True)
        print("len(train_dataloader)",len(tsload))
        return Infinitecycle(tsload, self.hparams.steps_per_epoch)

    def training_step(self, batch, batch_idx):
        batch = self.Batch(*batch)
        y_hat = self.forward(batch)
        loss = y_hat.mean()
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss}

    def configure_optimizers(self):
        lr= self.hparams.lr / (1852 * 60)
        lrs = self.hparams.lr / C
        lralt = self.hparams.lr
        lrC = self.hparams.lr * 1e-6
        optimizer = torch.optim.Adam([
            {
            'params':self.loc_sensors.parameters(),'lr':lr},
            {'params':self.shift_sensors.parameters(),'lr':lrs},
            {'params':self.alt_sensors.parameters(),'lr':lralt},
            {'params':self.C,'lr': lrC}
        ])
        scheduler = {
            'scheduler':torch.optim.lr_scheduler.StepLR(optimizer,step_size=100,gamma=self.hparams.gamma),
            'interval':'step'
        }
        return [optimizer], [scheduler]
    def after_train(self):
        self.cuda()
        self.eval()
        self.print_sensorsparams()
        def predict():
            val_data = TensorDataLoader(self.ts, batch_size = 100000,shuffle=False,pin_memory=True)
            def prepare_batch(batch):
                return self.Batch(*tuple(x.cuda() for x in batch))
            pred = torchmapf(self.forward,prepare_batch,val_data)
            pred = pred.detach().cpu().numpy()
            print(np.mean(pred),np.median(pred))
        predict()
        if self.hparams.save_sensorsparams!="":
            dsave = {
                "loc":self.loc_sensors,
                "alt":self.alt_sensors,
                "shift":self.shift_sensors,
                "C":self.C,
            }
            torch.save(dsave,self.hparams.save_sensorsparams)
