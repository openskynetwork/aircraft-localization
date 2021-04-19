import csaps
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import pickle
import common
import networkx as nx
from argparse import ArgumentParser
import graph_tool as gt
from graph_tool import topology
from collections import namedtuple

MIN_REQUIRED_NB = 10

SmoothedTraj = namedtuple('SmoothedTraj',['slat','slon','trajff'])

def add_args(parser):
    parser.add_argument('--inputfile', type=str, required=True)
    parser.add_argument('--outputfile', type=str, default='')
    parser.add_argument('--thr_error', type=float, default=1.)
    parser.add_argument('--smooth', type=float, default=1e-3)
    parser.add_argument('--speed_limit', type=float, default=300.)
    parser.add_argument('--ts', action='store_true',default=False)
    parser.add_argument('--pbname', type=str,default='')

def precompute_distance(lat, lon, t,speedmin,speedmax):
    '''compute a matrix telling if points i and j are reachable within speed limits'''
    xyz = np.stack(common.numpylla2ecef(lat,lon,np.zeros_like(lon)),-1)
    d = scipy.spatial.distance_matrix(xyz,xyz)
    t = np.transpose(np.array([t]))
    dt = scipy.spatial.distance_matrix(t,t)
    return np.maximum(d-speedmax*dt,0)+np.maximum(speedmin*dt-d,0)


# could be speed-up , i run shortest* two times
def get_gtlongest(dd):
    '''compute the longest path of points complying with the speed limits'''
    v,g,prop_dist = compute_gtgraph(dd)
    c = gt.topology.shortest_distance(g,source=None,target=None,weights=prop_dist,negative_weights=True,directed=True,pred_map=True,dag=True)
    i=min(list(range(dd.shape[0])),key=lambda i:np.min(c[v[i]].a))
    j=min(list(range(dd.shape[0])),key=lambda j:c[v[i]].a[j])
    mindist=c[i].a[j]
    longest_path,_ =gt.topology.shortest_path(g,source=v[i],target=v[j],weights=prop_dist,negative_weights=True,dag=True)
    longest_path =list(map(int,longest_path))
    return longest_path


def compute_gtgraph(dd):
    '''compute the graph of points complying with the speed limits: i and j are adjacent if i can be reached by j within the speed limits'''
    g=gt.Graph(directed=True)
    eprop_dist = g.new_edge_property("int")
    d={}
    for i in range(dd.shape[0]):
        d[i]=g.add_vertex()
    for i in range(dd.shape[0]):
        for j in range(i+1,dd.shape[1]):
            if dd[i,j]==0:
                e=g.add_edge(d[i],d[j])
                eprop_dist[e] = -1
    return d,g,eprop_dist

def filter_error(error, thr):
    if thr <= 1.:
        thr = np.sort(error,axis=None)[int(error.shape[0]*thr)-1]
    return error <= thr



def filter_speedlimit(lat,lon,t,speedmin,speedmax):
    '''returns boolean vector giving the longest sequence of points complying with the speed limits'''
    print("filter trajectory to keep the longest sequence of points complying with the speed limits")
    dd = precompute_distance(lat,lon,t,speedmin,speedmax)
    print("initial number of points",dd.shape[0])
    longest_path = get_gtlongest(dd)
    res=np.array([i in longest_path for i in range(dd.shape[-1])])
    print("longest sequence", np.sum(res))
    return res

def comparewithtrue(dspred, smoothedtraj, vdftrue=None):
    t = dspred.timeAtServer.values
    if vdftrue is None:
        truelat=dspred.latitude.values
        truelon=dspred.longitude.values
    else:
        vdf = vdftrue.merge(dspred,on='id',suffixes=('','nan'),how='inner')
        truelat=vdf.latitude.values
        truelon=vdf.longitude.values
    slat,slon = smoothedtraj.predict(t)
    d=common.haversine_distance(slat,slon,truelat,truelon)
    return d

class SmoothedTrajInterface:
    def __init__(self,trajff, smooth=None):
        self.trajff = trajff
    def predict(self, t):
        raise NotImplemented

class SmoothedTraj(SmoothedTrajInterface):
    def __init__(self,trajff, smooth=None):
        super().__init__(trajff,smooth)
        self.slat =  csaps.CubicSmoothingSpline(xdata=trajff.timeAtServer.values,ydata=trajff.nnpredlatitude.values,smooth=smooth).spline
        self.slon =  csaps.CubicSmoothingSpline(xdata=trajff.timeAtServer.values,ydata=trajff.nnpredlongitude.values,smooth=smooth).spline
    def predict(self, t):
        return self.slat(t),self.slon(t)


def main():
    parser = ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    ds = pickle.load(open(args.inputfile,'rb'))
    aircrafts = ds.aircraft.unique()
    if args.pbname != '':
        fdf,edf,m = common.loadall(args.pbname)
        vdftrue = None if args.ts else pd.read_csv("./Data/{}_result/{}_result.csv".format(args.pbname, args.pbname))
    d={}
    ld = []
    for aircraft in aircrafts:
        print("aircraft",aircraft)
        traj = ds.query("aircraft=="+str(aircraft)).reset_index(drop=True)
        if traj.shape[0]>MIN_REQUIRED_NB:
            error=traj.error.values
            # discard points with a large multilateration error
            filterror = filter_error(error,args.thr_error)
            trajf = traj.loc[filterror]
            # keep the longest sequence satisfying speed constraints
            if  trajf.shape[0]>MIN_REQUIRED_NB:
                filtspeed = filter_speedlimit(trajf.nnpredlatitude.values,trajf.nnpredlongitude.values,trajf.timeAtServer.values,0.,args.speed_limit)
                trajff = trajf.loc[filtspeed]
                drawtrue=common.haversine_distance(trajff.latitude,trajff.longitude,trajff.nnpredlatitude.values,trajff.nnpredlongitude.values)
                smoothedtraj = SmoothedTraj(trajff, args.smooth)
                t = trajff.timeAtServer.values
                slat, slon = smoothedtraj.predict(t)
                dsmoothraw = common.haversine_distance(slat,slon,trajff.nnpredlatitude.values,trajff.nnpredlongitude.values)
                tmin=np.min(t)
                tmax=np.max(t)
                if args.pbname!='':
                    traje=edf.query("aircraft=="+str(aircraft)).query(str(tmin)+"<=timeAtServer").query("timeAtServer<="+str(tmax)).reset_index(drop=True)
                    dsmoothtrue = comparewithtrue(traje,smoothedtraj,vdftrue)#[300:-300]
                    ld.append(dsmoothtrue)
                    print(common.rmse(ld[-1]),common.rmse90(ld[-1]),common.rmse50(ld[-1]))
                print(traj.shape,trajff.shape)
                d[aircraft]=smoothedtraj
    if len(ld)>0:
        dsmoothtrue = np.concatenate(ld)
        print(dsmoothtrue.shape[0],common.rmse(dsmoothtrue),common.rmse90(dsmoothtrue))
        e =np.sort(dsmoothtrue,axis=None)[:int(dsmoothtrue.shape[0]*0.6)+1]
        print(e.shape[0],common.rmse(e),common.rmse90(e))
    if args.outputfile != '':
        # save dict[aircraft]=SmoothedTraj
        with open(args.outputfile,'wb') as f:
            pickle.dump(d,f)

if __name__=='__main__':
    main()
