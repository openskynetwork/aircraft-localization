import pickle
import common
import csaps
import pandas as pd
import numpy as np
from argparse import ArgumentParser
import pickle
from splineaircraftpos import SmoothedTraj, get_gtlongest
import scipy


def add_args(parser):
    parser.add_argument('--inputfile', type=str, required=True)
    parser.add_argument('--outputfile', type=str, default='')
    parser.add_argument('--altsmooth', type=float, default=0.001)
    parser.add_argument('--pbname', type=str,default='')

def precompute_diffalt(alt, t,speedmax):
    '''compute a matrix telling if i and j are complying with speedmax'''
    alt = np.transpose(np.array([alt]))
    d=scipy.spatial.distance_matrix(alt,alt)
    t=np.transpose(np.array([t]))
    dt=scipy.spatial.distance_matrix(t,t)
    return np.maximum(d-speedmax*dt,0)#+np.maximum(speedmin*dt-d,0)

def barosmooth(trajff,smooth):
    '''fit a spline with x=timeAtServer and y=baroAltitude'''
    speedmax = 50.8 # 10000ft/min
    dd=precompute_diffalt(trajff.baroAltitude.values,trajff.timeAtServer.values,speedmax)
    if np.sum(dd)>0:#    count edges number, if "full" graph we can keep it all without computing longest path
        longest_path = smooth.get_gtlongest(dd)
        keep = np.array([i in longest_path for i in range(dd.shape[-1])])
    else:
        keep = np.array([True]*dd.shape[-1])
    t = trajff.timeAtServer.values[keep]
    h = trajff.baroAltitude.values[keep]
    smo = csaps.CubicSmoothingSpline(xdata=t,ydata=h,smooth=smooth).spline
    return smo


def main():
    parser = ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    dsmoothedTraj = pickle.load(open(args.inputfile,'rb'))
    fdf,edf,m = common.loadall(args.pbname)
    edf=edf.sort_values(by=["aircraft","timeAtServer"]).reset_index(drop=True)
    lbandwidth = list(range(1,20))+list(range(20,100,20))
    for aircraft in dsmoothedTraj:
        trajedf=edf.query('aircraft=='+str(aircraft))#.reset_index(drop=True)
        smo = dsmoothedTraj[aircraft]
        trajedf=trajedf.query('timeAtServer<='+str(np.max(smo.trajff.timeAtServer.values))).query(str(np.min(smo.trajff.timeAtServer.values))+'<=timeAtServer')
        slat,slon=smo.predict(trajedf.timeAtServer.values)
        dist2derror=common.haversine_distance(slat,slon,trajedf.latitude.values,trajedf.longitude.values)
        trajedf=trajedf.assign(smoothedlatitude=slat,smoothedlongitude=slon,dist2derror=dist2derror)
        dle={i:[] for i in lbandwidth}
        ln=[]
        le=[]
        lt0=[]
        lt1=[]
        ls = {'mean':[],'max':[],'min':[]}
        lc = {'mean':[],'max':[],'min':[]}
        n=smo.trajff.timeAtServer.values.shape[0]
        ddslat = smo.slat.derivative(nu=2)
        ddslon = smo.slon.derivative(nu=2)
        dslat = smo.slat.derivative(nu=1)
        dslon = smo.slon.derivative(nu=1)
        def update(d,v):
            d['mean'].append(np.mean(v))
            d['min'].append(np.min(v))
            d['max'].append(np.max(v))
        # several points (>2) of trajedf are inside trajff.timeAtserver.values[i] and trajff.timeAtserver.values[i+1], so for trajff[i], we compute statistics on all the points between t0 and t1, these statistics will give us new feature for the point trajff[i]
        for i in range(n):
            t0=(smo.trajff.timeAtServer.values[max(i - 1,0)]+smo.trajff.timeAtServer.values[i])/2
            t1=(smo.trajff.timeAtServer.values[min(i + 1,n-1)]+smo.trajff.timeAtServer.values[i])/2
            trajedft0t1 = trajedf.query(str(t0)+"<=timeAtServer").query("timeAtServer<="+str(t1))
            t = trajedft0t1.timeAtServer.values
            lat = smo.slat(t)
            lon = smo.slon(t)
            dlat = dslat(t)
            dlon = dslon(t)
            ddlat = ddslat(t)
            ddlon = ddslon(t)
            h = trajedft0t1.baroAltitude.values
            speed, c = common.speed_curvature(lat,lon,dlat,dlon,ddlat,ddlon,h)
            update(ls,speed)
            update(lc,c)
            ln.append(trajedft0t1.shape[0])
            lt0.append(t0)
            lt1.append(t1)
            le.append(np.mean(trajedft0t1.dist2derror.values))
            draw = smo.trajff.timeAtServer.values-smo.trajff.timeAtServer.values[i]
            for bandwidth in lbandwidth:
                d =(draw/bandwidth)**2
                dle[bandwidth].append(np.sum(np.exp(-d)))
#            ld.append(density)
        sbaroalt=barosmooth(smo.trajff,args.altsmooth)
        sdbaroalt = sbaroalt.derivative(nu=1)
        smo.trajff.loc[:,"smoothedbaroAltitude"]=sbaroalt(smo.trajff.timeAtServer.values)
        smo.trajff.loc[:,"dbaroAltitude"]=sdbaroalt(smo.trajff.timeAtServer.values)
        # error between true traj and smoothed one on points between t0 and t1
        smo.trajff.loc[:,"smoothedtrueerror"]=np.array(le)
        slat,slon=smo.predict(smo.trajff.timeAtServer.values)
        # distance between smoothed traj and raw traj, gives an idea of how spreads the raw points are
        smo.trajff.loc[:,"smoothedrawerror"]=common.haversine_distance(slat,slon,smo.trajff.nnpredlatitude.values,smo.trajff.nnpredlongitude.values)
        # number of points between t0 and t1
        smo.trajff.loc[:,"nb"]=np.array(ln)
        smo.trajff.loc[:,"t0"]=np.array(lt0)
        smo.trajff.loc[:,"t1"]=np.array(lt1)
        # min speed between t0 and t1
        smo.trajff.loc[:,"speedmin"]=np.array(ls['min'])
        # mean speed between t0 and t1
        smo.trajff.loc[:,"speedmean"]=np.array(ls['mean'])
        # max speed between t0 and t1
        smo.trajff.loc[:,"speedmax"]=np.array(ls['max'])
        # min curvature between t0 and t1
        smo.trajff.loc[:,"curvaturemin"]=np.array(lc['min'])
        # mean curvature between t0 and t1
        smo.trajff.loc[:,"curvaturemean"]=np.array(lc['mean'])
        # max curvature between t0 and t1
        smo.trajff.loc[:,"curvaturemax"]=np.array(lc['max'])
        smo.trajff.loc[:,"dt01"]=smo.trajff.loc[:,"t1"].values - smo.trajff.loc[:,"t0"]
        smo.trajff.loc[:,"dspeed"]=smo.trajff.speedmax.values-smo.trajff.speedmin.values
        smo.trajff.loc[:,"dspeeddt01"]=smo.trajff.dspeed.values/smo.trajff.dt01.values
        for bandwidth in lbandwidth:
            # density of measurement in trajff across timeAtServer. The more point per unit of time, the more precise the prediction should be
            smo.trajff.loc[:,"density"+str(bandwidth)]=np.array(dle[bandwidth])
    if args.outputfile != '':
        with open(args.outputfile,'wb') as f:
            pickle.dump(dsmoothedTraj,f)


if __name__=='__main__':
    main()
