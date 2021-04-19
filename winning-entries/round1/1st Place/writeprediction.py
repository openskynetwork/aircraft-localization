import pickle
import common
import pandas as pd
import numpy as np
from argparse import ArgumentParser
import pickle
from splineaircraftpos import SmoothedTraj
import learnfilter
from learnfilter import MyLGBMClassifier
from math import ceil


latn = "latitudepredictionfile"
lonn = "longitudepredictionfile"

def add_args(parser):
    parser.add_argument('--inputfile', type=str, required=True)
    parser.add_argument('--outputfile', type=str, default='')
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--coverage', type=float,default=None)
    parser.add_argument('--min_continuous_to_keep', type=int,default=10)
    parser.add_argument('--pbname', type=str,default='')


def merge_with_result(pred,pbname):
    '''build the dataframe to be written as the prediction file'''
    vdf = pd.read_csv("./Data/{0}_result/{0}_result.csv".format(pbname))
    vdf.loc[:,"longitude"] = np.nan
    vdf.loc[:,"latitude"] = np.nan
    vdf=vdf.merge(pred,on="id",how="left")
    vdf=vdf.drop(columns=["longitude","latitude"])
    vdf=vdf.rename(columns={latn:"latitude",lonn:"longitude"})
    return vdf

def merge_intervals(lt):
    '''merges valid time intervals to speed up predictions'''
    lt = sorted(lt)
    l = []
    tdold = None
    tfold = None
    for (td,tf) in lt:
        if tfold == td:
            tfold = tf
        else:
            if tdold is not None:
                l.append((tdold,tfold))
            tdold = td
            tfold = tf
    if tdold is not None:
        l.append((tdold,tfold))
    return l

def merge_continuous_filter_intervals(times, keeps, nb_continuous):
    '''merges valid time intervals to speed up predictions, but with a constraint on the number of contiguous points that are judged to be worth keeping'''
    l=[]
    for i, k in enumerate(keeps):
        if k:
            if len(l)==0 or l[-1][-1]!=i-1:
                l.append([i])
            else:
                l[-1].append(i)
    return [(times[x[0]],times[x[-1]])for x in l if len(x)>=nb_continuous]


def compute_dicot_from_proba_thresh(dsmoothedTraj, dproba, thresh, min_continuous_to_keep):
    '''compute merged valid intervals from points that are judged to be better than a given threshold'''
    d={}
    for aircraft,smo in dsmoothedTraj.items():
        keep = dproba[aircraft][0]<=thresh
        d[aircraft]=list(zip(smo.trajff.t0[keep], smo.trajff.t1[keep]))
        merged=merge_continuous_filter_intervals(smo.trajff.timeAtServer.values,keep,min_continuous_to_keep)
        d[aircraft]=merged
    return d

def search_proba_thresh(edf, dsmoothedTraj, tokeep, dproba, min_continuous_to_keep):
    '''search the minimum threshold that gives at least [tokeep] number of points'''
    def count_points(edf,dicot):
        s = 0
        for aircraft in dicot:
            edfaircraft = edf.query("aircraft=="+str(aircraft))
            for (td,tf) in dicot[aircraft]:
                s += edfaircraft.query(str(td)+"<=timeAtServer").query("timeAtServer<="+str(tf)).shape[0]
        return s
    def f(proba_thresh):
        dicot=compute_dicot_from_proba_thresh(dsmoothedTraj,dproba, proba_thresh, min_continuous_to_keep)
        return count_points(edf,dicot)
    vproba = np.concatenate([np.repeat(prob,nb) for (prob,nb) in dproba.values()])
    starting_proba_thresh = np.quantile(vproba, tokeep/vproba.shape[0])
    possible_proba_thresh = np.sort(np.unique(vproba[vproba>=starting_proba_thresh]))
    def dicho(i,j):
        while i+1 < j:
            print(i,j)
            m = (i+j)//2
            if f(possible_proba_thresh[m])>=tokeep:
                j = m
            else:
                i = m
        return possible_proba_thresh[j]
    return dicho(0,len(possible_proba_thresh)-1)

def compute_dicot_from_model(edf, dsmoothedTraj, model, tokeep, min_continuous_to_keep):
    d={}
    dproba={}
    for aircraft,smo in dsmoothedTraj.items():
        _,X = learnfilter.makeX(smo.trajff,model.lvar, model.lsensors)
        dproba[aircraft] = (model.predict(X),smo.trajff.nb.values)
    print("searching probability threshold")
    proba_thresh = search_proba_thresh(edf, dsmoothedTraj, tokeep, dproba, min_continuous_to_keep)
    print("found probability threshold", proba_thresh)
    return compute_dicot_from_proba_thresh(dsmoothedTraj, dproba, proba_thresh, min_continuous_to_keep)

def build_predictionfile(edf,dsmoothedTraj, dicot):
    edf = edf.copy()
    l=[]
    for aircraft in dicot:
        smo = dsmoothedTraj[aircraft]
        edfaircraft = edf.query("aircraft=="+str(aircraft))
        for (td, tf) in dicot[aircraft]:
            trajedf=edfaircraft.query(str(td)+"<=timeAtServer").query("timeAtServer<="+str(tf))
            slat,slon = smo.predict(trajedf.timeAtServer.values)
            trajedf.loc[:,latn]=slat
            trajedf.loc[:,lonn]=slon
            l.append(trajedf.loc[:,["id",latn,lonn,"latitude","longitude"]])
    pred = pd.concat(l)
    return pred

def main():
    parser = ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    dsmoothedTraj = pickle.load(open(args.inputfile,'rb'))
    learnfilter.addsmooth(dsmoothedTraj)
    fdf,edf,m = common.loadall(args.pbname)
    if args.model=='' or args.coverage is None:
        #compute valid time intervals for each aircraft, without filtering
        dicot = {aircraft:[(np.min(smo.trajff.timeAtServer),np.max(smo.trajff.timeAtServer))] for aircraft,smo in dsmoothedTraj.items()}
    else:
        #compute valid time intervals for each aircraft with filtering using the model
        with open(args.model,'rb') as f:
            model = pickle.load(f)
        vdf = pd.read_csv("./Data/{0}_result/{0}_result.csv".format(args.pbname))
        tokeep = ceil(vdf.shape[0]*args.coverage)
        print("# of points to keep",tokeep)
        dicot = compute_dicot_from_model(edf,{k:smo for (k,smo) in dsmoothedTraj.items() if smo.trajff.shape[0]>0},model, tokeep, args.min_continuous_to_keep)
    print("compute prediction")
    pred= build_predictionfile(edf,dsmoothedTraj, dicot)
    print("compute distance")
    d = common.haversine_distance(pred.loc[:,latn].values,pred.loc[:,lonn].values,pred.latitude.values,pred.longitude.values)
    print(d.shape[0],common.rmse(d),common.rmse90(d))
    if args.outputfile != '':
        print("writing prediction file")
        pred=pred.drop(columns=["longitude","latitude"])
        df=merge_with_result(pred,args.pbname)
        print("actual coverage",df.query("longitude==longitude").shape[0]/df.shape[0])
        df.to_csv(args.outputfile,float_format="%.12f",index=False)


if __name__=='__main__':
    main()
