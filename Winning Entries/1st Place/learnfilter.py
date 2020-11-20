import pandas as pd
import numpy as np
from argparse import ArgumentParser
import pickle
import random
import scipy
import lightgbm as lgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import PredefinedSplit

from sklearn.model_selection import RandomizedSearchCV

from collections import namedtuple

Problem = namedtuple('Problem',['model','makeX','makey','parameters'])

def add_args(parser):
    parser.add_argument('--inputfile', type=str, required=True)
    parser.add_argument('--outputfile', type=str, default='')
    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--n_iter', type=int, default=100)
    parser.add_argument('--latlon', action='store_true',default=False)
    parser.add_argument('--classif', action='store_true',default=False)
    parser.add_argument('--dbaro', action='store_true',default=False)

# cross entropy with continuous label does not work with LGBMClassifier...
class MyLGBMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, lsensors,feature_fraction,num_leaves,learning_rate,min_child_samples,subsample,reg_lambda):
        super().__init__()
        self.lsensors=lsensors
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.reg_lambda = reg_lambda
        self.feature_fraction=feature_fraction
    def fit(self,X,y):
        paramdata={'max_bin':255}
        train_set_  = lgb.Dataset(X,y,categorical_feature=list(range(len(self.lsensors))),params=paramdata)
        params={
            'num_leaves':self.num_leaves,
            'learning_rate': self.learning_rate,
            'min_child_samples':self.min_child_samples,
            'subsample':self.subsample,
            'reg_lambda': self.reg_lambda,
            'objective':'xentropy',
            'subsample_freq':1,
            'feature_fraction':self.feature_fraction,
        }
        self.model = lgb.train(params,train_set_,num_boost_round=4000,categorical_feature=list(range(len(self.lsensors))))
    def predict(self,X,num_iteration=None):
        return self.model.predict(X,num_iteration=num_iteration)

    def score(self,X,y,sample_weight=None):
        y_pred = self.predict(X)
        return np.mean(y*np.log(y_pred)+(1-y)*np.log(1-y_pred))


def makeX(df,lvar,lsensors=None):
    m = len( [x for x in list(df) if x.startswith("sensor")])
    # assert below is just sanity check on round1_competition, it should be deleted otherwise
#    assert m==13
    vsensors = ["sensor"+str(i) for i in range(m)]
    sensors=df.loc[:,vsensors].values
    if lsensors is None:
        lsensors = sorted(np.unique(sensors))
    def containsensor(x):
        b=np.transpose(sensors==x)
        res=np.logical_or.reduce(list(b))
        return res
    l=[]
    for x in lsensors:
        l.append(containsensor(x))
    # matrix containing as many columns as different receivers in the file. each column is 1 if the receiver of this column has a measurement for the considered line
    isensors=np.stack(l,axis=-1)
    return lsensors,np.concatenate( (isensors,df.loc[:,lvar].values), axis=-1)

def makey(df):
    error=np.copy(df.smoothedtrueerror.values)
    # error >5000 are considered as ADS-B outlier
    error[error>5000]=np.median(error)
    order=np.argsort(error)
    rank_perc=np.argsort(order)/error.shape[0]
    return rank_perc

def addsmooth(dsmoothedTraj):
    for smo in dsmoothedTraj.values():
        slat,slon=smo.predict(smo.trajff.timeAtServer.values)
        smo.trajff.loc[:,"smoothedlatitude"]=slat
        smo.trajff.loc[:,"smoothedlongitude"]=slon

def main():
    parser = ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    dsmoothedTraj = pickle.load(open(args.inputfile,'rb'))
    addsmooth(dsmoothedTraj)
    ltraj =[smo for smo in dsmoothedTraj.values() if smo.trajff.shape[0]>100]
    print([smo.trajff.shape[0] for smo in ltraj])
    ntraj = len(ltraj)
    random.shuffle(ltraj)
    ts=pd.concat([smo.trajff for smo in ltraj])
    print(list(ts))
    # hyperparameters used for the 25.02 submissions
    # parameters ={'feature_fraction': 0.837266468665352, 'learning_rate': 0.0013782873851139932, 'min_child_samples': 33, 'num_leaves': 4, 'reg_lambda': 5.725801055525217e-12, 'subsample': 0.4944846046759285}
    # parameters={k:[v] for k,v in parameters.items()}
    parameters = {
        'num_leaves':scipy.stats.randint(2,11),
        'learning_rate': scipy.stats.loguniform(1e-4,1e-2),
        'min_child_samples':scipy.stats.randint(10,60),
        'subsample':scipy.stats.uniform(loc=0.3,scale=0.4),
        'reg_lambda': scipy.stats.loguniform(1e-14,1e-10),
        'feature_fraction':scipy.stats.uniform(loc=0.7,scale=0.3),
    }
    print(parameters)
    lvar=["error","smoothedrawerror","nb","dt01","countmeasure","countmeasurecorrected","baroAltitude"]+[x for x in list(ts) if "density" in x]+[x for x in list(ts) if "speed" in x]+[x for x in list(ts) if "curvature" in x]
    if args.latlon:
        lvar=lvar+["smoothedlatitude","smoothedlongitude"]#"nnpredlatitude","nnpredlongitude"]
    if args.dbaro:
        lvar=lvar+["dbaroAltitude"]
    # compute folds so that each aircraft is inside only one fold
    test_fold=np.concatenate([np.repeat(i//30,smo.trajff.shape[0]) for i,smo in enumerate(ltraj)])#[keep]
    ps = PredefinedSplit(test_fold)
    print("number of folds",ps.get_n_splits())
    lsensors,X = makeX(ts,lvar)
    X = X
    y = makey(ts)
    model = MyLGBMClassifier(lsensors,feature_fraction=1,num_leaves=7,learning_rate=0.1,min_child_samples=10,subsample=1.,reg_lambda=0.) if args.classif else lgb.LGBMRegressor(n_estimators=4000,subsample_freq=10,random_state=0,n_jobs=1,objective='l2',importance_type='gain',max_bin=511)
    model = RandomizedSearchCV(model, parameters, cv=ps, n_jobs=args.n_jobs, verbose=1, n_iter=args.n_iter, random_state=0)
    # 3 dirty lines below... just close your eyes and skip it
    model.argslearnmodel = args
    model.lsensors = lsensors
    model.lvar = lvar
    model.fit(X,y)
    print(model.score(X,y))
    print(model.cv_results_)
    print(model.best_params_)
    print(model.best_score_)
    if args.outputfile != '':
        with open(args.outputfile,'wb') as f:
            pickle.dump(model,f)


if __name__=='__main__':
    main()
