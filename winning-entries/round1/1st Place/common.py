import torch
from collections import namedtuple
from pytorch_lightning import loggers, callbacks
import numpy as np
import scipy
from scipy import stats
import os
import pandas as pd
import pickle
import time

# speed of light in vaccum in m/nanoseconds
C=0.299792458

torch.set_num_threads(4)

def freeze(model):
    for params in model.parameters():
        params.requires_grad = False

def addxyz(df,height=""):
#    x,y,z=lla2xyz(df.longitude.values,df.latitude.values,df.loc[:,height].values)
    x,y,z=numpylla2ecef(df.latitude.values,df.longitude.values,df.loc[:,height].values)
    df=df.assign(x=x,y=y,z=z)
    return df


def speed_curvature_from_slat(slat,slon,t,h):
    lat = slat(t)
    lon = slon(t)
    dslat = slat.derivative(nu=1)
    dslon = slon.derivative(nu=1)
    ddslat =slat.derivative(nu=2)
    ddslon =slon.derivative(nu=2)
    return speed_curvature(lat,lon,dslat(t),dslon(t),ddslat(t),ddslon(t),h)


def speed_curvature(lat,lon,dlat,dlon,ddlat,ddlon,h):
    # theta = lon
    # phi = inc
    R = 6378137
    r = R+h
    deg2rad = np.pi/180
    lat = deg2rad * lat
    lon = deg2rad * lon
    dlat = deg2rad * dlat
    dlon = deg2rad * dlon
    inc = np.pi/2-lat
    dinc = -dlat
    ddinc = -ddlat
    dx = r * dlon * np.sin(inc)
    dy = r * dinc
    ddx = r * ddlon * np.sin(inc) + 2 * r * dlon*dinc *np.cos(inc)
    ddy = r * ddinc - r * dlon**2 *np.sin(inc)*np.cos(inc)
    c = (dx *ddy - dy *ddx)/(dx**2+dy**2)**(3/2)
    return np.sqrt(dx**2+dy**2), np.abs(c)


# there might be a way to speed-up this by vectorizing it, please find a commented version of what i tried below, but it does not work due to the difficulty to deal with the "same_sensors" thing. But I would bet there is a way to make it work. In
def compute_multilateration_error(dists, times, same_sensors, c):
    r = 0.
    for i,di in enumerate(dists):
        for j in range(i+1,len(dists)):
            b = same_sensors[j]
            if b.byte().any():
                dxyz = di - dists[j]
                dtime = times[i]-times[j]
                e = torch.abs(dxyz - c*dtime)*b
                r += e
            else:
                break
    return r

# I tried to vectorize it here !
# def compute_multilateration_error(dists, times, same_sensors, c):
#     ddists = dists.unsqueeze(1)-dists.unsqueeze(2)
#     dt = times.unsqueeze(1)-times.unsqueeze(2)
# #    sames = sensors.unsqueeze(1) != sensors.unsqueeze(2)
# #    return torch.sum(torch.abs(ddists-c*dt)*sames,dim=[1,2])*0.5
# #    print(ds.shape,dt.shape,sames.shape)
# #    raise Exception
#     r = 0.
#     for i in range(dists.shape[-1]):
#         for j in range(i+1,dists.shape[-1]):
#             b=same_sensors[j]
#             if torch.sum(b).item()>0:
#                 r+=torch.abs(ddists[:,i,j]-c*dt[:,i,j])*b
#             else:
#                 break
#     return r


def mse(e):
    return np.mean(e**2)
def rmse(e):
    return np.sqrt(mse(e))

def rmse90(x):
    return rmse(np.sort(np.abs(x),axis=None)[:int(x.shape[0]*0.9)+1])
def rmse50(x):
    return rmse(np.sort(np.abs(x),axis=None)[:int(x.shape[0]*0.5)+1])

def get_close_sensors(loc,thresh,serials):
    xyz = lla2ecef(loc.weight[:,0],loc.weight[:,1],torch.zeros_like(loc.weight[:,1]))
#    print(xyz.shape)
    xyz = xyz.numpy()
    d=scipy.spatial.distance_matrix(xyz,xyz)
    res=[(i,j) for (i,di) in enumerate(d) for (j,dij) in enumerate(di) if dij<thresh and i<j and (i in serials) and (j in serials)]
    print("list of close sensors", res)
    return res

def get_sensors_embedding(sensors):
    maxemb = sensors.serial.max()
    loc_sensors = torch.nn.Embedding(maxemb+1, 2,scale_grad_by_freq=True)
    alt_sensors = torch.nn.Embedding(maxemb+1, 1,scale_grad_by_freq=True)
    shift_sensors = torch.nn.Embedding(maxemb+1, 1,scale_grad_by_freq=True)
    with torch.no_grad():
        torch.nn.init.zeros_(loc_sensors.weight)
        torch.nn.init.zeros_(alt_sensors.weight)
        torch.nn.init.zeros_(shift_sensors.weight)
        for _,line in sensors.iterrows():
            loc_sensors.weight[line.serial,:]=torch.Tensor((line.latitude,line.longitude))
            alt_sensors.weight[line.serial,:]=torch.Tensor((line.height,))
    return {"loc":loc_sensors,"alt":alt_sensors,"shift":shift_sensors,"C":torch.nn.Parameter(torch.Tensor([C]))}


def rad(degrees):
    return degrees*np.pi/180

# convert radians to decimal degrees
def deg(radians):
    return 180*radians/np.pi

def lla2ecef(lat, lon, h):
    lat = rad(lat)#[:,:,0])
    lon = rad(lon)#:,:,1])
#    h = h[:,:,0]
    # WSG84 ellipsoid constants
    a = 6378137
    e = 8.1819190842622e-2
    # prime vertical radius of curvature
    N = a / torch.sqrt(1 - e**2 * torch.sin(lat)**2)
    x = (N+h) * torch.cos(lat) * torch.cos(lon)
    y = (N+h) * torch.cos(lat) * torch.sin(lon)
    z = ((1-e**2) * N +h) * torch.sin(lat)
    return torch.stack((x, y, z),dim=-1)



def numpylla2ecef(lat,lon,h):
    lat = rad(lat)#[:,:,0])
    lon = rad(lon)#:,:,1])
#    h = h[:,:,0]
    # WSG84 ellipsoid constants
    a = 6378137
    e = 8.1819190842622e-2
    # prime vertical radius of curvature
    N = a / np.sqrt(1 - e**2 * np.sin(lat)**2)
    x = (N+h) * np.cos(lat) * np.cos(lon)
    y = (N+h) * np.cos(lat) * np.sin(lon)
    z = ((1-e**2) * N +h) * np.sin(lat)
    return (x, y, z)

def load_sensors(pbname,load_loc,final=False):
    if load_loc!="":
        loc = torch.load(load_loc)
        for v in loc.values():
            v.cpu()
        return loc
    else:
        sensorname= "./Data/{}/{}.csv".format(pbname,"sensors")
        sensors = pd.read_csv(sensorname)
        return get_sensors_embedding(sensors)

def expandrow(df):
    l = []
    for i, line in df.iterrows():
        lm=eval(line.measurements)
        for id_time_signal in lm:
            l.append(list(line.values)+id_time_signal)
    return pd.DataFrame(l,columns=list(df)+["sensor","stimestamp","strength"])

def expandcol(df):
    ''' load the data, all the measurements are grouped, one line <=> one point, if not enough measurement, the last measurement is repeated. Might not be the best way to do it, In retrospect, np.nan would be better, would ease debug and maybe ease vectorizing of compute_multilateration_error'''
    m = df.numMeasurements.max()
    df = df.sort_values(by=["id","stimestamp"]).reset_index(drop=True)
    g = df.groupby("id")
    s = ["sensor","stimestamp","strength"]
#    indextime = s.index("stimestamp")
    ns = len(s)
    what = list(df)[:8]#+["x","y","z"]
    nwhat = len(what)
    res = np.zeros((g.ngroups,len(what)+len(s)*m))
    c = what+[ x+str(i) for i in range(m) for x in s]
    lg = []
    tstart = time.time()
    for j,(name,gdf) in enumerate(g):
        resline =res[j]
#        print(name)

#        print("gdf",gdf)
#        gdf.assign({})
#        gdf = gdf.sort_values(by="stimestamp").reset_index(drop=True)
        resline[:nwhat]=gdf.loc[:,what].values[0]
#        tmin = resline[7+indextime]
        n = gdf.shape[0]
        l=[]
#        print(gdf.loc[:,s].values)
#        print(gdf.loc[:,s].values.reshape(n*ns))
#        raise Exception
        resline[nwhat:nwhat+n*ns] = gdf.loc[:,s].values.reshape(n*ns)
        resline[nwhat+n*ns:] = np.tile(resline[nwhat+(n-1)*ns:nwhat+n*ns],m-n)
        # if j > 1000:
        #     break
#    time1000=time.time()-tstart
    res = pd.DataFrame(res,columns = c)
    # print(time1000*g.ngroups/1000)
    # print(res[:10])
    # raise Exception
    return (s,res)#g.apply(toapply)#m,pd.DataFrame(l,columns=list(df)+cols)

def loadset(pbname):
    '''load the data, one measurement <=> one line'''
    filename= "./Data/{}/{}.csv".format(pbname,pbname)
#    sensorname= "./Data/{}/{}.csv".format(pbname,"sensors")
#    sensors = pd.read_csv(sensorname)
#    codes,uniques = pd.factorize(sensors["type"])
    pklname = filename+".pkl"
    if not os.path.exists(pklname):
        df = pd.read_csv(filename)
        fdf = expandrow(df)
        pickle.dump(fdf, open(pklname,"wb"))
    else:
        fdf= pickle.load(open(pklname,"rb"))
    # print(fdf.describe())
    # print(fdf.head())
    # mintime = fdf.groupby(['id'])["stimestamp"].min()
    # print(mintime)
    # print(type(mintime))
    # fdf = fdf.merge(mintime,on='id',suffixes=('','min'))
    # fdf = addxyz(fdf,"geoAltitude")
    # print(fdf.head())
    # sensors["type"]=codes
    # sensors = addxyz(sensors,"height")
    # fdf = fdf.merge(sensors,left_on="sensor",right_on="serial",suffixes=("","sensor"))
    # p1 = (fdf.x.values,fdf.y.values,fdf.z.values)
    # p2 = (fdf.xsensor.values,fdf.ysensor.values,fdf.zsensor.values)
    # fdf = fdf.assign(dist3D = dist(p1,p2))
    return fdf

def filter_edf(edf,m,srepeat,threshold):
    '''filter the measurement in edf, discard measurement that are made [threshold] nanoseconds after the first measurement'''
    for i in range(1,m):
        b= edf.loc[:,"stimestamp"+str(i)].values-edf.loc[:,"stimestamp0"].values >= threshold
        if b.any():
            si = [x+str(i) for x in srepeat]
            si1 = [x+str(i-1) for x in srepeat]
            edf.loc[b,si] = edf.loc[b,si1].values
            edf.loc[b,"sensor"+str(i)] = edf.loc[b,"sensor"+str(i-1)].values
            edf.loc[b,"strength"+str(i)] = edf.loc[b,"strength"+str(i-1)].values
    return edf



def loadall(pbname):
    '''load all the data, fdf and edf'''
    fdf = loadset(pbname)
    print(fdf.head())
    edfname = "./Data/{}/{}_edf.pkl".format(pbname,pbname)
    if not os.path.exists(edfname):
        srepeat,edf = expandcol(fdf)
        pickle.dump((srepeat,edf), open(edfname,"wb"))
    else:
        srepeat,edf = pickle.load(open(edfname,"rb"))
    print(edf.head())
    m = fdf.numMeasurements.max()
    for i in range(m-1,-1,-1):
        edf.loc[:,"stimestamp"+str(i)]= edf.loc[:,"stimestamp"+str(i)].values - edf.loc[:,"stimestamp0"].values
    threshold = 1e6
    edf = filter_edf(edf,m,srepeat,threshold)
    return fdf,edf,m

def haversine_distance(lat1, lon1, lat2, lon2):
    r = 6371000
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) *   np.sin(delta_lambda / 2)**2
    res = r * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
    return res


def conv(dgroup, dico, fields):
    '''utility function for loading input data for pytorch'''
    def f(df):
        return  tuple(dico.get(k,torch.Tensor)(df.loc[:,dgroup[k]].values) for k in fields)
    return f

def torchmapf(f,prepare_batch,vs):
    l=[]
    with torch.no_grad():
        for batch in vs:
            batch = prepare_batch(batch)
            l.append(f(batch))
    pred=torch.cat(l)
    return pred
