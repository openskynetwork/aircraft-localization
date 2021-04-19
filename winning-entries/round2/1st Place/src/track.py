import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from collections import defaultdict
import itertools

from tqdm import tqdm

from src.geo import P3, haversine_distance
from src.stations import Stations
from src.filters import filter_speedlimit
from optimize import solve_point

from scipy.optimize import fmin_l_bfgs_b

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import HuberRegressor


## Predict track using points after solving multilateration equations
def check_point(full, part, t0, Teps=30, Nmin=10, ds_max=50):
    # select points from the neighbourhood
    ii = np.where(abs(part.timeAtServer.values - t0) < Teps)[0]
    
    if len(ii) < Nmin:
        return -1
    
    # Prepare polynomial features for time
    poly = PolynomialFeatures(2)
    t0 = np.min(part.timeAtServer.values[ii])
    t = (part.timeAtServer.values[ii] - t0).reshape(-1, 1)
    poly.fit(t)
    poly_t = poly.transform(t)

    # two huber regressions for latitude and longitude
    hr_lat = HuberRegressor().fit(poly_t, part.lat_pred.values[ii])
    hr_lon = HuberRegressor().fit(poly_t, part.lon_pred.values[ii])

    # Predict locations in full
    id_min, id_max = np.min(part.id.values[ii]), np.max(part.id.values[ii])
    t_full = (full.loc[(full.id >= id_min) & (full.id <= id_max)].timeAtServer.values - t0).reshape(-1, 1)
    poly_t_full = poly.transform(t_full)

    full.loc[(full.id >= id_min) & (full.id <= id_max), 'lon_pred2'] = hr_lon.predict(poly_t_full)
    full.loc[(full.id >= id_min) & (full.id <= id_max), 'lat_pred2'] = hr_lat.predict(poly_t_full)
    
    return id_max


## class to work with a track
class Track:
    def __init__(self, df, stations=None):
        self._aircraft = df.aircraft.values[0]
        
        self._telemetry = df.reset_index(drop=True)
        if stations is not None:
            self._st = stations
        
        # store index for all stations and their pairs like (st_1, st_2)
        self._stations = defaultdict(list)
        # store time records for single stations
        self._times = defaultdict(list)
        
        for i, meas in enumerate(self._telemetry.measurements):
            if stations is not None:
                sts = [w[0] for w in eval(meas) if w[0] in stations.inventory]
                time = [w[1] for w in eval(meas) if w[0] in stations.inventory]
            else:
                sts = [w[0] for w in eval(meas)]
                time = [w[1] for w in eval(meas)]

            for j in range(1, 4):
                for x in itertools.combinations(sorted(sts), r=j):
                    self._stations[x].append(i)
                    
            for s, t in zip(sts, time):
                self._times[s].append(t*1e-9)
                
                
    def __repr__(self):
        return f'Track {self._aircraft}: {self._telemetry.shape[0]} points'
    
    
    @property
    def aircraft(self):
        return self._aircraft
    
    
    def single_stations(self):
        return self._times.keys()
    
    
    def stations_stat(self, station=None):
        if not station:
            output = {}
        
            for key in self._stations:
                output[key] = len(self._stations[key])
                
            return output
            
        elif type(station) is int:
            return len(self._stations[(station,)])
        else:
            return len(self._stations[station])
    
    
    def stations_comb(self):
        return self._stations.keys()
    
    
    def telemetry(self, station=None):
        cols = ['latitude', 'longitude', 'geoAltitude', 'baroAltitude']
        
        if station is None:
            return self._telemetry[cols]
        elif type(station) is int:
            return self._telemetry.loc[self._stations[(station,)], cols].reset_index(drop=True)
        else:
            return self._telemetry.loc[self._stations[station], cols].reset_index(drop=True)
        
    
    def time(self, station):
        if type(station) is int:
            return self._times[station]
        else:
            station = sorted(station)
            
            p = (pd.DataFrame()
                 .assign(ind=self._stations[(station[0],)],
                         time=self._times[station[0]])
                 .set_index('ind')
                 .rename(columns={'time':str(station[0])})
                )
            
            for s in station[1:]:
                p1 = (pd.DataFrame()
                      .assign(ind=self._stations[(s,)],
                              time=self._times[s])
                      .set_index('ind')
                      .rename(columns={'time':str(s)})
                     )
                p = pd.concat([p, p1], axis=1, join='inner')
                
            return p.reset_index(drop=True)
        
        
    def correct_time(self, stations):
        # time correction for each stations
        for s in self._times:
            self._times[s] = stations.correct_time(s, self._times[s])
            
        # prepare and add stations and times to self._telemetry
        times = defaultdict(list)
        sts = defaultdict(list)

        for i in range(self._telemetry.shape[0]):
            times[i] = []
            sts[i] = []

        for station in sorted([s for s in self._stations.keys() if len(s)==1]):
            station = station[0]
            for i, j in enumerate(self._stations[(station,)]):
                if ~np.isnan(self._times[station][i]):
                    times[j].append(self._times[station][i])
                    sts[j].append(station)
                
        out_s = []
        out_t = []

        for i in range(self._telemetry.shape[0]):
            out_s += [sts[i]]
            out_t += [times[i]]
            
        self._telemetry['stations'] = out_s
        self._telemetry['times'] = out_t
        
    
    def plot(self, cmin=0, cmax=15000):
        plt.scatter(self._telemetry.longitude,
                    self._telemetry.latitude,
                    c = self._telemetry.geoAltitude)
        plt.clim(cmin, cmax)
        
        
## class to work with a collection of tracks        
class TrackCollection():
    def __init__(self, df, stations=None):
        self._tracks = []
        
        for aircraft in tqdm(df.aircraft.unique()):
            tr = Track(df[df.aircraft==aircraft], stations)
            ## correct time if Stations object provided 
            if stations is not None:
                tr.correct_time(stations)
                
            self._tracks.append(tr)
    
    
    def __repr__(self):
        return f'Collection of {len(self._tracks)} tracks'
    
    
    def __iter__(self):
        for tr in self._tracks:
            yield tr
            
            
    def __len__(self):
        return len(self._tracks)
            
            
    def __getitem__(self, i):
        return self._tracks[i]
    
    
    # aggregate telemetry and/or time across tracks
    def aggregate(self, comb, return_telemetry=False, return_time=False):
        track_stat = {}

        if return_telemetry:
            __lat, __lon, __hgt = [], [], []
            
        if return_time:
            __t1 = np.array([])
        
        for i, tr in enumerate(self):
            if tr.stations_stat(comb) > 0:
                track_stat[i] = tr.stations_stat(comb)

                if return_telemetry:
                    tel = tr.telemetry(comb)
                    __lat += tel.latitude.values.tolist()
                    __lon += tel.longitude.values.tolist()
                    __hgt += tel.geoAltitude.values.tolist()
                    
                if return_time:
                    t = tr.time(comb).values
                    if __t1.shape[0] == 0:
                        __t1 = t
                    else:
                        __t1 = np.r_[__t1, t]
                
        if return_telemetry or return_time:
            track_telemetry = pd.DataFrame()
            if return_telemetry:
                track_telemetry = track_telemetry.assign(latitude = __lat,
                                                         longitude = __lon,
                                                         geoAltitude = __hgt)
            if return_time:
                for i, s in enumerate(comb):
                    track_telemetry[str(s)] = __t1[:, i]
                    
            return (track_stat, track_telemetry)
        else:
            return track_stat
        
        
    def get_pairs(self, combs):
        pairs = defaultdict(int)

        for tr in self:
            for w in combs:
                if tr.stations_stat(w) > 0:
                    pairs[w] += tr.stations_stat(w)

        return pairs
    
    
    def get_aircraft_times(self, stations, comb, A0_B=None, verbose=False, ylim=[-1e3, 1e3]):
        _, tel = self.aggregate(comb, return_telemetry=True, return_time=True)

        st1, st2 = comb
        st1_str, st2_str = str(comb[0]), str(comb[1])

        output = np.zeros([tel.shape[0], 2])

        p3 = [P3(lat, lon, hgt) for lat, lon, hgt in zip(tel['latitude'], tel['longitude'], tel['geoAltitude'])]

        t1 = stations.correct_time(st1, tel[st1_str].values)
        t2 = stations.correct_time(st2, tel[st2_str].values)

        output[:, 0] = np.array([t - stations.time_to(st1, p) for t, p in zip(t1, p3)])
        output[:, 1] = np.array([t - stations.time_to(st2, p) for t, p in zip(t2, p3)])

        if verbose:
            print('Approx Median error [m]:', 3e8*np.median(np.fabs(output[:, 0] - output[:, 1])))

            plt.plot(output[:, 0], 1e9*(output[:, 0] - output[:, 1]), '.')
            plt.title('Delta time between stations', fontsize=16)
            plt.xlabel('Time, s', fontsize=14)
            plt.ylabel('Delta time, ns', fontsize=14)
            plt.ylim(ylim)
            plt.grid()
        
        return output
    
    