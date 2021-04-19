from IPython.core.pylabtools import figsize
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from optimize import *
from scipy.optimize import fmin_l_bfgs_b

from src.geo import P3, dist3d, eff_velocity
from src.stations import Stations
from src.filters import median_filter
from src.track import Track, TrackCollection

from tqdm import tqdm

import itertools
from collections import defaultdict

from scipy.interpolate import splrep, splev
from sklearn.linear_model import RANSACRegressor

import copy
import json
import os


### Solver class to carry out optimization of good stations
class GoodStationsSolver():
    def __init__(self):
        self.result = np.array([])
        
        self.st_list = {}
        self.mp = {}
        self.st_N = {}
        self.points = {}
    
    
    def __repr__(self):
        return str(self.result)

    
    ## Prepare a subset of data for optimization
    def prepare_data(self, tracks, stations, threshold=1500*1e-9, N=10000):
        self.__prepare_telemetry_for_combinations(tracks, stations, [x for x in itertools.combinations(stations.inventory, r=2)])
        self.__filter_by_delta_time(threshold)
        self.__calculate_station_statistics()
        self.__prepare_points(stations, N)
        
        
    ## Optimize:
    # 1) A0, B, shifts and stations locations (optimize_location_flag = True)
    # 2) or stations shifts only (optimize_location_flag = False)
    def optimize(self, stations, A0, B, shifts, optimize_locations_flag=False):
        # Save setup
        self.optimize_locations = optimize_locations_flag
        
        if not optimize_locations_flag:  ## time shifts only
            # stations locations in cartesian coordinates + heights
            st_loc = np.array([])
            st_hgt = np.array([])
            for s in stations.inventory:
                st_loc = np.r_[st_loc, np.array(stations.carts(s))]
                st_hgt = np.r_[st_hgt, np.array(stations.locs(s)[2])]
                
            # Save setup
            self.st_loc = st_loc
            self.st_hgt = st_hgt

            # Prepare x0
            x0 = np.array(shifts)

            self.result = fmin_l_bfgs_b(optimize_shifts, x0,
                                        args=(self.points['cartesian'],
                                              self.points['geoalt'],
                                              st_loc, st_hgt,
                                              np.array(self.points['st1'], dtype=np.int32),
                                              np.array(self.points['st2'], dtype=np.int32),
                                              self.points['t1'],
                                              self.points['t2'],
                                              A0, B),
                                        approx_grad=1,
                                        factr=10,
                                        maxfun=1000000)
            print(self.result[:-1])
            
        else:
            ## A0, B, time shifts and stations' locations
            st_loc = []
            for s in stations.inventory:
                st_loc += list(stations.locs(s))

            M = len(stations.inventory)
            x0 = np.zeros(4*M+2)
            x0[0] = A0
            x0[1] = B
            x0[2:M+2] = shifts
            x0[M+2:] = st_loc

            self.result = fmin_l_bfgs_b(optimize_locations, x0,
                                        args=(self.points['cartesian'],
                                              self.points['geoalt'],
                                              np.array(self.points['st1'], dtype=np.int32),
                                              np.array(self.points['st2'], dtype=np.int32),
                                              self.points['t1'],
                                              self.points['t2']),
                                        approx_grad=1,
                                        factr=10,
                                        maxfun=1000000)
            print(self.result[:-1])

        
    def plot_errors_distribution(self, hist_lim=200, hist_bins=100):
        
        if not self.optimize_locations:
            output = optimize_AB_shifts(self.result[0],
                                        self.points['cartesian'],
                                        self.points['geoalt'],
                                        self.st_loc, self.st_hgt,
                                        np.array(self.points['st1'], dtype=np.int32),
                                        np.array(self.points['st2'], dtype=np.int32),
                                        self.points['t1'],
                                        self.points['t2'],
                                        return_errors=1)
        else:
            output = optimize_locations(self.result[0],
                                        self.points['cartesian'],
                                        self.points['geoalt'],
                                        np.array(points['st1'], dtype=np.int32),
                                        np.array(points['st2'], dtype=np.int32),
                                        self.points['t1'],
                                        self.points['t2'],
                                        return_errors=1)


        _ = plt.hist([y for y in output if y < hist_lim], bins=hist_bins)
                           
    
    ## Prepare telemetry for given pairs of stations (combinations)
    def __prepare_telemetry_for_combinations(self, tracks, stations, combinations):
        st_list = {}  # station: combinations
        mp = {}  # telemetry for combinations

        for tr in tqdm(tracks):
            for comb in set(combinations).intersection(tr.stations_comb()):
                if tr.stations_stat(comb) < 10:
                    continue
                    
                if comb in stations.close_stations:
                    continue

                # add stations
                if comb[0] in st_list:
                    if comb not in st_list[comb[0]]:
                        st_list[comb[0]].append(comb)
                else:
                    st_list[comb[0]] = [comb]

                if comb[1] in st_list:
                    if comb not in st_list[comb[1]]:
                        st_list[comb[1]].append(comb)
                else:
                    st_list[comb[1]] = [comb]
                
                cartesian = []

                # prepare telemetry
                tel = tr.telemetry(comb)

                generator = (P3(lat, lon, hgt).to_cartesian() for lat, lon, hgt in zip(tel.latitude, tel.longitude, tel.geoAltitude))
                for val in generator:
                    cartesian += list(val)

                geoalt = tel.geoAltitude.values.tolist()

                t = tr.time(comb)
                
                st1_str, st2_str = str(comb[0]), str(comb[1])

                t1 = t[st1_str].values.tolist()
                t2 = t[st2_str].values.tolist()

                tel = pd.concat([tel, t], axis=1)
                tel = tel.assign(time1 = lambda x: [t - stations.time_to(comb[0], [lat, lon, hgt]) for t, lat, lon, hgt in zip(x[st1_str], x['latitude'], x['longitude'], x['geoAltitude'])],
                                 time2 = lambda x: [t - stations.time_to(comb[1], [lat, lon, hgt]) for t, lat, lon, hgt in zip(x[st2_str], x['latitude'], x['longitude'], x['geoAltitude'])])

                dt = (tel.time2 - tel.time1).values.tolist()

                if comb in mp:
                    mp[comb]['cartesian'] += cartesian
                    mp[comb]['geoalt'] += geoalt
                    mp[comb]['t1'] += t1
                    mp[comb]['t2'] += t2
                    mp[comb]['dt'] += dt
                else:
                    mp[comb] = {'cartesian':cartesian,
                                'geoalt':geoalt,
                                't1':t1,
                                't2':t2,
                                'dt':dt}
                    
        # Save results
        self.st_list = st_list
        self.mp = mp
                    
            
    ## Filter telemetry by applying a threshold around median value
    def __filter_by_delta_time(self, threshold=1500):
        # Filtering for good stations
        for key in tqdm(self.mp):
            med = np.median(self.mp[key]['dt'])
            ind = np.where((self.mp[key]['dt'] > med - threshold) & (self.mp[key]['dt'] < med + threshold))[0]
            self.mp[key]['N'] = len(ind)

            for col in ['geoalt', 't1', 't2', 'dt']:
                self.mp[key][col] = [self.mp[key][col][i] for i in ind]

            ind_cart = []
            for i in ind:
                ind_cart += [3*i, 3*i+1, 3*i+2]

            self.mp[key]['cartesian'] = [self.mp[key]['cartesian'][i] for i in ind_cart]

    
    ## Prepare statistics on filtered telemetry
    def __calculate_station_statistics(self):
        # Calculate statistics 
        st_N = defaultdict(int)

        for s in self.st_list:
            for comb in self.st_list[s]:
                st_N[s] += len(self.mp[comb]['t1'])
                
        self.st_N = st_N

        
    ## Prepare a subset of telemetry points of a given size 
    def __prepare_points(self, stations, N=10000):
        # Prepare output
        cartesian = []
        geoalt = []
        t1, t2, st1, st2 = [], [], [], []

        for s in tqdm(self.st_N):
            # prepare all data for a station
            __cartesian, __geoalt = [], []
            __t1, __t2, __st1, __st2 = [], [], [], []

            for comb in self.st_list[s]:
                __cartesian += self.mp[comb]['cartesian']
                __geoalt += self.mp[comb]['geoalt']
                __t1 += self.mp[comb]['t1']
                __t2 += self.mp[comb]['t2']
                __st1 += [float(stations.i(comb[0]))] * len(self.mp[comb]['t1'])
                __st2 += [float(stations.i(comb[1]))] * len(self.mp[comb]['t1'])

            if self.st_N[s] > N:
                ind = np.random.permutation(np.arange(0, self.st_N[s], 1))[:N]
            else:
                ind = np.arange(0, self.st_N[s], 1)

            generator = ((__cartesian[3*i], __cartesian[3*i+1], __cartesian[3*i+2]) for i in ind)
            for val in generator:
                cartesian += list(val)

            geoalt += [__geoalt[i] for i in ind]
            t1 += [__t1[i] for i in ind]
            t2 += [__t2[i] for i in ind]
            st1 += [__st1[i] for i in ind]
            st2 += [__st2[i] for i in ind]

        self.points = {'cartesian':np.array(cartesian),
                       'geoalt':np.array(geoalt),
                       't1':np.array(t1),
                       't2':np.array(t2),
                       'st1':np.array(st1),
                       'st2':np.array(st2)}
    


### Solver class to carry out optimization of not so good stations
class SingleStationSolver():
    
    def __init__(self):
        pass
    
    
    ## Find new stations and their pairs with already synchronized ones
    def find_new_stations(self, tracks, stations, bad_stations=[], N=1000):
        new_stations = {}

        for tr in tqdm(tracks):
            for comb in tr.stations_comb():
                if len(comb) != 2:
                    continue

                if tr.stations_stat(comb) < N:
                    continue

                if (comb[0] in stations.inventory) and (comb[1] not in stations.inventory):
                    st1, st2 = comb
                elif (comb[0] not in stations.inventory) and (comb[1] in stations.inventory):
                    st2, st1 = comb
                else:
                    st1, st2 = None, None
                    
                if (st2 is not None) and (st2 not in bad_stations):
                    if (st2 in new_stations) and (comb not in new_stations[st2]):
                        new_stations[st2].append(comb)

                    if st2 not in new_stations:
                        new_stations[st2] = [comb]
        
        self.new_stations = new_stations
        return new_stations
        
        
    ## Prepare a subset of telemetry points for given pairs of stations of a given size
    def prepare_data(self, tracks, stations, single_station, N=10000):
        __cartesian = []
        __geoalt = []
        __t1, __t2, __st2 = [], [], []
        
        N_points = 0
        
        self.stations = stations
        self.single_station = single_station
        self.N = N
        
        for tr in tracks:
            for comb in self.new_stations[single_station]:
                if tr.stations_stat(comb) < 10:
                    continue
                
                t = tr.time(comb)
                if comb[0] == single_station:
                    st1, st2 = comb
                else:
                    st2, st1 = comb
                
                time2 = stations.correct_time(st2, t[str(st2)].values)
                ind_nan = np.where(np.isnan(time2))[0]
                
                if len(ind_nan) > 0:
                    tel = tr.telemetry(comb).drop(ind_nan, inplace=False)
                    t = tr.time(comb).drop(ind_nan, inplace=False)
                else:
                    tel = tr.telemetry(comb)
                    t = tr.time(comb)
                
                generator = (P3(lat, lon, hgt).to_cartesian() for lat, lon, hgt in zip(tel.latitude, tel.longitude, tel.geoAltitude))
                for val in generator:
                    __cartesian += list(val)

                __geoalt += tel.geoAltitude.values.tolist()

                if comb[0] == single_station:
                    __st2 += [stations.i(comb[1])] * t.shape[0]
                else:
                    __st2 += [stations.i(comb[0])] * t.shape[0]
                    
                __t1 += t[str(st1)].values.tolist()
                __t2 += time2[np.where(~np.isnan(time2))[0]].tolist()
            
                N_points += t.shape[0]
                
        self.all_points = {'cartesian':np.array(__cartesian),
                           'geoalt':np.array(__geoalt),
                           't1':np.array(__t1),
                           't2':np.array(__t2),
                           'st2':np.array(__st2)}
        
        if N_points > N:
            ind = np.random.permutation(np.arange(0, N_points, 1))[:N]
        else:
            ind = np.arange(0, N_points, 1)

        cartesian = []
        generator = ((__cartesian[3*i], __cartesian[3*i+1], __cartesian[3*i+2]) for i in ind)
        for val in generator:
            cartesian += list(val)

        points = {'cartesian':np.array(cartesian),
                  'geoalt':np.array(__geoalt)[ind],
                  't1':np.array(__t1)[ind],
                  't2':np.array(__t2)[ind],
                  'st2':np.array(__st2)[ind]}
        
        self.points = points
        
        
    ## Return aircraft time given station measured values
    def prepare_aircraft_time(self, all_points=False):
        if all_points is True:
            arr = self.all_points
        else:
            arr = self.points
            
        N = arr['t2'].shape[0]
        A0 = self.stations.A0
        B = self.stations.B
        
        t_0 = np.zeros(N)

        # Prepare stations' locations
        st_loc = []
        for s in self.stations.inventory:
            st_loc += list(self.stations.carts(s))

        st_hgt = []
        for s in self.stations.inventory:
            st_hgt += [self.stations.locs(s)[2]]

        # Calculate aircraft times for good stations
        for i in range(N):
            j = int(arr['st2'][i])
            d2 = np.sqrt(pow(arr['cartesian'][3*i] - st_loc[3*j], 2) + \
                         pow(arr['cartesian'][3*i+1] - st_loc[3*j+1], 2) + \
                         pow(arr['cartesian'][3*i+2] - st_loc[3*j+2], 2))
            t_0[i] = arr['t2'][i] - d2 / eff_velocity(st_hgt[j], arr['geoalt'][i], A0, B)
    
        return t_0
    
    
    ## Return station time given estimated aircraft time values
    def prepare_station_time(self, t2_0, all_points=False):
        if all_points is True:
            arr = self.all_points
        else:
            arr = self.points
            
        N = arr['t1'].shape[0]
        A0 = self.stations.A0
        B = self.stations.B
        
        cart = self.stations.carts(self.single_station)
        hgt = self.stations.locs(self.single_station)[2]
        t1_1 = np.zeros(N)
        
        for i in range(N):
            d1 = np.sqrt(pow(arr['cartesian'][3*i] - cart[0], 2) + \
                         pow(arr['cartesian'][3*i+1] - cart[1], 2) + \
                         pow(arr['cartesian'][3*i+2] - cart[2], 2))
            t1_1[i] = t2_0[i] + d1 / eff_velocity(hgt, arr['geoalt'][i], A0, B)
            
        return t1_1

    
    ## Optimization method:
    # - optimize station location and linear clock drift using a subset of points
    # - select the best spline approximation of clock random walk on all available points
    def optimize(self):
        # aircraft time according to eq.2 from Theory
        t2_0 = np.array(self.prepare_aircraft_time(all_points=False))
        # station time according to eq.2 from Theory
        t1_1 = np.array(self.prepare_station_time(t2_0, all_points=False))
        
        # eliminate linear drift using eq.2 from Theory
        lr = RANSACRegressor().fit(t1_1.reshape(-1, 1), self.points['t1'] - t1_1)
        # use predicted values below
        shift = lr.predict(t1_1.reshape(-1, 1))
        
        N = t1_1.shape[0]
        A0 = self.stations.A0
        B = self.stations.B
        
        # Define a parameter for median filter
        med_par = int(N * 30. / 3600.)
        med_par = med_par if (med_par % 2 != 0) else (med_par + 1)
        
        # optimization function
        def func(x, cart_points, heights, t1, t2_0, shift, A0, B, med_par, return_arrays=False):
            N = t2_0.shape[0]
            t1_1 = np.zeros(N)
            cart = P3(*x).to_cartesian()
            for i in range(N):
                # distance from aircraft location to station
                d1 = np.sqrt(pow(cart_points[3*i] - cart[0], 2) + \
                             pow(cart_points[3*i+1] - cart[1], 2) + \
                             pow(cart_points[3*i+2] - cart[2], 2))
                # aircraft time
                t1_1[i] = t2_0[i] + d1 / eff_velocity(x[2], heights[i], A0, B)

            ind = np.argsort(t1_1)
            # apply median filter to random walk values
            med = median_filter(t1[ind] - t1_1[ind] - shift[ind], med_par)
            
            if return_arrays:
                return t1_1, med, ind
            else:
                # minimize residual error using linear drift predicted values and median values for random wallk
                return 1e9*np.sum(np.fabs(t1[ind] - t1_1[ind] - shift[ind] - med)) / N
        
        # optimization
        self.result = fmin_l_bfgs_b(func,
                                    self.stations.locs(self.single_station),
                                    args=(self.points['cartesian'],
                                          self.points['geoalt'],
                                          self.points['t1'],
                                          t2_0,
                                          shift,
                                          A0, B, med_par),
                                    maxfun=300,
                                    approx_grad=1)
        
        ## If optimization error too big -> bad station
        if self.result[1] > 1e7:
            self.delta_dist = np.nan
            self.med_error = np.nan
            self.spl = (np.array([]), np.array([]), 3)
            self.spl_s_param = np.nan
            self.max_dt_gap = np.nan
            self.lr = lr
            return
        
        # if new location is too far from the initial value
        self.delta_dist = dist3d(P3(*self.result[0]), self.stations.p3s(self.single_station))
        if self.delta_dist > 100:  #[m]
            # rerun optimization one more time
            t1_1, _, _ = func(self.result[0],
                              self.points['cartesian'],
                              self.points['geoalt'],
                              self.points['t1'],
                              t2_0, shift,
                              A0, B, med_par, return_arrays=True)
            
            lr = RANSACRegressor().fit(t1_1.reshape(-1, 1), self.points['t1'] - t1_1)
            shift = lr.predict(t1_1.reshape(-1, 1))
            
            self.result = fmin_l_bfgs_b(func,
                                        self.result[0],
                                        args=(self.points['cartesian'],
                                              self.points['geoalt'],
                                              self.points['t1'],
                                              t2_0,
                                              shift,
                                              A0, B, med_par),
                                        maxfun=300,
                                        approx_grad=1)
        
        # get station time values etc
        self.t1_1, self.med, self.ind = func(self.result[0],
                                              self.points['cartesian'],
                                              self.points['geoalt'],
                                              self.points['t1'],
                                              t2_0, shift,
                                              A0, B, med_par, return_arrays=True)
        
        self.t1_1 = self.t1_1[self.ind]
        # deduplicate values
        self.t1_1, ind_dup = np.unique(self.t1_1, return_index=True)
        self.ind = self.ind[ind_dup]
        self.med = self.med[ind_dup]
        
        # max time gap
        self.max_dt_gap = np.max(np.diff(self.t1_1))
        # save linear regression
        self.lr = lr
        # delta location
        self.delta_dist = dist3d(P3(*self.result[0]), self.stations.p3s(self.single_station))
        
        # update station's location
        self.stations.update_location(self.single_station, self.result[0])
        
        # fit the best spline using all_points available
        t2_0 = np.array(self.prepare_aircraft_time(all_points=True))
        t1_1 = np.array(self.prepare_station_time(t2_0, all_points=True))
        
        min_error = np.inf
        
        for v in np.arange(2, 21, 0.5):
            spl = splrep(self.t1_1, self.med, s=v*1e-12, k=3)
            if any(np.isnan(spl[1])):
                continue
            else:
                # estimate median error in meters
                err = 3e8*np.median(np.fabs(self.all_points['t1'] - t1_1 - lr.predict(t1_1.reshape(-1, 1)) - splev(t1_1, spl)))
                if err < min_error:
                    min_error = err
                    self.spl = spl
                    self.med_error = err
                    self.spl_s_param = v*1e-12
                    
        if min_error == np.inf:
            self.spl = (np.array([]), np.array([]), 3)
            self.med_error = np.nan
            self.spl_s_param = np.nan
    
        
        
    def check_result(self, ylim1=[-1e4, 1e4], ylim2=[-1e3, 1e3], ylim3=[-1e3, 1e3]):
        print(f'Median error: {self.med_error:.4}m using s={self.spl_s_param:.3}')
        print(f'Max time gap: {self.max_dt_gap:.4}s')
        print(f'Delta distance: {self.delta_dist:.4}m')
        
        figsize(15, 10)
        f = plt.figure()
        
        ax1 = f.add_subplot(311)
        ax1.plot(self.t1_1, 1e9*(self.points['t1'][self.ind] - self.t1_1 - self.lr.predict(self.t1_1.reshape(-1, 1))), '.')
        ax1.set_ylim(ylim1)
        ax1.grid()

        ax2 = f.add_subplot(312)
        ax2.plot(self.t1_1, 1e9*(self.points['t1'][self.ind] - self.t1_1 - self.lr.predict(self.t1_1.reshape(-1, 1)) - splev(self.t1_1, self.spl)), '.')
        ax2.set_ylim(ylim2)
        ax2.grid()
        
        ax3 = f.add_subplot(313)
        ax3.plot(self.t1_1, 1e9*self.med, '.')
        ax3.plot(self.t1_1, 1e9*splev(self.t1_1, self.spl), 'r-')
        ax3.set_ylim(ylim3)
        ax3.grid()
        
        
    def save(self, filename = 'stations_params.json'):
        # if there is no file -> create a new one
        if filename not in os.listdir():
            with open(filename, 'w') as f:
                json.dump({}, f)

        # Load stations parameters
        with open(filename, 'r') as f:  
            st_params = json.load(f)

        
        # prepare parameters for a new stations
        st_params[self.single_station] = {'location':self.result[0].tolist(),
                                          'lr':[self.lr.estimator_.coef_[0], self.lr.estimator_.intercept_],
                                          'spl_knots':self.spl[0].tolist(),
                                          'spl_coefs':self.spl[1].tolist(),
                                          'med_error':self.med_error,
                                          'max_dt_gap':self.max_dt_gap,
                                          'delta_dist':self.delta_dist}
        
        
        ## case of time gaps
        if self.max_dt_gap > 25:
            ind = (np.concatenate([np.array([i, i+1]) for i in np.where(np.diff(self.t1_1) > 25)[0]])).tolist()
            
            # indent to add from both sides of a time gap
            indent = 1  # [s]
            gaps = []
            for i in range(0, len(ind), 2):
                gaps.append([self.t1_1[ind[i]]-indent, self.t1_1[ind[i+1]]+indent])

            if self.t1_1[0] > 25:
                gaps = [[0, self.t1_1[0]+indent]] + gaps

            if self.t1_1[-1] < 3600 - 25:
                gaps = gaps + [[self.t1_1[-1]-indent, 3600]]
                
            ## join small gaps together
            new_gaps = []
            curr_gap = copy.copy(gaps[0])
            N, i = len(gaps), 1
            while i < N:
                # in case of overlaping gaps
                if gaps[i][0] - curr_gap[1] < 50:
                    curr_gap[1] = gaps[i][1]
                else:
                    new_gaps += [curr_gap]
                    curr_gap = copy.copy(gaps[i])

                i += 1
    
            new_gaps += [curr_gap]
            
            # save bounds
            st_params[self.single_station]['gaps'] = new_gaps
    

        # Write to json file
        with open(filename, 'w') as f:
            json.dump(st_params, f)
            