import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from src.geo import P3, dist3d, eff_velocity
from collections.abc import Iterable
from scipy.interpolate import splev


## class to work with stations and appling time correction
class Stations:
    
    def __init__(self, stations_list, sensors_path='data/round2_sensors.csv'):
        ## Input checks
        if type(stations_list) is not list:
            raise Exception('Stations must be a list')
            
        if len(stations_list) == 0:
            raise Exception('Stations list is empty')
            
        self._inventory = sorted(stations_list)
        
        self._sensors_data = pd.read_csv(sensors_path)
        serials = self._sensors_data.serial.values
        
        ## Variables
        self.A0 = 3e-4
        self.B = 1.1e-4
        
        self.st_dict = {}  # map station to index
        self.st_rev = {}  # map index to station
        
        self.st_loc = {}  # station location
        self.st_p3 = {}  # stations location in cartesian coordinates
        
        self.st_params = None
        
        for i, s in enumerate(self._inventory):
            if s not in serials:
                print(f'Warning: station {s} is not available')
                continue
                
            self.st_dict[s] = i
            self.st_rev[i] = s

            station_geo = self._sensors_data.loc[self._sensors_data.serial==s, ['latitude', 'longitude', 'height']].values[0]
            self.st_loc[s] = station_geo
            self.st_p3[s] = P3(*station_geo)
            
        self.close_stations = []

        for i, s1 in enumerate(self._inventory):
            for s2 in self._inventory[i+1:]:
                if self.distance_to(s1, self.locs(s2)) <= 15000:
                    self.close_stations.append((s1, s2))
            
    
    def undup_stations(self, a):
        if len(a) < 2:
            return len(a)

        for i in range(len(a)-1):
            for j in range(i+1, len(a)):
                if (a[i], a[j]) in self.close_stations:
                    return max(self.undup_stations(a[:i] + a[i+1:]), self.undup_stations(a[:j] + a[j+1:]))   
        return len(a)
    
    
    def add_station(self, station):
        if station in self._inventory:
            print(f'Warning: station {station} is already in inventory, nothing changed')
        else:
            self._inventory = sorted(self._inventory + [station])

            station_geo = self._sensors_data.loc[self._sensors_data.serial==station,
                                                 ['latitude', 'longitude', 'height']].values[0]
            self.st_loc[station] = station_geo
            self.st_p3[station] = P3(*station_geo)

            self.st_dict = {}
            self.st_rev = {}
            for i, s in enumerate(self._inventory):
                self.st_dict[s] = i
                self.st_rev[i] = s
            
    
    def __repr__(self):
        return 'Stations: ' + ', '.join(str(x) for x in self._inventory)
        
        
    @property
    def inventory(self):
        return self._inventory
    
    
    ## Decorators
    def __check_station(func):  #  check that station is in inventory
        def inner(self, *args):
            if args[0] in self.st_dict:
                return func(self, *args)
            else:
                raise Exception(f'Station {args[0]} is not in inventory')
        return inner
            
        
    def __check_index(func):  # check that index exists
        def inner(self, *args):
            if args[0] in self.st_rev:
                return func(self, *args)
            else:
                raise Exception(f'Index {args[0]} is out of bounds')
        return inner
    
            
    @__check_station
    def i(self, station):  # index for station
        return self.st_dict[station]
        
        
    @__check_index
    def s(self, ind):  # station for given index
        return self.st_rev[ind]
        
        
    @__check_station
    def locs(self, station):  # station location
        return self.st_loc[station]
        
        
    @__check_station
    def carts(self, station):  # cartesian coordinates for a station
        return list(self.st_p3[station].to_cartesian())
    
    
    @__check_station
    def p3s(self, station):  # station location in cartesian coordinates
        return self.st_p3[station]
        
    
    @__check_station
    def distance_to(self, station, location):  # distance to a station
        if not isinstance(location, P3):
            location = P3(*location)
            
        return dist3d(self.p3s(station), location)
    
    
    @__check_station
    def time_to(self, station, location, ns=False):  # time-of-flight to a station
        if not isinstance(location, P3):
            height_1 = location[2]
        else:
            height_1 = location.height
        
        time = self.distance_to(station, location) / eff_velocity(height_1, self.st_loc[station][2], self.A0, self.B)
            
        if ns:
            time *= 1e9
            
        return time
    
        
    @__check_station
    def update_location(self, station, location):
        if not isinstance(location, P3):
            self.st_loc[station] = location
            self.st_p3[station] = P3(*location)
        else:
            self.st_loc[station] = [location.latitude, location.longitude, location.height]
            self.st_p3[station] = location
    
    
    @__check_station
    def correct_time(self, station, time):  # correct time
        if type(time) != np.ndarray:
            time = np.array(time)
            
        if self.st_params is not None:
            s = str(station)
            if s in self.st_params:
                # apply shift for good stations
                if 'shift' in self.st_params[s]:
                    return time + self.st_params[s]['shift']
                else:  # apply eq.3 from Theory for all the other stations
                    a = self.st_params[s]['lr'][0]
                    b = self.st_params[s]['lr'][1]
                    spl = (np.array(self.st_params[s]['spl_knots']), np.array(self.st_params[s]['spl_coefs']), 3)

                    output = (time - b - splev((time - b)/(1 + a), spl)) / (1 + a)
                    
                    # if there gaps -> fill with nan
                    if 'gaps' in self.st_params[s]:
                        ind_nan = np.concatenate([np.where((output>=lb)&(output<=rb))[0] for lb, rb in self.st_params[s]['gaps']])
                        output[ind_nan] = np.nan
                        
                    return output
            else:
                return time
            
        raise Exception('No st_params!')
        
    
    ## Plot all stations
    def plot(self, ax=None):
        if not ax:
            ax = plt.gca()
            plt.grid()
            plt.xlabel('Longitude', fontsize=14)
            plt.ylabel('Latitude', fontsize=14)
            
        ax.scatter([x[1] for x in self.st_loc.values()],
                   [x[0] for x in self.st_loc.values()],
                    c='r', marker='x')
        for s in self.inventory:
            plt.text(self.st_loc[s][1], self.st_loc[s][0], str(s))
