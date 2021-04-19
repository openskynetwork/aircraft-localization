#############################################
#
# geo.py
#
#############################################
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')


## Class to work with geolocations and transformation to cartesian coordinates
class P3:
    def __init__(self, latitude, longitude, height):
        self.latitude = latitude
        self.longitude = longitude
        self.height = height
        
    def __repr__(self):
        return f'P3 class of coordinates: ({self.latitude}, {self.longitude}, {self.height})'
        
    # convert decimal degrees to radians
    def rad(self, degrees):
        return degrees * np.pi / 180.

    # convert radians to decimal degrees
    def deg(self, radians):
        return 180 * radians / np.pi

    # convert WSG84 coordinates to cartesian ones
    def to_cartesian(self):
        lat = self.rad(self.latitude)
        lon = self.rad(self.longitude)

        # WSG84 ellipsoid constants
        a = 6378137
        e2 = 6.69437999014e-3 #8.1819190842622e-2

        # prime vertical radius of curvature
        N = a / np.sqrt(1 - e2 * pow(np.sin(lat),2))

        return( (N + self.height) * np.cos(lat) * np.cos(lon),
                (N + self.height) * np.cos(lat) * np.sin(lon),
                ((1 - e2) * N + self.height) * np.sin(lat) )

    
## 3D distance between WSG84 coordinates
def dist3d(p1, p2):
    x1, y1, z1 = p1.to_cartesian()
    x2, y2, z2 = p2.to_cartesian()
        
    return np.sqrt(pow(x1-x2,2) + pow(y1-y2,2) + pow(z1-z2,2))


## Effective velocity (see Theory in README)
def eff_velocity(height_1, height_2, A0=3e-4, B=1.1e-4):
    light_speed = 299792458.  # [m/s]
    h_min, h_max = min(height_1, height_2), max(height_1, height_2)
    if h_max == h_min:
        return light_speed / (1 + A0)
    else:
        return light_speed / (1 + A0 * (np.exp(-B*h_min) - np.exp(-B*h_max)) / B / (h_max - h_min))


## Copy of code from @richardalligier round_1 of the competition
def haversine_distance(lat1, lon1, lat2, lon2):
    r = 6371000
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) *   np.sin(delta_lambda / 2)**2
    res = r * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
    return res


## Copy of code from @richardalligier round_1 of the competition
def rad(degrees):
    return degrees*np.pi/180


## Copy of code from @richardalligier round_1 of the competition
def numpylla2ecef(lat,lon,h):
    lat = rad(lat)
    lon = rad(lon)
    
    # WSG84 ellipsoid constants
    a = 6378137
    e = 8.1819190842622e-2
    # prime vertical radius of curvature
    N = a / np.sqrt(1 - e**2 * np.sin(lat)**2)
    x = (N+h) * np.cos(lat) * np.cos(lon)
    y = (N+h) * np.cos(lat) * np.sin(lon)
    z = ((1-e**2) * N +h) * np.sin(lat)
    return (x, y, z)


## Plot aplitude profiles of refractive index and effective velocity
## to check optimization results
def plot_altitude_profiles(A0=3e-4, B=1.1e-4):
    h = np.arange(100, 12100, 100)  # altitudes

    f, ax1 = plt.subplots()
    
    # plot refractive index
    y = 1 + A0 * np.exp(-B*h)
    l1 = ax1.plot(h, y, 'b.')
    
    # plot effective refractive index
    y1 = 1 + A0 * (1 - np.exp(-B*h)) / B / h
    l2 = ax1.plot(h, y1, 'g.')
    
    ax1.set_xlabel('Altitude, m')
    ax1.set_ylabel('Refractive index, N')
    ax1.grid()
    ax1.set_title('Altitude profiles')
    
    ax2 = ax1.twinx()
    
    # plot average (effective) transition velocity
    z = 1 / (1 + A0 * (1 - np.exp(-B*h)) / B / h)
    l3 = ax2.plot(h, z, 'r.')
    ax2.set_ylabel('Effective transition velocity, V/c')
    
    ax1.legend(['Refractive index', 'Effective refractive index'], loc='upper right')
    ax2.legend(['Effective transition velocity'], loc='lower right')
    