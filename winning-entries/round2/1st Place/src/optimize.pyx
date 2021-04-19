### Cython code

# load libraries
cimport cython

import numpy as np
cimport numpy as np

from libc.math cimport fabs, exp, sqrt, cos, sin, pi, pow

from cython import boundscheck, wraparound
from cython.parallel import prange


## Number of threads
cdef:
    int N_THREADS = 4


# WSG84 ellipsoid constants
cdef:
    double wgs_a = 6378137  # [m]
    double wgs_e2 = 0.00669437999014

cdef:
    double light_speed = 299792458.


# --------------------------------------------------------------------
# Cython function to convert WGS84 coordinates to cartesian
# --------------------------------------------------------------------

cdef wgs2cart_c(double latitude, double longitude, double height):
    cdef:
        double lat = pi * latitude / 180.
        double lon = pi * longitude / 180.
        
        # prime vertical radius of curvature
        double N = wgs_a / sqrt(1 - wgs_e2 * pow(sin(lat), 2))
        double cart[3]

    cart[0] = (N + height) * cos(lat) * cos(lon)
    cart[1] = (N + height) * cos(lat) * sin(lon)
    cart[2] = ((1 - wgs_e2) * N + height) * sin(lat)
    
    return cart
    

# --------------------------------------------------------------------
# Cython function to calculate distance in 3D space
# --------------------------------------------------------------------

cdef dist3d_c(double latitude1, double longitude1, double height1,
              double latitude2, double longitude2, double height2):
    
    cdef double cart1[3], cart2[3]
    
    cart1 = wgs2cart_c(latitude1, longitude1, height1)
    cart2 = wgs2cart_c(latitude2, longitude2, height2)
    
    return sqrt(pow(cart1[0] - cart2[0], 2) + pow(cart1[1] - cart2[1], 2) + pow(cart1[2] - cart2[2], 2))


# --------------------------------------------------------------------
# Cython function to calculate effective wave velocity
# --------------------------------------------------------------------
cdef inline double eff_velocity(double h_min, double h_max, double A0, double B) nogil:
    if h_min < h_max:
        return light_speed / (1 + A0 * (exp(-B*h_min) - exp(-B*h_max)) / B / (h_max - h_min))
    elif h_min > h_max:
        return light_speed / (1 + A0 * (exp(-B*h_max) - exp(-B*h_min)) / B / (h_min - h_max))
    else:
        return light_speed / (1 + A0 / B)
    

# --------------------------------------------------------------------
# Cython function to optimize stations shifts only
# --------------------------------------------------------------------
@boundscheck(False)
@wraparound(False)
def optimize_shifts(double[:] x,  # shifts to be optimized
                    double[:] points,  # aicraft locations in cartesian coordinates
                    double[:] heights, # aircraft geoAltitude values
                    double[:] st_loc,  # stations locations in cartesian coordinates
                    double[:] st_hgt,  # stations height
                    int[:] st_1,  # station 1 indexes
                    int[:] st_2,  # station 2 indexes
                    double[:] t_1,  # station 1 measured times
                    double[:] t_2,  # station 2 measured times
                    double A0,  # A0 parameter
                    double B,  # B parameter
                    int loss_l1=1,  # L1 or L2 loss
                    int return_errors=0):
    
    cdef:
        int N = t_1.shape[0]
        int i, j1, j2
        double d1, d2, dt
        
        double[:] err = np.zeros(N)
        # transform shifts to normal format
        double[:] st_shift = np.array([10**y for y in x])
        
    for i in prange(N, nogil=True, schedule='guided', num_threads=N_THREADS):
        # stations' indexes
        j1 = st_1[i]
        j2 = st_2[i]
        
        # distances from aircraft to station 1 and 2
        d1 = sqrt((points[3*i]-st_loc[3*j1])**2 + (points[3*i+1]-st_loc[3*j1+1])**2 + (points[3*i+2]-st_loc[3*j1+2])**2)
        d2 = sqrt((points[3*i]-st_loc[3*j2])**2 + (points[3*i+1]-st_loc[3*j2+1])**2 + (points[3*i+2]-st_loc[3*j2+2])**2)
        # delta time * velocity
        dt = eff_velocity(0.5*st_hgt[j1]+0.5*st_hgt[j2], heights[i], A0, B) * (t_1[i] - t_2[i] + st_shift[j1] - st_shift[j2])
        
        if loss_l1 == 1 or return_errors == 1:
            err[i] = fabs(d1 - d2 - dt)
        else:
            err[i] = (d1 - d2 - dt)**2
        
    if return_errors == 1:
        return np.array(err)
    elif loss_l1 == 1:
        return np.sum(err) / N
    else:
        return np.sqrt(np.sum(err) / N)
    
    
# --------------------------------------------------------------------
# Cython function to optimize stations shifts and parameters A0, B
# --------------------------------------------------------------------
@boundscheck(False)
@wraparound(False)
def optimize_AB_shifts(double[:] x,  # A0, B and shifts to be optimized
                       double[:] points,  # aicraft locations in cartesian coordinates
                       double[:] heights,  # aircraft geoAltitude values
                       double[:] st_loc,  # stations locations in cartesian coordinates
                       double[:] st_hgt,  # stations height
                       int[:] st_1,  # station 1 indexes
                       int[:] st_2,  # station 2 indexes
                       double[:] t_1,  # station 1 measured times
                       double[:] t_2,  # station 2 measured times
                       int loss_l1=1,
                       int return_errors=0):
    
    cdef:
        int N = t_1.shape[0]
        int i, j1, j2
        double d1, d2, dt
        
        double[:] err = np.zeros(N)
        
        # transform to normal format
        double A0 = 10**x[0]
        double B = 10**x[1]
        double[:] st_shift = np.array([10**y for y in x[2:]])
        
    for i in prange(N, nogil=True, schedule='guided', num_threads=N_THREADS):
        # stations' indexes
        j1 = st_1[i]
        j2 = st_2[i]
        
        # distances from aircraft to station 1 and 2
        d1 = sqrt((points[3*i]-st_loc[3*j1])**2 + (points[3*i+1]-st_loc[3*j1+1])**2 + (points[3*i+2]-st_loc[3*j1+2])**2)
        d2 = sqrt((points[3*i]-st_loc[3*j2])**2 + (points[3*i+1]-st_loc[3*j2+1])**2 + (points[3*i+2]-st_loc[3*j2+2])**2)
        # delta time * velocity
        dt = eff_velocity(0.5*st_hgt[j1]+0.5*st_hgt[j2], heights[i], A0, B) * (t_1[i] - t_2[i] + st_shift[j1] - st_shift[j2])
        
        if loss_l1 == 1 or return_errors == 1:
            err[i] = fabs(d1 - d2 - dt)
        else:
            err[i] = (d1 - d2 - dt)**2
        
    if return_errors == 1:
        return np.array(err)
    elif loss_l1 == 1:
        return np.sum(err) / N
    else:
        return np.sqrt(np.sum(err) / N)
    

# --------------------------------------------------------------------
# Cython function to optimize stations locations, shifts and A0, B parameters
# --------------------------------------------------------------------
@boundscheck(False)
@wraparound(False)
def optimize_locations(double[:] x,  # A0, B, shifts and stations locations to be optimized
                       double[:] points,  # aicraft locations in cartesian coordinates
                       double[:] heights,  # aircraft geoAltitude values
                       int[:] st_1,  # station 1 indexes
                       int[:] st_2,  # station 2 indexes
                       double[:] t_1,  # station 1 measured times
                       double[:] t_2,  # station 2 measured times
                       int loss_l1=1,
                       int return_errors=0):
    
    cdef:
        int N = t_1.shape[0]
        int M = int((x.shape[0] - 2) / 4)
        int i, j1, j2
        double d1, d2, dt
        
        double[:] err = np.zeros(N)
        
        # transform to normal format
        double A0 = 10**x[0]
        double B = 10**x[1]
        double[:] st_shift = np.array([pow(10, y) for y in x[2:M+2]])
        double[:] st_loc = np.zeros(3*M)
        double[:] st_hgt = np.zeros(M)
        
    # cartesian coordinates for stations locations + heights
    for i in range(M):
        j1 = 3*i+M+2
        cart = wgs2cart_c(x[j1], x[j1+1], x[j1+2])
        st_loc[3*i] = cart[0]
        st_loc[3*i+1] = cart[1]
        st_loc[3*i+2] = cart[2]
        st_hgt[i] = x[j1+2]
    
    for i in prange(N, nogil=True, schedule='guided', num_threads=N_THREADS):
        # stations' indexes
        j1 = st_1[i]
        j2 = st_2[i]
        
        # distances from aircraft to station 1 and 2
        d1 = sqrt((points[3*i]-st_loc[3*j1])**2 + (points[3*i+1]-st_loc[3*j1+1])**2 + (points[3*i+2]-st_loc[3*j1+2])**2)
        d2 = sqrt((points[3*i]-st_loc[3*j2])**2 + (points[3*i+1]-st_loc[3*j2+1])**2 + (points[3*i+2]-st_loc[3*j2+2])**2)
        # delta time * velocity
        dt = eff_velocity(0.5*st_hgt[j1]+0.5*st_hgt[j2], heights[i], A0, B) * (t_1[i] - t_2[i] + st_shift[j1] - st_shift[j2])
        
        if loss_l1 == 1 or return_errors == 1:
            err[i] = fabs(d1 - d2 - dt)
        else:
            err[i] = (d1 - d2 - dt)**2
        
    if return_errors == 1:
        return np.array(err)
    elif loss_l1 == 1:
        return np.sum(err) / N
    else:
        return np.sqrt(np.sum(err) / N)

    
# --------------------------------------------------------------------
# Cython function to solve multilateration equations
# --------------------------------------------------------------------
@boundscheck(False)
@wraparound(False)
def solve_point(double[:] x,  # latitude and longitude to be optimized
                double baroAlt,  # baroAltitude
                double[:] times,  # stations measured times
                int[:] st,  # stations indexes
                double[:] st_cart,  # stations locations in cartesian coordinates
                double[:] st_hgt,  # stations heigts
                double A0,  # A0 parameter
                double B,  # B parameters
                int loss_l1 = 1
                ):
    
    cdef:
        int N = times.shape[0]
        
        double err = 0
        
        int i, j, j1, j2
        double dist1, dist2, t1, t2
        double cart[3]
    
    # cartesian coordinates for x
    cart = wgs2cart_c(x[0], x[1], baroAlt)
    
    # loops over each pair of stations
    for i in prange(N, nogil=True, schedule='guided', num_threads=N_THREADS):
        # index of station 1 (used for st_cart and st_hgt arrays)
        j1 = st[i]
        dist1 = sqrt((st_cart[3*j1] - cart[0])**2 + (st_cart[3*j1+1] - cart[1])**2 + (st_cart[3*j1+2] - cart[2])**2)
        t1 = times[i] - dist1 / eff_velocity(st_hgt[j1], baroAlt, A0, B)
        
        for j in range(i+1, N):
            # index of station 2 (used for st_cart and st_hgt arrays)
            j2 = st[j]
            
            dist2 = sqrt((st_cart[3*j2] - cart[0])**2 + (st_cart[3*j2+1] - cart[1])**2 + (st_cart[3*j2+2] - cart[2])**2)
            t2 = times[j] - dist2 / eff_velocity(st_hgt[j2], baroAlt, A0, B)
            
            err += fabs(t1 - t2)
    
    return err / N
