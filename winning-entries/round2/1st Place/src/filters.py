#############################################
#
# filters.py
#
#############################################

import numpy as np
from scipy.signal import medfilt

import graph_tool as gt
from graph_tool import topology
import scipy

from src.geo import numpylla2ecef

## Classic median filter
def median_filter(z, window):
    # Add flipped points to the left of the array
    lb = 2*np.median(z[:3])*np.ones(window) - z[3:window+3][::-1]
    # Add flipped pointe to the right of the array
    rb = 2*np.median(z[-3:])*np.ones(window) - z[-window-3:-3][::-1]

    # Combine together and apply median filter
    final = np.r_[lb, z, rb]
    return medfilt(final, window)[window:-window]


## Copy of code from @richardalligier round_1 of the competition
def precompute_distance(lat, lon, t,speedmin,speedmax):
    '''compute a matrix telling if points i and j are reachable within speed limits'''
    xyz = np.stack(numpylla2ecef(lat,lon,np.zeros_like(lon)),-1)
    d = scipy.spatial.distance_matrix(xyz,xyz)
    t = np.transpose(np.array([t]))
    dt = scipy.spatial.distance_matrix(t,t)
    return np.maximum(d-speedmax*dt,0)+np.maximum(speedmin*dt-d,0)


## Copy of code from @richardalligier round_1 of the competition
def get_gtlongest(dd):
    '''compute the longest path of points complying with the speed limits'''
    v,g,prop_dist = compute_gtgraph(dd)
    c = gt.topology.shortest_distance(g,source=None,
                                      target=None,
                                      weights=prop_dist,
                                      negative_weights=True,
                                      directed=True,
                                      pred_map=True,
                                      dag=True)
    i=min(list(range(dd.shape[0])),key=lambda i:np.min(c[v[i]].a))
    j=min(list(range(dd.shape[0])),key=lambda j:c[v[i]].a[j])
    mindist=c[i].a[j]
    longest_path,_ =gt.topology.shortest_path(g,source=v[i],
                                              target=v[j],
                                              weights=prop_dist,
                                              negative_weights=True,
                                              dag=True)
    longest_path =list(map(int,longest_path))
    return longest_path


## Copy of code from @richardalligier round_1 of the competition
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


## Copy of code from @richardalligier round_1 of the competition
def filter_speedlimit(lat,lon,t,speedmin,speedmax, verbose=True):
    '''returns boolean vector giving the longest sequence of points complying with the speed limits'''
    if verbose:
        print("filter trajectory to keep the longest sequence of points complying with the speed limits")
    
    dd = precompute_distance(lat,lon,t,speedmin,speedmax)
    
    if verbose:
        print("initial number of points",dd.shape[0])
    
    longest_path = get_gtlongest(dd)
    res=np.array([i in longest_path for i in range(dd.shape[-1])])
    
    if verbose:
        print("longest sequence", np.sum(res))
   
    return res
