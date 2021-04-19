import math
import numpy as np
from libs import geo

HEIGHT_THRESHOLD = 17313.746438852424 #14354.342100693497 #15000 #12200.114551713068 #12311.43251408517 #15713.560776812128 #20e3
DIST_THRESHOLD = 726856.4190648322 #882333.0711211708 #805678.3759864655 #277961.1414527998 #828609.5647409786 #400e3

def calc(receiver_coords, timestamps, powers, altitude=None, weight=None):
    if len(receiver_coords) < 4:
        return None

    n = len(receiver_coords)
    mask = [True for i in range(n)]
    # remove too closest sensors
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(receiver_coords[i] - receiver_coords[j])
            if d<300:
                mask[j] = False

    mask = np.array(mask, dtype=np.bool)
    receiver_coords = receiver_coords[mask]
    timestamps = timestamps[mask]
    if weight is not None:
        weight = weight[mask]

    if len(receiver_coords) < 4:
        return None

    timestamps = timestamps - timestamps.min()
    # calculate coords with Bancroft method
    pos = geo.bancroft_method(receiver_coords, timestamps, altitude, weight=weight)
    if pos is None:
        return None

    # remove predicted coords thats too high under Earth or too lower undeground
    height_thr = HEIGHT_THRESHOLD
    #height_thr = 100000
    # height_thr = 35000
    _, _, h = geo.cartesian_to_latlon(*pos)
    if abs(h) > height_thr:
        return None

    # remove predicted coords that too far from sensors
    dist_thr = DIST_THRESHOLD
    for rec in receiver_coords:
        d = np.linalg.norm(rec - pos)
        if d > dist_thr:
            return None

    return pos