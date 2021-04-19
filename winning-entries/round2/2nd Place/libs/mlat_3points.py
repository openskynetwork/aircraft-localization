import numpy as np
from libs import geo

def _get_TDoA(receiver_coords, pos_latlon, normalize=True):
    pos = geo.latlon_to_cartesian(*pos_latlon)
    TDoA = []
    for rec in receiver_coords:
        d = np.linalg.norm(rec - pos)
        TDoA.append(d/geo.LIGHT_SPEED)

    TDoA = np.array(TDoA)
    if normalize:
        TDoA -= TDoA.min()

    return TDoA

def _get_TDoA_cartesian(receiver_coords, pos_cartesian, normalize=True):
    pos = pos_cartesian
    TDoA = []
    for rec in receiver_coords:
        d = np.linalg.norm(rec - pos)
        TDoA.append(d/geo.LIGHT_SPEED)

    TDoA = np.array(TDoA)
    if normalize:
        TDoA -= TDoA.min()

    return TDoA


