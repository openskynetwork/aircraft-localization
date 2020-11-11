import numpy as np

from . import geo


def _haversine_distance(lat1, lon1, lat2, lon2):
   r = geo.EARTH_RADIUS
   phi1 = np.radians(lat1)
   phi2 = np.radians(lat2)
   delta_phi = np.radians(lat2-lat1)
   delta_lambda = np.radians(lon2-lon1)
   a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
   res = r * (2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)))
   return res


def _filter_nan(lat1, lon1, lat2, lon2):
    good = np.isfinite(lat1) & np.isfinite(lat2)
    return lat1[good], lon1[good], lat2[good], lon2[good]



def rmse(lat1, lon1, lat2, lon2, half_threshold = True, return_dist=False):
    l = len(lat1)
    lat2_orig = lat2.copy()
    lat1, lon1, lat2, lon2 = _filter_nan(lat1, lon1, lat2, lon2)
    if half_threshold and len(lat1) <= int(np.ceil(l*0.5)):
        return 1e100

    dist = _haversine_distance(lat1, lon1, lat2, lon2)
    score = np.sqrt(np.mean(np.square(dist)))

    if return_dist:
        d = np.zeros(len(lat2_orig))
        d[np.isfinite(lat2_orig)] = dist
        d[np.isnan(lat2_orig)] = np.nan
        return score, d
    else:
        return score


def rmse_90_cut(lat1, lon1, lat2, lon2, half_threshold = True):
    l = len(lat1)
    lat1, lon1, lat2, lon2 = _filter_nan(lat1, lon1, lat2, lon2)
    if half_threshold and len(lat1) <= int(np.ceil(l * 0.5)):
        return 1e100

    dist = _haversine_distance(lat1, lon1, lat2, lon2)
    dist = np.sort(dist)
    dist = dist[:int(np.ceil(len(dist)*0.9))]
    return np.sqrt(np.mean(np.square(dist)))
