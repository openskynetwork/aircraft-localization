import numpy as np
import math
import pyproj

EARTH_RADIUS = 6378140
LIGHT_SPEED = 299792458/1.0003
LIGHT_SPEED_1e9 = .299792458/1.0003 # light speed in air

ecef = pyproj.CRS("+proj=geocent +datum=WGS84 +towgs84=0,0,0")
lla = pyproj.CRS.from_epsg(4326)

lla_to_ecef = pyproj.Transformer.from_crs(lla, ecef, always_xy=True)
ecef_to_lla = pyproj.Transformer.from_crs(ecef, lla, always_xy=True)

def latlon_to_cartesian(lat, lon, h):
    """
    Convert Lat/Long coordinates to 3D cartesian
    :param lat:
    :param lon:
    :param h:
    :return:
    """
    x, y, z = lla_to_ecef.transform(lon, lat, h)
    return np.array([x, y, z])

def cartesian_to_latlon(x, y, z):
    """
    Convert 3D cartesian coords to Lat/Long (WGS84)
    :param x:
    :param y:
    :param z:
    :return:
    """
    long, lat, h = ecef_to_lla.transform(x, y, z)
    return lat, long, h

def _lorentz(x, y):
    p = x[0] * y[0] + x[1] * y[1] + x[2] * y[2] - x[3] * y[3]
    return p

def _bancroft(B, weight=None):
    """
    Bancroft multilateration method
    :param B:
    :return:
    """
    B = B.astype(np.float64)/EARTH_RADIUS   # normalization
    m = B.shape[0]

    e = np.ones((m, 1), dtype=np.float64)
    alpha = np.zeros((m, 1), dtype=np.float64)
    for i in range(m):
        alpha[i] = _lorentz(B[i], B[i]) / 2

    if m > 4:
        if weight is not None:
            B = B * weight.reshape(-1, 1)
            e = e * weight.reshape(-1, 1)
            alpha = alpha * weight.reshape(-1, 1)
        BBBe = np.linalg.lstsq(B, e, rcond=-1)[0]
        BBBalpha = np.linalg.lstsq(B, alpha, rcond=-1)[0]
    else:
        BBBe = np.linalg.solve(B, e)
        BBBalpha = np.linalg.solve(B, alpha)

    a = _lorentz(BBBe, BBBe)
    b = _lorentz(BBBe, BBBalpha) - 1
    c = _lorentz(BBBalpha, BBBalpha)
    if b * b - a * c < 0:
        return None

    root = math.sqrt(b * b - a * c)
    r = [(-b - root) / a,
         (-b + root) / a]
    possible_pos = np.zeros((4, 2), dtype=np.float64)
    for i in range(2):
        possible_pos[:, i] = (r[i] * BBBe + BBBalpha).flatten()

    possible_pos = possible_pos[0:3, :]

    # select a solution that placed at Earth surface
    if np.linalg.norm(possible_pos[:, 0]) > np.linalg.norm(possible_pos[:, 1]):
        pos = possible_pos[:, 0]
    else:
        pos = possible_pos[:, 1]
    return pos*EARTH_RADIUS

def bancroft_method(sensors_coords, time_of_arrivals, altitude=None, weight=None):
    pseudorange = (time_of_arrivals - np.min(time_of_arrivals))*LIGHT_SPEED
    if len(sensors_coords) >= 4:
        B = np.hstack([sensors_coords, pseudorange.reshape(-1, 1)])

        pos = _bancroft(B, weight=weight)
        if pos is None:
            return None
        return pos
    else:
        return None
