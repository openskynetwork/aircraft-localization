import math
import numpy as np
from libs import geo

def calc(receiver_coords, timestamps, powers, altitude=None):
    """
    Calculate approximated coords only by receiving signal power
    :param receiver_coords:
    :param timestamps:
    :param powers:
    :param altitude:
    :return:
    """
    res_coords = np.zeros(3, dtype=np.float32)

    k = 0.1
    sum_power = np.sum(powers + k)
    for i in range(len(receiver_coords)):
        res_coords += receiver_coords[i] * (powers[i] + k)

    if (sum_power > 0):
        res_coords /= sum_power

    if altitude is not None:
        res_coords = (res_coords / np.linalg.norm(res_coords))*(geo.EARTH_RADIUS + altitude)

    return res_coords