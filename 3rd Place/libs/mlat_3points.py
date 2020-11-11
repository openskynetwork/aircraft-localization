import numpy as np
from libs import geo, mlat_approx_by_power

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


def calc(receiver_coords, timestamps, powers, altitude=None, start_pos=None, start_step = None, end_step = None):
    """
    Iterative optimization method for points with 3 or more measurements
    :param receiver_coords: sensor coords
    :param timestamps: receiving time
    :param powers:
    :param altitude: baro altitude
    :param start_pos: initial search coords
    :param start_step: start lat/long step
    :param end_step: finish lat/long step
    :return: optimized coords
    """
    if start_pos is None:
        approx_pos = mlat_approx_by_power.calc(receiver_coords, timestamps, powers, altitude=altitude)
    else:
        approx_pos = start_pos
    approx_pos_latlon = geo.cartesian_to_latlon(*approx_pos)
    if altitude is not None:
        approx_pos_latlon = (approx_pos_latlon[0], approx_pos_latlon[1], altitude)

    target_TDoA = timestamps
    best_TDoA_diff = np.linalg.norm(target_TDoA - _get_TDoA(receiver_coords, approx_pos_latlon))
    best_pos_latlon = approx_pos_latlon

    pos_latlon = approx_pos_latlon
    updated = True

    step = 0.005 if start_step is None else start_step
    if end_step is None:
        end_step = 0.0001

    iter = 0
    while updated and (step > end_step) and (iter < 100):
        updated = False

        for i in range(-1, 2):
            for j  in range(-1, 2):
                if (i==0) and (j==0):
                    continue

                pos2_latlon = np.array([pos_latlon[0] + i*step, pos_latlon[1] + j*step, pos_latlon[2]])
                TDoA = _get_TDoA(receiver_coords, pos2_latlon)

                diff = np.linalg.norm(target_TDoA - TDoA)
                if diff < best_TDoA_diff:
                    best_TDoA_diff = diff
                    best_pos_latlon = pos2_latlon
                    updated = True

        pos_latlon = best_pos_latlon

        if not updated:
            step /= 2
            updated = True

        iter += 1

    if iter>=100:
        return None

    return geo.latlon_to_cartesian(*best_pos_latlon)



