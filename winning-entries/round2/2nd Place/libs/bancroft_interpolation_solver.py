from libs import geo
from libs import mlat_bancroft, mlat_3points
from libs import preprocessing
from libs import metrics
from libs.SensorSynchronization import SensorSynchronization

from typing import Dict

import numpy as np
import json
from scipy import signal, interpolate
from pyproj import geod
from numba import jit

OUTLIERS_THRESHOLD = 0.191288083293375 #0.02027213303512406 #0.10043493011984012 #0.013161539933980232 #0.1
OUTLIERS_COUNT = 4 #7 #10 #48 #10
POINTS_COUNT =  17 #65 #25
SPEED_LIMIT = 779.2001553664188 #1500e3/60/60 #714.5230457281422

SMOOTH_SAMPLE_FREQ = 64.80187573373688 #20
SMOOTH_CRITICAL_FREQ = 0.04109712678299951 #0.02

FILTER_BY_TDOA_THRESHOLD = 60
DIVIDE_PARTS = 100

IMPROVE_START_STEP = 0.001
IMPROVE_DIST_THRESHOLD = 100

COORD0_SHIFT = 0.00004
COORD1_SHIFT = 0.00004

ISOLATED_THRESHOLD = 3


def outlier_filter(time, coords):
    if (time is None) or (len(time) == 0):
        return time, coords

    thr = OUTLIERS_THRESHOLD
    k = OUTLIERS_COUNT
    masks = []
    for i in range(2):
        y = coords[:, i]
        y2 = np.pad(y, k, mode='edge')
        y_filtered = signal.medfilt(y2, 2*k-1)
        y_filtered = y_filtered[k:-k]
        mask = np.abs(y - y_filtered) < thr
        masks.append(mask)

    full_mask = np.logical_and(masks[0], masks[1])

    return np.array(time)[full_mask], coords[full_mask]

def interpolate_coords(x0, y0, x1):
    if len(x0) < 1:
        res = np.empty((len(x1), 3))
        res[:] = np.NaN
        return res, x1

    if len(x0) == 1:
        res = np.empty((len(x1), 3))
        res[:] = np.NaN

        for i in range(len(x1)):
            if abs(x1[i] - x0[0]) < 1e-3:
                res[i] = y0[0]

        return res, x1

    x0 = np.array(x0)
    y0 = np.array(y0)
    x1 = np.array(x1)

    x0, x0_ind = np.unique(x0,return_index=True)
    y0 = y0[x0_ind]

    if len(x0) < 1:
        res = np.empty((len(x1), 3))
        res[:] = np.NaN
        return res, x1

    if len(x0) == 1:
        res = np.empty((len(x1), 3))
        res[:] = np.NaN

        for i in range(len(x1)):
            if abs(x1[i] - x0[0]) < 1e-3:
                res[i] = y0[0]

        return res, x1

    time_idx = 0
    time_left, time_right = x0[time_idx], x0[time_idx + 1]

    res = np.empty((len(x1), 3))
    res[:] = np.NaN

    for i in range(len(x1)):
        time = x1[i]

        if time_idx + 1 >= len(x0):
            break
            print('skip')
            continue

        if time < time_left:
            continue
        if time > time_right:
            time_idx += 1
            if time_idx + 1 >= len(x0):
                continue
            time_left, time_right = x0[time_idx], x0[time_idx + 1]

        if (time_left <= time) and (time <= time_right):
            p_left = y0[time_idx]
            p_right = y0[time_idx+1]

            g = geod.Geod(ellps="WGS84")

            k = (time-time_left)/(time_right - time_left)
            forward, back, dist = g.inv(p_left[1], p_left[0], p_right[1], p_right[0])
            p_lon, p_lat, _, = g.fwd(p_left[1], p_left[0], forward, dist*k)
            alt = p_left[2] + k*(p_right[2] - p_left[2])

            res[i] = np.array([p_lat, p_lon, alt])

    return res, x1


def smooth_1d(x0, y0):
    mask = np.isfinite(y0)
    if np.sum(mask)<2:
        return y0

    x1 = x0[mask].copy()
    y1 = y0[mask].copy()

    x_left = x1.min()
    x_right = x1.max()
    sample_freq = SMOOTH_SAMPLE_FREQ
    n = int(round((x_right - x_left) / (1 / sample_freq)))
    x2 = np.linspace(x_left, x_right, n)
    y2 = interpolate.interp1d(x1, y1)(x2)

    filt = signal.butter(1, Wn = SMOOTH_CRITICAL_FREQ, btype='lowpass', fs=sample_freq)

    y2_filtered  = signal.filtfilt(filt[0], filt[1], y2, method='pad')
    if len(y2_filtered) > 30*sample_freq:
        k = int(15*SMOOTH_SAMPLE_FREQ)
        y2_filtered[:k] =y2[:k]
        y2_filtered[-k:] = y2[-k:]

    res = y0.copy()
    res[mask] = interpolate.interp1d(x2, y2_filtered)(x1)

    return res

def smooth_filter(x0, y0):
    if y0 is None:
        res = np.empty((x0.shape[0], 3))
        res[:] = np.NaN
        return res

    mask = np.isfinite(y0[:, 0])
    x0_f = x0[mask]
    y0_f = y0[mask]

    if np.sum(mask) <= 6:
        return y0

    res = y0

    for i in range(y0_f.shape[1]):
        smoothed = smooth_1d(x0_f, y0_f[:, i])
        res[mask, i] = smoothed

    return res

def adaptive_filter_by_score(predicted, dists, keep_part=0.5):
    dists[np.isnan(predicted[:, 0])] = 1e100
    res = predicted.copy()

    dists_pos = np.argsort(dists)
    res[dists_pos[int(np.ceil(len(dists)*keep_part)+1):], :] = np.nan

    return res

def parse_measurements(df, sensor_coords, sensor_synch:SensorSynchronization, sensors_filter_line=None):
    """
    Remove outliers in measurement data
    :param df: test dataset for aircraft
    :param sensor_coords:
    :param shift: time shift for each sensor
    :return: filtered test dataset
    """
    meas_time = []
    meas_received_time = []
    meas_sensor_coords = []
    meas_altitude = []

    iterator = df.iterrows()
    i =-1
    for _, row in iterator:
        i +=1
        # curr_t = receiver_time[0] if len(receiver_time) else row['timeAtServer']
        curr_t = row['timeAtServer']

        sensor_ids, receiver_time = _parse_json_meas(row['measurements'], sensor_synch,
                                                     sensors_filter_line=sensors_filter_line, server_time=curr_t)
        receiver_coords = [sensor_coords[id] for id in sensor_ids]


        meas_time.append(curr_t)
        meas_received_time.append(np.array(receiver_time))
        meas_sensor_coords.append(np.array(receiver_coords))
        meas_altitude.append(row['baroAltitude'])

    return np.array(meas_time), meas_sensor_coords, meas_received_time, np.array(meas_altitude)


def _parse_json_meas(meas_str, sensor_synch: SensorSynchronization, sensors_filter_line: Dict[int, float] = None, server_time=None):
    meas = json.loads(meas_str)
    sensor_ids = np.array([m[0] for m in meas])
    sensor_times = np.array([m[1] for m in meas], dtype=np.float64)

    sensor_ids, sensor_times = sensor_synch.synch(sensor_ids, sensor_times, points_count=POINTS_COUNT)

    all_meas = []
    for sensor_id, timestamp in zip(sensor_ids, sensor_times):
        sensor_component = 1
        all_meas.append((sensor_component, timestamp, sensor_id))

    all_meas = sorted(all_meas)

    best_seq_from = -1
    best_seq_to = -1
    best_seq_len = 0

    seq_from = 0
    for seq_to in range(len(all_meas)):
        while (seq_from < seq_to) & (all_meas[seq_from][0] != all_meas[seq_to][0]):
            seq_from += 1
        while (seq_from < seq_to) & ((all_meas[seq_to][1] - all_meas[seq_from][1])*geo.LIGHT_SPEED > 800e3):
            seq_from += 1

        seq_len = seq_to - seq_from + 1
        if seq_len > best_seq_len:
            best_seq_len = seq_len
            best_seq_from = seq_from
            best_seq_to = seq_to

    if best_seq_from==-1:
        return [], []

    return [x[2] for x in all_meas[best_seq_from:best_seq_to+1]], \
           [x[1] for x in all_meas[best_seq_from:best_seq_to+1]]

def mlat_4meas_points(meas_time, meas_sensor_coords, meas_received_time, meas_altitude=None):
    """
    Apply a Bancroft method for one point
    :param meas_time: server time
    :param meas_sensor_coords: sensor coords
    :param meas_received_time: receiving time for all sensors including shift time
    :return:
    """
    mlat_coord = []
    mlat_time = []
    for i in range(len(meas_time)):
        receiver_coords = meas_sensor_coords[i]
        receiver_time = meas_received_time[i]
        receiver_power = []

        if len(receiver_coords) < 4:
            continue

        receiver_time -= np.min(receiver_time)

        approx_coords_cartesian = mlat_bancroft.calc(receiver_coords, receiver_time, receiver_power)
        if (approx_coords_cartesian is not None) and (meas_altitude is not None):
            best_solution = approx_coords_cartesian
            #for w in np.linspace(0.1, 1, 10):
            w = 1
            approx_coords_latlon = geo.cartesian_to_latlon(*best_solution)
            surface_point_cartesian = geo.latlon_to_cartesian(approx_coords_latlon[0], approx_coords_latlon[1], 0)

            receiver_coords2 = np.concatenate([receiver_coords, surface_point_cartesian.reshape(1, 3)], axis=0)

            ToA = mlat_3points._get_TDoA_cartesian(receiver_coords, approx_coords_cartesian, normalize=False)
            alt = meas_altitude[i]

            receiver_time2 = np.concatenate([receiver_time, [alt/geo.LIGHT_SPEED - ToA.min()]])
            weight = np.concatenate([np.ones_like(receiver_time), np.array([w])])

            next_solution = mlat_bancroft.calc(receiver_coords2, receiver_time2, receiver_power, weight=weight)
            if next_solution is not None:
                best_solution = next_solution
            # else:
            #     break
            if best_solution is not None:
                approx_coords_cartesian = best_solution

        if approx_coords_cartesian is not None:
            mlat_coord.append(approx_coords_cartesian)
            mlat_time.append(meas_time[i])


    return mlat_time, mlat_coord


def solve_for_aircraft(df, aircraft, sensor_coords, filter_byTdoA=True, silent=False, sensors_filter_line=None, sensor_synch=None):
    """
    Base function for track calculation. It uses Bancroft multilateration method
    :param df: test dataset
    :param aircraft: aircraft id
    :param sensor_coords: 3D sensor coords
    :param sensors_filter_line - preprocessed data for outlier sensors detection
    :param sensor_synch: sensor time synchronization object
    :return: all restored coords for aircraft
    """

    # filter data for one test aircraft
    aircraft_df = preprocessing.get_data_for_aircrafts(df, [aircraft]).copy()
    if not silent:
        print(aircraft_df.shape)

    # Parse measurements and independent sensor time synchronization for each point.
    # Leave measurements data only for points with 4 or more correctly synchronized sensor measurements
    meas_time, meas_sensor_coords, meas_received_time, meas_altitude = \
        parse_measurements(aircraft_df, sensor_coords, sensor_synch=sensor_synch, sensors_filter_line=sensors_filter_line)


    # calculate coordinates with Bancroft method for all points with 4 or more measurements
    mlat_4meas_time, mlat_4meas_coords = mlat_4meas_points(meas_time, meas_sensor_coords, meas_received_time, meas_altitude)

    # convert 3D cartesian coordinates to WGS84
    mlat_4meas_coords_latlon = np.array([geo.cartesian_to_latlon(*x) for x in mlat_4meas_coords])
    # outlier filtering with median filtering
    mlat_4meas_time, mlat_4meas_coords_latlon = outlier_filter(mlat_4meas_time, mlat_4meas_coords_latlon)
    del mlat_4meas_coords # not valid data after outlier filtering

    # convert lat/lon coords to cartesian again
    mlat_4meas_coords = np.array([geo.latlon_to_cartesian(*x) for x in mlat_4meas_coords_latlon])

    # filter outliers by speed limit
    if len(mlat_4meas_time) > 2:
        # get mean aircraft speed
        mean_speed = calc_mean_speed(np.array(mlat_4meas_time, dtype=np.float64),
                                     np.array(mlat_4meas_coords, dtype=np.float64))
        # filter outliers based on aircraft speed
        mlat_4meas_time, mlat_4meas_coords = filter_by_speedlimit(np.array(mlat_4meas_time, dtype=np.float64),
                                                                  np.array(mlat_4meas_coords, dtype=np.float64),
                                                                  mean_speed)

    mlat_4meas_coords_latlon = np.array([geo.cartesian_to_latlon(*x) for x in mlat_4meas_coords])
    del mlat_4meas_coords  # not valid data after outlier filtering

    # save positions of correctly restored points with 4 or more measurements
    keypoint_mask = np.isin(aircraft_df['timeAtServer'], mlat_4meas_time)

    mlat_time = mlat_4meas_time
    mlat_coords_latlon = mlat_4meas_coords_latlon

    # interpolate WGS84 coords between points with 4 or more measurements
    interpolated_coords_latlon, interpolated_time = \
        interpolate_coords(mlat_time, mlat_coords_latlon, meas_time)

    # smooth aircraft track with low-pass filter
    interpolated_coords_latlon = smooth_filter(interpolated_time, interpolated_coords_latlon)

    # Calculate weighs for further global coverage filtering. Not used in current approach
    if isinstance(mlat_time, np.ndarray) and len(mlat_time)>1:
        time_dist_to_support = np.min(np.abs(meas_time.reshape((-1, 1)) - mlat_time.reshape((1, -1))), axis=-1)
        sort = np.argsort(np.abs(meas_time.reshape((-1, 1)) - mlat_time.reshape((1, -1))), axis=-1)
        pos = sort[:, 0]
        pos2 = sort[:, 1]
        diff_value = (
                             np.abs(mlat_coords_latlon[pos, 0] - mlat_coords_latlon[pos2, 0]) +
                             np.abs(mlat_coords_latlon[pos, 1] - mlat_coords_latlon[pos2, 1])
                     )/2
        time_dist_to_support = time_dist_to_support * diff_value

        time_dist_to_support[np.isnan(interpolated_coords_latlon[:,0])] = 1e100
    else:
        time_dist_to_support = np.zeros_like(interpolated_coords_latlon[:, 0])
        time_dist_to_support[:] = 1e100


    return interpolated_coords_latlon, interpolated_time, time_dist_to_support, keypoint_mask


@jit(nopython=True)
def filter_by_speedlimit(mlat_4meas_time, mlat_4meas_coords, mean_speed):
    """
    Remove outliers from predicted aircraft positions by speed thresholding using dynamic programming approach.

    :param mlat_4meas_time: server time for predicted points
    :param mlat_4meas_coords: predicted aircraft points in 3D
    :param mean_speed:
    :return:
    """
    if len(mlat_4meas_time) <= 2:
        return mlat_4meas_time, mlat_4meas_coords

    SPEED_LIMIT = 1.15*mean_speed

    n = int(len(mlat_4meas_time))
    dp = np.zeros((n, 2), dtype=np.int32)
    max_len = np.ones(n, dtype=np.int32)
    dp[:, 0] = 1
    dp[:, 1] = -1

    for i in range(n):
        for j in range(i-1, -1, -1):
            if max_len[j] + 1 < dp[i, 0]:
                break

            nodes = dp[j, 0] + 1
            if nodes <= dp[i,0]:
                continue

            dist = np.linalg.norm(mlat_4meas_coords[i] - mlat_4meas_coords[j])
            #dist = np.sqrt(np.sum(np.square(mlat_4meas_coords[i] - mlat_4meas_coords[j])))
            t = mlat_4meas_time[i] - mlat_4meas_time[j]
            if t > 0.01:
                speed = dist/t
                if speed < SPEED_LIMIT:
                    dp[i,0] = nodes
                    dp[i, 1] = j

        if i > 0:
            max_len[i] = max(max_len[i-1], dp[i, 0])
        else:
            max_len[i] = dp[i, 0]

    best_pos = np.argmax(dp[:, 0])
    mask = np.zeros(n, dtype=np.bool_)

    while(best_pos != -1):
        mask[best_pos] = True
        best_pos = dp[best_pos, 1]

    return mlat_4meas_time[mask], mlat_4meas_coords[mask]

def calc_mean_speed(time, coords):
    """
    Calculate meaian aircraft speed based on predicted points
    :param time: time at server for predicted points
    :param coords: predicted 3D cartesian coordinstes
    :return: aircraft median speed
    """
    coords_latlon = []
    for c in coords:
        coords_latlon.append(geo.cartesian_to_latlon(*c))

    speed = []
    for i in range(len(coords_latlon) - 1):
        p1 = coords_latlon[i]
        p2 = coords_latlon[i+1]
        dist = metrics._haversine_distance(p1[0], p1[1], p2[0], p2[1])
        dt = time[i+1] - time[i]
        if dt > 1e-3:
            speed.append(dist/dt)

    mean_speed = np.median(speed)
    return mean_speed
