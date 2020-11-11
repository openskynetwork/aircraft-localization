from libs import geo
from libs import mlat_bancroft, mlat_3points
from libs import preprocessing
from libs import metrics
from libs import piecewise_torch_filter

import numpy as np
from tqdm import tqdm
import json
from scipy import signal, interpolate
import math
from pyproj import geod

import matplotlib.pyplot as plt

OUTLIERS_THRESHOLD = 0.10043493011984012 #0.013161539933980232 #0.1
OUTLIERS_COUNT = 10 #48 #10

SMOOTH_SAMPLE_FREQ = 64.80187573373688 #20
SMOOTH_CRITICAL_FREQ = 0.04109712678299951 #0.02

FILTER_BY_TDOA_THRESHOLD = 60
DIVIDE_PARTS = 100

IMPROVE_START_STEP = 0.001
IMPROVE_DIST_THRESHOLD = 100

COORD0_SHIFT = 0.00004
COORD1_SHIFT = 0.00004

ISOLATED_THRESHOLD = 3

def filter_onepoint_predictions(coords, time, threshold=3):
    mask = np.zeros(coords.shape[0], dtype=np.int32)

    start_pos = -1

    for i in range(coords.shape[0]):
        v = coords[i,0]
        if np.isfinite(v) and start_pos==-1:
            start_pos = i
        if np.isnan(v) and start_pos!=-1:
            mask[start_pos:i] = len(np.unique(time[start_pos:i]))
            start_pos = -1
        if (i==coords.shape[0]-1) and start_pos != -1:
            mask[start_pos:i] = len(np.unique(time[start_pos:(i+1)]))

    thr = threshold#ISOLATED_THRESHOLD
    res = coords.copy()
    res[mask <= thr, :] = np.nan

    return res


def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]


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

    time_idx = 0
    time_left, time_right = x0[time_idx], x0[time_idx + 1]

    res = np.empty((len(x1), 3))
    res[:] = np.NaN

    for i in range(len(x1)):
        time = x1[i]

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

    if np.sum(mask) == 0:
        return y0

    res = y0

    for i in range(y0_f.shape[1]):
        smoothed = smooth_1d(x0_f, y0_f[:, i])
        res[mask, i] = smoothed

    return res

def filter_by_TDoA(df, sensor_coords, predicted, shift):
    res = predicted.copy()

    dists = []
    for i in range(len(df)):
        if np.isnan(predicted[i,0]):
            continue

        row = df.iloc[i]
        meas = json.loads(row['measurements'])

        if len(meas) < 2:
            continue

        receiver_coords = []
        receiver_time = []

        for m in meas:
            sensor_id, timestamp, power = m
            receiver_coords.append(sensor_coords[sensor_id])
            receiver_time.append(timestamp - shift[sensor_id] * 1e9)

        receiver_coords = np.array(receiver_coords)
        receiver_time = np.array(receiver_time)/1e9
        receiver_time -= np.min(receiver_time)

        TDoA = mlat_3points._get_TDoA(receiver_coords, predicted[i])
        d = np.linalg.norm(TDoA - receiver_time)
        if d > FILTER_BY_TDOA_THRESHOLD/geo.LIGHT_SPEED:
            res[i,:] = np.nan

    return res


def tdoa_distance(tdoa, times):
    return np.mean(np.abs(tdoa-times))/math.sqrt(times.shape[0]) / (len(times)-1)


def adaptive_filter_by_TDoA(df, sensor_coords, predicted, shift, keep_part=0.5, silent=False):
    res = predicted.copy()

    dists = np.zeros((len(df),), dtype=np.float)
    iterator = range(len(df)) if silent else tqdm(range(len(df)), "filtering")
    for i in iterator:
        if np.isnan(predicted[i,0]):
            dists[i] = 1e100
            continue

        row = df.iloc[i]
        meas = json.loads(row['measurements'])

        if len(meas) < 2:
            dists[i] = 1e80
            continue

        receiver_coords = []
        receiver_time = []

        for m in meas:
            sensor_id, timestamp, power = m
            receiver_coords.append(sensor_coords[sensor_id])
            receiver_time.append(timestamp - shift[sensor_id] * 1e9)

        receiver_coords = np.array(receiver_coords)
        receiver_time = np.array(receiver_time)/1e9
        receiver_time -= np.min(receiver_time)

        TDoA = mlat_3points._get_TDoA(receiver_coords, predicted[i])

        dists[i] = tdoa_distance(TDoA, receiver_time)

    dists_pos = np.argsort(dists)
    res[dists_pos[int(np.ceil(len(dists)*keep_part)+1):], :] = np.nan

    return res


def adaptive_filter_by_score(predicted, dists, keep_part=0.5):
    dists[np.isnan(predicted[:, 0])] = 1e100
    res = predicted.copy()

    dists_pos = np.argsort(dists)
    res[dists_pos[int(np.ceil(len(dists)*keep_part)+1):], :] = np.nan

    return res

def parse_measurements(df, sensor_coords, shift):
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
    for _, row in iterator:
        meas = json.loads(row['measurements'])

        receiver_coords = []
        receiver_time = []
        receiver_power = []

        curr_t = -1
        for m in meas:
            sensor_id, timestamp, power = m
            receiver_coords.append(sensor_coords[sensor_id])
            receiver_time.append(timestamp/1e9 - shift[sensor_id])
            receiver_power.append(power)

            if curr_t == -1:
                curr_t = receiver_time[0]

        if curr_t == -1:
            curr_t = row['timeAtServer']

        meas_time.append(curr_t)
        meas_received_time.append(np.array(receiver_time))
        meas_sensor_coords.append(np.array(receiver_coords))
        meas_altitude.append(row['baroAltitude'])

    return np.array(meas_time), meas_sensor_coords, meas_received_time, np.array(meas_altitude)


def mlat_4meas_points(meas_time, meas_sensor_coords, meas_received_time):
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

        if approx_coords_cartesian is not None:
            mlat_coord.append(approx_coords_cartesian)
            mlat_time.append(meas_time[i])

    return mlat_time, mlat_coord

def combine_coords(t1, c1, t2, c2):
    t = np.concatenate([t1, t2])
    c = np.concatenate([c1, c2])

    t, k = np.unique(t1, return_index=True)
    return t, c[k]

def improve_3meas_points(mlat_4meas_time, mlat_4meas_coords_latlon,
                         meas_time, meas_sensor_coords, meas_received_time, meas_altitude):
    interpolated_coords_latlon, interpolated_time = \
        interpolate_coords(mlat_4meas_time, mlat_4meas_coords_latlon, meas_time)
    """
    Improve predicted coords for points with 3 or more measurements
    """

    # smooth data applying low-pass filter
    interpolated_coords_latlon = smooth_filter(interpolated_time, interpolated_coords_latlon)

    mlat_3meas_time = []
    mlat_3meas_coords = []

    for i in range(len(meas_time)):
        receiver_coords = meas_sensor_coords[i]
        receiver_time = meas_received_time[i]
        receiver_power = []

        if len(receiver_coords) < 3:
            continue

        receiver_time -= np.min(receiver_time)

        if np.isnan(interpolated_coords_latlon[i, 0]):
            continue

        p = mlat_3points.calc(receiver_coords, receiver_time, receiver_power, meas_altitude[i],
                              geo.latlon_to_cartesian(*interpolated_coords_latlon[i]), IMPROVE_START_STEP, 0.000002)

        if p is not None:
            p_latlon = geo.cartesian_to_latlon(*p)
            d = metrics._haversine_distance(p_latlon[0], p_latlon[1], interpolated_coords_latlon[i, 0],
                                            interpolated_coords_latlon[i, 1])
            # remove coords when improved version is too far from interpolated version
            if d < IMPROVE_DIST_THRESHOLD:
                mlat_3meas_time.append(meas_time[i])
                mlat_3meas_coords.append(p_latlon)

    if len(mlat_3meas_time):
        mlat_3meas_time = np.array(mlat_3meas_time)
        mlat_3meas_coords = np.array(mlat_3meas_coords)

        return combine_coords(mlat_3meas_time, mlat_3meas_coords,
                              mlat_4meas_time, mlat_4meas_coords_latlon)
    else:
        return mlat_4meas_time, mlat_4meas_coords_latlon

def solve_for_aircraft(df, aircraft, sensor_coords, filter_byTdoA=True, silent=False):
    """
    Base function for track calculation. It uses Bancroft multilateration method
    :param df: testd taset
    :param aircraft: aircraft id
    :param sensor_coords: sensor coords
    :param filter_byTdoA: filter data by TDoA distance
    :param silent:
    :return: all restored coords for aircraft
    """

    # load time shifts for all sensors
    shift = np.load("shift_coefs.npz")["shift"]

    aircraft_df = preprocessing.get_data_for_aircrafts(df, [aircraft])
    if not silent:
        print(aircraft_df.shape)

    # remove outliers from test dataset
    meas_time, meas_sensor_coords, meas_received_time, meas_altitude = parse_measurements(aircraft_df, sensor_coords, shift)
    # calculate coordinates with Bancroft method for all points with 4 or more measurements
    mlat_4meas_time, mlat_4meas_coords = mlat_4meas_points(meas_time, meas_sensor_coords, meas_received_time)
    # convert 3D cartesian coordinates to WGS84
    mlat_4meas_coords_latlon = np.array([geo.cartesian_to_latlon(*x) for x in mlat_4meas_coords])
    # outlier filtering with median filtering
    mlat_4meas_time, mlat_4meas_coords_latlon = outlier_filter(mlat_4meas_time, mlat_4meas_coords_latlon)
    del mlat_4meas_coords # not valid data after outlier filtering

    # try to improve coordinates for points with 3 or more measurements
    mlat_time, mlat_coords_latlon = improve_3meas_points(mlat_4meas_time, mlat_4meas_coords_latlon,
                         meas_time, meas_sensor_coords, meas_received_time, meas_altitude)

    # interpolate WGS84 coords between points with 3 or more measurements
    interpolated_coords_latlon, interpolated_time = \
        interpolate_coords(mlat_time, mlat_coords_latlon, meas_time)

    # approximate track with piecewise-linear regression
    if isinstance(mlat_time, np.ndarray):
        weights = (np.min(np.abs(interpolated_time.reshape((-1, 1)) - mlat_time.reshape((1, -1))), axis=-1)<1e-15).astype(np.float)
        interpolated_coords_latlon[:, 0] = piecewise_torch_filter.filter_with_piecewise_with_nans(interpolated_time, interpolated_coords_latlon[:, 0], divide_parts=DIVIDE_PARTS, silent=silent, weights=weights)
        interpolated_coords_latlon[:, 1] = piecewise_torch_filter.filter_with_piecewise_with_nans(interpolated_time, interpolated_coords_latlon[:, 1], divide_parts=DIVIDE_PARTS, silent=silent, weights=weights)

    # filter interpolated coordinates by TDoA differences between actual and measured receiving time
    if filter_byTdoA:
        interpolated_coords_latlon = filter_by_TDoA(aircraft_df, sensor_coords, interpolated_coords_latlon, shift)

    # remove isolated predictions
    interpolated_coords_latlon = filter_onepoint_predictions(interpolated_coords_latlon, interpolated_time)

    # Calculate weighs for further global coverage filtering
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

    interpolated_coords_latlon[:, 0] += COORD0_SHIFT
    interpolated_coords_latlon[:, 1] += COORD1_SHIFT

    return interpolated_coords_latlon, interpolated_time, time_dist_to_support


def calc_TDoA_dist(row, sensor_coords, shift, predicted_coord):
    """
    Calculate a time distance between predicted coords and receiving time
    :param row:
    :param sensor_coords:
    :param shift:
    :param predicted_coord:
    :return:
    """
    meas = json.loads(row['measurements'])

    if len(meas) < 2:
        return np.NaN

    receiver_coords = []
    receiver_time = []

    for m in meas:
        sensor_id, timestamp, power = m
        receiver_coords.append(sensor_coords[sensor_id])
        receiver_time.append(timestamp - shift[sensor_id] * 1e9)

    receiver_coords = np.array(receiver_coords)
    receiver_time = np.array(receiver_time) / 1e9
    receiver_time -= np.min(receiver_time)

    TDoA = mlat_3points._get_TDoA(receiver_coords, predicted_coord)
    d = np.linalg.norm(TDoA - receiver_time)
    return d
