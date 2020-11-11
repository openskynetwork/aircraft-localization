import numpy as np
import pandas as pd
from tqdm import tqdm
import math
import json
from libs import geo


def get_train_part(df):
    return df[np.logical_not(np.isnan(df['latitude']))]

def get_test_part(df):
    return df[np.isnan(df['latitude'])]

def unique_aircrafts(df):
    return np.unique(df['aircraft'])

def get_data_for_aircrafts(df, aircrafts):
    return df[df['aircraft'].isin(aircrafts)]

def filter_unsuficcient_aircrafts_data(df, min_points_thr=200):
    cnt = df[['aircraft', 'measurements']].groupby('aircraft').count()
    mask = cnt['measurements'] > min_points_thr
    filtered_aircrafts = cnt[mask].index

    filtered_data = get_data_for_aircrafts(df, filtered_aircrafts)
    return filtered_data

def find_bad_sensors(df, sensor_coords):
    dists = dict()
    for _, row in tqdm(df.iterrows(), total=len(df)):
        geoAltitude = row['geoAltitude']
        meas = json.loads(row['measurements'])

        lat = row['latitude']
        long = row['longitude']
        xyz_gt = geo.latlon_to_cartesian(lat, long, geoAltitude)

        receiver_coords = []
        receiver_time = []

        for m in meas:
            sensor_id, timestamp, power = m
            receiver_coords.append(sensor_coords[sensor_id])
            receiver_time.append(timestamp)

            d = math.sqrt(np.sum(np.square(xyz_gt - sensor_coords[sensor_id])))
            if sensor_id not in dists:
                dists[sensor_id] = []
            dists[sensor_id].append(d)

    bad_sensors = []
    for k, v in dists.items():
        if len(v) < 100:
            bad_sensors.append(k)
        elif np.min(v) > 25000:
            bad_sensors.append(k)

    return bad_sensors

def parse_measurements(all_df):
    aircrafts = unique_aircrafts(all_df)

    res = []

    for aircraft_id in tqdm(aircrafts, desc='aircraft'):
        df = get_data_for_aircrafts(all_df, [aircraft_id]).copy(deep=True)

        col_receiver_time = []
        col_receiver_power = []
        col_sensor_ids = []

        # for _, row in tqdm(df.iterrows(), total=len(df)):
        for _, row in df.iterrows():
            meas = json.loads(row['measurements'])

            receiver_time = []
            receiver_power = []
            sensor_ids = []

            for m in meas:
                sensor_id, timestamp, power = m
                receiver_time.append(timestamp / 1e9)
                receiver_power.append(power)
                sensor_ids.append(sensor_id)

            receiver_time = np.array(receiver_time)
            receiver_power = np.array(receiver_power)
            sensor_ids = np.array(sensor_ids)

            col_receiver_time.append(receiver_time)
            col_receiver_power.append(receiver_power)
            col_sensor_ids.append(sensor_ids)

        time_diff_thr = 0.01
        for i in range(len(col_receiver_time)):
            d = col_receiver_time[i].max() - col_receiver_time[i].min()
            if d >= time_diff_thr:
                filtered_mask = []

                k = col_receiver_time[i]

                for j in range(len(col_receiver_time[i])):
                    curr_t = k[j]
                    curr_min = np.delete(k, j).min()

                    filtered_mask.append(abs(curr_min - curr_t) < time_diff_thr)

                filtered_mask = np.array(filtered_mask, dtype=np.bool)

                col_receiver_time[i] = col_receiver_time[i][filtered_mask]
                col_receiver_power[i] = col_receiver_power[i][filtered_mask]
                col_sensor_ids[i] = col_sensor_ids[i][filtered_mask]

        measurements = []
        numMeasurements = []
        for i in range(len(col_receiver_time)):
            new_meas = []
            for j in range(len(col_receiver_time[i])):
                new_meas.append([int(col_sensor_ids[i][j]), int(col_receiver_time[i][j]*1e9), int(col_receiver_power[i][j])])
            measurements.append(str(new_meas))
            numMeasurements.append(len(new_meas))

        df['measurements'] = measurements
        df['numMeasurements'] = numMeasurements
        res.append(df)

    res = pd.concat(res)
    res = res.sort_values(by='id')
    return res