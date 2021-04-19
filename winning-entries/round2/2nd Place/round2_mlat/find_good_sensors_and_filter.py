import numpy as np
from tqdm import tqdm
from sklearn.linear_model import RANSACRegressor
from sklearn.dummy import DummyRegressor
import os

from libs import data, preprocessing

def calc_ransac_inliers(x, y, threshold=0.1):
    ransac = RANSACRegressor(base_estimator=DummyRegressor(), min_samples=5)
    ransac.fit(x.reshape(-1, 1), y)
    return np.abs(y-ransac.estimator_.constant_[0][0]) < threshold, ransac.estimator_.constant_[0][0]


if __name__ == '__main__':
    sensor_coords = data.get_sensor_coords()
    sensor_coords_latlon = data.get_sensor_coords_cartesian()

    all_df = data.get_test_dataset()
    print("Read train part...")
    train_df = preprocessing.get_train_part(all_df)[0:1000000]
    print("Parse measurements...")
    df = preprocessing.parse_measurements_json(train_df)

    sensor_meas = dict()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Sensors data structure"):
        meas_id = row['id']
        ids = row['sensor_ids']
        times = row['sensor_times']
        for id, t in zip(ids, times):
            if id not in sensor_meas:
                sensor_meas[id] = {'x':[], 'y':[], 'meas_id': []}
            sensor_meas[id]['x'].append(row['timeAtServer'])
            sensor_meas[id]['y'].append(t)
            sensor_meas[id]['meas_id'].append(meas_id)

    good_sensors = []
    sensors_filter_line = {}

    for id, meas in tqdm(sensor_meas.items(), desc="Checking sensors"):
        x = np.array(meas['x'])
        y = np.array(meas['y'])
        m_ids = np.array(meas['meas_id'])

        y2 = y/1e9 - x
        if len(x) > 5:
            inliers, filter_constant = calc_ransac_inliers(x, y2, threshold=0.1)

            if np.sum(inliers) > 5:
                good_sensors.append(id)
                sensors_filter_line[id] = filter_constant


    print('tot sensors', len(sensor_meas), 'good sensors', len(good_sensors))
    good_sensors = np.array(good_sensors)
    if not os.path.exists('round2_mlat/good_sensors.npz'):
        np.savez_compressed('round2_mlat/good_sensors.npz', good_sensors=good_sensors, sensors_filter_line=sensors_filter_line)
