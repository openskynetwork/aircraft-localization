import numpy as np
import pandas as pd
from libs import geo
import os

TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'

def get_sensors(fixed=False):
    if fixed:
        return pd.read_csv(f'{TEST_DIR}/sensors_fixed.csv')
    return pd.read_csv(f'{TEST_DIR}/sensors.csv')

def get_sensor_coords_dataframe():
    sensors = get_sensors()
    return sensors

def get_sensor_coords(fixed=False):
    sensors = get_sensors(fixed=fixed)
    id = sensors['serial'].values
    coords = sensors[['latitude', 'longitude', 'height']].values
    return dict(zip(id, coords))

def get_sensor_coords_cartesian(fixed=False):
    sensors = get_sensors(fixed=fixed)
    id = sensors['serial'].values
    coords = sensors[['latitude', 'longitude', 'height']].values
    coords_cartesian = [geo.latlon_to_cartesian(*c) for c in coords]
    return dict(zip(id, coords_cartesian))

def get_test_dataset():
    return pd.read_csv(f'{TEST_DIR}/round1_competition.csv')

def get_train_dataset(filename):
    return pd.read_csv(filename)

def get_train_sensor_coords(filename):
    sensors = pd.read_csv(filename)
    id = sensors['serial'].values
    coords = sensors[['latitude', 'longitude', 'height']].values
    return dict(zip(id, coords))

def get_train_sensor_coords_cartesian(filename):
    sensors = pd.read_csv(filename)
    id = sensors['serial'].values
    coords = sensors[['latitude', 'longitude', 'height']].values
    coords_cartesian = [geo.latlon_to_cartesian(*c) for c in coords]
    return dict(zip(id, coords_cartesian))

def get_time_shifts():
    return np.load("shift_coefs.npz")["shift"]

def get_train_files():
    res = []
    for f in sorted(list(os.listdir(TRAIN_DIR))):
        if f.startswith('training_') and f.endswith('category_1'):
            base_path = TRAIN_DIR + '/' + f
            info = {'train': base_path + '/' + f + '.csv',
                    'sensors': base_path + '/' + 'sensors.csv',
                    'ground_truth': base_path + '_result/' + f + '_result.csv'
                    }
            res.append(info)

    return res

    