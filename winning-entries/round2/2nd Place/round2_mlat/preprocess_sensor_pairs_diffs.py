import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import gc
from argparse import ArgumentParser

from libs import data, preprocessing, mlat_3points

TEST_SIZE = 0.1

def preprocess_train_part(train_df, range_from, range_to, sensors_filter_line, sensor_coords):
    """
    Preprocess a part of the aircrafts ground-truth data
    :param train_df: aircraft's ground-truth data
    :param range_from: start position in file
    :param range_to: end position in file
    :param sensors_filter_line: bad sensor filtering infromation
    :param sensor_coords: sensor 3D coordinates
    :return:
    """
    df = train_df.iloc[range_from:range_to]

    if sensors_filter_line:
        df = preprocessing.parse_measurements_json_filtered(df, sensors_filter_line)
    else:
        df = preprocessing.parse_measurements_json(df)

    df = df[['timeAtServer', 'latitude', 'longitude', 'geoAltitude', 'sensor_ids', 'sensor_times']]
    gc.collect()

    res = dict()

    for _, row in tqdm(df.iterrows(), total=len(df), desc='Preprocess diffs'):
        receiver_ids = row['sensor_ids']
        receiver_time = row['sensor_times']
        if len(receiver_ids) < 2:
            continue

        lat = row['latitude']
        long = row['longitude']
        geoAltitude = row['geoAltitude']
        # filter wrong coords in aircraft ground truth data
        if (abs(lat) > 90) or (abs(long) > 180):
            continue

        receiver_ids = np.array(receiver_ids)
        receiver_time = np.array(receiver_time, dtype=np.float64) / 1e9
        order = np.argsort(receiver_ids)
        receiver_ids = receiver_ids[order]
        receiver_time = receiver_time[order]

        receiver_coords = np.array([sensor_coords[sensor_id] for sensor_id in receiver_ids], dtype=np.float64)
        TDoA = mlat_3points._get_TDoA(receiver_coords, (lat, long, geoAltitude), normalize=False)
        timeAtServer = row['timeAtServer']

        # preprocess data for all possible sensor pairs
        n = len(receiver_ids)
        for i in range(n):
            for j in range(i + 1, n):
                id1 = receiver_ids[i]
                id2 = receiver_ids[j]

                t1 = receiver_time[i]
                t2 = receiver_time[j]

                value = [t1, t2, timeAtServer, TDoA[i], TDoA[j]]
                key = (id1, id2)
                if key not in res:
                    res[key] = []
                res[key].append(value)

    for k, v in res.items():
        res[k] = np.array(v, dtype=np.float64)

    return res

def divide_on_chunks(n, num_chunks):
    chunk_size = (n + num_chunks - 1)//num_chunks
    res = []

    for i in range(num_chunks):
        a = i*chunk_size
        b = min(a + chunk_size, n)
        res.append((a,b))
    return res

def preprocess(validation=False, cut_level=None, deny_filtering=False):
    """
    Preprocess and save all time/distanse differences between actual aircraft position and received sensor time
    :param validation:
    :param cut_level:
    :param deny_filtering:
    :return:
    """
    # base sesnsor filtering
    sensor_coords = data.get_sensor_coords_cartesian()

    g_sensors_data = np.load('round2_mlat/good_sensors.npz', allow_pickle=True)
    good_sensors = g_sensors_data['good_sensors']
    sensors_filter_line = g_sensors_data['sensors_filter_line'][()] if 'sensors_filter_line' in g_sensors_data else None
    if deny_filtering:
        sensors_filter_line = None
    else:
        sensor_coords2 = dict()
        for id in good_sensors:
            if id in sensor_coords:
                sensor_coords2[id] = sensor_coords[id]

        sensor_coords = sensor_coords2

    # train data preprocessing
    all_df = data.get_test_dataset()
    train_df = preprocessing.get_train_part(all_df)

    # used for validation only
    if cut_level is not  None:
        train_df = train_df.head(cut_level)

    if validation:
        train_aircrafts, _ = preprocessing.divide_aircrafts(train_df, test_size=TEST_SIZE)
        train_df = preprocessing.filter_by_aircrafts(train_df, train_aircrafts)

    train_df = train_df[['timeAtServer', 'latitude', 'longitude', 'geoAltitude', 'measurements']]

    del all_df
    gc.collect()

    divide_cnt = 20
    n_jobs = 4
    chunks = divide_on_chunks(len(train_df), divide_cnt)
    print('start preprocessing')
    chunk_res = Parallel(n_jobs=n_jobs)(delayed(preprocess_train_part)
                                        (train_df, a, b, sensors_filter_line, sensor_coords) for a, b in chunks)
    # chunk_res = []
    # for a, b in chunks:
    #     chunk_res.append(preprocess_train_part(train_df, a, b, sensors_filter_line, sensor_coords))
    print('concat result')

    res = dict()
    for chunk in chunk_res:
        for k, v in chunk.items():
            if k not in res:
                res[k] = []
            res[k].append(v)

    for k, v in res.items():
        res[k] = np.concatenate(v)

    PREPROCESSED_FILE_NAME = 'round2_mlat/preprocessed_diffs'
    if validation:
        PREPROCESSED_FILE_NAME += "_v"
    if cut_level:
        PREPROCESSED_FILE_NAME += f"_{cut_level}"


    np.savez_compressed(PREPROCESSED_FILE_NAME, data=res)


if __name__ == '__main__':
    argparse = ArgumentParser()
    argparse.add_argument("--cut_level", type=int, default=None)
    argparse.add_argument("--deny_filtering", action='store_true')
    argparse.add_argument("--validation", action='store_true')
    args = argparse.parse_args()

    preprocess(args.validation, args.cut_level, args.deny_filtering)
