from libs import preprocessing
from libs import bancroft_interpolation_solver
from libs import SensorSynchronization
from libs import metrics

import numpy as np
from tqdm import tqdm

def calc_aircraft(aircraft, test_df, sensor_coords, silent=True, sensors_filter_line = None, sensor_synch=None):
    """
    Calculate a track for one aircraft
    :param aircraft: aircraft id
    :param sensor_synch: sensor synchronization object
    :return: all know coordinates for aircraft
    """
    df = test_df.copy()
    mlat_coord_latlong_interp, mlat_time_interp, scores, keypoint_mask = \
        bancroft_interpolation_solver.solve_for_aircraft(df, aircraft, sensor_coords, filter_byTdoA=True,
                                                         silent=silent, sensors_filter_line=sensors_filter_line, sensor_synch=sensor_synch)

    # print('result for aircraft', aircraft, '-', mlat_coord_latlong_interp.shape)

    aircraft_df = preprocessing.get_data_for_aircrafts(df, [aircraft])

    # show prediction charts
    if False:
        key_points = np.abs(scores) < 1e-6

        if mlat_time_interp is not None:
            d_lat = aircraft_df['latitude'] - mlat_coord_latlong_interp[:, 0]
            d_long = aircraft_df['longitude'] - mlat_coord_latlong_interp[:, 1]
            kp_coords = mlat_coord_latlong_interp[key_points]
            kp_time = mlat_time_interp[key_points]

        if np.sum(key_points) >= 2:
            track_vector = kp_coords[-1] - kp_coords[0]
            print('track vector:', track_vector[0:2], 'mean lat/long shift:', [np.mean(d_lat[key_points]) * 111e3, np.mean(d_long[key_points]) * 111e3])

        f, ax = plt.subplots(nrows=3, sharex=True, ncols=2)

        # ax[0].plot(aircraft_df['timeAtServer'], aircraft_df['latitude'], "g", label="lat_true")
        if mlat_time_interp is not None:
            ax[0, 0].plot(mlat_time_interp, mlat_coord_latlong_interp[:, 0], "r", label="lat_pred")
            ax[0, 0].plot(mlat_time_interp, aircraft_df['latitude'], "g", label="lat_pred")
            ax[0, 0].scatter(mlat_time_interp[key_points], mlat_coord_latlong_interp[:, 0][key_points])
            ax[0, 1].plot(mlat_time_interp, d_lat, "r", label="lat_pred")
            ax[0, 1].scatter(mlat_time_interp[key_points], d_lat[key_points])
        ax[0, 0].set_title(f"Lattitude")
        # plt.show()

        # ax[1].plot(aircraft_df['timeAtServer'], aircraft_df['longitude'], "g", label="lon_true")
        if mlat_time_interp is not None:
            ax[1, 0].plot(mlat_time_interp, mlat_coord_latlong_interp[:, 1], "r", label="lon_pred")
            ax[1, 0].plot(mlat_time_interp, aircraft_df['longitude'], "g", label="lon_pred")
            ax[1, 0].scatter(mlat_time_interp[key_points], mlat_coord_latlong_interp[:, 1][key_points])
            ax[1, 1].plot(mlat_time_interp, d_long, "r", label="long_pred")
            ax[1, 1].scatter(mlat_time_interp[key_points], d_long[key_points])
        ax[1, 0].set_title(f"Longitude")
        # plt.show()
        score, dist = metrics.rmse(aircraft_df['latitude'], aircraft_df['longitude'], mlat_coord_latlong_interp[:, 0], mlat_coord_latlong_interp[:, 1], half_threshold = False, return_dist=True)

        # ax[2].plot(aircraft_df['timeAtServer'], aircraft_df['geoAltitude'], "g", label="h_true")
        if mlat_time_interp is not None:
            ax[2, 0].plot(mlat_time_interp, mlat_coord_latlong_interp[:, 2], "r", label="h_pred")
            ax[2, 0].plot(mlat_time_interp, aircraft_df['baroAltitude'], "g", label="h_baro")
            ax[2, 1].plot(mlat_time_interp, dist, "r", label="h_pred")
        ax[2, 0].set_title(f"Height")
        ax[2, 1].set_title(f"RMSE {score}")

        plt.suptitle(f"Aircraft {aircraft}.")



        plt.show()

    return aircraft_df['id'].values, mlat_coord_latlong_interp, scores, keypoint_mask


def predict_all_aircrafts(test_df, aircrafts, sensor_coords, preprocessing_filename=None):
    """
    Locate all aircraft positions
    :param test_df: test data
    :param aircrafts: aircraft ids
    :param sensor_coords: 3D sensor coords
    :param preprocessing_filename: preprocessed file with sensor time shift data for all known sensor pairs
    :return:
    """

    # load preprocessed sensor pairs data
    ss = SensorSynchronization.SensorSynchronization(preprocessing_filename=preprocessing_filename)

    result = []
    for aircraft in tqdm(aircrafts):
        result.append(
            calc_aircraft(aircraft, test_df, sensor_coords, sensors_filter_line=None, sensor_synch=ss))

    predicted_ids = []
    predicted_coords = []
    predicted_scores = []
    predicted_keypoint_mask = []
    for r in result:
        predicted_ids.append(r[0])
        predicted_coords.append(r[1])
        predicted_scores.append(np.array(r[2]))
        predicted_keypoint_mask.append(np.array(r[3]))

    predicted_ids = np.concatenate(predicted_ids)
    predicted_coords = np.concatenate(predicted_coords)
    predicted_scores = np.concatenate(predicted_scores)
    predicted_keypoint_mask = np.concatenate(predicted_keypoint_mask)

    sort_pos = np.argsort(predicted_ids)
    predicted_ids = predicted_ids[sort_pos]
    predicted_coords = predicted_coords[sort_pos]
    predicted_scores = predicted_scores[sort_pos]
    predicted_keypoint_mask = predicted_keypoint_mask[sort_pos]

    return predicted_ids, predicted_coords, predicted_scores, predicted_keypoint_mask