from libs import data
from libs import preprocessing
from libs import metrics
from libs import bancroft_interpolation_solver

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

def find_shifts(gt, predicted, mlat_time_interp, df):
    predicted = predicted.copy()
    rmse, dist_m = metrics.rmse(gt[:, 0], gt[:, 1],
                                predicted[:, 0], predicted[:, 1], False,
                                return_dist=True)

    dist_thr = np.percentile(dist_m[np.isfinite(dist_m)], 90)
    mask = dist_m < dist_thr
    mask = np.logical_and(mask, df['numMeasurements'] >= 4)
    predicted[np.logical_not(mask), :] = np.nan

    idx = np.where(np.isfinite(predicted[:, 0]))
    start_idx = idx[0][0]
    end_idx = idx[0][-1]

    start_time, end_time = mlat_time_interp[start_idx], mlat_time_interp[end_idx]
    start_p, end_p = predicted[start_idx], predicted[end_idx]
    vp = (end_p - start_p)/(end_time - start_time)

    pa = gt[np.isfinite(predicted[:, 0])]
    pb = predicted[np.isfinite(predicted[:, 0])]
    dp = pb - pa
    dp = np.mean(dp, axis=0)
    print('time shift dp', dp)
    print('time shift dp/vp', dp/vp)




all_df = data.get_test_dataset()
sensor_coords = data.get_sensor_coords_cartesian()

known_df = preprocessing.get_train_part(all_df)
test_df = preprocessing.get_test_part(all_df)

known_df = preprocessing.filter_unsuficcient_aircrafts_data(known_df)

known_aircrafts = preprocessing.unique_aircrafts(known_df)
print(known_aircrafts)

train_aircrafts, val_aicrafts = train_test_split(known_aircrafts, random_state=42, test_size=0.2)
# val_aicrafts = [767]
train_df = preprocessing.get_data_for_aircrafts(known_df, train_aircrafts)
val_df = preprocessing.get_data_for_aircrafts(known_df, val_aicrafts)
print(train_df.shape, val_df.shape)

gt_coords = []
predicted_coords = []

for aircraft in val_aicrafts:
    mlat_coord_latlong_interp, mlat_time_interp, scores = bancroft_interpolation_solver.solve_for_aircraft(val_df, aircraft, sensor_coords, )

    aircraft_df = preprocessing.get_data_for_aircrafts(val_df, [aircraft])


    if mlat_coord_latlong_interp is not None:
        # find_shifts(aircraft_df[['latitude', 'longitude']].values,
        #             mlat_coord_latlong_interp[:, 0:2], mlat_time_interp, aircraft_df)

        rmse, dist_m = metrics.rmse(aircraft_df['latitude'], aircraft_df['longitude'],
                            mlat_coord_latlong_interp[:, 0], mlat_coord_latlong_interp[:, 1], False, return_dist=True)
        rmse90 = metrics.rmse_90_cut(aircraft_df['latitude'], aircraft_df['longitude'],
                            mlat_coord_latlong_interp[:, 0], mlat_coord_latlong_interp[:, 1], False)

        lat_diff = aircraft_df['latitude'] - mlat_coord_latlong_interp[:, 0]
        long_diff = aircraft_df['longitude'] - mlat_coord_latlong_interp[:, 1]
        lat_diff = lat_diff[np.isfinite(lat_diff)]
        long_diff = long_diff[np.isfinite(long_diff)]

        print('lat diff mean', np.mean(lat_diff) )
        print('long diff mean', np.mean(long_diff) )

        gt_coords.append(aircraft_df[['latitude', 'longitude', 'geoAltitude']].values)
        predicted_coords.append(mlat_coord_latlong_interp)
    else:
        rmse = None
        rmse90 = None

    if True:
        f, ax = plt.subplots(nrows=3, ncols=2, sharex=True)

        ax[0, 0].plot(aircraft_df['timeAtServer'], aircraft_df['latitude'], "g", label="lat_true")
        if mlat_time_interp is not None:
            ax[0, 0].plot(mlat_time_interp, mlat_coord_latlong_interp[:, 0], "r", label="lat_pred")
            ax[0, 1].plot(mlat_time_interp, mlat_coord_latlong_interp[:, 0] - aircraft_df['latitude'], "r+", label="lat_pred")
        ax[0, 0].set_title(f"Lattitude")
        # plt.show()

        ax[1, 0].plot(aircraft_df['timeAtServer'], aircraft_df['longitude'], "g", label="lon_true")
        if mlat_time_interp is not None:
            ax[1, 0].plot(mlat_time_interp, mlat_coord_latlong_interp[:, 1], "r", label="lon_pred")
            ax[1, 1].plot(mlat_time_interp, mlat_coord_latlong_interp[:, 1] - aircraft_df['longitude'], "r+", label="lon_pred")
        ax[1, 0].set_title(f"Longitude")
        # plt.show()

        ax[2, 0].plot(aircraft_df['timeAtServer'], aircraft_df['geoAltitude'], "g", label="h_true")
        if mlat_time_interp is not None:
            ax[2, 0].plot(mlat_time_interp, mlat_coord_latlong_interp[:, 2], "r", label="h_pred")
            ax[2, 1].plot(mlat_time_interp, dist_m, "r+", label="dist")

        ax[2, 0].set_title(f"Height")

        plt.suptitle(f"Aircraft {aircraft}. rmse={rmse} {rmse90}")
        plt.show()

gt_coords = np.concatenate(gt_coords)
predicted_coords = np.concatenate(predicted_coords)

rmse = metrics.rmse(gt_coords[:, 0], gt_coords[:, 1],
                    predicted_coords[:, 0], predicted_coords[:, 1], False)

rmse90 = metrics.rmse_90_cut(gt_coords[:, 0], gt_coords[:, 1],
                    predicted_coords[:, 0], predicted_coords[:, 1], False)

print('coverage:', 1 - np.sum(np.isnan(predicted_coords[:, 0]))/predicted_coords.shape[0])

print('full rmse', rmse)
print('full rmse90', rmse90)

