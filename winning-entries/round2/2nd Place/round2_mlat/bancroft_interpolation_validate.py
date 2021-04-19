from argparse import  ArgumentParser
import numpy as np
import pandas as pd

from libs import data, preprocessing, metrics, bancroft_interpolation_solver
from round2_mlat.bancroft_prediction_routines import predict_all_aircrafts


def calculate_score_coverage(df, ids, coords):
    gt_df = df[df["id"].isin(ids)]
    lat_gt = gt_df['latitude'].values
    lon_gt = gt_df['longitude'].values

    rmse = metrics.rmse(lat_gt, lon_gt, coords[:, 0], coords[:, 1], half_threshold=False)
    rmse_90 = metrics.rmse_90_cut(lat_gt, lon_gt, coords[:, 0], coords[:, 1], half_threshold=False)

    coverage = np.sum(np.isfinite(coords[:, 0]))/coords.shape[0]

    print("RMSE 100%:", rmse)
    print("RMSE 90%:", rmse_90)
    print("Coverage:", coverage)


def validate(cut_level):
    sensor_coords = data.get_sensor_coords_cartesian()
    all_df = data.get_test_dataset()
    all_df = preprocessing.get_train_part(all_df)

    if cut_level is not None:
        all_df = all_df.head(int(cut_level))

    _, test_aircrafts = preprocessing.divide_aircrafts(all_df, 0.2)
    test_df = preprocessing.filter_by_aircrafts(all_df, test_aircrafts)

    preprocessing_filename = 'round2_mlat/preprocessed_diffs_v'
    if cut_level:
        preprocessing_filename += f"_{cut_level}"
    preprocessing_filename += ".npz"

    predicted_ids, predicted_coords, predicted_scores, keypoint_mask = predict_all_aircrafts(test_df, test_aircrafts, sensor_coords,
                                                                              preprocessing_filename=preprocessing_filename)

    res = pd.DataFrame({'id': predicted_ids,
                        'aircraft': test_df['aircraft'],
                        'timeAtServer': test_df['timeAtServer'],
                        'latitude': predicted_coords[:, 0],
                        'longitude': predicted_coords[:, 1],
                        'geoAltitude': predicted_coords[:, 2],
                        'score': predicted_scores,
                        'keypoint': keypoint_mask,
                        'gt_latitude': test_df['latitude'],
                        'gt_longitude': test_df['longitude']
                        })

    res.to_csv('round2_mlat/validation_predict.csv', index=False, float_format="%.10f", na_rep="NaN")

    calculate_score_coverage(test_df, predicted_ids, predicted_coords)

    print("---------Cut 70-----------")
    predicted_coords2 = bancroft_interpolation_solver.adaptive_filter_by_score(predicted_coords,
                                                                               predicted_scores, keep_part=0.7)
    calculate_score_coverage(test_df, predicted_ids, predicted_coords2)



if __name__ == '__main__':
    argparse = ArgumentParser()
    argparse.add_argument("--cut_level", default=500000)
    args = argparse.parse_args()

    validate(args.cut_level)