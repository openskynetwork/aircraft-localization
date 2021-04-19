from libs import data
from libs import preprocessing
import numpy as np
import pandas as pd

from round2_mlat.bancroft_prediction_routines import predict_all_aircrafts


if __name__ == '__main__':
    all_df = data.get_test_dataset().copy()
    sensor_coords = data.get_sensor_coords_cartesian()

    test_df = preprocessing.get_test_part(all_df)

    aircrafts = preprocessing.unique_aircrafts(test_df)

    predicted_ids, predicted_coords, predicted_scores, predicted_keypoint = predict_all_aircrafts(test_df, aircrafts, sensor_coords)

    raw = pd.DataFrame({'id': predicted_ids,
                               'latitude': predicted_coords[:, 0],
                               'longitude': predicted_coords[:, 1],
                               'geoAltitude': predicted_coords[:, 2],
                               'score': predicted_scores})
    raw = raw.sort_values(by=['id'], ignore_index=True)
    raw.to_csv('round2_mlat/predicted_with_scores.csv', index=False, float_format="%.10f", na_rep="NaN")
    raw[['id', 'latitude', 'longitude', 'geoAltitude']].to_csv('round2_mlat/predicted_not_filtered.csv', index=False, float_format="%.10f", na_rep="NaN")

    print('coverage:', 1 - np.sum(raw['latitude'].isna())/raw.shape[0])

    # save all predicted points with additional information for further filtering
    res = pd.DataFrame({'id': predicted_ids,
                        'aircraft': test_df['aircraft'],
                        'timeAtServer': test_df['timeAtServer'],
                        'latitude': predicted_coords[:, 0],
                        'longitude': predicted_coords[:, 1],
                        'geoAltitude': predicted_coords[:, 2],
                        'score': predicted_scores,
                        'keypoint': predicted_keypoint
                        })

    res = res.sort_values(by=['id'], ignore_index=True)
    res.to_csv('round2_mlat/predicted_with_scores_keypoints.csv', index=False, float_format="%.10f", na_rep="NaN")
