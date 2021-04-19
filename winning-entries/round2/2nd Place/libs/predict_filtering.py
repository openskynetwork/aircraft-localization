import numpy as np
import pandas as pd
import math

from libs import preprocessing
from libs import bancroft_interpolation_solver


def filter(df):
    aircrafts = preprocessing.unique_aircrafts(df)

    scores = []
    ids = []
    for aircraft_id in aircrafts:
        aircraft_df = preprocessing.get_data_for_aircrafts(df, [aircraft_id])
        score = _calc_pred_score(aircraft_df)

        ids.append(aircraft_df['id'].values)
        scores.append(score)

    ids = np.concatenate(ids)
    scores = np.concatenate(scores)

    scores = scores[np.argsort(ids)]

    predicted_ids = df['id'].values
    predicted_coords = df[['latitude', 'longitude', 'geoAltitude']].values

    print('coverage before filtering:', 1 - np.sum(np.isnan(predicted_coords[:, 0])) / predicted_coords.shape[0])

    predicted_coords = bancroft_interpolation_solver.adaptive_filter_by_score(predicted_coords, scores, keep_part=0.7)
    print('coverage filter by score 0.7:', 1 - np.sum(np.isnan(predicted_coords[:, 0])) / predicted_coords.shape[0])

    df['latitude'] = predicted_coords[:, 0]
    df['longitude'] = predicted_coords[:, 1]
    df['geoAltitude'] = predicted_coords[:, 2]

    return df

def _calc_pred_score_old(df):
    return df['score'].astype(np.float32).values


def _calc_pred_score(df):
    keypoint = df['keypoint'].values
    coords = df[['latitude', 'longitude', 'geoAltitude']].values
    score = np.zeros_like(keypoint, dtype=np.float32)
    timeAtServer = df['timeAtServer'].values
    n = len(df)

    if np.sum(keypoint) >= 2:
        closest_left_time = np.zeros(n, dtype=np.float32)
        closest_right_time = np.zeros(n, dtype=np.float32)

        keypoint_time = []
        keypoint_coords = []

        ct = np.nan
        for i in range(n):
            if keypoint[i]:
                ct = timeAtServer[i]
                keypoint_time.append(ct)
                keypoint_coords.append(coords[i])
            closest_left_time[i] = ct

        ct = np.nan
        for i in range(n-1, -1, -1):
            if keypoint[i]:
                ct = timeAtServer[i]
            closest_right_time[i] = ct

        dist_between_keypoints = closest_right_time - closest_left_time

    pos = -1
    for i in range(n):
        if not np.isfinite(coords[i, 0]):
            score[i] = 1e100
            continue

        if keypoint[i]:
            score[i] = 0
            pos += 1
        else:
            left_angle = calc_angle(keypoint_time, keypoint_coords, pos)
            right_angle = calc_angle(keypoint_time, keypoint_coords, pos+1)
            angle = max(left_angle, right_angle)

            d = min(timeAtServer[i] - closest_left_time[i], closest_right_time[i] - timeAtServer[i])
            score[i] = (dist_between_keypoints[i] + d)*angle
#            score[i] = angle

    # if np.sum(keypoint) > 1:
    #     f, ax = plt.subplots(1, 4, sharex=False)
    #     ax[0].plot(keypoint_time, np.array(keypoint_coords)[:, 0])
    #     ax[0].scatter(keypoint_time, np.array(keypoint_coords)[:, 0])
    #     ax[1].plot(keypoint_time, np.array(keypoint_coords)[:, 1])
    #     ax[1].scatter(keypoint_time, np.array(keypoint_coords)[:, 1])
    #     ax[2].plot(timeAtServer[score<1e99], score[score<1e99])
    #     ax[3].plot(np.array(keypoint_coords)[:, 0], np.array(keypoint_coords)[:, 1])
    #
    #     plt.show()

    return score

def calc_angle(keypoint_time, keypoint_coords, pos):
    if pos >= len(keypoint_time):
        return 0

    pos_left = pos - 1
    while (pos_left >= 0) and (abs(keypoint_time[pos_left] - keypoint_time[pos])<1e-3):
        pos_left -= 1

    pos_right = pos + 1
    while (pos_right < len(keypoint_time)) and (abs(keypoint_time[pos_right] - keypoint_time[pos]) < 1e-3):
        pos_right += 1

    if (pos_left >= 0) and (pos_right < len(keypoint_time)):
        # a1 = math.atan2(keypoint_coords[pos][0] - keypoint_coords[pos_left][0], keypoint_coords[pos][1] - keypoint_coords[pos_left][1])
        # a2 = math.atan2(keypoint_coords[pos_right][0] - keypoint_coords[pos][0],
        #                 keypoint_coords[pos_right][1] - keypoint_coords[pos][1])
        #
        # da = a1 - a2
        # return abs(da)

        d_left = (keypoint_coords[pos_left] - keypoint_coords[pos])/(keypoint_time[pos_left] - keypoint_time[pos])
        d_right = (keypoint_coords[pos] - keypoint_coords[pos_right]) / (keypoint_time[pos] - keypoint_time[pos_right])
        dd = d_left - d_right
        dd = np.abs(dd)
        return dd[0] + dd[1]
    else:
        return 0
