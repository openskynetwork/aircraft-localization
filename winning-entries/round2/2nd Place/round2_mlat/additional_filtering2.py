import numpy as np
import pandas as pd

from libs import predict_filtering

if __name__ == '__main__':
    predict_df = pd.read_csv('round2_mlat/predicted_with_scores_keypoints.csv',low_memory=False)

    predict_df_filtered = predict_filtering.filter(predict_df.copy())

    res = predict_df_filtered[['id', 'latitude', 'longitude', 'geoAltitude']]
    res.to_csv('round2_mlat/predicted_filtered2.csv', index=False, float_format="%.10f", na_rep="NaN")