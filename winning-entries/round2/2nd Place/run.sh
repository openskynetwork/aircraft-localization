#!/bin/sh

export PYTHONPATH=$PYTHONPATH:.
echo "Data filtering"
python round2_mlat/find_good_sensors_and_filter.py

echo "Preprocessing"
python round2_mlat/preprocess_sensor_pairs_diffs.py

echo "Multilateration"
python round2_mlat/bancroft_interpolation_predict.py

echo "Submission filtering"
python round2_mlat/additional_filtering2.py

