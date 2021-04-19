# Aircraft Localization Competetion : Round 2

# Filestructure
```
Repository
└───data
│ mlat_results_test.csv( Performed Multilateration on test set , after modeling sensor clock drift.)
│ round2_sample_empty.csv ( Empty sample submission)
└───src
│   └───data(data processing functions)
│       │   data_extraction.py (data extraction from Opensky Servers
│   └───helper(Utility functions)
│   └───main(main scripts here)
│       │   modelsensordrift.py(Models clock drift for each sensor)
│       │   multilaterationParTest.py (performs multilateration after clock correction on test data)
│       │   test.py (interpolation and trajectory predication)
```
# Usage

1. Download mlat_results_test.csv from ''' https://drive.google.com/file/d/1hsQ-ZfrvDHcs9fo1EesJ7xmj7mKGJwrp/view?usp=sharing '''
and store it in /data/folder
2. Change directory to /src/main
3. ```python test.py```
4. Result stored as result_submission_out.csv

# Method Highlights
1. Models clock drifts of sensors from training data
2. Performs Multilaterion on test data after clock correction
3. Interpolates between predictions to estimate trajectory and intermidiate points 


