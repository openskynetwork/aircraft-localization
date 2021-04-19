# AC Localization competition

Scripts to create submission for the [Aircraft localization competition](https://www.aicrowd.com/challenges/cyd-campus-aircraft-localization-competition) 

## Filesystem

***data:***
- **res_test_MLAT3_001.pckl** : Localized points from Multilateration on competetion dataset
- **sensor_meta_filtered.pckl**: Metadata about good sensor combinations extracted from training data.
- **round1_competition.csv** : Competetion dataset (Given by organizers)
- **sensors.csv** : Metadata about sensors (Given by organizers)
- **round1_sample_empty.csv** : Sample Submission (Given by organizers)

***src:*** (Contains all the scripts)
- **helper:**
   - **MLAT3_1.py** : Functions to perform Multilateration
   utils.py : utility functions

- **main:**
   - **MLAT_test.py**: Run this to perform multilateration on test data. Will obtain res_test_MLAT3_001.pckl
   - **full_prediction_pipeline.py**: Run this to perform trajectory estimation and generate result submission
 
***results:***
   - **entry_0.csv** : Result submission 
 		 **RMSE ~ 59 meters, Coverage : 0.509**

## Usage
- Copy round1_competition.csv, sensors.csv, round1_sample_empty.csv into ~/data folder
- Set working directory to root of this filesystem for all scripts
- Run MLAT_test.py to perform multilateration and obtain res_test_MLAT3_001.pckl
- Run full_prediction_pipeline.py to create final submission

## Requirements

- numpy
- scipy
- pandas
- sklearn
- [pwlf](https://pypi.org/project/pwlf/)


## Author
Param Uttarwar : dataWizard