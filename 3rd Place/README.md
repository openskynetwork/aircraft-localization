# CYD Campus Aircraft Localization Competition
## 1st round solution of ck.ua team (3rd place)

####Authors
* Roman Chernenko
* Vitaly Bondar


####Instructions for running

* put `round1_competition.csv` and `sensors.csv` files to the `data/test` folder
* create virtual environment with the virtualenv/pip or conda mechanism and install packages from `requirements.txt`
* Optionally recalculate time shifts of sensors
    * delete file `shift_coefs.npz` for recalculating time shifts of sensors
    * delete file `shift_coefs_diffs.npz` for recollecting statistics of sensor pairs time lags
    * run `python calc_sensor_shift2.py`
* run `python bancroft_interpolation_predict.py`
* run `python additional_filtering.py`
* take predicted result in `submission_filtered.csv` if you use `additional_filtering.py` 
or in `submission_bancroft_interp_nans_smooth_butter.csv` otherwise.

####Algorithm overview

  First we calculate time shifts in seconds between sensors. This shifts may have different nature: ping difference, 
informing time of start/end of ADB-S message.
For this we found most frequent time error between calculated and observed for each pair of sensors. 
Then this time errors combined to form system of equations and all together solved with the least squared method 
(see more details in `calc_sensor_shift2.py`).
As the result we got some time delta for each sensor, and later this time shifts added to the each measurement timestamp.

Main part of prediction is made per aircraft track and consists of next calculations:

  * With Bancroft method all points with the 4 or more sensors data are calculated in 3D cartesian coordinates
  
  * Predicted coordinates reverted to latitude/longitude/height coordinates
  
  * Outliers in latitude and longitude are filtered with the median filter
  
  * Points with 3 or more measurements improved by Hooke-Jeeves method
  
  * Points with the less than 3 measurements filled with the linear interpolation
  
  * Whole aircraft track fitted with the piecewise linear function
  
  * Points with the big difference between measured and predicted TDoA are filtered out
  
  * Predicted single points without neighbours filtered out
  
  * Constant coordinates shift added
See more details in `bancroft_interpolation_predict.py` and `additional_filtering.py`
  
It is worth to note that `additional_filtering.py` procedures is fully experimental and was dirty changed many times in 
last night of competition. So it can be unuseful or even can have negative influence.
