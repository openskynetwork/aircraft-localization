# CYD Campus Aircraft Localization Competition
## 2nd round solution of ck.ua team (2nd(?) place)

###Authors
* Roman Chernenko
* Vitaly Bondar


###Instructions for running

* put `round2_competition.csv` and `round2_sensors.csv` files to the `data/test` folder
* create virtual environment with the conda mechanism and install packages from 
`environment.yml`. You can perform it with next command `conda env create -f environment.yml`
* activate new environment (`conda activate round2_final`)


* Optionally recalculate sensor points filtering
    * run `python round2_mlat/find_good_sensors_and_filter.py`
* Recalculate sensors pairs diffs (need to run only once)
    * delete file `round2_mlat/preprocessed_diffs.npz` if exists
    * run `python round2_mlat/preprocess_sensor_pairs_diffs.py`
* run `python round2_mlat/bancroft_interpolation_predict.py`
* run `python round2_mlat/additional_filtering2.py`
* you will find predicted result in `round2_mlat/predicted_filtered2.csv`

Alternatively, you can run end-to-end process with one script `run.sh`

###Algorithm overview

#### Core idea - LASS (Local Adaptive Sensor Synchronization)

The main problem of the challenge that all sensors are not synchronized in time. We have
only 'timeAtServer' feature with approx. accuracy +/- 1 second that obliviously not 
enough for multilateration approach. A lot attempts of global sensor time synchnonization
did not bring any significant results.

So instead of global sensors synchronization, as we did it on first round, 
we decided to synchronize sensors for each unknown point independently. We noticed that
time shift of each sensor at some small-time window is  possible to describe with 
linear relationship. A corrected time for each sensor looks next:

t'<sub>i</sub> = t<sub>i</sub> + shift<sub>i</sub>(t<sub>i</sub>)

shift<sub>i</sub>(t) = k<sub>i</sub>t + b<sub>i</sub>

where:

t<sub>i</sub> - original Time of Arrival (ToA) of i-th sensor, in seconds

t'<sub>i</sub> - synchronized ToA of i-th sensor, in seconds

shift<sub>i</sub>(t) - time shift of i-th sensor 

k<sub>i</sub>, b<sub>i</sub> - time shifting factor and bias

All k, b parameters of the model was calculated independently for each unknown 
aircraft point using least squares method based on known aircraft positions that relatively
closed in time. 

More precisely local shift parameters k and b calculation procedure is next:
- With preprocessed kdtree find closest N points in the time space (t<sub>1</sub>, t<sub>2</sub>), 
  where t<sub>1</sub> - reported time of the sensor<sub>1</sub>,  t<sub>2</sub> - reported time of the sensor<sub>2</sub>
- Find optimal k,b by least squares method using time shifts in the found N cases 

You can find more details about the approach in `libs/SensorSynchronization.py` file.

 


#### Preparation stage

* **Sensor points filtering:** For each sensor we find some constant that fit `sensorTime/1e9-timeAtServer` best. 
  Later with this constant we can filter most obvious outliers in the sensor time points.
* **Sensors pairs diff precalculation:** For the quick access in the future we precalculate 
  and save for each dataset point next mapping: 
  pair of sensors -> Time error between 2 measured times difference and signal travel based time difference. 
  This mapping collected separately for each pair of sensors with the additional information of sensors 
  internal time. This data structure used for quick implementation of LASS method.


#### Localization calculation and filtering

Main part of prediction is made per aircraft track and consists of next calculations:

  * With Bancroft method all points with the 4 or more sensors data are calculated in 3D cartesian
    coordinates. Important that sensor time synchronized with the LASS method.
  
  * Predicted coordinates reverted to latitude/longitude/height coordinates
  
  * Outliers in latitude and longitude are filtered with the median filter

  * Points filtered out with the speed limit.
    * Calculate median among calculated speeds of passing distance between predicted neighbour points
    * With the dynamic programming find flight path with the 1.15*median speed limit and usage of maximum 
      amount of points
    * Filter out points that was unused in the found path 
  
  * Points with the less than 4 measurements filled with the linear interpolation
  
  * Smooth aircraft track with low-pass filter
  
  * Predicted points are filtered with the special scoring function to exactly match the minimum 
    requirement of the coverage. The score function take in account 2 values: 
    distance to the closest mlat-calculated point (smaller is better) and 
    distance between 2 neighbored mlat-calculated points (smaller is better).

See more details in `bancroft_interpolation_predict.py` and `additional_filtering2.py`
