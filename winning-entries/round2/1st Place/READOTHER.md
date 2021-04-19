# 1. Setup

#### To setup python environment:
```
>> conda env create --file environment.yml
```
#### To run notebooks:
```
>> conda activate gt
```
```
(gt) >> jupyter notebook
```

Some functions use Cython code, which needs to be compiled. Just run the `1. Synchronize good stations.ipynb` notebook first. Cython functions use 4 CPU cores by default, adjust `N_THREADS` variable in `src/optimize.pyx` file.  

#### Data sets

All data files should be in ./data folder. Only three files were used in this solution:
 - round2_competition.csv
 - round2_sensors.csv
 - round2_training1.csv


# /src files description

 | Filename     |  Description  |
 |--------------|---------------|
 | filters.py   | median and graph-based speed limit filters |
 | geo.py       | functions to calculate distance, effective wave velocity; to transform coordinates and to plot altitude profiles |
 | optimize.pyx | collection of cython functions to solve multelateral equations efficiently |
 | solvers.py   | GoodStationsSolver and SingleStationSolver classes |
 | stations.py  | Stations class including time correction method |
 | track.py     | Track and TrackCollection classes to work with tracks |

# /results files description

 | Filename     |  Description  |
 |--------------|---------------|
 | predictions_test.csv | Test track predictions created using predict_track function |
 | stations_params_final.json | Synchronized stations parameters used to make my submissions |
 | submission_78m.csv | The best submission on the public leaderboard showing about 78m accuracy |


# 2. Theory
## 2.1 Wave velocity model

In round 1 of the competition many participants used effective wave velocity instead of speed of light to estimate distance by time-of-flight. @richardalligier found its value using optimization technique. In round 2 I improved this model by introducing altitude dependence of wave velocity. 

Using altitude dependence of refractive index $n(h)$ from [1], velocity as a function of altitude can be written as follows: 

$$v(h) = \frac{c}{n(h)} = \frac{c}{1 + A_0\cdot e^{-B\cdot h}}$$

, where $c$ is the speed of light, $h$ - altitude, $n(h)$ - refractive index, $A_0$ and $B$ - some constants.

Instead of integrating velocity each time, let's consider some constant (average) effective velocity: 
$$\hat{v} = const = \frac{L}{\int{dt(h)}}$$

$$\int{dt(h)} = \int_0^L{\frac{1+A_0\cdot e^{-B\cdot l\cdot \sin{\phi}}}{c}\cdot dl} = \int_{h_1}^{h_2}{\frac{1+A_0\cdot e^{-B\cdot h}}{c\cdot \sin{\phi}}\cdot dh} = \frac{h_2 - h_1}{c\cdot \sin{\phi}} + \frac{A_0}{c\cdot B\cdot \sin{\phi}}\cdot(e^{-B\cdot h_1} - e^{-B\cdot h_2})$$

, where $L$ - wave path, $h_1$ and $h_2$ - initial and final altitudes of the wave path.

Finally, after inserting $L = \frac{h_2 - h_1}{\sin{\phi}}$, effective wave velocity will be:

$$(Eq. 1): \hat{v} = \frac{c}{1+\frac{A_0}{B\cdot (h_2 - h_1)}(e^{-B\cdot h_1}-e^{-B\cdot h_2})}, h_2 > h_1$$

New wave velocity model shows 0.1m less average residual error in solving multilateral equations for 35 good stations (`1. Synchronize good stations.ipynb` notebook) than altitude independent constant effective wave velocity.

[1] R. Purvinskis et al. Multiple Wavelength Free-Space Laser Communications. Proceedings of SPIE - The International Society for Optical Engineering, 2003. 

## 2.2 Stations clock drift

Stations are synchronized when there is no clock drift, so measured time is equal to aircraft time + time-of-flight:

$$t^{meas} = t^{aircraft} + \frac{L}{\hat{v}}$$

If station measurements have drift, then:

$$t^{meas} = t^{aircraft} + \frac{L}{\hat{v}} + drift(t^{aircraft} + \frac{L}{\hat{v}})$$

It's worth to notice here that drift is added at the moment of signal detection!

We have to have some already synchronized stations. Let's consider a synchronized station 1 and a drifted station 2.

$$drift(t_2) = t_2^{meas} - t_2^{aircraft} - \frac{L_2}{\hat{v}}$$

Considering $t_2^{aircraft} \triangleq t_1^{aircraft}$ and inserting corresponding equation for station 1, we get the final formula:

$$(Eq. 2): drift(t_2) = t_2^{meas} - (t_1^{meas} - \frac{L_1}{\hat{v}}) - \frac{L_2}{\hat{v}}$$

### Clock drift approximation

Clock drift consists of linear drift and random walk. Clock drift was approximated by a sum of a linear function and a spline:

$$drift(t) = A\cdot t + B + spline(t)$$

So,

$$t^{meas} = t^{aircraft} + \frac{L}{\hat{v}} + A\cdot(t^{aircraft} + \frac{L}{\hat{v}}) + B + spline(t^{aircraft} + \frac{L}{\hat{v}})$$

$$t^{meas} = (A+1)\cdot(t^{aircraft} + \frac{L}{\hat{v}}) + B + spline(t^{aircraft} + \frac{L}{\hat{v}})$$

It would be difficult to solve the last nonlinear equation directly. Instead, we will use the fact that spline approximates the slow component (random walk) of clock drift and therefore in the first approximation we can simply ignore it:

$$t^{aircraft} + \frac{L}{\hat{v}} = \frac{t^{meas} - B}{A+1}$$

Finally, we can synchronize station measurements by applying the following trasformation to measured time values:

$$(Eq. 3): t^{aircraft} + \frac{L}{\hat{v}} \triangleq t^{sync} = \frac{t^{meas} - B - spline(\frac{t^{meas} - B}{A+1})}{A+1}$$

# 3. Solution steps

## `1. Synchronize good stations.ipynb`

Here I computed parameters of the wave velocity model, sensors positions and clock shifts for 35 good stations out of 45 marked as 'good'. A good station shouldn't have visible clock drift or random walk and should have combined measurements (pairs) with several other good stations (we should be able to optimize its location).

For 35 selected stations a subset of points (20,000 per station) was prepared to reduce computation complexity. On this subset averaged L1 loss $|L_1 - L_2 - \hat{v}\cdot(t_1^{meas} + t_1^{shift} - t_2^{meas} - t_2^{shift})|$ was minimized. Here $L_1$ and $L_2$ are distances from an aircraft to two stations, $t_1^{shift}$ and $t_2^{shift}$ are constant stations time shifts.


## `2. Add station 150 using round2_training1 dataset.ipynb`

I decided to add station 150 separately because it has only one pair with a good station (station 14) from 35 synchronized. It's not enough to update station's location, so I didn't want to include it to the 35 good stations list. So, I found clock shift for station 150 and used its default location.

## `3. Synchronize all other stations.ipynb`

Starting from the stations closest to 36 synchronized, I was able to synchronize more than 200 other stations. Initially I considered only candidates having at least 3 pairs with synchronized stations each of which having at least 1000 points. By the end I had to reduce these constrains in order to add more stations.

This notebook contains many runs of synchronization of new stations. After each run new stations' measurements were corrected and these stations are used further as synchronized. Many stations showing time gaps or big estimated median error were checked visually before adding to the list. So, even stations with time gaps were added (and their gaps recorded) and used later for tracks prediction.

## `4. Predict and filter tracks.ipynb`

Tracks prediction is based on the algorithm I developed in the round 1 of the competition. The main idea is to use HuberRegression to fit points after solving multilateration equations. This method showed better accuracy on almost all training tracks in comparison with spline approximation. Also I used a brilliant graph-based filter developed by @richardalligier in the round 1 to filter points after solving multilateration equations by aircraft speed limit.

After this stage algorithm should produce about 71.5% coverage of test tracks with accuracy about 85m on the public leaderboard.

In order to improve accuracy further, I removed points with big error and added new points with small error. Splines for latitude and longitude as functions of timeAtServer were fitted for each track. Distance between points predicted earlier and splines gave me estimation of error in points. New points may be added to fill gaps in tracks. Error of new points depends on gap duration, so the best result on the public leaderboard was achieved by using gaps of 60s or less.


## Computation time

Synchronization time for good stations depends on initial values and may be between 30min and several hours. Synchronization of all stations took at least 8 hours on my machine. Prediction of test tracks required about 3 hours, while tracks filtering is pretty fast.

I'm sure there is a big potential in optimization especially for stations synchronization.

# 4. External code usage


 - Transformation of coordinates from WGS84 to Cartesian was taken from [https://competition.opensky-network.org/documentation.html];

 - Some useful functions (haversine_distance, numpylla2ecef) were taken from the solution of @richardalligier from round 1;

 - Functions related to graph filter to aircraft speed (precompute_distance, get_gtlongest, compute_gtgraph, filter_speedlimit) were also taken from the solution of @richardalligier from round 1 of the competition.
