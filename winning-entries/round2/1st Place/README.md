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

Using altitude dependence of refractive index <img src="svgs/0b700b6ef9752b739fe4ee8dc2925d28.svg?invert_in_darkmode" align=middle width=32.12352pt height=24.65759999999998pt/> from [1], velocity as a function of altitude can be written as follows: 

<p align="center"><img src="svgs/78ba6690fac5dc48d1b3aacfade2f2f3.svg?invert_in_darkmode" align=middle width=213.41924999999998pt height=33.583769999999994pt/></p>

, where <img src="svgs/3e18a4a28fdee1744e5e3f79d13b9ff6.svg?invert_in_darkmode" align=middle width=7.113876000000004pt height=14.155350000000013pt/> is the speed of light, <img src="svgs/2ad9d098b937e46f9f58968551adac57.svg?invert_in_darkmode" align=middle width=9.471165000000003pt height=22.831379999999992pt/> - altitude, <img src="svgs/0b700b6ef9752b739fe4ee8dc2925d28.svg?invert_in_darkmode" align=middle width=32.12352pt height=24.65759999999998pt/> - refractive index, <img src="svgs/2e5cace905a61fe431f7b898becb0be1.svg?invert_in_darkmode" align=middle width=18.881445000000006pt height=22.46574pt/> and <img src="svgs/61e84f854bc6258d4108d08d4c4a0852.svg?invert_in_darkmode" align=middle width=13.293555000000003pt height=22.46574pt/> - some constants.

Instead of integrating velocity each time, let's consider some constant (average) effective velocity: 
<p align="center"><img src="svgs/324d302c449c8b7a25e54fbe21a471f8.svg?invert_in_darkmode" align=middle width=143.40314999999998pt height=38.810145pt/></p>

<p align="center"><img src="svgs/7668dde8336ca86314c642afcfb541ab.svg?invert_in_darkmode" align=middle width=752.4626999999999pt height=42.92277pt/></p>

, where <img src="svgs/ddcb483302ed36a59286424aa5e0be17.svg?invert_in_darkmode" align=middle width=11.187330000000003pt height=22.46574pt/> - wave path, <img src="svgs/5a95dbebd5e79e850a576db54f501ab8.svg?invert_in_darkmode" align=middle width=16.023645000000005pt height=22.831379999999992pt/> and <img src="svgs/0f7cea0b89929faf20eda59174bc247f.svg?invert_in_darkmode" align=middle width=16.023645000000005pt height=22.831379999999992pt/> - initial and final altitudes of the wave path.

Finally, after inserting <img src="svgs/09167708b6619ba3b4d57545fe6e5937.svg?invert_in_darkmode" align=middle width=73.57482pt height=29.461410000000004pt/>, effective wave velocity will be:

<p align="center"><img src="svgs/0fbbdcce21393b8c1ab4ddd4b39b5e89.svg?invert_in_darkmode" align=middle width=379.61055pt height=39.53796pt/></p>

New wave velocity model shows 0.1m less average residual error in solving multilateral equations for 35 good stations (`1. Synchronize good stations.ipynb` notebook) than altitude independent constant effective wave velocity.

[1] R. Purvinskis et al. Multiple Wavelength Free-Space Laser Communications. Proceedings of SPIE - The International Society for Optical Engineering, 2003. 

## 2.2 Stations clock drift

Stations are synchronized when there is no clock drift, so measured time is equal to aircraft time + time-of-flight:

<p align="center"><img src="svgs/43580024997948d3e68c2ff8a486aa37.svg?invert_in_darkmode" align=middle width=150.28794pt height=33.629475pt/></p>

If station measurements have drift, then:

<p align="center"><img src="svgs/2550b51fa5c8c827afe0d44672968a2d.svg?invert_in_darkmode" align=middle width=315.3315pt height=33.629475pt/></p>

It's worth to notice here that drift is added at the moment of signal detection!

We have to have some already synchronized stations. Let's consider a synchronized station 1 and a drifted station 2.

<p align="center"><img src="svgs/1eed11d33672dd90ecde0d32e49238d6.svg?invert_in_darkmode" align=middle width=241.69529999999997pt height=33.629475pt/></p>

Considering <img src="svgs/31d18a2424dd7476a46822fd19f48a1b.svg?invert_in_darkmode" align=middle width=135.345375pt height=31.780980000000003pt/> and inserting corresponding equation for station 1, we get the final formula:

<p align="center"><img src="svgs/ace9190b24c492a962ba4f2c97120a65.svg?invert_in_darkmode" align=middle width=338.22855pt height=33.629475pt/></p>

### Clock drift approximation

Clock drift consists of linear drift and random walk. Clock drift was approximated by a sum of a linear function and a spline:

<p align="center"><img src="svgs/40d75a8025d335645062e323b7d5e5ea.svg?invert_in_darkmode" align=middle width=225.20685pt height=16.438356pt/></p>

So,

<p align="center"><img src="svgs/e1dc6ec661976b0794dd68ee39114674.svg?invert_in_darkmode" align=middle width=504.68385pt height=33.629475pt/></p>

<p align="center"><img src="svgs/0a8b9ea411938f2f635b8208b0cdaafb.svg?invert_in_darkmode" align=middle width=433.3411499999999pt height=33.629475pt/></p>

It would be difficult to solve the last nonlinear equation directly. Instead, we will use the fact that spline approximates the slow component (random walk) of clock drift and therefore in the first approximation we can simply ignore it:

<p align="center"><img src="svgs/316867f5e8fd4f771c3322e48f63a2d1.svg?invert_in_darkmode" align=middle width=187.61819999999997pt height=34.999305pt/></p>

Finally, we can synchronize station measurements by applying the following trasformation to measured time values:

<p align="center"><img src="svgs/a76e37ce12efe81e832da24b771003e1.svg?invert_in_darkmode" align=middle width=439.15409999999997pt height=41.067015pt/></p>

# 3. Solution steps

## `1. Synchronize good stations.ipynb`

Here I computed parameters of the wave velocity model, sensors positions and clock shifts for 35 good stations out of 45 marked as 'good'. A good station shouldn't have visible clock drift or random walk and should have combined measurements (pairs) with several other good stations (we should be able to optimize its location).

For 35 selected stations a subset of points (20,000 per station) was prepared to reduce computation complexity. On this subset averaged L1 loss <img src="svgs/5d6189b601b6b15604e05866ec8efa5c.svg?invert_in_darkmode" align=middle width=331.86565499999995pt height=31.780980000000003pt/> was minimized. Here <img src="svgs/929ed909014029a206f344a28aa47d15.svg?invert_in_darkmode" align=middle width=17.739810000000002pt height=22.46574pt/> and <img src="svgs/4327ea69d9c5edcc8ddaf24f1d5b47e4.svg?invert_in_darkmode" align=middle width=17.739810000000002pt height=22.46574pt/> are distances from an aircraft to two stations, <img src="svgs/ed3d6a7ea65a223451a604b6372c870a.svg?invert_in_darkmode" align=middle width=37.15305pt height=31.780980000000003pt/> and <img src="svgs/87d5c3931435576d25da229aa5fbd5f3.svg?invert_in_darkmode" align=middle width=37.15305pt height=31.780980000000003pt/> are constant stations time shifts.


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
