# Setup

```
conda env create --file environment.yml
```

In the Makefile, you might want change `NJOBS_TRAIN_FILTER` to the number of cores you have. You can adjust accordingly the number of hyperparameters tested `N_ITER`. Here, with the default parameters, it takes ~90 minutes to run. You can reduce `N_ITER` to speed-up things at the cost of a possibly worse prediction.

All the data files should in a folder ./Data.

For instance, for round1_competition, we have:

```
./Data/round1_competition_result/round1_competition_result.csv
./Data/round1_competition/sensors.csv
./Data/round1_competition/round1_competition.csv
```

Where `round1_competition_result.csv` is the empty result file.

Then to compute the prediction without filtering, run:

```
make result
```

This should give you a ~ 35.5 meters error with a 0.881 coverage.


Then to compute the prediction with filtering, run (takes a lot of time):

```
make filteredresult
```

This should give you a ~ 25-26 meters error with a 0.5 coverage.

# How it works

## Step 1 Compute sensors positions, time shift and wave speed: `sensorsparams.py`
Find the radio wave speed, and the latitude, longitude, altitude and time shift of each sensors by minimizing the multi-lateration error. So, if we have p points, we minimize:

$$
\sum_{i=1}^p \sum_{j=1}^{n^{\text{measurement}}_i} \sum_{k=j+1}^{n^{\text{measurement}}_i} | d(\mathrm{sensor}_k,pos_i) -d(\mathrm{sensor}_j,pos_i)  - (t^{\text{measurement}}_{i,k}+\mathrm{shift}_k-t^{\text{measurement}}_{i,j}-\mathrm{shift}_j)C |
$$


Fun fact, at the end of the training, we have a radio wave speed consistant with a 1.000216 refractive index which is somewhat close to the 1.0003 refractive index for visible light in the air. I did not find any data about the expected refractive index in the air for 1090MHz.

## Step 2 Estimate aircraft position with multilateration error: `aircraftpos.py`
Here we only consider the points with at least 4 measurements. As a poor man's geometry sanity check, two close sensors only count as one.

In this step, we consider an altitude equal to baroAltitude, and we perform a gradient descent to find latitude and longitude minimizing the multi-lateration error minimized in Step 1 plus a continuity soft constraint.

For $p_{\text{traj}}$ consecutive points in the same trajectory, this continuity soft constraint is:
$$
\sum_{i=1}^{p_{\text{traj}}-1} \mathrm{max}(d(\mathrm{pos_{i}},\mathrm{pos_{i+1}})-(t^{\mathrm{timeAtServer}}_{i+1}-t^{\mathrm{timeAtServer}}_{i}) \mathrm{speedlimit},0)
$$.


It penalizes points that are too far away compared with their temporal difference. I observed that this soft constraint reduced the average multi-lateration error.

## Step 3 Discard bad positions and fit a spline on aircraft positions: `splineaircraftpos.py`
In this step, we discard points with a large multi-lateration error. However, there are outliers with a low multi-lateration error, so we discard them using a speed limit. I did not want to write an algorithm for discarding outliers, these are painful to write and tweak. So instead, for each trajectory, I build a directed graph. In this graph, there is a directed edge from $i$ to $j$ (noted $i\rightarrow j$), if $t^{\mathrm{timeAtServer}}_i$ < $t^{\mathrm{timeAtServer}}_j$ and $d(\mathrm{pos_{i}},\mathrm{pos_{j}}) \leqslant (t^{\mathrm{timeAtServer}}_j-t^{\mathrm{timeAtServer}}_i)\mathrm{speedlimit}$. Then I keep the points that are in the longest path in this graph. Fun fact, the selected points should be a clique because with the triangular inequality we have: $(i\rightarrow j ~\text{and}~ j\rightarrow k) \Rightarrow i\rightarrow k$.

Then we fit a cubic spline on the remaining points, this spline is stored in dictionary: dict[aircraft]=SmoothedTraj, where SmoothedTraj contains the spline and the points that were not discarded.

## Step 4 Add features to the estimated positions: `splineaircraftposwithfeature.py`
Using the spline we compute the interpolated on all the points, even those discarded. We compute statistics on these points like the speed, the curvature. These statistics are computed on a temporal range around each point that was not discarded. The prediction error made around each of these points is also computed. The idea is to predict this error and keep only the temporal ranges that are associated with a low predicted error.


## Step 5 Learn a model predicting the error made: `learnfilter.py`
Here we want to fit a model that keeps the temporal ranges with a low prediction error. As the range of the prediction error is really large and "unbalanced" (a small numbers of errors are really large), I decided to compute the quantile of the error, and then used this quantile as a probability label of being a large error. So using LightGBM, I trained a gradient boosting classifier with the cross-entropy on these continuous labels.

## Step 6 Write the prediction file: `writeprediction.py`
In this step, using the model learned in the previous step, we can rank each temporal ranges. Using this, we consider only the n temporal ranges with the smallest probability, then among these n ranges, we keep those that are part of a temporally contiguous sequence of 10 ranges. This "smoothen" across time the criteria used to discard the ranges.

The n is selected as the smallest value that allows us to comply with the coverage requirement.