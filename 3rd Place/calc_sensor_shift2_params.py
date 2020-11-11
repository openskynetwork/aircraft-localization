from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger, ScreenLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

import numpy as np
import math
from sklearn.linear_model import LinearRegression, Lasso, Ridge

from libs import geo, data

LOAD_PREV = False


sensor_coords = data.get_sensor_coords_cartesian()
sensors = np.max(list(sensor_coords.keys()))
diffs = np.load("shift_coefs_diffs.npz", allow_pickle=True)["diffs"][()]

def black_box_function(repeats,
                       bins,
                       half_window,
                       cut_count,
                       break_dist,
                       ridge_alpha
                       ):

    repeats = int(round(repeats))
    bins = int(round(bins))
    half_window = int(round(half_window))

    def peak_by_hist(array):
        array = np.array(array)

        for i in range(repeats):
            counts, edges = np.histogram(array, bins)
            max_pos = np.argmax(counts)
            pos1 = max(0, max_pos - half_window)
            pos2 = min(len(edges) - 1, max_pos + 1 + half_window)
            array = array[np.logical_and(array >= edges[pos1], array <= edges[pos2])]
            if edges[pos2]-edges[pos1] <= break_dist:
                break

        return np.median(array)
        # return (edges[max_pos] + edges[max_pos + 1]) / 2

    try:
        A = []
        B = []

        for k, v in diffs.items():
            if len(v) < cut_count:
                continue

            id1, id2 = k
            row = np.zeros(sensors + 1, dtype=np.float32)
            row[id1] = 1
            row[id2] = -1
            A.append(row)
            B.append(peak_by_hist(v))

        A = np.array(A)
        B = np.array(B)

        # lr = Lasso(fit_intercept=False, alpha=1e-9)
        # lr = LinearRegression(fit_intercept=False)
        lr = Ridge(fit_intercept=False, alpha=ridge_alpha, random_state=42, solver="svd")
        lr.fit(A, B)
        shift = lr.coef_

        score = math.sqrt(np.mean(np.square(A @ shift - B))) * geo.LIGHT_SPEED
        return -score

    except Exception as e:
        print(e)
        return -1e100

# Bounded region of parameter space
pbounds = {
    "repeats": (1, 10),
    "bins": (2, 200),
    "half_window": (0, 5),
    "cut_count": (0, 400),
    "break_dist" : (0, 1e-4),
    "ridge_alpha": (0, 1e-3)
}
start_points = [
    {
        "repeats": 3,
        "bins": 50,
        "half_window": 1,
        "cut_count": 50,
        "break_dist": 0,
        "ridge_alpha": 0
    },
    {
        "repeats": 3,
        "bins": 17,
        "half_window": 4,
        "cut_count": 395,
        "break_dist": 0,
        "ridge_alpha": 0
    },
    {
        "repeats": 3,
        "bins": 9,
        "half_window": 2,
        "cut_count": 333,
        "break_dist": 0,
        "ridge_alpha": 0
    },
    {
        "repeats": 6,
        "bins": 11,
        "half_window": 0,
        "cut_count": 382,
        "break_dist": 0,
        "ridge_alpha": 0
    },
    {
        "repeats": 5,
        "bins": 14,
        "half_window": 5,
        "cut_count": 342,
        "break_dist": 0,
        "ridge_alpha": 0
    },
    {
        'bins': 34,
        'break_dist': 2.9320173499588157e-06,
        'cut_count': 323,
        'half_window': 2,
        'repeats': 10,
        'ridge_alpha': 0.0003769905628035679
    },
    {
        'bins': 32,
        'break_dist': 1.9245350613262425e-05,
        'cut_count': 311,
        'half_window': 3,
        'repeats': 5,
        'ridge_alpha': 0.0002026956982888747
    },
    { # 11.9392
        'bins': 30,
        'break_dist': 2.9705053123740167e-06,
        'cut_count': 309,
        'half_window': 3,
        'repeats': 6,
        'ridge_alpha': 0.0005692718745725846
    },
    { #11.91
        'bins': 31,
        'break_dist': 3.343855359025361e-06,
        'cut_count': 307,
        'half_window': 4,
        'repeats': 6,
        'ridge_alpha': 0.0006696012127733874
    }
]

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=42,
)
for point in start_points:
    optimizer.probe(
        params=point,
        lazy=True,
    )

if LOAD_PREV:
    load_logs(optimizer, logs=["./calc_shifts_params.json"]);

logger = JSONLogger(path="./calc_shifts_params.json")
scrlogger = ScreenLogger()
optimizer.subscribe(Events.OPTIMIZATION_STEP, scrlogger)
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

optimizer.maximize(
    init_points=500,
    n_iter=500,
)

print(optimizer.max)

