import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

from libs import geo

STD_THRESHOLD = 120.07552229629951 #100

PREPROCESSED_DIFFS_FILENAME = 'round2_mlat/preprocessed_diffs.npz'

class SensorSynchronization:
    def __init__(self, preprocessing_filename=None):
        if preprocessing_filename is None:
            preprocessing_filename = PREPROCESSED_DIFFS_FILENAME
        data = np.load(preprocessing_filename, allow_pickle=True)['data'][()]

        self.sensor_pairs = dict()

        # load all preprocessed diffs for all know sensor pairs and build KD-trees for fast search
        for pair, values in tqdm(data.items(), desc='build KDTrees', total=len(data)):
            #[t1, t2, timeAtServer, TDoA[i], TDoA[j]]
            times = values[:, 0:2].copy()
            TDoA = values[:, 3:5]

            #v = (receiver_time[i] - receiver_time[j]) - (TDoA[i] - TDoA[j])
            diff = (times[:, 0] - times[:, 1]) - (TDoA[:, 0] - TDoA[:, 1])
            #diff = TDoA[:, 0] - TDoA[:, 1] + times[:, 1] - times[:, 0]

            self.sensor_pairs[pair] = {'time': times,
                                       'diff': diff,
                                       'tree': cKDTree(times)}

    @staticmethod
    def biggest_component(G):
        """
        Return one biggsst connected component from graph G
        :param G: Sensor connection graph
        :return:
        """
        node = len(G)
        component = np.zeros(node, dtype=np.int)

        component_idx = 0
        for i in range(node):
            if component[i] != 0:
                continue

            if len(G[i]):
                component_idx += 1
                component[i] = component_idx
                Q = [i]

                while len(Q):
                    u = Q.pop(0)
                    for v in G[u]:
                        if component[v] == 0:
                            component[v] = component_idx
                            Q.append(v)

        c_t = list(filter(lambda v: v > 0, component))
        unique_components, counts = np.unique(c_t, return_counts=True)
        return np.argwhere(component == unique_components[np.argmax(counts)])


    def synch(self, sensor_ids, sensor_times, points_count=25):
        """
        Precise time synchronization of all sensors for one point
        :param sensor_ids: used senso ids
        :param sensor_times: raw time measurements
        :param points_count: number of samples for each sensor pair from train dataset for synchronization
        :return:
        """
        n = len(sensor_ids)

        if n < 4:
            return np.array([]), np.array([])
        sensor_ids = np.array(sensor_ids, dtype=np.int)
        sensor_times = np.array(sensor_times, dtype=np.float64) / 1e9

        order = np.argsort(sensor_ids)
        sensor_ids = sensor_ids[order]
        sensor_times = sensor_times[order]

        A = []
        B = []
        G = [[] for _ in range(n)]

        # enumerate all possible sensor pairs
        for i in range(n):
            for j in range(i+1, n):
                # select closest measurements in time for current sensor pair
                time, diff = self._closest_measurements(sensor_ids[i], sensor_ids[j], sensor_times[i], sensor_times[j], n=points_count)
                if time is None:
                    continue

                # plt.scatter(time[:, 0], time[:, 1])
                # plt.scatter([sensor_times[i]], [sensor_times[j]])
                # plt.show()

                # filter outlier measurements
                time, diff = self._check_and_filter_measurements(time, diff, np.array([sensor_times[i], sensor_times[j]], dtype=np.float64))
                if time is not None:
                    G[i].append(j)
                    G[j].append(i)

                    row = np.zeros((diff.shape[0], 2*n) , dtype=np.float64)
                    row[:, i] = 1
                    row[:, j] = -1

                    row[:, n + i] = time[:, 0]
                    row[:, n + j] = -time[:, 1]

                    A.append(row)
                    B.append(diff)

        if len(A) < 1:
            return np.array([]), np.array([])

        A = np.concatenate(A, axis=0)
        B = np.concatenate(B, axis=0)

        # lr = LinearRegression(fit_intercept=False)
        # lr.fit(A, B)

        # Calculate a linear regression model of time shifts for each sensor.
        # shift_i = k*t_i + b, where:
        #   shift_i - time shift of i-th sensor in seconds
        #   k - time offset factor for i-th sensor
        #   b - bias
        coef,_,_,_ = np.linalg.lstsq(A, B, rcond=None)
        d = A@coef - B
        residues = np.linalg.norm(A@coef - B)
        std = np.std(d) * geo.LIGHT_SPEED
        # print(std)
        #
        # plt.scatter(np.arange(0, len(d)), d * geo.LIGHT_SPEED)
        # plt.show()

        # check if we have a good solutions for all sensors
        if std>100:
            return np.array([]), np.array([])

        shift = [coef[n:],
                 coef[0:n]]

        shift = np.array(shift, dtype=np.float64).transpose()
        # used_sensors = np.max(np.abs(A[:, :n]), axis=0) > 0
        used_sensors = np.zeros((n,), dtype=bool)
        used_sensors[self.biggest_component(G)] = True

        # return measurements only for 4 or more synchronized measurements
        if np.sum(used_sensors) < 4:
            return np.array([]), np.array([])

        res_sensor_ids = sensor_ids[used_sensors]
        res_sensor_times = []
        # sensor time synchronization itself calculated on sensor shift model
        for i in range(n):
            if used_sensors[i]:
                # timestamp_diff = timestamp * shift[sensor_id, 0] + shift[sensor_id, 1]
                # meas_time = timestamp - timestamp_diff

                timestamp_diff = sensor_times[i]*shift[i, 0] + shift[i, 1]
                fixed_sensor_t = sensor_times[i] - timestamp_diff
                res_sensor_times.append(fixed_sensor_t)

                # fixed_sensor_t = sensor_times[i] + (sensor_times[i]*shift[i,0] + shift[i, 1])
                # res_sensor_times.append(fixed_sensor_t)

        return res_sensor_ids, res_sensor_times

    def _closest_measurements(self, id1, id2, t1, t2, n=25):
        """
        Return n most revelant measurements from train dataset for sensor pair (id1, id2) in time position (t1, t2)
        """
        key = (id1, id2)
        if key not in self.sensor_pairs:
            return None, None

        curr_pair = self.sensor_pairs[key]
        if curr_pair['time'].shape[0] < n:
            return None, None

        _, closest_meas_idx = curr_pair['tree'].query(np.array([t1, t2]), k=n)
        return curr_pair['time'][closest_meas_idx], curr_pair['diff'][closest_meas_idx]

    def _check_and_filter_measurements(self, time, diff, curr_point):
        """
        Filter sensor measurements outliers from train dataset
        """
        A = np.concatenate([time, np.ones((time.shape[0],1), dtype=np.float64)], axis=1)
        coefs,residues,_,_ = np.linalg.lstsq(A, diff, rcond=None)

        #np.std(diff - d2)
        d2 = A @ coefs
        err = np.abs(diff - d2)
        k = 2
        order = np.argsort(err)[0:-k]

        time = time[order].copy()
        diff = diff[order].copy()

        A = np.concatenate([time, np.ones((time.shape[0], 1), dtype=np.float64)], axis=1)
        coefs, residues, _, _ = np.linalg.lstsq(A, diff, rcond=None)

        # plt.scatter(time[:, 0], (diff - np.median(diff))*geo.LIGHT_SPEED)
        # plt.show()

        # d2 = A @ coefs
        # plt.scatter(time[:, 0], (diff - d2)*geo.LIGHT_SPEED)
        # plt.show()

        std = np.sqrt(residues/A.shape[0])*geo.LIGHT_SPEED
        std_thr = STD_THRESHOLD
        # std_thr = 1000
        if std > std_thr:
            return None, None

        A = np.vstack([time[:, 0], np.ones(len(time))]).transpose()
        B = time[:, 1]
        coefs, residues, _, _ = np.linalg.lstsq(A, B, rcond=None)
        p = curr_point[0]*coefs[0] + coefs[1]
        err_m = (p - curr_point[1])*geo.LIGHT_SPEED
        if err_m > 800e3:
            return None, None
        #d = A@coefs - time[:, 1]

        # plt.scatter(time[:, 0], d*geo.LIGHT_SPEED)
        # plt.scatter([curr_point[0]], [(p - curr_point[1])*geo.LIGHT_SPEED])
        # plt.show()

        return time, diff