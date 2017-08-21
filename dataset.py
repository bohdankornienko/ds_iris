import csv
import os.path
import numpy as np

class_names = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
class_ids = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}


class DataSet:
    def __init__(self, data_set_path):
        if not os.path.exists(data_set_path):
            RuntimeError('Invalid path to data set: ' + data_set_path)

        self._data_set_path = data_set_path
        self._is_end_of_data = True
        self._request_new_reader()

    @property
    def is_end_of_data(self):
        return self._is_end_of_data

    def next_batch(self, batch_size=100):
        self._request_new_reader()

        features = []
        labels = []

        for i in range(batch_size):
            try:
                row = next(self._iris_reader)

                if not row:
                    self._is_end_of_data = True
                    break
            except StopIteration:
                self._is_end_of_data = True
                break

            features_vec = row[:4]
            if 0 == len(features_vec):
                continue

            features_vec = [float(x) for x in features_vec]
            label = class_ids[row[4]]

            features.append(features_vec)
            labels.append(label)

        return np.array(features), np.array(labels)

    def _request_new_reader(self):
        if not self._is_end_of_data:
            return

        self._is_end_of_data = False

        csvfile = open(self._data_set_path, 'r')
        self._iris_reader = csv.reader(csvfile)
