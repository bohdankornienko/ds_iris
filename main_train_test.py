from model import TensorFlowModel
from dataset import DataSet

import os
import shutil


def main():
    if os.path.exists('./iris'):
        shutil.rmtree('./iris')

    data_set = DataSet('data/iris.data.train')

    iris = TensorFlowModel(input_size=4, learning_rate=0.005, classes_count=3, n_epoch=500, batch_size=100, skip_step=20)
    iris.build_graph()
    iris.train_model(data_set=data_set)

    data_set = DataSet('data/iris.data.test')
    iris.test_model(data_set=data_set)

if __name__ == '__main__':
    main()
