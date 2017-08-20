#!/bin/bash

if [[ ! -d "data" ]] ; then
    mkdir data
fi

cd data

wget http://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data
wget http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
wget http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.names
