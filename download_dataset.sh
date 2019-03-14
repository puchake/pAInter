#!/bin/bash

wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -O cifar-10-python.tar.gz
mkdir -p data/dataset_raw
tar xvzf cifar-10-python.tar.gz --strip-components=1 -C data/dataset_raw/
rm cifar-10-python.tar.gz