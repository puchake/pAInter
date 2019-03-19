#!/bin/bash

if [[ -z "$1" ]]
then
      echo "You have to specify type of pytorch installation. CUDA will install pytorch with CUDA support and CPU will install pytorch without it."
      exit 1
fi

python3 -m venv venv
source venv/bin/activate

if [[ "$1" == "CPU" ]]
then
    pip install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp35-cp35m-linux_x86_64.whl
    pip install torchvision
elif [[ "$1" == "CUDA" ]]
then
    pip install torch torchvision
else
    echo "Unsupported value of first argument = $1. Supported values are CPU and CUDA."
    exit 1
fi

pip install -r requirements.txt
deactivate