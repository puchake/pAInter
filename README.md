# p**AI**nter

p**AI**nter is a tool which enables colouring of grayscale images by using GAN. 
It is done as an assignment for "Systems with machine learning" at GUT.

## Requirements

Your system has to have Python 3.5 installed and available as `python3` command 
(on Windows you can achieve that with `mklink` command). If you want to use 
another python you have to change `setup_env.sh`/`setup_env.bat` to install
appropriate version of PyTorch.

## Setup

Simply run `setup_env.sh` or `setup_env.bat`. Both of these scripts require 
passing 1 positional argument which can have value of "CUDA" or "CPU". Depending
on the value of this argument, PyTorch with/without GPU support will be 
installed. These scripts will create virtual env in `venv` folder with `python3`
present in the system. It will also install requirements specified in 
`requirements.txt` in this virtual env next to the PyTorch mentioned above.

## Dataset download

For sake of simplicity, right now we use CIFAR-10 dataset. To automatically 
download python version of this set run `download_dataset.sh`. It will create 
`data/dataset_raw` directories if they are not already present and extract
contents of downloaded archive there.