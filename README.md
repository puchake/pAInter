# p**AI**nter

p**AI**nter is a tool which enables colouring of grayscale images by using GAN. It is done as an assignment for
"Systems with machine learning" at GUT.

## Requirements

Your system has to have Python 3 installed and available as `python3` command. If you want to use another python simply
change it in `setup_env.sh` script.

## Setup

Simply run `setup_env.sh` or `setup_env.bat`. It will create virtual env in `venv` folder with `python3` present in the system. It will
also install requirements specified in `requirements.txt` in this virtual env.

## Dataset download

For sake of simplicity, right now we use CIFAR-10 dataset. To automatically download python version of this set run
`download_dataset.sh`. It will create `data/dataset_raw` directories if they are not already present and extract
contents of downloaded archive there.