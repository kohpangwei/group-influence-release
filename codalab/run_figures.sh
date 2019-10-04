#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)/src
export JUPYTER_PATH=$JUPYTER_PATH:$(pwd)/src
export INFLUENCE_PLOT_DIR=$(pwd)/plots
export INFLUENCE_OUT_DIR=$(pwd)/output

jupyter nbconvert paper-figures.ipynb --to notebook --execute
