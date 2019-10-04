#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)/src

python2 src/scripts/data_valuation.py \
  --data-dir ./data \
  --out-dir ./output \
  --dataset-id reduced_cdr \
  --sample-weights
