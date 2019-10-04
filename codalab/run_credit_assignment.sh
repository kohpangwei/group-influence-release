#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)/src

python2 src/scripts/credit_assignment.py \
  --data-dir ./data \
  --out-dir ./output \
  --dataset-id mnli \
  --no-intermediate-results
