#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)/src
DATASET_ID=$1

python2 src/scripts/subsets_infl_logreg.py \
  --data-dir ./data \
  --out-dir ./output \
  --dataset-id $DATASET_ID \
  --subset-choice-type range \
  --subset-min-rel-size 0.0025 \
  --subset-max-rel-size 0.25 \
  --num-subsets 100 \
  --max-memory 100000000 \
  --no-intermediate-results
