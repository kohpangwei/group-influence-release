#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)/src
DATASET_IDS="ortho2 gauss4"

for DATASET_ID in $DATASET_IDS; do
  python2 src/scripts/counterexamples.py \
    --out-dir ./output \
    --dataset-id $DATASET_ID \
    --no-intermediate-results
done
